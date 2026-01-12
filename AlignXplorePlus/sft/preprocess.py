import os
import json
import random
import torch
import logging
import argparse
from tqdm import tqdm
from collections import defaultdict

from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import gather_object

from torch.utils.data import DataLoader, Dataset


# --- 配置 ---
class Config:
    STRONG_MODEL_PATH = "checkpoints/Qwen3-1.7B-SFT"
    INPUT_FILE = None  # 输入数据，每行为一个用户JSON
    OUTPUT_FILE = None
    BATCH_SIZE_PER_DEVICE = 4
    TOP_K_PERCENTAGE = 0.20  # 保留前20%
    MAX_SEQUENCE_LENGTH = 16384


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name. [AlignX, MIND, Amazon]",
    )
    return parser.parse_args()


# --- 日志设置 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- 数据集类 ---
class UserHistoryDataset(Dataset):
    def __init__(self, data_file):
        self.data = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.accelerator = Accelerator()

        # 在所有进程上加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.STRONG_MODEL_PATH,
            trust_remote_code=True,
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 加载模型
        self.strong_model = self._load_model(config.STRONG_MODEL_PATH)
        # self.weak_model = self._load_model(config.WEAK_MODEL_PATH)

        self.strong_model.eval()
        # self.weak_model.eval()

    def _load_model(self, model_path):
        """使用 bfloat16 加载模型以节省显存"""
        # device_map="auto" 让 accelerate 自动处理模型分片
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": self.accelerator.device},  # 每个进程加载到自己的device
            trust_remote_code=True,
        )
        return model

    def format_and_tokenize_batch(self, batch_data):
        """
        格式化一个batch的数据，生成长文本并记录关键信息
        """
        formatted_texts = []
        metadata_list = []

        for user_idx, user_data in enumerate(batch_data):
            full_sequence = "Given a sequence of user preference records, each containing:\n- A prompt text enclosed in <prompt> tags (may be empty)\n- Two options marked with <option_A> and <option_B> tags\n- User's preference (A or B) enclosed in <preference> tags\n\nYour task is to predict each preference by analyzing patterns from all previous records in the sequence."
            user_metadata = {
                "uid": user_data["uid"],
                "original_histories": [],
                "pref_indices": [],
            }

            user_data["behaviors"] = user_data["behaviors"][-100:]

            for history_idx, item in enumerate(user_data["behaviors"]):
                p = item["query"]
                options = [item["chosen"], item["rejected"]]
                # 随机打乱 A 和 B
                random.shuffle(options)
                rA, rB = options

                # 确定正确的偏好是 A 还是 B
                q = "A" if rA == item["chosen"] else "B"

                # 构建单次交互的文本片段
                # 使用特殊分隔符可以帮助模型更好地理解结构
                segment = f"<prompt>{p}</prompt><option_A>{rA}</option_A><option_B>{rB}</option_B><preference> "
                postfix = " </preference>|SEP|"

                # 在拼接前记录当前序列长度，用于计算偏好token的位置
                prefix_len = len(self.tokenizer.encode(full_sequence + segment))

                full_sequence += segment + q + postfix  # 添加 preference 和 EOS

                # 记录偏好token的索引（在token化后）
                # 注意：空格作为最后一个字符时会单独编码成一个token，id为220，而如果后续加入了q，则会和q合并成一个token，因此q在整体编码的token_ids中的坐标需要-1
                user_metadata["pref_indices"].append(prefix_len - 1)

                # 保存原始历史记录和映射关系，以便重组
                item["query"] = "Please recommend some movies to me."
                user_metadata["original_histories"].append(item)

            user_metadata["sequence"] = full_sequence

            formatted_texts.append(full_sequence)
            metadata_list.append(user_metadata)

        # Tokenize整个batch
        inputs = self.tokenizer(
            formatted_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.MAX_SEQUENCE_LENGTH,
        )
        return inputs, metadata_list

    def get_log_probs(self, model, inputs):
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            # 计算 log softmax
            log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs

    def run(self):
        """主执行流程"""
        dataset = UserHistoryDataset(self.config.INPUT_FILE)

        def collate_fn(batch):
            return batch

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE_PER_DEVICE,
            shuffle=False,  # 保持顺序以便于重组
            collate_fn=collate_fn,
        )

        dataloader = self.accelerator.prepare(dataloader)

        all_signals = []

        # 使用tqdm在主进程上显示进度条
        progress_bar = tqdm(
            dataloader,
            disable=not self.accelerator.is_main_process,
            desc=f"Processing batches {self.config.INPUT_FILE}",
        )

        for batch in progress_bar:
            # 1. 格式化和Tokenize
            # 这一步在每个进程中独立完成，但由于batch是同步的，所有进程处理的是同一批用户
            inputs, metadata_list = self.format_and_tokenize_batch(batch)

            try:
                inputs_on_device = {
                    k: v.to(self.accelerator.device) for k, v in inputs.items()
                }
            except AttributeError:
                # 如果 inputs 里的某个 value不是 tensor (例如，一个列表)，这个会报错。
                # 但对于 tokenizer 的标准输出来说，这不应该发生。
                # 这是一个健壮性检查。
                self.accelerator.print(
                    f"Warning: A value in the tokenizer output is not a tensor. Skipping batch. Keys: {inputs.keys()}"
                )
                continue

            # 2. 并行模型推理 (使用已在正确设备上的 inputs_on_device)
            strong_log_probs = self.get_log_probs(self.strong_model, inputs_on_device)

            # 3. 计算信号强度
            # batch_signals = []
            input_ids = inputs_on_device["input_ids"]
            user_batch = []

            for i, user_meta in enumerate(metadata_list):  # 遍历batch中的每个用户
                user_data = []
                for j, pref_idx in enumerate(user_meta["pref_indices"]):
                    # 确保索引在序列长度内 (因为可能被截断)
                    if pref_idx >= input_ids.shape[1] - 1:
                        continue

                    # 获取 preference token (A or B) 的 ID
                    pref_token_id = input_ids[i, pref_idx]  # No need for + 1

                    # Debug
                    # if i == 0:
                    #     print(self.tokenizer.decode(pref_token_id))

                    # 提取 logprob
                    logprob_m1 = strong_log_probs[i, pref_idx - 1, pref_token_id].item()

                    # 记录信号和其来源
                    user_data.append(
                        {
                            "item": user_meta["original_histories"][j],
                            "logprob1": logprob_m1,
                            "pref_idx": pref_idx,
                            "hid": j,
                        }
                    )
                user_batch.append(
                    {
                        "uid": user_meta["uid"],
                        "history": user_data,
                    }
                )

            # 4. 分布式数据收集和筛选
            # all_signals_list = gather_object(batch_signals)
            all_user_list = gather_object(user_batch)
            # print(all_signals_list[0])
            # breakpoint()

            if self.accelerator.is_main_process:
                # all_signals_flat = [
                #     item for sublist in all_user_list for item in sublist
                # ]
                all_signals_flat = all_user_list
                # breakpoint()

                if not all_signals_flat:
                    continue

                all_signals.extend(all_signals_flat)
                # Saving all meta data
                # TODO: save after inference
                # with open(self.config.OUTPUT_FILE, "w", encoding="utf-8") as fm:
                #     for item in all_signals:
                #         fm.write(json.dumps(item, ensure_ascii=False) + "\n")
                with open(self.config.OUTPUT_FILE, "a+", encoding="utf-8") as fm:
                    for item in all_signals_flat:
                        fm.write(json.dumps(item, ensure_ascii=False) + "\n")
                progress_bar.set_description(
                    f"Processing batches (user saved: {len(all_signals)})"
                )


def main():
    config = Config()
    args = parse_args()
    dataset = args.dataset
    config.INPUT_FILE = f"data/{dataset}/train.jsonl"
    config.OUTPUT_FILE = f"{dataset}_with_signal.jsonl"

    # 等待所有进程，确保文件已创建
    Accelerator().wait_for_everyone()

    processor = DataProcessor(config)
    processor.run()


if __name__ == "__main__":
    # 使用 accelerate launch 启动脚本
    # accelerate launch --num_processes=8 your_script_name.py
    main()
