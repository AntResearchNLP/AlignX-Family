import os
import json
import re
import argparse
import random
import math

import numpy as np

from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass, field

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm


# load template for Qwen2.5-7B-Instruct

with open(
    "prompt_templates/preference.txt",
    "r",
    encoding="utf-8",
) as f:
    system_prompt = f.read()


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Evaluation model name.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Dataset name.",
    )
    parser.add_argument("--thinking", action="store_true", help="Enable thinking")
    return parser.parse_args()


def build_prompt(
    tokenizer: AutoTokenizer,
    history: List[Dict[str, str]],
    prompt: str = "",
    response_a: str = "",
    response_b: str = "",
    enable_thinking: bool = False,
) -> str:
    """Build prompt for the model."""

    prompt_str = "\n\n".join(
        [
            f"=====Triple {idx+1}=====\n\n*QUERY:*\n {triple['query']}\n{'*'*5}\n*CHOSEN ITEM BY THE USER:*\n{triple['chosen']}\n{'*'*5}\n*REJECTED ITEM BY THE USER:*\n{triple['rejected']}"
            for idx, triple in enumerate(history)
        ]
    )
    persona = f"""The user's prefernce will be provided as a sequence of triples, where each triple is `(QUERY, CHOSEN ITEM BY THE USER, REJECTED ITEM BY THE USER)`.\n\n=====Interaction History=====\n{prompt_str}\n\n=====END====="""
    full_prompt = (
        f"Determine which response the user prefers based on the user's preferences. "
        f"Please output your selection below in a json format by filling in the placeholders in []:"
        f'{{"selection": "[Item A / Item B]"}}\n'
        # f"{system_prompt}\n"
        f"<Prompt>\n{prompt}\n</Prompt>\n\n"
        f"<Preference>\n{persona}</Preference>\n\n"
        f"<Item A>\n{response_a}\n</Item A>\n\n"
        f"<Item B>\n{response_b}\n</Item B>\n\n"
        f"Now, ONLY output your selection without any other text outside of this specified structure."
    )
    # 构建消息格式 (适用于 Qwen3)
    messages = [
        {"role": "system", "content": "Generate a task-specific response."},
        {"role": "user", "content": full_prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return text


@dataclass
class EvaluationConfig:
    """评估配置类"""

    input_file: str
    enable_thinking: bool = False
    model_name: str = "public_ckpts/Qwen/Qwen3-8B"
    system_prompt_path: str = (
        "prompt_templates/preference.txt"
    )
    tensor_parallel_size: int = 8
    max_model_len: int = 16384

    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    max_tokens: int = 16

    def __post_init__(self):

        path = self.input_file.replace("/upi_benchmark/", "/results/direct/")
        dir_name = os.path.dirname(path)
        file_name = os.path.basename(path)
        file_name = file_name.replace("_preference_", "_resultv2_")
        file_name = (
            f"direct_thinking_v2_{file_name}"
            if self.enable_thinking
            else f"direct_v2_{file_name}"
        )
        self.output_path: str = os.path.join(dir_name, file_name)
        self.rm = os.path.basename(self.model_name)
        self.performance_file: str = os.path.join(
            dir_name, f"performance_{self.rm}.txt"
        )
        if "gpt" in self.rm:
            self.max_tokens = 2048
        if self.enable_thinking:
            self.max_tokens = 4096


class PreferenceEvaluator:
    """偏好评估器类"""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.tokenizer = None
        self.llm = None
        self.system_prompt = ""

    def preprocess_data(self, test_data: List[Dict]) -> List[Dict]:
        """预处理数据"""
        processed_data = []

        def truncate_history(original_history, target):
            while True:
                prompt = target["query"]
                chosen = target["chosen"]
                rejected = target["rejected"]
                try:
                    prompt = build_prompt(
                        self.tokenizer,
                        original_history,
                        prompt,
                        chosen,
                        rejected,
                        self.config.enable_thinking,
                    )
                    length = len(self.tokenizer.encode(prompt, add_special_tokens=True))
                except Exception as e:
                    half = len(original_history) // 2
                    original_history = original_history[-half:]
                    continue

                if length < self.config.max_model_len:
                    return original_history
                if len(original_history) == 1:
                    return []
                half = len(original_history) // 2
                original_history = original_history[-half:]

        for idx, item in enumerate(test_data):
            item["history"] = truncate_history(item["history"], item["history"][-1])
            if not item["history"]:
                continue

            # transferring testing setting
            history = item["history"]
            target = item["target"]

            prompt = target["query"]
            chosen = target["chosen"]
            rejected = target["rejected"]
            flag = random.randint(0, 1)
            if flag:
                responseA = chosen
                responseB = rejected
                answer = "Item A"
            else:
                responseA = rejected
                responseB = chosen
                answer = "Item B"

            # history = truncate_history(history, target)
            formatted_input1 = build_prompt(
                self.tokenizer,
                history,
                prompt,
                responseA,
                responseB,
                self.config.enable_thinking,
            )
            # 创建新的数据项
            processed_item1 = {}
            processed_item1["idx"] = idx
            processed_item1["instruction"] = formatted_input1
            processed_item1["answer"] = answer
            processed_item1["target"] = target
            processed_item1["history"] = history
            processed_item1["label"] = flag

            processed_data.append(processed_item1)

        return processed_data

    def parse_model_output(self, output_text: str) -> Optional[str]:
        """解析模型输出"""
        try:
            # 尝试直接解析JSON
            parsed = json.loads(output_text)
            return parsed.get("selection")
        except json.JSONDecodeError:
            try:
                # 使用正则表达式提取JSON
                match = re.search(r"\{.*\}", output_text)
                if match:
                    json_str = match.group()
                    parsed_json = json.loads(json_str)
                    return parsed_json.get("selection")
            except (json.JSONDecodeError, AttributeError):
                try:
                    answer = output_text.strip("{}").split(":")[1].strip()
                    return answer
                except AttributeError:
                    print(f"Failed to parse output: {output_text}")
                    return None
        return None

    def parse_gpt_output(self, output_text: str) -> Optional[str]:
        """解析模型输出"""
        marker = "assistantfinal"
        if marker in output_text:
            # breakpoint()
            json_str = output_text.split(marker)[-1].strip()
            try:
                return json.loads(json_str).get("selection")
            except (json.JSONDecodeError, AttributeError):
                try:
                    # 使用正则表达式提取JSON
                    match = re.search(r"\{.*\}", output_text)
                    if match:
                        json_str = match.group()
                        parsed_json = json.loads(json_str)
                        return parsed_json.get("selection")
                except (json.JSONDecodeError, AttributeError):
                    return None
        return None

    def extract_confidence(self, logprobs_content: List[Dict]) -> float:
        """
        A -> Positive; B -> Negative
        """

        def normalize_token(tok: str) -> str:
            return tok.replace("Ġ", "").replace("Ċ", "\n")

        confidence_of_A = None
        confidence_of_B = None

        # print(logprobs_content)

        for item in reversed(logprobs_content):
            for k, v in item.items():
                if normalize_token(v.decoded_token).strip() == "A":
                    logprob_of_A = v.logprob
                    # 第4步：转换成标准概率 (0-1)
                    confidence_of_A = math.exp(logprob_of_A)
                if normalize_token(v.decoded_token).strip() == "B":
                    logprob_of_B = v.logprob
                    confidence_of_B = math.exp(logprob_of_B)
            if confidence_of_A is not None or confidence_of_B is not None:
                break
        if confidence_of_A is None and confidence_of_B is None:
            # 打印所有 token 的信息，便于调试
            print("\n--- 所有生成Token的详细信息 ---")
            return 0.5
        return confidence_of_A or (1 - confidence_of_B)

    def process_outputs(self, data: List[Dict], outputs: List[Any]) -> List[Dict]:
        """处理模型输出"""
        for idx, output in enumerate(outputs):
            output_text = output.outputs[0].text
            output_logprobs = output.outputs[0].logprobs

            data[idx]["original_output_text"] = output_text

            if "gpt" in self.config.rm:
                selection = self.parse_gpt_output(output_text)
            else:
                selection = self.parse_model_output(output_text)
            # breakpoint()
            confidence = self.extract_confidence(output_logprobs)
            if selection:
                data[idx]["selection"] = selection
                data[idx]["confidence"] = confidence
            else:
                data[idx]["selection"] = output_text  # 保存原始输出作为备选
                data[idx]["confidence"] = confidence

        return data

    def calculate_auc_acc(self, data: List[Dict]) -> Tuple[float, float, int, int]:
        from sklearn.metrics import roc_auc_score

        y_true = [item["label"] for item in data]
        y_score = [item["confidence"] for item in data]

        auc = roc_auc_score(y_true, y_score)

        total = len(data)
        correct = 0

        for item in data:
            if "selection" in item and "answer" in item:
                if item["selection"] == item["answer"]:
                    correct += 1

        accuracy = correct / total if total > 0 else 0.0

        return auc, accuracy, correct, total

    def run_evaluation(self) -> Dict[str, Any]:
        """运行完整的评估流程"""
        print("Starting preference evaluation...")

        # 1. 加载模型和数据

        # with open(self.config.input_file, "r", encoding="utf-8") as f:
        #     test_data = json.load(f)
        test_data = []
        with open(self.config.input_file, "r") as f:
            for line in f:
                test_data.append(json.loads(line))
        with open(self.config.system_prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

        print(f"Loading tokenizer from {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        print(f"Loading LLM model from {self.config.model_name}")
        self.llm = LLM(
            model=self.config.model_name,
            max_model_len=self.config.max_model_len,
            tensor_parallel_size=self.config.tensor_parallel_size,
            disable_custom_all_reduce=True,
        )
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            n=1,
            max_tokens=self.config.max_tokens,
            logprobs=1,
            # truncate_prompt_tokens=8192,
        )

        # 2. 准备推理文本
        processed_data = self.preprocess_data(test_data)
        # texts = self.prepare_texts_for_inference(processed_data)
        texts = [data["instruction"] for data in processed_data]

        # 3. 生成模型响应
        print("Generating responses...")
        outputs = self.llm.generate(texts, sampling_params)
        # breakpoint()

        # 4. Calculating accuracy
        results = self.process_outputs(processed_data, outputs)
        auc, accuracy, correct, total = self.calculate_auc_acc(results)

        # 5. Saving output
        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        print(f"Results saved to {output_path}")

        evaluation_results = {
            "AUC": auc,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        with open(self.config.performance_file, "a+", encoding="utf-8") as ff:
            ff.write(f"Results saved to {output_path}.\n")
            ff.write(json.dumps(evaluation_results) + "\n")
        print(f"Evaluation completed!")
        print(f"AUC: {auc:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Correct: {correct}/{total}")

        return evaluation_results


def set_seed(seed: int = 42):
    """
    固定 Python、NumPy、PyTorch 等随机种子，确保结果可复现
    """

    # 1. 固定 Python 内置随机种子
    random.seed(seed)

    # 2. 固定环境变量随机因子（部分依赖）
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 3. 固定 numpy 随机种子
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多 GPU

        # 确保 cudnn 使用确定性算法
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # 没有安装 torch 则跳过

    print(f"Random seed set to {seed}")


def main(args):
    """主函数"""
    set_seed()
    # 创建配置
    input_file = args.input_file
    model_name = args.model_name
    config = EvaluationConfig(
        model_name=model_name, input_file=input_file, enable_thinking=args.thinking
    )

    # 创建评估器并运行评估
    evaluator = PreferenceEvaluator(config)
    _ = evaluator.run_evaluation()


if __name__ == "__main__":
    args = parse_args()
    main(args)
