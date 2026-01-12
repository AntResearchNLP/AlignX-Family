import os
import json
import re
import argparse
import random
import math

from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass, field

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, sampling_params
from tqdm import tqdm


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


@dataclass
class EvaluationConfig:
    """评估配置类"""

    input_file: str
    enable_thinking: bool = False
    model_name: str = "Qwen/Qwen3-8B"
    system_prompt_path: str = "prompt_templates/preference.txt"
    
    tensor_parallel_size: int = 8
    max_model_len: int = 16384

    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    max_tokens: int = 4096

    def __post_init__(self):
        if self.enable_thinking:
            self.max_tokens = 4096
        path = self.input_file.replace("/upi_benchmark/", "/results/direct/")
        dir_name = os.path.dirname(path)
        file_name = os.path.basename(path)
        file_name = file_name.replace("_preference_", "_result_")
        file_name = (
            f"direct_thinking_{file_name}"
            if self.enable_thinking
            else f"direct_{file_name}"
        )
        self.output_path: str = os.path.join(dir_name, file_name)
        self.rm = os.path.basename(self.model_name)
        self.performance_file: str = os.path.join(
            dir_name, f"performance_{self.rm}.txt"
        )


class PreferenceEvaluator:
    """偏好评估器类"""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.tokenizer = None
        self.llm = None
        self.system_prompt = ""

    def format_prompt(
        self, prompt: str, response_a: str, response_b: str, persona: str = ""
    ) -> str:
        """格式化输入提示"""
        return (
            f"Determine which response the user prefers based on the user's preferences. "
            f"Please output your selection below in a json format by filling in the placeholders in []:"
            f'{{"selection": "[Item A / Item B]"}}\n'
            # f"{self.system_prompt}\n"
            f"<Prompt>\n{prompt}\n</Prompt>\n\n"
            f"<Preference>\n{persona}</Preference>\n\n"
            f"<Item A>\n{response_a}\n</Item A>\n\n"
            f"<Item B>\n{response_b}\n</Item B>\n\n"
            f"Now, ONLY output your selection without any other text outside of this specified structure."
        )

    def format_profile(self, history):
        prompt_str = "\n\n".join(
            [
                f"=====Tripe {idx+1}=====\n\n*QUERY:*\n {triple['query']}\n{'*'*5}\n*CHOSEN ITEM BY THE USER:*\n{triple['chosen']}\n{'*'*5}\n*REJECTED ITEM BY THE USER:*\n{triple['rejected']}"
                for idx, triple in enumerate(history)
            ]
        )
        full_text = f"""The user's history will be provided as a sequence of triples, where each triple is `(QUERY, CHOSEN ITEM BY THE USER, REJECTED ITEM BY THE USER)`.\n\n=====Interaction History=====\n{prompt_str}\n\n=====END====="""

        return full_text

    def preprocess_data(self, test_data: List[Dict]) -> List[Dict]:
        """预处理数据"""
        processed_data = []

        for idx, item in tqdm(
            enumerate(test_data), total=len(test_data), desc="Preprocessing data"
        ):
            target = item["target"]
            history = item["history"]
            # profile = item["profile"]
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

            # 格式化输入
            profile = self.format_profile(history)
            formatted_input = self.format_prompt(prompt, responseA, responseB, profile)

            # 创建新的数据项
            processed_item = {}
            processed_item["idx"] = idx
            processed_item["instruction"] = formatted_input
            processed_item["answer"] = answer
            processed_item["target"] = target
            processed_item["profile"] = profile

            processed_data.append(processed_item)

        return processed_data

    def prepare_texts_for_inference(
        self, data: List[Dict], enable_thinking=False
    ) -> List[str]:
        """准备推理文本"""
        texts = []

        for item in data:
            prop = item["instruction"]

            # 构建消息格式 (适用于 Qwen3)
            messages = [
                {"role": "system", "content": "Generate a task-specific response."},
                {"role": "user", "content": prop},
            ]

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )

            texts.append(text)

        return texts

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

    def process_outputs(self, data: List[Dict], outputs: List[Any]) -> List[Dict]:
        """处理模型输出"""
        for idx, output in enumerate(outputs):
            output_text = output.outputs[0].text

            if "gpt" in self.config.rm:
                selection = self.parse_gpt_output(output_text)
            else:
                selection = self.parse_model_output(output_text)
            if selection:
                data[idx]["selection"] = selection
            else:
                data[idx]["selection"] = output_text  # 保存原始输出作为备选

        return data

    def calculate_accuracy(self, data: List[Dict]) -> Tuple[float, int, int]:
        """计算准确率"""
        total = len(data)
        correct = 0

        for item in data:
            if "selection" in item and "answer" in item:
                if item["selection"] == item["answer"]:
                    correct += 1

        accuracy = correct / total if total > 0 else 0.0
        return accuracy, correct, total

    def save_results(self, data: List[Dict]) -> None:
        """保存结果"""
        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"Results saved to {output_path}")

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

        # 2. 准备推理文本
        processed_data = self.preprocess_data(test_data)
        texts = self.prepare_texts_for_inference(
            processed_data, self.config.enable_thinking
        )

        # 3. 生成模型响应
        print("Generating responses...")
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            n=1,
            max_tokens=self.config.max_tokens,
            logprobs=1,
        )
        outputs = self.llm.generate(texts, sampling_params)
        # breakpoint()

        # 4. Calculating accuracy
        results = self.process_outputs(processed_data, outputs)
        accuracy, correct, total = self.calculate_accuracy(results)

        # 5. Saving output
        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        print(f"Results saved to {output_path}")

        evaluation_results = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            # "data": results,
        }
        print(f"Evaluation completed!")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Correct: {correct}/{total}")
        with open(self.config.performance_file, "a+", encoding="utf-8") as ff:
            ff.write(f"Results saved to {output_path}.\n")
            ff.write(json.dumps(evaluation_results) + "\n")

        return evaluation_results


def main(args):
    """主函数"""
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
