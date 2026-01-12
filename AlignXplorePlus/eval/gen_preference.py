import os
import re
import json
import argparse
from typing import List, Dict, Optional
from pathlib import Path

import random
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def build_prompt(
    tokenizer: AutoTokenizer,
    history: List[Dict[str, str]],
    past: Optional[str],
) -> str:

    system_prompt = """You are an expert User Preference Analyst. Your sole task is to analyze a user's past preference summary (possibly not provided) and new interaction history and summarize the user preferences.\n\nThe user's history will be provided as a sequence of triples, where each triple is `(QUERY, CHOSEN ITEM BY THE USER, REJECTED ITEM BY THE USER)`.\n\nDo not include any introductory phrases, greetings, or any other text outside of this specified structure."""

    prompt_str = "\n\n".join(
        [
            f"=====Triple {idx+1}=====\n\n*QUERY:*\n {triple['query']}\n{'*'*5}\n*CHOSEN ITEM BY THE USER:*\n{triple['chosen']}\n{'*'*5}\n*REJECTED ITEM BY THE USER:*\n{triple['rejected']}"
            for idx, triple in enumerate(history)
        ]
    )

    full_text = f"""Analyze the past preference summary and the following user interaction history to summarize the comprehensive user preferences in concise language. If past preferences are provided, adjust the preferences by combining past preferences with those reflected in current behavior, removing conflicting parts, and integrating new insights. If no past preferences are provided, derive the final preferences solely from user behavior. The user's history will be provided as a sequence of triples, where each triple is `(QUERY, CHOSEN ITEM BY THE USER, REJECTED ITEM BY THE USER)`.\n=====Past Preference Summary=====\n\n"Not Provided."\n\n=====Interaction History=====\n{prompt_str}\n\n=====END=====\n\nNow, given the above user's past prefernce summary and the interaction history, summarize the user preferences."""

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": full_text.strip(),
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt


def parse_output(response):
    return response.strip().split("</think>")[-1].strip()


def main():
    ap = argparse.ArgumentParser(
        description="Two-stage streaming preference summarization (concise)."
    )
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tensor-parallel-size", type=int, default=8)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

    # Load dataset
    dataset = []
    with open(args.input, "r") as f:
        for line in f:
            dataset.append(json.loads(line))

    dataset_name = os.path.splitext(os.path.basename(args.input))[0].split("_")[0]
    last_set = {"amazon", "movielens", "mind"}
    last = dataset_name in last_set
    print(f"{dataset_name}::Last:{last}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    sampling = SamplingParams(
        temperature=0.6,
        top_k=20,
        top_p=0.95,
        n=1,
        max_tokens=4096,
    )

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        disable_custom_all_reduce=True,
    )

    def truncate_history(original_history):
        while True:
            try:
                prompt = build_prompt(tokenizer, original_history, None)
                length = len(tokenizer.encode(prompt, add_special_tokens=True))
            except Exception as e:
                half = len(original_history) // 2
                original_history = original_history[-half:]
                continue

            if length < args.max_model_len:
                return original_history
            if len(original_history) == 1:
                return []
            half = len(original_history) // 2
            original_history = original_history[-half:]

    # Preprocess history
    for idx, item in tqdm(enumerate(dataset), total=len(dataset)):
        filtered_history = item["history"]  # [:-1]
        filtered_history = truncate_history(filtered_history)
        item["history"] = filtered_history

    data = []
    texts = []
    for idx, item in tqdm(enumerate(dataset), total=len(dataset)):
        if last:
            mid_position = len(item["history"]) // 3
            history1 = item["history"][
                : mid_position * 2
            ]  # whole history contains target
        else:
            history1 = item["history"]

        prompt = build_prompt(tokenizer, history1, None)
        rec = dict(item)
        rec["idx"] = idx
        rec["raw_prompt"] = prompt
        texts.append(prompt)
        data.append(rec)
    outputs = llm.generate(texts, sampling_params=sampling)

    for i, out in enumerate(outputs):
        raw = out.outputs[0].text
        data[i]["first_response"] = raw
        data[i]["profile"] = parse_output(raw)

    model_name = os.path.basename(args.model)
    model_name = model_name.replace("-", "_")
    data_source = os.path.basename(args.input).split(".")[0]
    output_file = f"preferences/{model_name}/{data_source}_preference_{model_name}.json"
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Results saved to {output_path}")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


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


if __name__ == "__main__":
    set_seed()
    main()
