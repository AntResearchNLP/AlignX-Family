import random
import json
from transformers import AutoTokenizer
from typing import DefaultDict, List, Dict, Optional, Tuple


def read_file(train_file):
    data = []
    with open(train_file, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def build_input_prompt(history: List[Dict[str, str]], past: Optional[str]):

    system_prompt = """You are an expert User Preference Analyst. Your sole task is to analyze a user's past preference summary (possibly not provided) and new interaction history and summarize the user preferences.\n\nThe user's history will be provided as a sequence of triples, where each triple is `(QUERY, CHOSEN ITEM BY THE USER, REJECTED ITEM BY THE USER)`.\n\nDo not include any introductory phrases, greetings, or any other text outside of this specified structure."""

    prompt_str = "\n\n".join(
        [
            f"=====Triple {idx+1}=====\n\n*QUERY:*\n {triple['query']}\n{'*'*5}\n*CHOSEN ITEM BY THE USER:*\n{triple['chosen']}\n{'*'*5}\n*REJECTED ITEM BY THE USER:*\n{triple['rejected']}"
            for idx, triple in enumerate(history)
        ]
    )

    full_text = f"""Analyze the past preference summary and the following user interaction history to summarize the comprehensive user preferences in concise language. If past preferences are provided, adjust the preferences by combining past preferences with those reflected in current behavior, removing conflicting parts, and integrating new insights. If no past preferences are provided, derive the final preferences solely from user behavior. The user's history will be provided as a sequence of triples, where each triple is `(QUERY, CHOSEN ITEM BY THE USER, REJECTED ITEM BY THE USER)`.\n=====Past Preference Summary=====\n\n{past if past is not None else "Not Provided."}\n\n=====Interaction History=====\n{prompt_str}\n\n=====END=====\n\nNow, given the above user's past prefernce summary and the interaction history, summarize the user preferences."""

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
    return messages


def build_output_prompt(reasoning, preference):
    completion = f"""<think>\n{reasoning}\n</think>\n\n{preference}"""
    messages = [{"role": "assistant", "content": completion}]
    return messages


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name. [amazon, mind]",
    )
    parser.add_argument("--split_num", type=int, default=1)
    args = parser.parse_args()

    train_file_list = [
        f"stage2/{args.dataset}_stage_2_{split}.jsonl"
        for split in range(1, args.split_num + 1)
    ]
    output_file = f"sft/{args.dataset}_sft.jsonl"
    with open(output_file, "w") as f:
        pass

    dataset = []
    for train_file in train_file_list:
        data = read_file(train_file)
        dataset.extend(data)
    print(f"Total user: {len(dataset)}")

    for user in dataset:
        history1 = [his["item"] for his in user["stage1"]]
        prompt1 = build_input_prompt(history1, None)
        completion1 = build_output_prompt(user["thinking_1"], user["preference_1"])
        dict1 = {
            "prompt": prompt1,
            "completion": completion1,
        }

        history2 = [his["item"] for his in user["stage2"]]
        prompt2 = build_input_prompt(history2, user["preference_1"])
        completion2 = build_output_prompt(user["thinking_2"], user["preference_2"])
        dict2 = {
            "prompt": prompt2,
            "completion": completion2,
        }
        with open(output_file, "a+", encoding="utf-8") as f:
            f.write(json.dumps(dict1, ensure_ascii=False) + "\n")
            f.write(json.dumps(dict2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
