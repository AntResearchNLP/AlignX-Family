import os
import re
import json
import random
import math
from tqdm import tqdm

from collections import defaultdict
from typing import DefaultDict, List, Dict, Optional, Tuple

from transformers import AutoTokenizer


def read_file(train_file):
    data = []
    with open(train_file, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def process(data: List[Dict]):
    """
    Split the user's behaviors into three stages.
    """
    filtered_data = []
    for user in data:
        split = len(user["history"]) // 3
        if split < 3:
            continue

        item = dict(user)
        item["stage1"] = user["history"][:split]
        item["stage2"] = user["history"][split : 2 * split]
        item["stage3"] = user["history"][2 * split :]
        filtered_data.append(item)
    return filtered_data


def build_preference_prompt(
    tokenizer: AutoTokenizer,
    history: List[Dict[str, str]],
    target: Dict[str, str],
) -> str:

    system_prompt = """You are an expert User Preference Analyst. Your sole task is to analyze a user's past preference summary (possibly not provided) and new interaction history and summarize the user preferences.\n\nThe user's history will be provided as a sequence of triples, where each triple is `(QUERY, CHOSEN ITEM BY THE USER, REJECTED ITEM BY THE USER)`.\n\nDo not include any introductory phrases, greetings, or any other text outside of this specified structure."""

    prompt_str = "\n\n".join(
        [
            f"=====Triple {idx+1}=====\n\n*QUERY:*\n {triple['query']}\n{'*'*5}\n*CHOSEN ITEM BY THE USER:*\n{triple['chosen']}\n{'*'*5}\n*REJECTED ITEM BY THE USER:*\n{triple['rejected']}"
            for idx, triple in enumerate(history)
        ]
    )

    chosen = target["chosen"]
    rejected = target["rejected"]
    flag = random.randint(0, 1)
    if flag:
        responseA = chosen
        responseB = rejected
    else:
        responseA = rejected
        responseB = chosen

    target_str = f"*QUERY:*\n {target['query']}\n{'*'*5}\n*ITEM A:*\n{responseA}\n{'*'*5}\n*ITEM B:*\n{responseB}"

    full_text = f"""Analyze the past preference summary, the following user interaction history, and the provided target triple to summarize the comprehensive user preferences in concise language. If past preferences are provided, adjust the preferences by combining past preferences with those reflected in current behavior, removing conflicting parts, and integrating new insights. If no past preferences are provided, derive the final preferences solely from user behavior.\nThe user's history will be provided as a sequence of triples, where each triple is `(QUERY, CHOSEN ITEM BY THE USER, REJECTED ITEM BY THE USER)`. When resolving conflicts between chosen and rejected items in the interaction history, always give priority to the preference associated with chosen items.\nThe target triple is a test triple in the format `(QUERY, ITEM A, ITEM B)`. Use the target triple as a guiding example to ensure the summarized preferences are aligned with distinguishing characteristics relevant to the target triple, but do not directly predict positive or negative samples to ensure generalizability.\n====Past Preference Summary=====\n"Not Provided."\n\n=====Interaction History=====\n{prompt_str}\n\n=====Target Triple=====\n{target_str}\n\n=====END=====\n\nNow, given the above user's past preference summary, the interaction history, and the target triple, summarize the user preferences and DO NOT make any prediction or classification about which item in the Target Triple is positive or negative during the preference summarization."""
    messages = [
        {
            "role": "system",
            "content": system_prompt.strip(),
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


def build_evaluation_prompt(
    tokenizer, prompt: str, response_a: str, response_b: str, persona: str = ""
) -> str:
    """格式化输入提示"""
    full_text = (
        f"Determine which response the user prefers based on the user's preferences. "
        f"Please output your selection below in a json format by filling in the placeholders in []:"
        f'{{"selection": "[Item A / Item B]"}}\n'
        f"<Prompt>\n{prompt}\n</Prompt>\n\n"
        f"<Preference>\n{persona}</Preference>\n\n"
        f"<Item A>\n{response_a}\n</Item A>\n\n"
        f"<Item B>\n{response_b}\n</Item B>\n\n"
    )
    messages = [
        {"role": "system", "content": "Generate a task-specific response."},
        {"role": "user", "content": full_text},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    return text


def prepare_user_data(tokenizer, test_data: List[Dict]) -> List[Dict]:
    """预处理数据"""
    processed_data = []

    for idx, item in enumerate(test_data):
        target = item["target"]["item"]
        prompt = target["query"]
        chosen = target["chosen"]
        rejected = target["rejected"]
        profile = item["preference"]

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
        formatted_input = build_evaluation_prompt(
            tokenizer, prompt, responseA, responseB, profile
        )

        # 创建新的数据项
        processed_item = dict(item)
        processed_item["instruction"] = formatted_input
        processed_item["answer"] = answer
        processed_item["label"] = flag

        # {uid, target, preference, instruction, answer, label}
        processed_data.append(processed_item)

    return processed_data


def parse_model_output(output_text: str) -> Optional[str]:
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
            print(f"Failed to parse output: {output_text}")
            return None
    return None


def build_merge_prompt(tokenizer: AutoTokenizer, raw_outputs_list: List[str]):
    system_prompt = """
    You are an expert information synthesizer. Your task is to merge multiple "Reasoning-Preference" pairs (separated by `|PreferenceSEP|`), which were all generated from the same source data, into one comprehensive, non-redundant pair, while faithfully preserving the narrative style of the original reasoning and preference.

    **Final Output Format:**
    The output MUST include two parts separated by `|PreferenceSEP|`.
    1.  **Merged Reasoning Process**: The merged, comprehensive, non-redundant reasoning process.
    2.  **Merged Preference Summary**: The merged, comprehensive, non-redundant preference summary.

    Your merging principle is **informational set union**. Think of it this way:
    - **Completeness is your top priority.** You MUST preserve every unique detail, reason, and preference from ALL inputs. The final merged reasoning process or preference should be **more comprehensive and information-rich** than any single input it was created from. No unique information should be lost.
    - **Deduplication is mandatory.** If the exact same point or idea or dimension is mentioned in multiple inputs (even with slightly different wording), it must appear only **once** in the final output. Eliminate all redundancy.

    Follow these strict rules:
    1.  **Merge, Don't Summarize**: Your goal is to create a complete union of information, not a high-level summary. Retain specific examples and nuances.
    2.  **No Invention**: Do not add any information or make any assumptions that are not explicitly stated in the provided inputs.
    3. Do not include any headers, introductory phrases, or explanations in your response.
    """

    outputs_str = "\n\n".join(
        [
            f"**Reasoning process and Preferenc Summary {idx+1}:**\n{output}"
            for idx, output in enumerate(raw_outputs_list)
        ]
    )

    full_text = f"""Your task is to merge multiple "Reasoning-Preference" pairs (separated by `|PreferenceSEP|`), which were all generated from the same source data, into one comprehensive, non-redundant pair, while faithfully preserving the **narrative style** of the original reasoning and preference.\n\nHere are {len(raw_outputs_list)} "Reasoning-Preference" pairs generated from the same user history.\n\n=====Reasoning and Preference List=====\n{outputs_str}\n=====END=====\n\nNow, apply the **informational set union** principle to merge these pairs. Your final output must be a single, maximally detailed, and non-redundant "Reasoning-Preference" pair, following the specified format.
    """

    messages = [
        {
            "role": "system",
            "content": system_prompt.strip(),
        },
        {
            "role": "user",
            "content": full_text.strip(),
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return prompt


def chunk_list(lst, chunk_size):
    """辅助函数：将列表按 chunk_size 分块"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def main():
    from vllm import LLM, SamplingParams
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name. [amazon, mind]",
    )
    parser.add_argument("--split", type=int, required=True)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    sampling = SamplingParams(
        temperature=0.6,
        top_k=20,
        top_p=0.95,
        n=1,
        max_tokens=4096,
    )
    model = "Qwen/Qwen3-32B"
    tensor_parallel_size = 8
    llm = LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        disable_custom_all_reduce=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model)

    BATCH_USER_SIZE = 1000  # 每批处理多少个用户

    # ===== 数据读取与预处理 =====
    train_file = (
        f"{args.dataset}_with_signal.jsonl"
    )
    output_file = f"stage1/{args.dataset}_stage_1_{args.split}.jsonl"
    with open(output_file, "w") as f:
        pass

    batching = 50000
    start_id = (args.split - 1) * batching
    end_id = start_id + batching

    data = read_file(train_file)
    print(train_file)
    print(f"Total user: {len(data)}")
    print(f"Processing {start_id}~{end_id}")

    data = data[start_id:end_id]
    processed_data = process(data=data)

    print(f"加载数据: {len(processed_data)} 用户")
    user_batches = list(chunk_list(list(enumerate(processed_data)), BATCH_USER_SIZE))
    total_batches = len(user_batches)
    print(f"总批次数: {total_batches}，每批 {BATCH_USER_SIZE} 用户")

    # tqdm 进度条
    pbar = tqdm(total=total_batches, desc="总体进度", unit="batch")

    # ===== 按批处理用户 =====
    for batch_idx, user_batch in enumerate(user_batches, start=1):
        print(f"\n=== 处理 Batch {batch_idx}，用户数: {len(user_batch)} ===")

        # ===== Stage 1 =====
        batch_texts1, batch_meta1 = [], []
        for u_idx, user in user_batch:
            targets = [t for t in user["stage2"] if math.exp(t["logprob1"]) > 0.9]
            if len(targets) > 5:
                targets = random.sample(targets, 5)
            if len(targets) <= 3:
                continue
            for target in targets:
                history = [his["item"] for his in user["stage1"]]
                prompt = build_preference_prompt(tokenizer, history, target["item"])
                batch_texts1.append(prompt)
                batch_meta1.append((u_idx, target))

        if not batch_texts1:
            continue

        try:
            outputs1 = llm.generate(batch_texts1, sampling)
        except Exception as e:
            print(f"[Stage1] Batch {batch_idx} 生成失败: {e}")
            continue  # 跳过当前批次

        user_stage1_data = defaultdict(list)
        for i, output in enumerate(outputs1):
            response = output.outputs[0].text
            # seperate reasoning and preference by `|PreferenceSEP|`
            reasoning = response.strip().split("</think>")[0].strip("<think>").strip()
            preference = response.strip().split("</think>")[-1].strip()
            raw = f"{reasoning}\n|PreferenceSEP|\n\n{preference}"
            u_idx, target = batch_meta1[i]
            user_stage1_data[u_idx].append(
                {
                    "uid": processed_data[u_idx]["uid"],
                    "target": target,
                    "preference": preference,
                    "raw": raw,
                }
            )

        # ===== Stage 2 =====
        batch_texts2, batch_meta2 = [], []
        processed_user_by_uid = {}
        for u_idx, items in user_stage1_data.items():
            processed_user = prepare_user_data(tokenizer, items)
            processed_user_by_uid[u_idx] = processed_user
            for inst_idx, item in enumerate(processed_user):
                batch_texts2.append(item["instruction"])
                batch_meta2.append((u_idx, inst_idx))

        if not batch_texts2:
            continue

        try:
            outputs2 = llm.generate(batch_texts2, sampling)
        except Exception as e:
            print(f"[Stage2] Batch {batch_idx} 生成失败: {e}")
            continue

        for i, output in enumerate(outputs2):
            selection = parse_model_output(output.outputs[0].text)
            u_idx, inst_idx = batch_meta2[i]
            processed_user_by_uid[u_idx][inst_idx]["selection"] = selection

        # ===== Stage 3 =====
        merge_prompts, merge_meta = [], []
        reasoning_pref_by_uid = {}
        selected_targets_by_uid = {}
        for u_idx, processed_user in processed_user_by_uid.items():
            reasoning_list = []
            selected_target_ids = []
            final_targets = []
            for pu in processed_user:
                if pu["selection"] == pu["answer"]:
                    reasoning_list.append(pu["raw"])
                    selected_target_ids.append(pu["target"]["hid"])
                    final_targets.append(
                        {"uid": processed_data[u_idx]["uid"], "target": pu["target"]}
                    )
            if len(selected_target_ids) < 3:
                continue
            merge_prompt = build_merge_prompt(tokenizer, reasoning_list)
            merge_prompts.append(merge_prompt)
            merge_meta.append(u_idx)
            reasoning_pref_by_uid[u_idx] = reasoning_list
            selected_targets_by_uid[u_idx] = (selected_target_ids, final_targets)

        if not merge_prompts:
            continue

        try:
            merge_outputs = llm.generate(merge_prompts, sampling)
        except Exception as e:
            print(f"[Stage3] Batch {batch_idx} 生成失败: {e}")
            continue

        final_pref_by_uid = {}
        final_think_by_uid = {}
        merge_response_by_uid = {}
        for i, output in enumerate(merge_outputs):
            raw_merge_response = output.outputs[0].text
            final_preferences = (
                raw_merge_response.strip().split("|PreferenceSEP|")[-1].strip()
            )
            final_thinking = (
                raw_merge_response.strip().split("|PreferenceSEP|")[0].strip()
            )
            u_idx = merge_meta[i]
            final_pref_by_uid[u_idx] = final_preferences
            final_think_by_uid[u_idx] = final_thinking
            merge_response_by_uid[u_idx] = raw_merge_response

        # ===== Stage 4 =====
        batch_texts4, batch_meta4 = [], []
        final_targets_processed_by_uid = {}
        for u_idx, (selected_ids, final_targets) in selected_targets_by_uid.items():
            pref = final_pref_by_uid[u_idx]
            for ft in final_targets:
                ft["preference"] = pref
            processed_final = prepare_user_data(tokenizer, final_targets)
            final_targets_processed_by_uid[u_idx] = processed_final
            for inst_idx, item in enumerate(processed_final):
                batch_texts4.append(item["instruction"])
                batch_meta4.append((u_idx, item["answer"]))

        if not batch_texts4:
            continue

        try:
            outputs4 = llm.generate(batch_texts4, sampling)
        except Exception as e:
            print(f"[Stage4] Batch {batch_idx} 生成失败: {e}")
            continue

        pass_count_by_uid = defaultdict(int)
        total_count_by_uid = defaultdict(int)
        for i, output in enumerate(outputs4):
            selection = parse_model_output(output.outputs[0].text)
            u_idx, answer = batch_meta4[i]
            total_count_by_uid[u_idx] += 1
            if selection == answer:
                pass_count_by_uid[u_idx] += 1

        passed_uid = {
            uid
            for uid in pass_count_by_uid
            if pass_count_by_uid[uid] >= total_count_by_uid[uid] * 0.8
        }

        # ===== 实时写入结果 =====
        with open(output_file, "a+", encoding="utf-8") as fout:
            for u_idx in passed_uid:
                final_user = dict(processed_data[u_idx])
                final_user["reasoning_preference_list_1"] = reasoning_pref_by_uid[u_idx]
                final_user["preference_1"] = final_pref_by_uid[u_idx]
                final_user["thinking_1"] = final_think_by_uid[u_idx]
                final_user["merge_output_1"] = merge_response_by_uid[u_idx]
                final_user["selected_target_ids"] = selected_targets_by_uid[u_idx][0]
                fout.write(json.dumps(final_user, ensure_ascii=False) + "\n")

        pbar.update(1)
        pbar.set_description(
            f"Batch_idx:{batch_idx}, Processing batches (user saved: {len(passed_uid)})"
        )

    print("全部完成！")
    pbar.close()


if __name__ == "__main__":
    main()
