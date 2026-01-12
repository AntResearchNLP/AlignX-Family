# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import json
import math
import random

from verl.utils.reward_score.math import is_equiv
from openai import OpenAI


urls = {
    "Qwen3-8B": "127.0.0.1:8000/v1"
}
client = OpenAI(
    base_url=urls["Qwen3-8B"],
    api_key="EMPTY",
)
with open(
    "preference.txt",
    "r",
    encoding="utf-8",
) as f:
    system_prompt = f.read()


def format_score(output_string):
    if not (
        output_string.count("<think>") == 1 and output_string.count("</think>") == 1
    ):
        return False
    pattern = re.compile(r"^<think>.+<\/think>.+$", re.DOTALL)
    # 或者 re.S
    if pattern.match(output_string):
        return True
    return False


def extract_preference(response):
    return response.strip().split("</think>")[-1].strip()


def _reward_template_upi(post, preference):
    chosen = post["chosen"]
    rejected = post["rejected"]
    task = post["query"]
    flag = random.randint(0, 1)
    if flag:
        responseA = chosen
        responseB = rejected
        answer = "Item A"
    else:
        responseA = rejected
        responseB = chosen
        answer = "Item B"

    # prop = f'Determine which response the user prefers based on the user’s preferences. Please output your selection below in a json format by filling in the placeholders in []:{{"selection": "[Item A / Item B]"}}\n{system_prompt}\n<Prompt>\n{task}\n</Prompt>\n\n<Preference>\n{preference}</Preference>\n\n<Item A>\n{responseA}\n</Item A>\n\n<Item B>\n{responseB}\n</Item B>\n\nNow, ONLY output your selection without any other text outside of this specified structure.'
    prop = (
        f"Determine which response the user prefers based on the user's preferences. "
        f"Please output your selection below in a json format by filling in the placeholders in []:"
        f'{{"selection": "[Item A / Item B]"}}\n'
        f"<Prompt>\n{task}\n</Prompt>\n\n"
        f"<Preference>\n{preference}</Preference>\n\n"
        f"<Item A>\n{responseA}\n</Item A>\n\n"
        f"<Item B>\n{responseB}\n</Item B>\n\n"
        f"Now, ONLY output your selection without any other text outside of this specified structure."
    )

    messages = [
        {"role": "system", "content": "Generate a task-specific response."},
        {"role": "user", "content": prop},
    ]

    return messages, answer


def generate_final_answer(messages):
    models = client.models.list()
    model = models.data[0].id
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=16,
        stream=False,  # 在此模板中，我们等待完整响应
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        logprobs=True,
    )
    return response.choices[0].message.content, response.choices[0].logprobs.content


def extract_confidence(logprobs_content, answer):
    def normalize_token(tok: str) -> str:
        return tok.replace("Ġ", "").replace("Ċ", "\n")

    confidence_of_A = None
    confidence_of_B = None

    for item in logprobs_content:
        # 注意：Token可能包含前导空格，例如 " A"。使用 strip() 来处理这种情况。
        if normalize_token(item.token).strip() == "A":
            logprob_of_A = item.logprob
            # 第4步：转换成标准概率 (0-1)
            confidence_of_A = math.exp(logprob_of_A)
        if normalize_token(item.token).strip() == "B":
            logprob_of_B = item.logprob
            confidence_of_B = math.exp(logprob_of_B)
    if confidence_of_A is None and confidence_of_B is None:
        # 打印所有 token 的信息，便于调试
        print("\n--- 所有生成Token的详细信息 ---")
        for item in logprobs_content:
            # breakpoint()
            print(f"Token: '{item.token}', LogProb: {item.logprob:.4f}")
        return 0.0
    if answer == "Item A":
        confidence_of_answer = confidence_of_A or (1 - confidence_of_B)
    elif answer == "Item B":
        confidence_of_answer = confidence_of_B or (1 - confidence_of_A)
    return confidence_of_answer


def compute_score(solution_str, ground_truth=None, data_source=None, extra_info=None):
    """The scoring function for UPI.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
    """
    stage_id = extra_info["stage_id"]
    # print(stage_id)
    if stage_id == 1:
        target = extra_info["target1"]
    elif stage_id == 2:
        target = extra_info["target2"]
    else:
        raise NotImplementedError("Stage id must be 1 or 2.")

    if not format_score(solution_str):
        # formatting score
        return {
            "score": 0.0,
            "format_socre": -1.0,
            "confidence": 0.0,
        }
    preference = extract_preference(solution_str)

    # confidence reward
    reward_prompt, answer = _reward_template_upi(target, preference)
    reward_response, logprobs = generate_final_answer(reward_prompt)
    if reward_response is None:
        raise ValueError("======== API receives none content. ========")
    reward = extract_confidence(logprobs, answer)
    return {
        "score": reward,
        "format_socre": 0.0,
        "confidence": reward,
    }
