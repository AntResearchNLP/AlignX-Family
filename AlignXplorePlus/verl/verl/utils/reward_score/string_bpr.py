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
import ast
import json
import warnings
import random

from openai import OpenAI
import numpy as np


urls = {
    "Qwen3-8B-H20-server-1": "http://33.184.123.246:8000/v1",
    "Qwen3-8B-H209": "http://33.184.124.128:8000/v1",
    "Qwen3-8B-h2011": "http://33.184.124.189:8000/v1",
}
client = OpenAI(
    base_url=urls["Qwen3-8B-H209"],
    api_key="EMPTY",
)
with open(
    "/ossfs/workspace/nas/yuting/code/PreInf/reinforcement learning/playground/rating_list.txt",
    "r",
    encoding="utf-8",
) as f:
    system_prompt = f.read()


def extract_preference_list(response):
    import re

    pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    matches = re.findall(pattern, response)
    if matches:
        result = matches[-1].replace("answer here", "")
    else:
        result = ""

    try:
        ret = eval(result)
        return ret
    except:
        if "," in result:
            return [item.strip() for item in result.split(",") if item.strip()]
        elif ";" in result:
            return [item.strip() for item in result.split(";") if item.strip()]
        else:
            return [result.strip()]


def _reward_template(post, preference):

    chosen = post["chosen"]
    rejected = post["rejected"]
    task = post["post"]
    flag = random.randint(0, 1)
    if flag:
        responseA = chosen
        responseB = rejected
        answer = 1.0
    else:
        responseA = rejected
        responseB = chosen
        answer = -1.0
    prop = f"""You are an AI evaluator. Your task is to compare two responses, A and B, and score their alignment with each dimension in the provided preference in two steps:\n1. Summarize the following user preference description into a Python list of strings. Each string should represent a single, core preference, capturing the original meaning.\n2. Analyze the given response and score its alignment with the list preference list.\n\nFor each dimension in the preference list, assign a score from -1.0 to 1.0:\n*   **-1.0:** The response clearly exhibits a characteristic opposite to the described preference.\n*   **0.0:** The response is neutral regarding the preference, or the preference is not applicable/observable.\n*   **1.0:** The response perfectly aligns with or exemplifies the described preference.\n*   Intermediate values (e.g., 0.5, -0.7): The response partially aligns/misaligns with the described preference.\n\nPlease output your scores below in a JSON format by filling in the placeholders in []:{{"List of Preference": "[]", "Score A": "[]", "Score B": "[]"}} with the scores in the exact same order as the preference dimensions. \n\n{system_prompt}\n\n<Prompt>\n{task}\n</Prompt>\n\n<Preference>\n{preference}\n</Preference>\n\n<Response A>\n{responseA}\n</Response A>\n\n<Response B>\n{responseB}\n</Response B>\n"""

    messages = [
        {"role": "system", "content": "Generate a task-specific response."},
        {"role": "user", "content": prop},
    ]

    return messages, answer


def generate(messages):
    models = client.models.list()
    model = models.data[0].id
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        # temperature=1.0,
        # max_completion_tokens=2048,
        stream=False,  # 在此模板中，我们等待完整响应
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return response.choices[0].message.content


def extract_ratings(response: str):
    json_pattern = re.compile(r"\{.*?\}", re.DOTALL)

    # 查找所有匹配的JSON字符串
    all_json_strings = json_pattern.findall(response)

    if not all_json_strings:
        print("未找到任何JSON格式的字符串。")
        return None

    # 获取最后一个匹配的JSON字符串
    last_json_str = all_json_strings[-1]
    # print(f"提取到的最后一个JSON字符串（未解析）:\n{last_json_str}")

    try:
        # 尝试将字符串解析为JSON对象
        last_json_object = json.loads(last_json_str)
    except json.JSONDecodeError as e:
        print(f"错误: 最后一个匹配到的字符串不是有效的JSON。\n字符串内容:\n{last_json_str}\n错误信息: {e}")
        return None
    except Exception as e:
        print(f"发生未知错误: {e}")
        return None
    score_a = last_json_object["Score A"]
    score_b = last_json_object["Score B"]
    if type(score_a) is str:
        score_a = ast.literal_eval(score_a)
    if type(score_b) is str:
        score_b = ast.literal_eval(score_b)

    if type(score_a) is float:
        score_a = [score_a]
    if type(score_b) is float:
        score_b = [score_b]
    if len(score_a) != len(score_b):
        print(f"Score维度不一致{len(score_a)} vs. {len(score_b)}")
        return None
    ret = np.array(score_a, dtype="float64") - np.array(score_b, dtype="float64")
    return ret


def regularization(preferences: list, lambda1: float, lambda2: float):
    return lambda1 * len(preferences) + lambda2 * sum(len(p) for p in preferences)


def compute_score(solution_str, ground_truth, data_source, extra_info):
    """The scoring function for AlingXplore.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
    """
    posts = extra_info["posts"]
    score = 0.0
    # preference = extract_preference_list(solution_str)
    preference = solution_str
    if preference is None or len(preference) == 0:
        print("=*" * 32)
        print(solution_str)
        print("*=" * 32)
        return -2.0
    reward_mode = "last"
    n = 1
    # posts = random.sample(posts, n)
    if reward_mode == "last":
        posts = posts[-n:]
    elif reward_mode == "random":
        posts = random.sample(posts, n)
    elif reward_mode == "all":
        pass
    else:
        raise NotImplementedError
    for post in posts:
        rating_prompt, flag = _reward_template(post, preference)
        rating_response = generate(rating_prompt)
        if rating_response is None:
            raise ValueError("======== API receives none content. ========")
        rating_delta = extract_ratings(rating_response)
        if rating_delta is None:
            warnings.simplefilter("always")
            warnings.warn("======== Rating is none. ========")
        else:
            score += flag * rating_delta.sum() / len(rating_delta)
    return score / n
