import json
import re

def parse_posts(text):
    """
    解析包含多个post的文本，提取每个post的'Post', 'Chosen', 'Rejected'部分。

    Args:
        text (str): 包含多个post的原始文本。

    Returns:
        list: 一个列表，其中每个元素都是一个字典，代表一个post，
              字典包含 'post', 'chosen', 'rejected' 键及其对应的内容。
    """

    # 定义正则表达式模式来捕获每个部分。
    # re.DOTALL 标志使得 '.' 也能匹配换行符。
    # \s* 匹配零个或多个空白字符，包括换行符，用于处理格式上的差异。
    # (.*?) 是非贪婪匹配，确保只捕获到下一个标记出现之前的内容。
    pattern = re.compile(
        r"\n\n\d+\.\s*\*Post:\*\s*\n"  # 匹配post的起始标记，如 "1. *Post:*"
        r"(.*?)\s*\n\*Chosen:\*\s*\n"  # 捕获Post内容 (Group 1)，直到 "*Chosen:*" 标记
        r"(.*?)\s*\n\*Rejected:\*\s*\n"  # 捕获Chosen内容 (Group 2)，直到 "*Rejected:*" 标记
        r"(.*?)(?=\n\n\d+\.\s*\*Post:\*|\Z)",  # 捕获Rejected内容 (Group 3)，直到下一个post的起始或文本末尾
        re.DOTALL,
    )

    # 查找所有匹配项
    matches = pattern.findall(text)

    all_posts_data = []

    # 遍历所有匹配项，将捕获到的内容整理成字典格式
    for match in matches:
        post_content = match[0].strip()  # 移除内容两端的空白字符和换行符
        chosen_content = match[1].strip()
        rejected_content = match[2].strip()

        all_posts_data.append(
            {
                "query": post_content,
                "chosen": chosen_content,
                "rejected": rejected_content,
            }
        )

    return all_posts_data


def process_dialogue(history):
    full_text = f"Summarize comprehensive user preferences based on their behavior in as concise language as possible. If past preferences are provided, adjust the preferences by combining past preferences with those reflected in current behavior, removing conflicting parts, and integrating new insights. If no past preferences are provided, derive the final preferences solely from user behavior.\n\n**Past Preferences:**\n\nNot provided\n\n**This person has chosen or rejected some items or some comments on queries (may be empty):**\n\n"

    for index, item in enumerate(history):
        query = item["query"]
        chosen = item["chosen"]
        rejected = item["rejected"]
        full_text = f"{full_text}{index+1}. *Query:*\n{query}\n\n*Chosen:*\n{chosen}\n\n*Rejected:*\n{rejected}\n\n"

    return full_text


with open(
    "/ossfs/workspace/nas/yuting/data/AlignXplore/streaming_rl_train.json", "r"
) as f:
    data_list = json.load(f)

obj_list = []
hist_len = 0
for idx, line in enumerate(data_list):
    obj = line[0]["history"]
    value1 = obj[0]["value"]
    value2 = obj[1]["value"]
    posts1 = parse_posts(value1)
    posts2 = parse_posts(value2)
    target = {
        "query": obj[0]["task"],
        "chosen": obj[0]["chosen"],
        "rejected": obj[0]["rejected"],
    }
    history = []
    history.extend(posts1)
    # history.append(target)
    history.extend(posts2)
    history.append(target)
    uid = str(114821 + idx)
    record = {
        "logprob1": 0.0,
        "logprob2": 0.0,
        "signal": 0.0,
    }
    item = {
        "uid": str(uid),
        "history": history,
        "mid_position": 4,
        "mid_record": record,
        "last_record": record,
        "signal_gap": 0.0,
        "distance_to_end": 4,
    }
    hist_len += len(history)
    obj_list.append(item)

with open("/ossfs/workspace/nas/yuting/data/AlignXplore/streaming_alignxplore_upi.jsonl", "w") as fout:
    for item in obj_list:
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")
    
print(hist_len/len(obj_list))