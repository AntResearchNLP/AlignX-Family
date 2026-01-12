import json
import random
import argparse
from tqdm import tqdm

from transformers import AutoTokenizer

def json2string(user_data, tokenizer):
    full_sequence = "Given a sequence of user preference histories, each containing:\n- A prompt text enclosed in <prompt> tags (may be empty)\n- Two options marked with <option_A> and <option_B> tags\n- User's preference (A or B) enclosed in <preference> tags\n\nYour task is to predict each preference by analyzing patterns from all previous records in the sequence."
    user_metadata = {
        "uid": user_data["uid"],
        "pref_indices": [],
    }

    if len(user_data["behaviors"]) > 100:
        user_data["behaviors"] = user_data["behaviors"][-100:]

    for _, item in enumerate(user_data["behaviors"]):
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
        prefix_len = len(tokenizer.encode(full_sequence + segment))

        full_sequence += segment + q + postfix  # 添加 preference 和 EOS

        # 记录偏好token的索引（在token化后）
        # 注意：CausalLM预测第N个token用的是前N-1个token的输出，所以索引是 prefix_len - 1
        user_metadata["pref_indices"].append(prefix_len - 1)

    return {"text": full_sequence, "indices": user_metadata["pref_indices"]}


tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-1.7B",
    trust_remote_code=True,
)


parser = argparse.ArgumentParser(description="Process some integers.")

parser.add_argument("--split", type=int, help="")
parser.add_argument("--filename", type=str, help="")

args = parser.parse_args()

split = args.split
filename = args.filename


file = f"data/{filename}/train.jsonl"


print(f"Processing {file}")
with open(file, "r") as f:
    all_user = []
    for line in f:
        user = json.loads(line)
        user_str = json2string(user, tokenizer)
        all_user.append(user_str)

with open(f"sft.jsonl", "a+", encoding="utf-8") as fw:
    for u in all_user:
        fw.write(json.dumps(u, ensure_ascii=False) + "\n")

print(f"Done processing {file}")
