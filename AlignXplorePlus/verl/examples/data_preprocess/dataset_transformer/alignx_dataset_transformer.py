"""
{
    "uid": user,
    "input": prompt,
    "output": chosen,
    "chosen": chosen,
    "rejected": rejected,
    "task": "Generate the user's preference based on their historical behaviors.\n\n",
    "prompt": "**This person has chosen or rejected comments on some posts:**\n\n",
    "behaviors": behaviors,
    "topic": topic
}
"""

import os
import json
from datasets import load_dataset
from tqdm.auto import tqdm


# Hugging Face 数据集名称
DATASET_HUB_NAME = "/ossfs/workspace/nas/yuting/data/AlignX/"

# 输出目录和文件名
OUTPUT_DIR = "/ossfs/workspace/nas/yuting/data/AlignX/"
TRAIN_FILENAME = "train.jsonl"
TEST_FILENAME = "test.jsonl"

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- 2. 数据转换函数 ---
# 这个函数负责将原始数据条目转换为你想要的格式
def format_example(example, topic, index):
    """
    将原始数据条目转换为目标 JSON 格式。

    Args:
        example (dict): 从 Hugging Face 数据集加载的单条数据, e.g., {'input': '...', 'output': '...'}
        topic (str): 这条数据所属的主题.

    Returns:
        dict: 格式化后的数据.
    """
    # 原始数据集的 'input' 字段是用户的历史写作内容
    # 原始数据集的 'output' 字段是生成的偏好总结
    input_str = example["prompt"]
    chosen = example["chosen"]
    rejected = example["rejected"]
    historical = example["Pair-wise Comparative Feedback"]
    if len(historical) < 8:
        return None
    behaviors = []

    for _, his in enumerate(historical):
        behavior_entry = {
            "query": his["prompt"],
            "chosen": his["chosen"],
            "rejected": his["rejected"],
        }
        behaviors.append(behavior_entry)

    # 构建最终的 JSON 对象
    formatted_json = {
        "uid": index,
        "input": input_str,
        "chosen": chosen,
        "output": chosen,
        "rejected": rejected,
        "task": "Generate the user's preference based on their historical behaviors.\n\n",
        "prompt": "**This person has chosen or rejected comments on some posts:**\n\n",
        "behaviors": behaviors,
        "topic": topic,
    }
    return formatted_json


# --- 3. 主处理逻辑 ---
def main():
    # 初始化用于存放所有数据的列表
    all_train_data = []
    # all_test_data = []

    print("--- 开始处理所有数据集 ---")

    data = load_dataset("/ossfs/workspace/nas/yuting/data/AlignX/")
    train_split = data["train"]
    # breakpoint()
    index = 0
    for example in tqdm(train_split, desc="转换 train 数据"):
        formatted_data = format_example(example, "alignx", index)
        if formatted_data is None:
            continue
        all_train_data.append(formatted_data)
        index += 1

    # --- 4. 保存为 JSONL 文件 ---
    # 保存合并后的训练集
    train_output_path = os.path.join(OUTPUT_DIR, TRAIN_FILENAME)
    print(f"\n正在保存合并后的训练集到: {train_output_path}")
    with open(train_output_path, "w", encoding="utf-8") as f:
        for record in tqdm(all_train_data, desc="保存 train.jsonl"):
            # ensure_ascii=False 保证中文字符正确显示
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 保存合并后的测试集
    # test_output_path = os.path.join(OUTPUT_DIR, TEST_FILENAME)
    # print(f"正在保存合并后的测试集到: {test_output_path}")
    # with open(test_output_path, "w", encoding="utf-8") as f:
    #     for record in tqdm(all_test_data, desc="保存 test.jsonl"):
    #         f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n--- 全部完成！---")
    print(f"训练集文件: {train_output_path}")
    # print(f"测试集文件: {test_output_path}")
    with open("/ossfs/workspace/nas/yuting/data/AlignX/statistics", "w+") as f:
        f.writelines(
            f"Num of training users: {len(all_train_data)}\nNum of testing users: 0"
        )


# --- 运行主程序 ---
if __name__ == "__main__":
    main()
