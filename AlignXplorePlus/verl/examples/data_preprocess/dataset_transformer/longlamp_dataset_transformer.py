"""
{
    "uid": user,
    "input": input_str,
    "output": output_str,
    "task": "Generate the user's preference based on their historical behaviors.\n\n",
    "prompt": f"**This person has written some {top}:**\n\n",
    "behaviors": behaviors,
    "topic": topic
}
"""

import os
import json
from datasets import load_dataset
from tqdm.auto import tqdm

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# --- 1. 配置信息 ---
# 定义我们要处理的所有数据集及其对应的 topic 标签
DATASET_CONFIGS = [
    {"config_name": "topic_writing_user", "topic": "topic_writing"},
    {"config_name": "abstract_generation_user", "topic": "abstract_generation"},
    {"config_name": "product_review_user", "topic": "product_review"},
]

# Hugging Face 数据集名称
DATASET_HUB_NAME = "/ossfs/workspace/nas/yuting/data/LongLaMP"

# 输出目录和文件名
OUTPUT_DIR = "/ossfs/workspace/nas/yuting/data/LongLaMP/"
TRAIN_FILENAME = "train.jsonl"
TEST_FILENAME = "test.jsonl"

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- 2. 数据转换函数 ---
# 这个函数负责将原始数据条目转换为你想要的格式
def format_example(example, topic):
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
    input_str = example["input"]
    output_str = example["output"]
    historical = example["profile"]
    behaviors = []
    if topic == "topic_writing":
        user_key = "author"
        prefix = "Generate content for the reddit post"
        query_key = "summary"
        content_key = "content"
        top = "reddit posts"
    elif topic == "abstract_generation":
        user_key = "name"
        prefix = "Generate an abstract for the title"
        query_key = "title"
        content_key = "abstract"
        top = "abstracts"
    elif topic == "product_review":
        user_key = "reviewerId"
        query_key = "description"
        content_key = "reviewText"
        prefix = "Generate the review text for the product with description"
        top = "reviews"
    else:
        raise KeyError

    for _, his in enumerate(historical):
        behavior_entry = {
            "query": f"{prefix} {his[query_key]}",
            "chosen": his[content_key],
            "rejected": None,
        }
        behaviors.append(behavior_entry)

    # 构建最终的 JSON 对象
    formatted_json = {
        "uid": example[user_key],
        "input": input_str,
        "output": output_str,
        "task": "Generate the user's preference based on their historical behaviors.\n\n",
        "prompt": f"**This person has written some {top}:**\n\n",
        "behaviors": behaviors,
        "topic": topic,
    }
    return formatted_json


# --- 3. 主处理逻辑 ---
def main():
    # 初始化用于存放所有数据的列表
    all_train_data = []
    all_test_data = []

    print("--- 开始处理所有数据集 ---")

    # 遍历我们定义好的数据集配置
    for config in DATASET_CONFIGS:
        config_name = config["config_name"]
        topic = config["topic"]
        print(f"\n>>> 正在处理配置: {config_name} (Topic: {topic})")

        # --- 处理训练集 ---
        train_split = load_dataset(DATASET_HUB_NAME, name=config_name, split="train")
        print(f"  - 已加载 'train' split，包含 {len(train_split)} 条数据。")

        # 逐条转换数据并添加到总列表
        # 使用 tqdm 显示进度条
        for example in tqdm(train_split, desc=f"  - 转换 train 数据 ({config_name})"):
            formatted_data = format_example(example, topic)
            all_train_data.append(formatted_data)

        # --- 处理验证集 ---
        val_split = load_dataset(DATASET_HUB_NAME, name=config_name, split="val")
        print(f"  - 已加载 'val' split，包含 {len(val_split)} 条数据。")

        # 逐条转换数据并添加到总列表
        for example in tqdm(val_split, desc=f"  - 转换 val 数据 ({config_name})"):
            formatted_data = format_example(example, topic)
            all_test_data.append(formatted_data)

        # --- 处理测试集 ---
        test_split = load_dataset(DATASET_HUB_NAME, name=config_name, split="test")
        print(f"  - 已加载 'test' split，包含 {len(test_split)} 条数据。")

        # 逐条转换数据并添加到总列表
        for example in tqdm(test_split, desc=f"  - 转换 test 数据 ({config_name})"):
            formatted_data = format_example(example, topic)
            all_test_data.append(formatted_data)

    print("\n--- 所有数据集处理完毕 ---")
    print(f"总计训练数据条数: {len(all_train_data)}")
    print(f"总计测试数据条数: {len(all_test_data)}")

    # --- 4. 保存为 JSONL 文件 ---
    # 保存合并后的训练集
    train_output_path = os.path.join(OUTPUT_DIR, TRAIN_FILENAME)
    print(f"\n正在保存合并后的训练集到: {train_output_path}")
    with open(train_output_path, "w", encoding="utf-8") as f:
        for record in tqdm(all_train_data, desc="保存 train.jsonl"):
            # ensure_ascii=False 保证中文字符正确显示
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 保存合并后的测试集
    test_output_path = os.path.join(OUTPUT_DIR, TEST_FILENAME)
    print(f"正在保存合并后的测试集到: {test_output_path}")
    with open(test_output_path, "w", encoding="utf-8") as f:
        for record in tqdm(all_test_data, desc="保存 test.jsonl"):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n--- 全部完成！---")
    print(f"训练集文件: {train_output_path}")
    print(f"测试集文件: {test_output_path}")
    with open("/ossfs/workspace/nas/yuting/data/LongLaMP/statistics", "a+") as f:
        f.writelines(
            f"Num of training users: {len(all_train_data)}\nNum of testing users: {len(all_test_data)}"
        )


# --- 运行主程序 ---
if __name__ == "__main__":
    main()
