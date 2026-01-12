import os
import pandas as pd
import random
import json
from tqdm import tqdm

# --- 1. 配置 ---
DATA_PATHS = {
    "train": "/ossfs/workspace/nas/yuting/data/MIND/MIND_train",
    "dev": "/ossfs/workspace/nas/yuting/data/MIND/MIND_dev",  # 我们将dev集作为测试集
}
OUTPUT_DIR = "/ossfs/workspace/nas/yuting/data/MIND/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TRAIN_RATIO = 0.8
USER_INTERACTION_THRESHOLD = 9

# --- 2. 加载所有数据 ---
print("Step 1: Loading all data (this may take a while for MIND-large)...")
news_cols = [
    "NewsID",
    "Category",
    "SubCategory",
    "Title",
    "Abstract",
    "URL",
    "TitleEntities",
    "AbstractEntities",
]
behaviors_cols = ["ImpressionID", "UserID", "Time", "History", "Impressions"]

all_news_df = pd.concat(
    [
        pd.read_csv(
            os.path.join(path, "news.tsv"), sep="\t", header=None, names=news_cols
        )
        for path in DATA_PATHS.values()
    ],
    ignore_index=True,
).drop_duplicates("NewsID")
print(f"Loaded {len(all_news_df)} unique news articles.")

all_behaviors_df = pd.concat(
    [
        pd.read_csv(
            os.path.join(path, "behaviors.tsv"),
            sep="\t",
            header=None,
            names=behaviors_cols,
        )
        for path in DATA_PATHS.values()
    ],
    ignore_index=True,
)
print(f"Loaded {len(all_behaviors_df)} total impression logs.")

# --- 3. 创建全局新闻资源 ---
print("\nStep 2: Creating global news map and ID set...")
all_news_df["Title"] = all_news_df["Title"].fillna("")
all_news_df["Abstract"] = all_news_df["Abstract"].fillna("")
news_content_map = (
    all_news_df.set_index("NewsID")
    .apply(lambda r: f"Title: {r['Title']}\nAbstract: {r['Abstract']}", axis=1)
    .to_dict()
)
all_news_ids = set(all_news_df["NewsID"].tolist())
del all_news_df  # 释放内存

# --- 4. NEW: 用户级别聚合 ---
print("\nStep 3: Aggregating behaviors at the user level...")
all_behaviors_df.fillna({"History": "", "Impressions": ""}, inplace=True)

# 定义一个聚合函数来获取最长的历史记录和所有曝光的列表
def get_longest_history(series):
    return max(series, key=len)


# 按UserID分组并聚合
user_agg_df = (
    all_behaviors_df.groupby("UserID")
    .agg(
        {
            "History": get_longest_history,
            "Impressions": list,  # 将所有impression字符串收集到一个列表中
        }
    )
    .reset_index()
)

print(f"Aggregated behaviors for {len(user_agg_df)} unique users.")
del all_behaviors_df  # 释放内存

# --- 5. NEW: 从聚合数据中解析交互 ---
print("\nStep 4: Parsing interactions from aggregated user data...")
parsed_interactions = []

for _, user_row in tqdm(
    user_agg_df.iterrows(), total=len(user_agg_df), desc="Processing aggregated users"
):
    user_id = user_row["UserID"]

    # 1. 处理唯一的历史记录 (全局负采样)
    history_news = user_row["History"].split()
    if history_news:
        clicked_history_set = set(history_news)
        available_for_neg_sampling = list(all_news_ids - clicked_history_set)
        if available_for_neg_sampling:
            for pos_news_id in history_news:
                neg_news_id = random.choice(available_for_neg_sampling)
                parsed_interactions.append(
                    {
                        "UserID": user_id,
                        "PositiveNewsID": pos_news_id,
                        "NegativeNewsID": neg_news_id,
                    }
                )

    # 2. 处理合并的曝光列表 (曝光内负采样)
    for impression_str in user_row["Impressions"]:
        if not impression_str:
            continue

        impressions = impression_str.split()
        positive_impressions = [imp[:-2] for imp in impressions if imp.endswith("-1")]
        negative_impressions = [imp[:-2] for imp in impressions if imp.endswith("-0")]

        if positive_impressions and negative_impressions:
            for pos_news_id in positive_impressions:
                neg_news_id = random.choice(negative_impressions)
                parsed_interactions.append(
                    {
                        "UserID": user_id,
                        "PositiveNewsID": pos_news_id,
                        "NegativeNewsID": neg_news_id,
                    }
                )

interactions_df = pd.DataFrame(parsed_interactions)
del user_agg_df, parsed_interactions  # 释放内存
print(f"Created {len(interactions_df)} total interaction pairs from aggregated data.")

# --- 6. 过滤用户 & 合并新闻内容 (与之前相同) ---
print("\nStep 5: Filtering users and merging news content...")
user_interaction_counts = interactions_df["UserID"].value_counts()
eligible_users = user_interaction_counts[
    user_interaction_counts >= USER_INTERACTION_THRESHOLD
].index.tolist()
filtered_df = interactions_df[interactions_df["UserID"].isin(eligible_users)].copy()
del interactions_df  # 释放内存


def get_news_content(news_id):
    return news_content_map.get(news_id, "")


tqdm.pandas(desc="Mapping chosen content")
filtered_df["chosen"] = filtered_df["PositiveNewsID"].progress_map(get_news_content)
tqdm.pandas(desc="Mapping rejected content")
filtered_df["rejected"] = filtered_df["NegativeNewsID"].progress_map(get_news_content)

print(
    f"Found {len(eligible_users)} eligible users (>= {USER_INTERACTION_THRESHOLD} interactions)."
)
print(f"Total interactions after filtering & merging: {len(filtered_df)}")

# --- 7. 严格的用户级别划分 (与之前相同) ---
print(
    f"\nStep 6: Splitting users into train/test sets ({TRAIN_RATIO*100}%/{100-TRAIN_RATIO*100}%)..."
)
all_unique_users = filtered_df["UserID"].unique().tolist()
random.shuffle(all_unique_users)
split_index = int(len(all_unique_users) * TRAIN_RATIO)
train_user_ids = set(all_unique_users[:split_index])
test_user_ids = set(all_unique_users[split_index:])
print(f"New training users: {len(train_user_ids)}")
print(f"New test users: {len(test_user_ids)}")

# --- 8. 生成最终的 JSONL 文件 (与之前相同) ---
print("\nStep 7: Generating train.jsonl and test.jsonl files...")


def generate_jsonl_file(dataframe, user_id_set, output_path):
    subset_df = dataframe[dataframe["UserID"].isin(user_id_set)]
    with open(output_path, "w", encoding="utf-8") as f:
        # 使用 groupby 以保证每个用户的行为被组织在一起
        for user_id, group in tqdm(
            subset_df.groupby("UserID"), desc=f"Writing {os.path.basename(output_path)}"
        ):
            user_behaviors = [
                {"query": None, "chosen": row["chosen"], "rejected": row["rejected"]}
                for _, row in group.iterrows()
            ]

            f.write(
                json.dumps(
                    {
                        "uid": user_id,
                        "task": "Generate users preference based on their histrorical behaviors.\n\n",
                        "prompt": "**This person has chosen or rejected some news:**\n\n",
                        "behaviors": user_behaviors,
                        "topic": "news",
                    }
                )
                + "\n"
            )


generate_jsonl_file(
    filtered_df, train_user_ids, os.path.join(OUTPUT_DIR, "train.jsonl")
)
generate_jsonl_file(filtered_df, test_user_ids, os.path.join(OUTPUT_DIR, "test.jsonl"))

print("\nProcessing complete!")
print(f"Output files are saved in the '{OUTPUT_DIR}' directory.")

with open("/ossfs/workspace/nas/yuting/data/MIND/statistics", "a+") as f:
    f.writelines(
        f"Num of training users: {len(train_user_ids)}\nNum of testing users: {len(test_user_ids)}"
    )
