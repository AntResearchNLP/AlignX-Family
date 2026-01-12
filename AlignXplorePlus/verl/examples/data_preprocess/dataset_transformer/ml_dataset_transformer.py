"""
    UniformDataFormat: [
        {
            uid: user index,
            task: "Generate users preference based on their histrorical behaviors.",
            topic: f"**This person has chosen or rejected comments on some posts/items/...:**\n\n",
            behaviors: [
                {
                    query(optioinal): Query of posts,
                    chosen: User-liked item/post/...,
                    rejected: User-disliked item/post/...,
                },...
            ]
        },...
    ]
"""

import json
import pandas as pd
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# --- 前序步骤精简代码 ---
print("正在执行前序数据处理步骤...")
DATA_DIR = "/ossfs/workspace/nas/yuting/data/MovieLens/ml-32m/"
# 加载数据
df_ratings = pd.read_csv(f"{DATA_DIR}ratings.csv")
df_movies = pd.read_csv(f"{DATA_DIR}movies.csv")

# ---------------------------------------------------------------------------
# 全流程代码：加载、聚合、过滤、采样、划分、格式化
# ---------------------------------------------------------------------------

# --- 1. 加载和聚合数据 ---
df_ratings.sort_values(by=["userId", "timestamp"], inplace=True)
user_interactions = df_ratings.groupby("userId")["movieId"].apply(list).reset_index()
user_interactions.rename(columns={"movieId": "positive_sequence"}, inplace=True)

# --- 2. 核心修改：过滤交互过少的用户 ---
print(f"过滤前的总用户数: {len(user_interactions)}")
# 计算每个用户的交互数量
user_interactions["interaction_count"] = user_interactions["positive_sequence"].apply(
    len
)
# 保留交互数量大于等于17的用户
min_interactions = 17
user_interactions = user_interactions[
    user_interactions["interaction_count"] >= min_interactions
].copy()
print(f"过滤后 (交互次数 >= {min_interactions}) 的用户数: {len(user_interactions)}")
# ----------------------------------------------


# --- 3. 负采样 ---
all_movie_ids = set(df_ratings["movieId"].unique())

print("Negative sampling...")


def sample_negatives(pos_list, all_items):
    neg_candidates = list(all_items - set(pos_list))
    return random.sample(neg_candidates, min(len(pos_list), len(neg_candidates)))


user_interactions["negative_samples"] = user_interactions["positive_sequence"].apply(
    lambda p: sample_negatives(p, all_movie_ids)
)

# user_interactions['negative_samples'] = user_interactions.apply(
#     lambda row: sample_negatives(
#         row['positive_sequence'],
#         all_movie_ids,
#         row['interaction_count']
#     ),
#     axis=1
# )

# --- 4. 用户划分（训练/测试） ---
train_user_ids, test_user_ids = train_test_split(
    user_interactions["userId"].unique(), test_size=0.2, random_state=42
)
train_data = user_interactions[user_interactions["userId"].isin(train_user_ids)].copy()
test_data = user_interactions[user_interactions["userId"].isin(test_user_ids)].copy()
print(f"训练集用户数: {len(train_data)}")
print(f"测试集用户数: {len(test_data)}")

# --- 5. 合并电影元数据 ---
movies_lookup = df_movies.set_index("movieId")


def enrich_sequence(id_list, lookup):
    enriched = []
    for mid in id_list:
        try:
            info = lookup.loc[mid]
            enriched.append(
                {"movieId": mid, "title": info["title"], "genres": info["genres"]}
            )
        except KeyError:
            continue
    return enriched


train_data["history_sequence"] = train_data["positive_sequence"].apply(
    lambda ids: enrich_sequence(ids, movies_lookup)
)
test_data["history_sequence"] = test_data["positive_sequence"].apply(
    lambda ids: enrich_sequence(ids, movies_lookup)
)
print("前序数据处理完成。")
# --- 前序步骤结束 ---

# --- 6. 格式化并保存为 JSONL ---
def format_and_save_as_jsonl(data_df, output_filename, movies_lookup_df):
    """
    将用户交互数据格式化为指定的JSONL格式并保存。
    """
    with open(output_filename, "w", encoding="utf-8") as f:
        # 遍历每个用户（DataFrame的每一行）
        for _, row in tqdm(
            data_df.iterrows(), total=len(data_df), desc=f"正在写入 {output_filename}"
        ):
            # 如果没有正样本或负样本，则跳过该用户
            if not row["history_sequence"] or not row["negative_samples"]:
                continue

            behaviors = []
            prompt = "**This person has chosen or rejected some movies:**\n\n"
            # 遍历用户的每个正向交互（chosen item）
            for i, chosen_movie_info in enumerate(row["history_sequence"]):
                # 为每个正样本随机匹配一个负样本
                rejected_movie_id = random.choice(row["negative_samples"])
                try:
                    rejected_movie_info = movies_lookup_df.loc[rejected_movie_id]
                except KeyError:
                    continue  # 如果找不到负样本信息，则跳过此配对

                # 格式化chosen和rejected的描述字符串
                chosen_str = f"Title: {chosen_movie_info['title']}, Genres: {chosen_movie_info['genres']}"
                rejected_str = f"Title: {rejected_movie_info['title']}, Genres: {rejected_movie_info['genres']}"

                # prompt = (
                #     f"{prompt}"
                #     f"{i+1}.\n"
                #     f"*Chosen:*\n{chosen_str}\n\n"
                #     f"*Rejected:*\n{rejected_str}\n\n"
                # )

                behavior_entry = {
                    "query": None,
                    "chosen": chosen_str,
                    "rejected": rejected_str,
                }
                behaviors.append(behavior_entry)

            # 构建最终的用户JSON对象
            # prompt = (
            #     "Generate the user's preference based on their historical behavior.\n\n"
            #     f"{prompt}"
            # )
            user_json_obj = {
                "uid": int(row["userId"]),
                "task": "Generate the user's preference based on their historical behaviors.\n\n",
                "prompt": prompt,
                "behaviors": behaviors,
                "topic": "movie",
            }
            f.write(json.dumps(user_json_obj) + "\n")


print("\n开始格式化并保存训练数据...")
format_and_save_as_jsonl(
    train_data,
    "/ossfs/workspace/nas/yuting/data/MovieLens/ml-32m/train.jsonl",
    movies_lookup,
)

print("\n开始格式化并保存测试数据...")
format_and_save_as_jsonl(
    test_data,
    "/ossfs/workspace/nas/yuting/data/MovieLens/ml-32m/test.jsonl",
    movies_lookup,
)

with open("/ossfs/workspace/nas/yuting/data/MovieLens/ml-32m/statistics", "a+") as f:
    f.writelines(
        f"Num of training users: {len(train_data)}\nNum of testing users: {len(test_data)}"
    )
