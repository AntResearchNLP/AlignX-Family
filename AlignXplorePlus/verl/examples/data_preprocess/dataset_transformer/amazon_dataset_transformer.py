import pandas as pd
import json
import gzip
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# from multiprocessing import Pool, cpu_count # Optional for further parallelization

# --- 1. 配置参数 ---
REVIEWS_FILE = "/ossfs/workspace/nas/yuting/data/Amazon/Books_5.json.gz"
META_FILE = "/ossfs/workspace/nas/yuting/data/Amazon/meta_Books.json.gz"
OUTPUT_TRAIN_FILE = "/ossfs/workspace/nas/yuting/data/Amazon/train.jsonl"
OUTPUT_TEST_FILE = "/ossfs/workspace/nas/yuting/data/Amazon/test.jsonl"
MIN_INTERACTIONS = 17
TEST_SET_SIZE = 0.2
RANDOM_STATE = 42

# --- 2. 数据加载辅助函数 ---
def load_data(file_path):
    data = []
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)


print("--- Step 1: Loading Data ---")
reviews_df = load_data(REVIEWS_FILE)
meta_df = load_data(META_FILE)
print(f"Loaded {len(reviews_df)} reviews and {len(meta_df)} meta entries.")

# --- 3. 数据预处理 (已优化) ---
print("\n--- Step 2: Pre-processing Data (Optimized) ---")

meta_df = meta_df[["asin", "title", "description"]]
meta_df["title"] = meta_df["title"].fillna("").astype(str)
meta_df["description"] = meta_df["description"].fillna("").astype(str)
meta_df = meta_df[meta_df["title"].str.strip() != ""]

### OPTIMIZATION 1: 使用向量化操作创建查找字典，避免iterrows() ###
# 先将title和description合并为单一的文本列
meta_df["full_text"] = (
    "Title: " + meta_df["title"] + "\nDescription: " + meta_df["description"]
)
# 将'asin'设为索引，然后直接转换为字典，速度极快
meta_lookup = meta_df.set_index("asin")["full_text"].to_dict()
print(f"Created metadata lookup for {len(meta_lookup)} items with titles.")

all_items = set(meta_lookup.keys())
interactions_df = reviews_df[["reviewerID", "asin", "unixReviewTime"]].copy()
interactions_df = interactions_df[interactions_df["asin"].isin(all_items)]
print(f"Kept {len(interactions_df)} interactions with corresponding metadata.")

# --- 4. 过滤用户并进行负采样 (已优化) ---
print("\n--- Step 3: Filtering Users and Negative Sampling (Optimized) ---")

user_interaction_counts = interactions_df["reviewerID"].value_counts()
eligible_users = user_interaction_counts[
    user_interaction_counts >= MIN_INTERACTIONS
].index
print(
    f"Found {len(eligible_users)} users with at least {MIN_INTERACTIONS} interactions."
)

filtered_reviews = interactions_df[interactions_df["reviewerID"].isin(eligible_users)]
filtered_reviews = filtered_reviews.sort_values(by=["reviewerID", "unixReviewTime"])

# 预先计算每个用户交互过的物品集合，这步本身是高效的
user_interacted_items = filtered_reviews.groupby("reviewerID")["asin"].apply(set)

user_histories = {}

### OPTIMIZATION 2: 使用 groupby() 迭代，避免在循环中反复筛选DataFrame ###
# 将DataFrame按用户ID分组，这样我们就可以高效地处理每个用户的子集
grouped_reviews = filtered_reviews.groupby("reviewerID")

pbar = tqdm(grouped_reviews, total=len(eligible_users), desc="Processing users")
for user_id, user_group_df in pbar:
    # 'user_group_df' 已经是该用户的所有评论，不再需要从大表中筛选
    interacted_set = user_interacted_items[user_id]
    negative_pool = list(all_items - interacted_set)

    if not negative_pool:
        continue

    positive_asins = user_group_df["asin"].tolist()
    num_positives = len(positive_asins)
    num_to_sample = min(num_positives, len(negative_pool))

    # 如果可用的负样本少于正样本，则只采样可用的数量
    if num_to_sample < num_positives:
        # print(f"Warning: User {user_id} has insufficient negative samples. Required {num_positives}, available {len(negative_pool)}. Sampling {num_to_sample}.")
        positive_asins = positive_asins[:num_to_sample]

    negative_asins = random.sample(negative_pool, num_to_sample)

    behaviors = [
        {"query": None, "chosen": meta_lookup[pos], "rejected": meta_lookup[neg]}
        for pos, neg in zip(positive_asins, negative_asins)
    ]

    if behaviors:
        user_histories[user_id] = behaviors

print(f"Processed histories for {len(user_histories)} users.")

# --- 5. 划分数据集并保存 ---
print("\n--- Step 4: Splitting and Saving Data ---")

final_user_ids = list(user_histories.keys())
train_users, test_users = train_test_split(
    final_user_ids, test_size=TEST_SET_SIZE, random_state=RANDOM_STATE
)
print(
    f"Splitting into {len(train_users)} training users and {len(test_users)} test users."
)

all_users_list = train_users + test_users
user_id_to_uidx = {user_id: i for i, user_id in enumerate(all_users_list)}


def save_to_jsonl(user_list, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for user_id in tqdm(user_list, desc=f"Saving {filename}"):
            if user_id not in user_histories:
                continue

            output_record = {
                "uid": user_id_to_uidx[user_id],
                "task": "Generate users preference based on their histrorical behaviors.\n\n",
                "prompt": "**This person has chosen or rejected some books:**\n\n",
                "behaviors": user_histories[user_id],
                "topic": "book",
            }
            f.write(json.dumps(output_record) + "\n")


save_to_jsonl(train_users, OUTPUT_TRAIN_FILE)
save_to_jsonl(test_users, OUTPUT_TEST_FILE)
print(f"\n✅ All done! Data saved to {OUTPUT_TRAIN_FILE} and {OUTPUT_TEST_FILE}")


with open("/ossfs/workspace/nas/yuting/data/Amazon/statistics", "a+") as f:
    f.writelines(
        f"Num of training users: {len(train_users)}\nNum of testing users: {len(test_users)}"
    )
