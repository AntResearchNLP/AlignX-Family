import os
from datasets import load_dataset, load_from_disk, concatenate_datasets
from transformers import AutoTokenizer
import torch.distributed as dist
import torch
import psutil
import datetime


# ========== 1. 初始化分布式 ==========
def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="gloo", timeout=datetime.timedelta(seconds=36000))  # 用CPU用gloo后端
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


# ========== 2. 切分数据 ==========
def split_dataset(dataset, rank, world_size):
    total = len(dataset)
    per_rank = total // world_size
    start = rank * per_rank
    end = (rank + 1) * per_rank if rank != world_size - 1 else total
    return dataset.select(range(start, end))


# ========== 3. 批量tokenizer+mask ==========
def tokenize_and_mask_batch(batch):
    tokenized = tokenizer(
        batch["text"],
        truncation=True,
        max_length=8192,
        padding="max_length"
    )

    all_labels = []
    for input_ids, keep_indices in zip(tokenized["input_ids"], batch["indices"]):
        labels = [-100] * len(input_ids)
        for idx in keep_indices:
            if 0 <= idx < len(labels):
                labels[idx] = input_ids[idx]
        all_labels.append(labels)

    tokenized["labels"] = all_labels
    return tokenized


# ========== 4. config ==========
MODEL_NAME = "Qwen/Qwen3-1.7B"
RAW_DATA_FILE = "sftdata.jsonl"
PROCESSED_DATA_DIR = "processed_dataset"

# 动态设定 CPU 进程内并行数（给 map 用）
TOTAL_CPU = psutil.cpu_count(logical=True)
# 在多进程 torchrun 下，每个rank占一部分CPU核心
def get_num_proc_per_rank():
    if "WORLD_SIZE" in os.environ:
        procs_per_node = int(os.environ["nproc_per_node"]) if "nproc_per_node" in os.environ else 1
        return max(1, TOTAL_CPU // procs_per_node)
    else:
        return max(1, TOTAL_CPU)
    

if __name__ == "__main__":
    # 禁用 GPU（只用CPU）
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_fast=True
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    rank, world_size = setup_distributed()

    # 如果数据已经处理过
    if os.path.exists(PROCESSED_DATA_DIR):
        if rank == 0:
            print(f"[Rank {rank}] 已存在处理数据 {PROCESSED_DATA_DIR}，直接退出")
        exit(0)

    if rank == 0:
        print(f"[Rank {rank}] 加载原始数据...")
    raw_dataset = load_dataset("json", data_files=RAW_DATA_FILE)["train"]

    # 切片
    dataset_shard = split_dataset(raw_dataset, rank, world_size)

    # map 处理
    num_proc_local = get_num_proc_per_rank()
    print(f"[Rank {rank}] 开始处理 {len(dataset_shard)} 条数据，用 {num_proc_local} CPU core(s)")
    dataset_shard = dataset_shard.map(
        tokenize_and_mask_batch,
        batched=True,
        batch_size=16,
        num_proc=num_proc_local,
        remove_columns=["text", "indices"]
    )

    # 保存自己的shard
    shard_dir = f"temp/temp_dataset_rank{rank}"
    dataset_shard.a(shard_dir)

    # 同步
    if dist.is_initialized():
        dist.barrier()

    # rank0 合并保存
    if rank == 0:
        shards = [load_from_disk(f"temp/temp_dataset_rank{i}") for i in range(world_size)]
        merged_dataset = concatenate_datasets(shards)
        merged_dataset.save_to_disk(PROCESSED_DATA_DIR)
        print(f"[Rank 0] 数据已保存到 {PROCESSED_DATA_DIR}")

    if dist.is_initialized():
        dist.barrier()
