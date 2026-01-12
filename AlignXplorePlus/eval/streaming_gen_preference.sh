#!/bin/bash

set -x

# 定义数据集列表
datasets=(
    "amazon"
    "mind"
    "movielens"
    "alignx"
    "student_phd"
    "friendly_unfriendly"
    "concise_detail"
    # 按需添加更多数据集
)

# 循环执行
for dataset in "${datasets[@]}"; do
    echo "Processing $dataset ..."
    python -u streaming_gen_preference.py \
        --model=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --input=upi_benchmark/${dataset}_upi.jsonl \
        --tensor-parallel-size=4
done

