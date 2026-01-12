#!/bin/bash
set -x

# --- 用户需要配置的变量 ---
# 节点总数 (机器数量)
export NUM_NODES=8
# 每台机器使用的GPU数量
export NUM_GPUS_PER_NODE=8
# 主节点的IP地址 (rank 0 的机器)
export MASTER_ADDR="127.0.0.1"
# 主节点的通信端口
export MASTER_PORT=12233
# Python 脚本的路径
PYTHON_SCRIPT="preprocess.py" # <-- 修改为你的Python脚本文件名
# 数据集参数
DATASET="MIND" # 或 MIND, AlignX, Amazon

# --- 不需要修改的部分 ---
# 节点排名(rank)，将通过命令行参数传入
NODE_RANK=$1
if [ -z "$NODE_RANK" ]; then
    echo "错误: 请提供节点排名 (NODE_RANK) 作为第一个参数, 例如: ./run_distributed.sh 0"
    exit 1
fi

# 计算总进程数
TOTAL_PROCESSES=$((NUM_NODES * NUM_GPUS_PER_NODE))

echo "启动节点: RANK=${NODE_RANK} of ${NUM_NODES}"
echo "主节点地址: ${MASTER_ADDR}:${MASTER_PORT}"
echo "总进程数: ${TOTAL_PROCESSES}"

# 使用 accelerate launch 启动
accelerate launch \
    --multi_gpu \
    --num_machines=${NUM_NODES} \
    --num_processes=${TOTAL_PROCESSES} \
    --machine_rank=${NODE_RANK} \
    --main_process_ip=${MASTER_ADDR} \
    --main_process_port=${MASTER_PORT} \
    ${PYTHON_SCRIPT} --dataset ${DATASET}

echo "Finish, strat blocking GPU"

