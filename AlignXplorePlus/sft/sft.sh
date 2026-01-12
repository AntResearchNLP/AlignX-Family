#!/bin/bash
set -x

pip uninstall apex -y

nnodes=$1
node_rank=$2
torchrun --nnodes=${nnodes} --nproc_per_node=8 --node_rank=${node_rank} --master_addr=127.0.0.1 --master_port=29500 sft.py
