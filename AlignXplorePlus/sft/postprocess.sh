set -x

# pip install -U vllm trl
dataset=$1
split_num=$2

python postprocess.py --dataset=$dataset --split_num=$split_num