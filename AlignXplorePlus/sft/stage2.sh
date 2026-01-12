set -x

# pip install -U vllm trl
dataset=$1
split=$2

python stage2.py --dataset=$dataset --split=$split

