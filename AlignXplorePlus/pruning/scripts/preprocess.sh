set -x

node_rank=$1

torchrun \
  --nnodes=8 \
  --nproc_per_node=8 \
  --node_rank=${node_rank} \
  --master_addr=127.0.0.1 \
  --master_port=29500 \
  preprocess.py

