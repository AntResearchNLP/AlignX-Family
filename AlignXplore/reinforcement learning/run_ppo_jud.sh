export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_P2P_DISABLE=1

DEBUG_MODE=False python -m playground.orz_7b_grpo_jud \
