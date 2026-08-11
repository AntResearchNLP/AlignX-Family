<div align="center">
  <h1 style="font-size: 40px;">AlignXada</h1>
  <p>Learning Preference Adaptation for Large Language Model Personalization via Verbal Reinforcement Learning</p>

  [![arXiv](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2608.09507)
  [![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Benchmark-yellow)](https://huggingface.co/datasets/VanillaH1/AlignXada-Benchmark)
</div>

## 📖 Introduction

<strong>AlignXada</strong> is a training-free verbal meta-learning framework for adapting a task-agnostic universal user profile to a specific downstream personalization task. A meta model learns a reusable textual refinement policy from a small support set, a frozen rewrite model applies that policy to unseen profiles, and a frozen downstream model consumes the refined profiles.

This directory contains the minimal runnable release. It intentionally excludes dataset construction, baseline implementations, SFT/RL experiments, analysis code, private endpoints, and result artifacts.

<p align="center">
  <img src="figures/alignxada_framework.png" width="100%" alt="AlignXada framework"/>
</p>
<p align="center">
  <em>Figure 2. Overview of AlignXada. A task-specific refinement policy is induced from a small support set and then frozen for held-out deployment; all model weights remain frozen throughout.</em>
</p>

## ✨ Key Features

- <strong>Training-Free Adaptation</strong>: AlignXada refines natural-language user profiles without updating model parameters.
- <strong>Task-Adaptive</strong>: A textual policy is learned for each downstream task and model configuration, then reused across held-out users.
- <strong>Model-Agnostic</strong>: Separate meta, rewrite, and downstream model roles can use OpenAI-compatible endpoints; the Python runtime also supports local vLLM models.
- <strong>Adaptive Support Sampling</strong>: The optimizer focuses policy updates on informative support examples and can gate candidate policies on a held-out development set.

## 📊 Main Results

The table below summarizes the main results across 13 tasks. Performance columns report the average task-level percentage-point change relative to the raw universal-profile baseline; higher is better. Token ratio (TR) is the refined profile length divided by the original profile length; lower is better.

| Downstream Model | RAG Avg. Δ vs. Raw | AlignXada Avg. Δ vs. Raw | Token Ratio (TR) ↓ |
|---|---:|---:|---:|
| Qwen3-8B | -3.59 | **+2.27** | 20.8% |
| DeepSeek-V4-Flash | -4.86 | **+7.00** | 25.5% |
| GPT-5-mini | -1.92 | **+2.19** | 22.2% |
| **Overall** | **-3.46** | **+3.82** | **22.8%** |

Across 39 task-model cells, AlignXada improves 33 cells over the raw-profile baseline and outperforms RAG in 36 cells while retaining only 22.8% of the original profile tokens. See the [paper](https://arxiv.org/abs/2608.09507) for complete task-level results and significance analysis.

## 🗂️ Repository Layout

```text
profile_meta_core.py                Shared data structures and utilities
profile_meta_prompts.py             Rewrite and meta-learning prompts
profile_meta_tasks.py               Task adapters and registry
profile_meta_runtime.py             OpenAI-compatible and local vLLM runtimes
profile_meta_optimizer.py           Core policy-learning implementation
profile_meta_optimizer_adaptive.py  Adaptive support-sampling entry point
personamem_v2.py                    PersonaMem-v2 task adapter
memorycd.py                         MemoryCD task adapter
evaluate_select_openai.py           Held-out evaluation of rewritten profiles
scripts/run_adaptive.sh             Run one adaptive AlignXada experiment
scripts/evaluate.sh                 Evaluate one rewritten-query file
```

## 🚀 Quick Start

### Requirements

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The supplied scripts use OpenAI-compatible chat-completions APIs. To use local vLLM models directly through the CLI, install `transformers` and `vllm` separately.

### Input Data

AlignXada consumes a pre-built JSON array. For the main three-way protocol, every row must contain `metadata.user_split` with one of the following values:

- `train_support` (or `support`) for policy induction;
- `selection` for development-time policy selection;
- `query` for final held-out evaluation.

Users must be disjoint across these splits. A minimal PersonaMem-v2-compatible row has the following form:

```json
{
  "profile": "A task-agnostic user profile...",
  "persona_id": "user_001",
  "question_id": "question_001",
  "pref_type": "preference_type",
  "target": {
    "query": "The downstream user query...",
    "correct_answer": "The preferred response...",
    "incorrect_answers": [
      "Distractor response 1...",
      "Distractor response 2...",
      "Distractor response 3..."
    ],
    "scenario": "personal_email"
  },
  "metadata": {
    "user_split": "train_support"
  }
}
```

MemoryCD rows use the same top-level `profile` and `metadata.user_split` fields. Their `target` object contains `task`, `domain`, `target_item`, `target_interaction`, `candidate_items`, and `user_id`; see `memorycd.py` for the accepted item and interaction fields.

Registered task keys include:

- `personamem_v2_<scenario>` for the nine PersonaMem-v2 scenarios;
- `memorycd_item_ranking`, `memorycd_rating_prediction`, `memorycd_review_title`, and `memorycd_review_generation`;
- `pairwise` and `lamp1` through `lamp7` for compatible pre-built files.

### Run Adaptive Policy Learning

Set the dataset, task, three model roles, and OpenAI-compatible endpoints. Never commit real credentials.

```bash
PROFILE_INPUT=path/to/profile_input.json \
TASK=personamem_v2_personal_email \
QUERY_SIZE=100 \
META_MODEL=YOUR_META_MODEL \
META_BASE_URL=https://your-meta-endpoint.example/v1 \
META_API_KEY=YOUR_META_API_KEY \
REWRITE_MODEL=YOUR_REWRITE_MODEL \
REWRITE_BASE_URL=https://your-rewrite-endpoint.example/v1 \
REWRITE_API_KEY=YOUR_REWRITE_API_KEY \
TARGET_MODEL=YOUR_TARGET_MODEL \
TARGET_BASE_URL=https://your-target-endpoint.example/v1 \
TARGET_API_KEY=YOUR_TARGET_API_KEY \
bash scripts/run_adaptive.sh
```

Important optional settings:

```bash
SUPPORT_SIZE=40
DEV_SUPPORT_SIZE=20
FEEDBACK_BATCH_SIZE=20
NUM_UPDATES=5
SUPPORT_SAMPLING_STRATEGY=pref_type_stratified
ADAPTIVE_SELECTION_STRATEGY=raw_role_weighted
ADAPTIVE_DEV_GATE=strict_improve
BEST_PATTERN_STRATEGY=best_epoch
PROFILE_META_PROMPT_VERSION=ideal
```

The optimizer writes files next to the input dataset:

```text
<input-directory>/res/<input-stem>_adaptive_process.json
<input-directory>/res/<input-stem>_adaptive_rewritten_query.json
<input-directory>/res/<input-stem>_adaptive_optimizer.log
```

The process file contains the update trajectory and selected policy. The rewritten-query file contains original and refined profiles for held-out rows.

### Evaluate Held-Out Profiles

Evaluate the rewritten-query file with the same downstream model used during policy induction:

```bash
INPUT_FILE=path/to/input_adaptive_rewritten_query.json \
TASK=personamem_v2_personal_email \
EVALUATOR_MODEL=YOUR_TARGET_MODEL \
EVALUATOR_BASE_URL=https://your-target-endpoint.example/v1 \
EVALUATOR_API_KEY=YOUR_TARGET_API_KEY \
bash scripts/evaluate.sh
```

Set `OUTPUT_FILE` to override the default output path. The evaluation reports task-specific raw and refined scores together with profile and context compression statistics.

## 📝 Notes

- All model weights remain frozen; only the textual refinement policy changes.
- One policy is learned for each task/downstream-model configuration and reused across held-out users in that configuration.
- API keys are redacted from optimizer logs. Profiles and model outputs may still contain sensitive user information, so do not publish generated logs or result files without an independent privacy review.

## ✍️ Citation

```bibtex
@misc{liu2026learningpreferenceadaptationlarge,
      title={Learning Preference Adaptation for Large Language Model Personalization via Verbal Reinforcement Learning},
      author={Yuting Liu and Wei Wu and Jianzhe Zhao and Guibing Guo},
      year={2026},
      eprint={2608.09507},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2608.09507},
}
```
