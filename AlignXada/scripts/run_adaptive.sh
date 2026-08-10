#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Dataset and task. PROFILE_INPUT must be a pre-built JSON array.
TASK="${TASK:-personamem_v2_personal_email}"
PROFILE_INPUT="${PROFILE_INPUT:-path/to/profile_input.json}"

# Policy-learning configuration.
SUPPORT_SIZE="${SUPPORT_SIZE:-40}"
SUPPORT_POSITIVE_COUNT="${SUPPORT_POSITIVE_COUNT:-0}"
DEV_SUPPORT_SIZE="${DEV_SUPPORT_SIZE:-20}"
DEV_SUPPORT_POSITIVE_COUNT="${DEV_SUPPORT_POSITIVE_COUNT:-0}"
QUERY_SIZE="${QUERY_SIZE:-100}"
NUM_UPDATES="${NUM_UPDATES:-5}"
FEEDBACK_BATCH_SIZE="${FEEDBACK_BATCH_SIZE:-20}"
SEED="${SEED:-42}"
SUPPORT_SAMPLING_STRATEGY="${SUPPORT_SAMPLING_STRATEGY:-pref_type_stratified}"
ADAPTIVE_SELECTION_STRATEGY="${ADAPTIVE_SELECTION_STRATEGY:-raw_role_weighted}"
ADAPTIVE_DEV_GATE="${ADAPTIVE_DEV_GATE:-strict_improve}"
BEST_PATTERN_STRATEGY="${BEST_PATTERN_STRATEGY:-best_epoch}"
USE_EXPLICIT_USER_SPLIT="${USE_EXPLICIT_USER_SPLIT:-1}"
USE_RECORDED_SAMPLING="${USE_RECORDED_SAMPLING:-0}"

export PROFILE_META_PROMPT_VERSION="${PROFILE_META_PROMPT_VERSION:-ideal}"

# Replace every placeholder below, or provide the corresponding environment
# variable when invoking this script.
META_BACKEND="${META_BACKEND:-openai}"
META_MODEL="${META_MODEL:-YOUR_META_MODEL}"
META_BASE_URL="${META_BASE_URL:-https://your-meta-endpoint.example/v1}"
META_API_KEY="${META_API_KEY:-YOUR_META_API_KEY}"
META_MAX_OUTPUT_TOKENS="${META_MAX_OUTPUT_TOKENS:-4096}"

REWRITE_BACKEND="${REWRITE_BACKEND:-openai}"
REWRITE_MODEL="${REWRITE_MODEL:-YOUR_REWRITE_MODEL}"
REWRITE_BASE_URL="${REWRITE_BASE_URL:-https://your-rewrite-endpoint.example/v1}"
REWRITE_API_KEY="${REWRITE_API_KEY:-YOUR_REWRITE_API_KEY}"
REWRITE_MAX_OUTPUT_TOKENS="${REWRITE_MAX_OUTPUT_TOKENS:-4096}"

TARGET_BACKEND="${TARGET_BACKEND:-openai}"
TARGET_MODEL="${TARGET_MODEL:-YOUR_TARGET_MODEL}"
TARGET_BASE_URL="${TARGET_BASE_URL:-https://your-target-endpoint.example/v1}"
TARGET_API_KEY="${TARGET_API_KEY:-YOUR_TARGET_API_KEY}"
TARGET_MAX_OUTPUT_TOKENS="${TARGET_MAX_OUTPUT_TOKENS:-128}"

if [[ "${PROFILE_INPUT}" == "path/to/profile_input.json" || ! -f "${PROFILE_INPUT}" ]]; then
  echo "Set PROFILE_INPUT to an existing pre-built dataset JSON file." >&2
  exit 1
fi

for value in "${META_MODEL}" "${REWRITE_MODEL}" "${TARGET_MODEL}"; do
  if [[ "${value}" == YOUR_* ]]; then
    echo "Replace all model placeholders before running." >&2
    exit 1
  fi
done

for value in "${META_API_KEY}" "${REWRITE_API_KEY}" "${TARGET_API_KEY}"; do
  if [[ "${value}" == YOUR_* ]]; then
    echo "Replace all API-key placeholders before running." >&2
    exit 1
  fi
done

sampling_args=()
if [[ "${USE_EXPLICIT_USER_SPLIT}" == "1" ]]; then
  sampling_args+=(--use-explicit-user-split)
fi
if [[ "${USE_RECORDED_SAMPLING}" == "1" ]]; then
  sampling_args+=(--debug-use-recorded-pairwise-sampling)
fi

python3 profile_meta_optimizer_adaptive.py \
  "${sampling_args[@]}" \
  --task "${TASK}" \
  --profile-input "${PROFILE_INPUT}" \
  --support-size "${SUPPORT_SIZE}" \
  --support-positive-count "${SUPPORT_POSITIVE_COUNT}" \
  --dev-support-size "${DEV_SUPPORT_SIZE}" \
  --dev-support-positive-count "${DEV_SUPPORT_POSITIVE_COUNT}" \
  --query-size "${QUERY_SIZE}" \
  --num-epochs "${NUM_UPDATES}" \
  --adaptive-num-updates "${NUM_UPDATES}" \
  --feedback-batch-size "${FEEDBACK_BATCH_SIZE}" \
  --adaptive-probe-scope pool \
  --adaptive-selection-strategy "${ADAPTIVE_SELECTION_STRATEGY}" \
  --adaptive-dev-gate "${ADAPTIVE_DEV_GATE}" \
  --best-pattern-strategy "${BEST_PATTERN_STRATEGY}" \
  --support-sampling-strategy "${SUPPORT_SAMPLING_STRATEGY}" \
  --seed "${SEED}" \
  --meta-backend "${META_BACKEND}" \
  --meta-model "${META_MODEL}" \
  --meta-base-url "${META_BASE_URL}" \
  --meta-api-key "${META_API_KEY}" \
  --meta-temperature 0.2 \
  --meta-max-output-tokens "${META_MAX_OUTPUT_TOKENS}" \
  --rewrite-backend "${REWRITE_BACKEND}" \
  --rewrite-model "${REWRITE_MODEL}" \
  --rewrite-base-url "${REWRITE_BASE_URL}" \
  --rewrite-api-key "${REWRITE_API_KEY}" \
  --rewrite-temperature 0.0 \
  --rewrite-max-output-tokens "${REWRITE_MAX_OUTPUT_TOKENS}" \
  --target-backend "${TARGET_BACKEND}" \
  --target-model "${TARGET_MODEL}" \
  --target-base-url "${TARGET_BASE_URL}" \
  --target-api-key "${TARGET_API_KEY}" \
  --target-temperature 0.0 \
  --target-max-output-tokens "${TARGET_MAX_OUTPUT_TOKENS}"
