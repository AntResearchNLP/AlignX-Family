#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TASK="${TASK:-personamem_v2_personal_email}"
INPUT_FILE="${INPUT_FILE:-path/to/adaptive_rewritten_query.json}"
OUTPUT_FILE="${OUTPUT_FILE:-}"
SEED="${SEED:-42}"

EVALUATOR_BACKEND="${EVALUATOR_BACKEND:-openai}"
EVALUATOR_MODEL="${EVALUATOR_MODEL:-YOUR_TARGET_MODEL}"
EVALUATOR_BASE_URL="${EVALUATOR_BASE_URL:-https://your-target-endpoint.example/v1}"
EVALUATOR_API_KEY="${EVALUATOR_API_KEY:-YOUR_TARGET_API_KEY}"
EVALUATOR_MAX_OUTPUT_TOKENS="${EVALUATOR_MAX_OUTPUT_TOKENS:-128}"

if [[ "${INPUT_FILE}" == "path/to/adaptive_rewritten_query.json" || ! -f "${INPUT_FILE}" ]]; then
  echo "Set INPUT_FILE to an existing *_adaptive_rewritten_query.json file." >&2
  exit 1
fi
if [[ "${EVALUATOR_MODEL}" == YOUR_* || "${EVALUATOR_API_KEY}" == YOUR_* ]]; then
  echo "Replace the evaluator model and API-key placeholders before running." >&2
  exit 1
fi

output_args=()
if [[ -n "${OUTPUT_FILE}" ]]; then
  output_args+=(--output-file "${OUTPUT_FILE}")
fi

python3 evaluate_select_openai.py \
  --task "${TASK}" \
  --input-file "${INPUT_FILE}" \
  "${output_args[@]}" \
  --seed "${SEED}" \
  --evaluator-backend "${EVALUATOR_BACKEND}" \
  --evaluator-model "${EVALUATOR_MODEL}" \
  --evaluator-base-url "${EVALUATOR_BASE_URL}" \
  --evaluator-api-key "${EVALUATOR_API_KEY}" \
  --evaluator-temperature 0.0 \
  --evaluator-max-output-tokens "${EVALUATOR_MAX_OUTPUT_TOKENS}"
