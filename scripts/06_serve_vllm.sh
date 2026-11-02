#!/usr/bin/env bash
# Step 7: serve with vLLM's OpenAI-compatible server.
#   MODE=lora   -> base model + LoRA adapter loaded at runtime (default)
#   MODE=merged -> merged standalone model from step 6
# Run inside the serving venv: source ~/venvs/serve/bin/activate
set -euo pipefail

MODE="${MODE:-lora}"
BASE_MODEL="${BASE_MODEL:-models/base}"
ADAPTER_PATH="${ADAPTER_PATH:-outputs/qwen7b-lora/final}"
ADAPTER_NAME="${ADAPTER_NAME:-my-lora}"
MERGED_MODEL="${MERGED_MODEL:-models/merged-qwen7b}"
SERVED_NAME="${SERVED_NAME:-my-model}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"
VLLM_API_KEY="${VLLM_API_KEY:?Set VLLM_API_KEY, e.g. export VLLM_API_KEY=\$(openssl rand -hex 24)}"

COMMON=(--host "$HOST" --port "$PORT"
        --gpu-memory-utilization "$GPU_UTIL"
        --max-model-len "$MAX_MODEL_LEN"
        --api-key "$VLLM_API_KEY"
        --dtype auto)

if [[ "$MODE" == "lora" ]]; then
  # Base model is reachable as "$SERVED_NAME", the adapter as "$ADAPTER_NAME".
  exec vllm serve "$BASE_MODEL" "${COMMON[@]}" \
    --served-model-name "$SERVED_NAME" \
    --enable-lora \
    --lora-modules "$ADAPTER_NAME=$ADAPTER_PATH" \
    --max-lora-rank "$MAX_LORA_RANK" \
    --max-loras 2
elif [[ "$MODE" == "merged" ]]; then
  exec vllm serve "$MERGED_MODEL" "${COMMON[@]}" --served-model-name "$SERVED_NAME"
else
  echo "MODE must be 'lora' or 'merged'" >&2; exit 1
fi
