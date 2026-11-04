#!/usr/bin/env bash
# Steps 4, 8, 11, 12: serve with vLLM's OpenAI-compatible server.
#   MODE=model -> serve any full model dir: base, merged or quantized (default)
#   MODE=lora  -> serve BASE_MODEL plus a LoRA adapter loaded at runtime
# Extra arguments are passed straight to `vllm serve`, e.g. --max-num-seqs 64
# Run inside the serving venv: source ~/venvs/serve/bin/activate
set -euo pipefail

MODE="${MODE:-model}"
MODEL_PATH="${MODEL_PATH:-models/base}"
SERVED_NAME="${SERVED_NAME:-my-model}"
BASE_MODEL="${BASE_MODEL:-models/base}"
ADAPTER_PATH="${ADAPTER_PATH:-outputs/qwen7b-lora/final}"
ADAPTER_NAME="${ADAPTER_NAME:-my-lora}"
# More adapters on the same base, space-separated: EXTRA_LORAS="ckpt250=outputs/qwen7b-lora/checkpoint-250"
EXTRA_LORAS="${EXTRA_LORAS:-}"
# Localhost only by default; reach it from a laptop with: ssh -L 8000:localhost:8000 <server>
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"
# vLLM reads the key from the environment, which keeps it out of `ps` (unlike --api-key).
: "${VLLM_API_KEY:=$(cat ~/.vllm_api_key 2>/dev/null)}"
: "${VLLM_API_KEY:?Create a key first: openssl rand -hex 24 > ~/.vllm_api_key && chmod 600 ~/.vllm_api_key}"
export VLLM_API_KEY

COMMON=(--host "$HOST" --port "$PORT"
        --gpu-memory-utilization "$GPU_UTIL"
        --max-model-len "$MAX_MODEL_LEN"
        --served-model-name "$SERVED_NAME")

if [[ "$MODE" == "lora" ]]; then
  # Base model is reachable as "$SERVED_NAME", each adapter by its name.
  # shellcheck disable=SC2086  # EXTRA_LORAS is a list
  exec vllm serve "$BASE_MODEL" "${COMMON[@]}" \
    --enable-lora \
    --lora-modules "$ADAPTER_NAME=$ADAPTER_PATH" $EXTRA_LORAS \
    --max-lora-rank "$MAX_LORA_RANK" \
    --max-loras 2 "$@"
elif [[ "$MODE" == "model" ]]; then
  # Quantized models (FP8 / INT8 / INT4) are detected from their config automatically.
  exec vllm serve "$MODEL_PATH" "${COMMON[@]}" "$@"
else
  echo "MODE must be 'model' or 'lora'" >&2; exit 1
fi
