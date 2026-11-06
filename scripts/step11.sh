#!/usr/bin/env bash
# Step 11: measure one model end to end: perplexity, KV cache, speed, extraction accuracy.
#   bash scripts/step11.sh models/merged-qwen7b-fp8 fp8
# Run in tmux, in the serve venv, with the GPU free (no other vLLM running).
#
# 1. evaluate.py (offline vLLM)            -> results/eval_<label>.json        (skip: SKIP_PPL=1)
# 2. start the server, stop it, start again: the second start is warm (compile cache, Step 4),
#    so its KV-cache numbers are the ones to compare. Logs: ~/vllm_<label>.log (.cold.log = 1st)
# 3. benchmark.py on fresh prompts          -> results/bench_<label>.json
# 4. eval_extraction.py + comparison with merged -> results/extract_<label>.json (skip: SKIP_EXTRACT=1)
# 5. one general prompt, then stop the server.
#
# Options (environment): SERVE_ARGS="..." extra `vllm serve` flags; SKIP_BENCH=1;
# EXTRACT_RUNS=N runs the extraction N times (labels <label>, <label>-r2, ...) to measure the
# run-to-run noise of one setting. Example (★ experiment, generation config):
#   SKIP_PPL=1 SKIP_BENCH=1 EXTRACT_RUNS=2 SERVE_ARGS="--generation-config vllm" \
#     bash scripts/step11.sh models/merged-qwen7b-w8a8 gen-vllm
set -euo pipefail
M="${1:?usage: step11.sh <model dir> <label>}"
L="${2:?usage: step11.sh <model dir> <label>}"
# 6 levels x 64 requests = 384 prompts: more than the 287 in eval.jsonl, so take them from the
# training prompts. Reused prompts would hit the prefix cache (bigger with smaller weights) and
# flatter TTFT. Only the prompts are used, so it doesn't matter that the model trained on them.
CONC="${CONC:-1,4,8,16,32,64}"
BENCH_PROMPTS="${BENCH_PROMPTS:-data/train.jsonl}"
SKIP_PPL="${SKIP_PPL:-0}"
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"
SKIP_BENCH="${SKIP_BENCH:-0}"
EXTRACT_RUNS="${EXTRACT_RUNS:-1}"
SERVE_ARGS="${SERVE_ARGS:-}"
PID=""

gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }

wait_gpu_free() {
  for _ in $(seq 120); do (( $(gpu_used) < 1000 )) && return 0; sleep 1; done
  echo "GPU still busy ($(gpu_used) MiB used)"; exit 1
}

start_server() {  # $1 = log file
  # shellcheck disable=SC2086  # SERVE_ARGS is a list of flags
  MODE=model MODEL_PATH="$M" SERVED_NAME="$L" bash scripts/serve_vllm.sh $SERVE_ARGS >"$1" 2>&1 &
  PID=$!
  for _ in $(seq 900); do
    curl -sf localhost:8000/health >/dev/null 2>&1 && return 0
    kill -0 "$PID" 2>/dev/null || { echo "server exited; last lines of $1:"; tail -30 "$1"; exit 1; }
    sleep 1
  done
  echo "server not healthy after 15 min"; exit 1
}

stop_server() {
  [[ -n "$PID" ]] || return 0
  kill -INT "$PID" 2>/dev/null || true
  wait "$PID" 2>/dev/null || true
  PID=""
  wait_gpu_free
}
trap stop_server EXIT

if curl -sf localhost:8000/health >/dev/null 2>&1; then
  echo "A server is already running on :8000. Stop it first."; exit 1
fi
wait_gpu_free

if [[ "$SKIP_PPL" != 1 ]]; then
  echo "### [$L] 1/5 perplexity"
  python scripts/evaluate.py --model "$M" --label "$L"
  wait_gpu_free
fi

echo "### [$L] 2/5 server: first start (may compile), then a warm restart"
start_server ~/"vllm_$L.cold.log"; stop_server
start_server ~/"vllm_$L.log"
for f in ~/"vllm_$L.cold.log" ~/"vllm_$L.log"; do
  echo "  -- $(basename "$f")"
  grep -E "Model loading took|torch.compile took|Available KV cache memory|GPU KV cache size|peak activation|Default sampling parameters" "$f" \
    | sed -E 's/^.*\] //; s/(peak activation).*/\1/' || true
done

if [[ "$SKIP_BENCH" != 1 ]]; then
  echo "### [$L] 3/5 benchmark"
  python scripts/benchmark.py --model "$L" --label "$L" --concurrency "$CONC" --max-tokens 128 \
    --eval-file "$BENCH_PROMPTS"
fi

if [[ "$SKIP_EXTRACT" != 1 ]]; then
  for i in $(seq "$EXTRACT_RUNS"); do
    R="$L"; (( i > 1 )) && R="$L-r$i"
    echo "### [$R] 4/5 extraction (run $i of $EXTRACT_RUNS)"
    python scripts/eval_extraction.py --model "$L" --label "$R"
    python scripts/compare_extraction.py results/extract_merged.json "results/extract_$R.json"
  done
fi

echo "### [$L] 5/5 general prompt"
python scripts/chat_client.py --model "$L" \
  --prompt "A train leaves at 3:40 and arrives at 6:15. How long is the trip? One sentence." | tail -n +2
