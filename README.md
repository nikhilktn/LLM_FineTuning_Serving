# Hands-on LLM Fine-Tuning, Quantization and Serving

A guided project for learning the three core skills of running your own LLM, on one GPU
server:

1. **Serving**: running a model behind an API with vLLM, and understanding what limits speed and concurrency.
2. **Fine-tuning**: teaching a model new behaviour with LoRA / QLoRA.
3. **Quantization**: shrinking a model to 8 or 4 bits and measuring what you gain and lose.

The approach is **measure, change one thing, measure again**. You start by benchmarking the
untouched base model. Each later step (fine-tuning, quantizing) is compared against that baseline
with the same two tools:

* `scripts/evaluate.py`: **quality**, measured as perplexity on held-out answers plus sample outputs you read yourself
* `scripts/benchmark.py`: **speed**, measured as latency and throughput at different concurrency levels

`scripts/report.py` puts every run into a single comparison table. By the end you'll have
real numbers from your own hardware, which teach far more than any blog post.

## Roadmap

| Step | What you do | What you learn |
|---|---|---|
| **A. Setup** | | |
| 0 | Check the server | What hardware limits you |
| 1 | Set up Python environments | Why the ML stack needs isolated environments |
| 2 | Download the base model | Model files, formats, memory math |
| 3 | Prepare training data | Chat templates, prompt/completion format |
| **B. Serving basics** | | |
| 4 | Serve the base model and call it | vLLM, the OpenAI-compatible API, the KV cache |
| 5 | Measure the baseline | Perplexity, TTFT, TPOT, throughput |
| **C. Fine-tuning** | | |
| 6 | Fine-tune with QLoRA | LoRA, QLoRA, the training loop, loss curves |
| 7 | Evaluate the adapter | Did fine-tuning actually help? |
| 8 | Serve the adapter (multi-LoRA) | Serving adapters without merging |
| **D. Quantization** | | |
| 9 | Merge the adapter | Folding LoRA into the weights |
| 10 | Quantize: FP8, INT8, INT4 | Post-training quantization, calibration |
| 11 | Evaluate and benchmark quantized models | The size / speed / quality trade-off |
| **E. Production** | | |
| 12 | Deploy the chosen model | systemd, API keys, exposing it safely |
| ★ | Experiments | Going deeper on your own |

Plan on roughly a weekend for A–C and another for D–E. Training time depends on your GPU and
dataset size.

---

## Key concepts (read once, refer back later)

**Memory math.** Model weights take *parameters × bytes per parameter*. For a 7B model:

| Precision | Bytes/param | 7B weights |
|---|---|---|
| FP32 | 4 | ~28 GB |
| BF16 / FP16 | 2 | ~15 GB |
| FP8 / INT8 | 1 | ~7.5 GB |
| INT4 | 0.5 | ~4.5 GB (incl. scales) |

Training needs much more than the weights: gradients, optimizer state and activations. That's
why full fine-tuning a 7B model needs more than 80 GB, while QLoRA fits in about 12–16 GB.

**LoRA.** Instead of updating a huge weight matrix `W` (d×d), you freeze it and learn a small
update `ΔW = B·A`, where `A` is r×d and `B` is d×r, with rank `r` around 8–64. That's about 1%
of the parameters, so gradients and optimizer state are tiny. `lora_alpha / r` scales how
strongly the update is applied.

**QLoRA.** Same as LoRA, but the frozen base is stored in 4-bit (NF4) and dequantized on the
fly during each forward pass. The LoRA weights stay in 16-bit. It saves a lot of memory and
costs a little speed and, usually, a negligible amount of quality.

**Post-training quantization (PTQ).** After training, store weights (and optionally activations)
in fewer bits:
* *Weight-only* (W4A16, W8A16): smaller model, faster when **memory-bound** (low concurrency, where decoding reads all the weights for every token).
* *Weights + activations* (W8A8 INT8, FP8): the matmuls themselves run in 8-bit, faster when **compute-bound** (high concurrency).
* *Calibration*: running a few hundred real samples through the model to pick good scales. It matters most for INT4 and INT8.

**Serving (vLLM).** Each in-flight request keeps a **KV cache** (the attention keys and values
of every token so far) in GPU memory. vLLM's *PagedAttention* stores it in blocks like virtual
memory, and *continuous batching* lets new requests join a running batch. VRAM left after the
weights goes to KV cache, and **more KV cache means more concurrent users and longer contexts**.
Quantization helps serving partly by freeing memory for KV cache.

**Serving metrics.**
* **TTFT** (time to first token): prefill plus queueing time. It's the lag a user notices.
* **TPOT** (time per output token): decode speed for each user.
* **Throughput** (output tokens/s across all users): what the GPU delivers in total.

As concurrency rises, throughput rises and per-user TPOT gets worse. Finding that balance is
the core skill of serving.

---

# A. Setup

## Step 0: Check the server

**Learn:** VRAM, not system RAM, decides which model you can use.

```bash
git clone git@github.com:nikhilktn/LLM_FineTuning_Serving.git && cd LLM_FineTuning_Serving
bash scripts/check_env.sh
```

**Look for:**
* **GPU name, memory and `compute_cap`.** Compute capability decides which quantization kernels are fast:
  * 8.0 / 8.6 (Ampere: A100, A10, RTX 30xx): INT8 and INT4 are fast. FP8 runs weight-only.
  * 8.9 (Ada: L4, L40S, RTX 40xx) and 9.0 (Hopper: H100): FP8 is fully accelerated.
* **"CUDA Version" in `nvidia-smi`**: 12.x. Newer drivers are better for recent vLLM.
* **Free disk**: at least 150 GB. You'll keep base, merged and three quantized copies.

**About your 64 GB RAM + 32 GB swap:** that's system memory. It matters when loading
checkpoints and merging. Swap never helps the GPU, and a job that starts swapping slows to a
crawl.

**Model by VRAM** (the default in all commands is `Qwen/Qwen2.5-7B-Instruct`):

| VRAM | Model | Train with |
|---|---|---|
| 12–16 GB | `Qwen/Qwen2.5-3B-Instruct` | QLoRA |
| 24 GB | `Qwen/Qwen2.5-7B-Instruct` | QLoRA |
| 40–48 GB | 7B, or `Qwen/Qwen2.5-14B-Instruct` | 7B: bf16 LoRA; 14B: QLoRA |
| 80 GB | 14B, or 32B | 14B: bf16 LoRA; 32B: QLoRA |

If you change the model, change every `models/...` / `--model` path below to match.

## Step 1: Set up Python environments

**Learn:** vLLM, the training stack and llm-compressor each pin their own PyTorch and
transformers versions. Putting them in one environment is the most common source of
breakage, so use three:

```bash
sudo apt-get update && sudo apt-get install -y python3-venv python3-dev git tmux

for env in train quant serve; do
  python3 -m venv ~/venvs/$env
  ~/venvs/$env/bin/pip install --upgrade pip
  ~/venvs/$env/bin/pip install -r requirements-$env.txt
done

~/venvs/train/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
~/venvs/serve/bin/python -c "import vllm; print(vllm.__version__)"
~/venvs/quant/bin/python -c "import llmcompressor; print(llmcompressor.__version__)"
```

If `cuda.is_available()` prints `False`, install the PyTorch wheel for your CUDA version from
https://pytorch.org/get-started/locally/ in that environment first.

## Step 2: Download the base model

**Learn:** what a model on the Hub actually is.

```bash
source ~/venvs/train/bin/activate
hf auth login        # token from https://huggingface.co/settings/tokens (older CLI: huggingface-cli login)
python scripts/download_model.py --model Qwen/Qwen2.5-7B-Instruct --out models/base
ls -lh models/base
cat models/base/config.json
```

**Look for:**
* `*.safetensors`: the weights, split into shards. Total size ≈ params × 2 bytes (bf16).
* `config.json`: the architecture. Note `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads` (GQA, which shrinks the KV cache) and `torch_dtype`.
* `tokenizer_config.json`: contains the **chat template**, the Jinja code that turns a
  `messages` list into the exact string the model was trained on, with tokens like `<|im_start|>`.

## Step 3: Prepare training data

**Learn:** what models actually see during training.

Training rows use conversational **prompt/completion** format. Loss is computed only on the
completion, so the model learns to *answer*, not to repeat the question:

```json
{"prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": "Question"}],
 "completion": [{"role": "assistant", "content": "Answer"}]}
```

The script converts Alpaca-style (`instruction` / `input` / `output`) and chat-style
(`messages`) data:

```bash
# Option 1: your own data (recommended once the pipeline works)
python scripts/prepare_data.py --input data/raw/my_data.jsonl --out-dir data --system-prompt "You are ..."

# Option 2: start with a public dataset to learn the pipeline
python scripts/prepare_data.py --hf-dataset yahma/alpaca-cleaned --max-samples 3000 --out-dir data
```

**Choose a task where you can see the change.** Generic instruction data barely changes an
instruct model. Good learning tasks have a distinctive output: a fixed JSON schema, a specific
tone or persona, domain Q&A from your own documents, or SQL generation for your schema.

**Try:** view the exact string the model trains on:

```bash
python -c "
import json; from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('models/base'); r = json.loads(open('data/train.jsonl').readline())
print(t.apply_chat_template(r['prompt'] + r['completion'], tokenize=False))"
```

---

# B. Serving basics

## Step 4: Serve the base model and call it

**Learn:** how vLLM uses GPU memory.

```bash
tmux new -s vllm
source ~/venvs/serve/bin/activate
export VLLM_API_KEY=$(openssl rand -hex 24); echo $VLLM_API_KEY   # save this
MODE=model MODEL_PATH=models/base SERVED_NAME=base bash scripts/serve_vllm.sh
# detach: Ctrl-b d
```

**Read the startup log.** It explains vLLM's memory use:
* `Model loading took X GiB`: the weights (about 15 GiB for 7B bf16).
* `# GPU blocks: N` / `KV cache size: N tokens` / `Maximum concurrency for 8192 tokens per request: Kx`:
  the memory left for KV cache after the weights, and how many full-length requests fit at
  once. **Write these down.** You'll compare them after quantizing.

Call it from a second shell:

```bash
source ~/venvs/serve/bin/activate && export VLLM_API_KEY=...
curl -s localhost:8000/v1/models -H "Authorization: Bearer $VLLM_API_KEY" | python -m json.tool
curl -s localhost:8000/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer $VLLM_API_KEY" \
  -d '{"model": "base", "messages": [{"role": "user", "content": "What is a KV cache?"}], "max_tokens": 200}' | python -m json.tool
python scripts/chat_client.py --model base --stream
curl -s localhost:8000/metrics | grep -E "^vllm:(num_requests|gpu_cache_usage)"   # Prometheus metrics
```

**Try:** restart with `MAX_MODEL_LEN=32768` and then `GPU_UTIL=0.7`, and watch how
"Maximum concurrency" changes.

## Step 5: Measure the baseline

**Learn:** the numbers every later step is judged against.

With the server from Step 4 still running:

```bash
python scripts/benchmark.py --model base --label base
```

Then stop the server (`tmux attach -t vllm`, Ctrl-C), because offline evaluation needs the GPU:

```bash
python scripts/evaluate.py --model models/base --label base
python scripts/report.py
```

**Look for:**
* **Benchmark:** from concurrency 1 to 32, throughput (tok/s) should rise a lot while TPOT gets
  only slightly worse. That's continuous batching at work. At concurrency 1, decoding is
  memory-bound: TPOT ≈ weight bytes ÷ GPU memory bandwidth. Check this against your GPU's
  bandwidth spec.
* **Perplexity:** how "surprised" the base model is by your reference answers. Fine-tuning
  should lower it.
* **Samples** in `results/eval_base.json`: read them. This is your "before" picture.

---

# C. Fine-tuning

## Step 6: Fine-tune with QLoRA

**Learn:** the training loop, LoRA hyperparameters, reading loss curves.

Make sure vLLM is stopped (`nvidia-smi` should show no processes). Then:

```bash
tmux new -s train
source ~/venvs/train/bin/activate
python scripts/train_lora.py \
    --model models/base --train-file data/train.jsonl --eval-file data/eval.jsonl \
    --output-dir outputs/qwen7b-lora --qlora \
    --epochs 2 --lr 2e-4 --batch-size 2 --grad-accum 8 --max-length 2048 \
    --lora-r 16 --lora-alpha 32
```

Watch it in two more shells:

```bash
watch -n 2 nvidia-smi                                   # memory and utilisation
tensorboard --logdir outputs/qwen7b-lora/runs --port 6006 --bind_all   # loss curves
# on your laptop: ssh -L 6006:localhost:6006 user@server, then open http://localhost:6006
```

**What each setting does:**
* `trainable params: ~40M || all params: ~7.6B || trainable%: ~0.5`: printed at start. That's LoRA.
* Effective batch = `batch-size × grad-accum` = 16. `grad-accum` trades time for memory.
* `--lr 2e-4` is high compared with full fine-tuning (~1e-5). LoRA tolerates and needs it.
* Gradient checkpointing recomputes activations in the backward pass instead of storing them, saving memory at about 30% extra time.

**Reading the curves:**
* `train/loss` drops fast at first, then flattens. A sudden jump to `nan` means the learning rate is too high or there's an fp16 overflow.
* `eval/loss` should follow it down. **If eval loss turns upward while train loss keeps
  falling, the model is overfitting.** Use the checkpoint from before the turn, fewer epochs,
  or more data.
* `outputs/qwen7b-lora/train_summary.json` records time, **peak GPU memory** and final eval
  loss. Keep it for the experiments.

**If you run out of memory:** `--batch-size 1 --grad-accum 16`, then lower `--max-length`.
**To resume:** `--resume outputs/qwen7b-lora/checkpoint-XXX`.

## Step 7: Evaluate the adapter

**Learn:** whether fine-tuning helped, judged by numbers and by reading outputs.

```bash
source ~/venvs/train/bin/activate
python scripts/test_adapter.py --base models/base --adapter outputs/qwen7b-lora/final \
    --prompt "<a prompt typical of your task>"

source ~/venvs/serve/bin/activate
python scripts/evaluate.py --model models/base --adapter outputs/qwen7b-lora/final --label lora-r16
python scripts/report.py
```

**Look for:**
* **Perplexity** of `lora-r16` should be clearly lower than `base`.
* **Side-by-side outputs:** does the fine-tuned model follow your format, tone or domain?
* **Side effects:** ask a few general questions outside your task. Heavy fine-tuning can cause
  *catastrophic forgetting*, where the model gets worse at things it used to do.

## Step 8: Serve the adapter (multi-LoRA)

**Learn:** vLLM can serve the base model plus several adapters from one GPU and pick one per
request. That's how one model can serve many customers or tasks.

```bash
source ~/venvs/serve/bin/activate
MODE=lora SERVED_NAME=base ADAPTER_NAME=my-lora ADAPTER_PATH=outputs/qwen7b-lora/final bash scripts/serve_vllm.sh
```

```bash
python scripts/chat_client.py --model my-lora     # fine-tuned
python scripts/chat_client.py --model base        # same server, base model
python scripts/benchmark.py --model my-lora --label lora-r16
python scripts/report.py
```

**Look for:** the throughput of `lora-r16` compared with `base`. Applying the adapter at
runtime adds a small overhead, which merging (Step 9) removes.

---

# D. Quantization

## Step 9: Merge the adapter

**Learn:** `W_merged = W + (alpha/r)·B·A`. After merging it's an ordinary model again, which
quantization tools need.

```bash
source ~/venvs/train/bin/activate
python scripts/merge_lora.py --base models/base --adapter outputs/qwen7b-lora/final --out models/merged-qwen7b
```

This runs on CPU in bf16 using system RAM: about 16 GB for 7B and 30 GB for 14B, both fine
with 64 GB. A 32B model would need about 65 GB and spill into swap.

**Sanity check:** `evaluate.py --model models/merged-qwen7b --label merged` should give almost
the same perplexity as `lora-r16`.

## Step 10: Quantize to FP8, INT8 and INT4

**Learn:** post-training quantization, calibration, and the difference between weight-only and
weight+activation quantization.

```bash
source ~/venvs/quant/bin/activate
python scripts/quantize.py --model models/merged-qwen7b --scheme fp8     # minutes, no calibration
python scripts/quantize.py --model models/merged-qwen7b --scheme w8a8    # SmoothQuant + GPTQ, ~15–40 min
python scripts/quantize.py --model models/merged-qwen7b --scheme w4a16   # GPTQ INT4, ~15–40 min
du -sh models/merged-qwen7b*
```

| Scheme | What's quantized | Calibration | Best on |
|---|---|---|---|
| `fp8` | weights + activations, 8-bit float | no | Ada / Hopper (compute capability ≥ 8.9) |
| `w8a8` | weights + activations, INT8 | yes | Ampere and newer, high concurrency |
| `w4a16` | weights only, INT4 (group size 128) | yes | small GPUs, low concurrency |

**What's happening:**
* **GPTQ** quantizes one layer at a time. It uses calibration activations to adjust the
  remaining weights so they make up for the rounding error of the ones already quantized.
* **SmoothQuant** moves activation outliers into the weights (a mathematically equivalent
  rescale), so activations quantize well to INT8.
* `lm_head` stays in 16-bit. It's sensitive to quantization and a small share of the model.
* Calibration uses **your own training data**, so the scales match the traffic the model will
  actually see.

Look inside `models/merged-qwen7b-w4a16/config.json` at `quantization_config`. vLLM reads it
to pick the right kernels automatically.

## Step 11: Evaluate and benchmark quantized models

**Learn:** the trade-off, measured on your own hardware.

For each of `merged`, `merged-qwen7b-fp8`, `merged-qwen7b-w8a8` and `merged-qwen7b-w4a16`:

```bash
source ~/venvs/serve/bin/activate
M=models/merged-qwen7b-w4a16; L=w4a16          # change per model

python scripts/evaluate.py --model $M --label $L                # server stopped
MODE=model MODEL_PATH=$M SERVED_NAME=$L bash scripts/serve_vllm.sh   # in tmux; note the KV-cache log lines
python scripts/benchmark.py --model $L --label $L                # second shell
# stop the server, move on to the next model
```

Then compare everything:

```bash
python scripts/report.py
```

**How to read the table:**
* **Size:** INT4 should be about ¼ of bf16, and FP8/INT8 about ½.
* **Perplexity:** 8-bit should be within about 1% of `merged`. INT4 typically loses a bit more.
  If it's much worse, try more calibration samples (`--num-calib 512`).
* **Concurrency 1 (TPOT):** memory-bound, so INT4 should be fastest since it reads the fewest bytes per token.
* **Concurrency 32 (throughput):** compute-bound, so W8A8 or FP8 often wins.
* **KV cache (from the logs):** smaller weights leave more room for cache, so "Maximum concurrency" goes up.
* **Read the samples** in `results/eval_<label>.json`. Perplexity doesn't catch everything.

**Choose your production model:** the smallest or fastest one whose quality you're happy with.

---

# E. Production

## Step 12: Deploy the chosen model

**Learn:** turning a manual process into a service.

Edit `deploy/vllm.service` (user, paths, `MODEL_PATH`, API key), then:

```bash
sudo cp deploy/vllm.service /etc/systemd/system/vllm.service
sudo systemctl daemon-reload && sudo systemctl enable --now vllm
sudo journalctl -u vllm -f
```

**Security:** always use `--api-key`, never expose port 8000 directly to the internet, and put
it behind nginx or Caddy with TLS, or restrict access with the firewall:
`sudo ufw allow from <your-ip> to any port 8000`.

---

# ★ Experiments for going deeper

Do these in any order. Use a different `--label` for each so `report.py` shows them side by side.

1. **LoRA vs QLoRA** (needs ≥ 40 GB for 7B bf16 LoRA): train without `--qlora` and compare
   `train_summary.json` (peak memory, time) and perplexity.
2. **LoRA rank:** train `--lora-r 8`, `16` and `64` (with `--lora-alpha` = 2×r). Where does
   perplexity stop improving? Remember vLLM needs `MAX_LORA_RANK` ≥ r.
3. **Data size:** train on 500, 2,000 and the full set. How much data does your task need?
4. **Quantized base + LoRA:** serve `MODE=lora BASE_MODEL=<an FP8/INT4 quantization of the base>`
   with your bf16 adapter, without merging. Compare its quality with quantize-after-merge.
5. **Calibration matters:** quantize W4A16 with `--calib-file` pointing at unrelated data (or
   `--num-calib 16`) and compare perplexity.
6. **Serving knobs:** benchmark with `--max-num-seqs 16` vs `256` (passed through
   `serve_vllm.sh`), different `MAX_MODEL_LEN`, and `--enable-prefix-caching` with a long shared
   system prompt.
7. **KV-cache quantization:** pass `--kv-cache-dtype fp8` to the server and check how
   "Maximum concurrency" changes.
8. **Beyond:** export to GGUF and run with llama.cpp on a CPU, try DPO with TRL after SFT, or
   train with Unsloth and compare speed.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` while training | `--batch-size 1 --grad-accum 16`, lower `--max-length`, use `--qlora` |
| vLLM: KV cache / `max_model_len` too large | Lower `MAX_MODEL_LEN` or raise `GPU_UTIL` |
| vLLM: LoRA rank greater than `max_lora_rank` | Set `MAX_LORA_RANK` ≥ trained rank |
| GPU busy / vLLM won't start | Another job is on the GPU. Check `nvidia-smi`. |
| `evaluate.py` OOM | A server is still running, or lower `--gpu-util` / `--max-model-len` |
| `bitsandbytes` errors | `pip install -U bitsandbytes` in the train env |
| FP8 model is slow on an A100 / RTX 30xx | Expected: no FP8 tensor cores, so it runs weight-only. Use `w8a8` there. |
| Gated model 401/403 | Accept the license on its HF page, then `hf auth login` |
| Fine-tuned model rambles or doesn't stop | Check your data's completions. Try fewer epochs. |

## Layout

```
scripts/       check_env, download_model, prepare_data, train_lora, test_adapter,
               merge_lora, quantize, evaluate, benchmark, report, serve_vllm, chat_client
deploy/        vllm.service
models/        base, merged and quantized models            (git-ignored)
outputs/       LoRA checkpoints, adapters, TensorBoard logs (git-ignored)
data/          prepared train/eval JSONL                     (git-ignored)
results/       eval_*.json, bench_*.json, committed so progress is visible in the PR
```
