# LLM Fine-Tuning (LoRA / QLoRA) and Serving with vLLM

The steps below take an open-source model from Hugging Face, fine-tune it with LoRA on your
own data, and serve it from this server through vLLM's OpenAI-compatible API.

```
Step 0  Check the server (GPU, driver, disk)        scripts/00_check_env.sh
Step 1  Set up two Python environments              requirements-train.txt / requirements-serve.txt
Step 2  Choose and download the base model          scripts/01_download_model.py
Step 3  Prepare the training data                   scripts/02_prepare_data.py
Step 4  Fine-tune with LoRA / QLoRA                 scripts/03_train_lora.py
Step 5  Test the adapter                            scripts/04_test_adapter.py
Step 6  (Optional) Merge the adapter into the base  scripts/05_merge_lora.py
Step 7  Serve with vLLM                             scripts/06_serve_vllm.sh
Step 8  Call the API                                scripts/07_test_client.py
Step 9  Run as a service (systemd)                  deploy/vllm.service
```

> **Important: 64 GB RAM + 32 GB swap is system memory, not GPU memory.** Your **GPU's VRAM**
> sets the model size you can train and serve. System RAM matters for loading checkpoints and
> merging adapters. Swap does nothing for GPU memory, and if a job ends up swapping it will slow
> to a crawl, so don't plan around it.

---

## Step 0: Check the server

Copy this folder to the server (`git clone` or `scp -r`), then run:

```bash
cd LLM_FineTuning_Serving
bash scripts/00_check_env.sh
```

Check these in the output:

| Check | What you need |
|---|---|
| `nvidia-smi` works | NVIDIA driver is installed. If not, install it from your distro or NVIDIA first. |
| "CUDA Version" in `nvidia-smi` | 12.x. Recent vLLM wheels are built against CUDA 12.8+, so newer drivers are better. |
| GPU memory | Sets the model size (table below). |
| Free disk | At least 100 GB. A 7B model is about 15 GB, a merged copy another 15 GB, plus checkpoints. |
| Python | 3.10 to 3.12. |

### Choose a model size for your GPU

| GPU VRAM | Train | Serve |
|---|---|---|
| 12–16 GB | 3B model (e.g. `Qwen/Qwen2.5-3B-Instruct`), QLoRA | 3B in bf16 |
| **24 GB** (RTX 3090/4090, A10, L4) | **7–8B with QLoRA** | 7–8B bf16, `--max-model-len 4096–8192` |
| 40–48 GB (A6000, L40S, A100-40G) | 7–8B LoRA in bf16, or 14B QLoRA | 7–14B bf16 |
| 80 GB (A100/H100) | 14B LoRA, 32B QLoRA | up to 32B bf16 |

The default in all scripts is **`Qwen/Qwen2.5-7B-Instruct`**. It's Apache-2.0 licensed, not
gated, supported by both PEFT and vLLM, and fits a 24 GB GPU. To use another model (for example
`meta-llama/Llama-3.1-8B-Instruct`, which is gated and needs license acceptance on HF), change
`--model` / `BASE_MODEL` in each step.

---

## Step 1: Create two Python environments

vLLM pins its own PyTorch version, which often conflicts with the training stack. Keep them
separate:

```bash
sudo apt-get update && sudo apt-get install -y python3-venv python3-dev git tmux

# Training env
python3 -m venv ~/venvs/train
source ~/venvs/train/bin/activate
pip install --upgrade pip
pip install -r requirements-train.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
deactivate

# Serving env
python3 -m venv ~/venvs/serve
source ~/venvs/serve/bin/activate
pip install --upgrade pip
pip install -r requirements-serve.txt
python -c "import vllm; print(vllm.__version__)"
deactivate
```

If `torch.cuda.is_available()` prints `False`, install the PyTorch wheel that matches your CUDA
version from https://pytorch.org/get-started/locally/ before installing the rest.

Optional: put the Hugging Face cache on a large disk:

```bash
echo 'export HF_HOME=/data/hf_cache' >> ~/.bashrc && source ~/.bashrc
```

---

## Step 2: Download the base model

```bash
source ~/venvs/train/bin/activate
hf auth login          # older huggingface_hub: `huggingface-cli login`. Paste a token from https://huggingface.co/settings/tokens
python scripts/01_download_model.py --model Qwen/Qwen2.5-7B-Instruct --out models/base
```

This saves the model to `models/base/`. Both training and serving read from this local copy, so
nothing is downloaded again later.

---

## Step 3: Prepare the training data

The training script expects JSONL in **conversational prompt/completion** format. Loss is then
computed only on the assistant's answer, not the prompt:

```json
{"prompt": [{"role": "system", "content": "You are ..."}, {"role": "user", "content": "Question"}],
 "completion": [{"role": "assistant", "content": "Answer"}]}
```

`02_prepare_data.py` converts the two most common formats for you:

* **Instruction format**: `{"instruction": ..., "input": ..., "output": ...}` (Alpaca style)
* **Chat format**: `{"messages": [{"role": ..., "content": ...}, ...]}`. The last assistant turn becomes the completion.

```bash
# Your own data
python scripts/02_prepare_data.py --input data/raw/my_data.jsonl --out-dir data \
    --system-prompt "You are a helpful assistant for QuantumStreet."

# Or a public dataset to try the pipeline end to end (2,000 samples)
python scripts/02_prepare_data.py --hf-dataset yahma/alpaca-cleaned --max-samples 2000 --out-dir data
```

This writes `data/train.jsonl` and `data/eval.jsonl` (5% held out).

**Data guidelines:**
* 500–5,000 high-quality examples usually beat 50,000 noisy ones.
* Keep the format and style of the answers consistent. The model copies them.
* Check that examples fit in `--max-length` tokens (default 2048). Longer ones get truncated.

---

## Step 4: Fine-tune with LoRA / QLoRA

Stop vLLM if it's running on the same GPU. Run training inside `tmux` so it survives an SSH
disconnect:

```bash
tmux new -s train
source ~/venvs/train/bin/activate

python scripts/03_train_lora.py \
    --model models/base \
    --train-file data/train.jsonl --eval-file data/eval.jsonl \
    --output-dir outputs/qwen7b-lora \
    --qlora \
    --epochs 2 --lr 2e-4 --batch-size 2 --grad-accum 8 \
    --max-length 2048 --lora-r 16 --lora-alpha 32
# Detach: Ctrl-b d     Reattach: tmux attach -t train
```

* `--qlora` loads the base in 4-bit (NF4). Use it on 24 GB GPUs. On 40 GB+ you can drop it for
  plain bf16 LoRA, which is slightly better quality and faster.
* Effective batch size = `batch-size × grad-accum` = 16.
* If you hit **CUDA OOM**, lower `--batch-size` to 1 and raise `--grad-accum` to 16, or
  lower `--max-length`.
* Watch the GPU in a second terminal: `watch -n 2 nvidia-smi`.
* Resume after a crash: add `--resume outputs/qwen7b-lora/checkpoint-XXX`.

**What to watch in the logs:** `loss` should drop steadily. `eval_loss` should drop and then
level off. If `eval_loss` starts rising while `loss` keeps falling, the model is overfitting:
use fewer epochs or more data.

The final adapter is saved to `outputs/qwen7b-lora/final/`. It's small (tens of MB) and contains
`adapter_config.json` and `adapter_model.safetensors`.

---

## Step 5: Test the adapter

```bash
python scripts/04_test_adapter.py --base models/base --adapter outputs/qwen7b-lora/final \
    --prompt "Explain what LoRA fine-tuning is in two sentences."
```

This prints the base model's answer and the fine-tuned model's answer side by side, so you can
see whether training changed what you wanted.

---

## Step 6 (optional): Merge the adapter into the base model

vLLM can serve an adapter directly (Step 7, option A). Merge only if you want a single standalone
model, slightly lower inference overhead, or to quantize it afterwards.

```bash
python scripts/05_merge_lora.py --base models/base --adapter outputs/qwen7b-lora/final \
    --out models/merged-qwen7b
```

The merge runs on CPU in bf16, so it uses **system RAM**: about 16 GB for 7–8B and 30 GB for 14B,
which both fit in your 64 GB. A 32B model needs about 65 GB and would spill into swap. It will
still finish, just slowly; for that size, serve the adapter instead of merging.

---

## Step 7: Serve with vLLM

```bash
tmux new -s vllm
source ~/venvs/serve/bin/activate

# Option A (recommended): base model + LoRA adapter, loaded at runtime
MODE=lora bash scripts/06_serve_vllm.sh

# Option B: the merged model from Step 6
MODE=merged bash scripts/06_serve_vllm.sh
```

The script starts an OpenAI-compatible server on `http://0.0.0.0:8000`. Key flags (set as env
vars, see the script):

| Flag | Meaning |
|---|---|
| `--gpu-memory-utilization 0.90` | Share of VRAM vLLM takes for weights + KV cache |
| `--max-model-len 8192` | Max prompt + output tokens. Lower it if startup fails with a KV cache error. |
| `--enable-lora --lora-modules name=path` | Serve the adapter under the model name `name` |
| `--max-lora-rank 16` | Must be ≥ the `--lora-r` you trained with |
| `--api-key` | Requires clients to send this key |

With option A you can serve the base model and several adapters at once. Add more
`name=path` pairs to `--lora-modules`.

---

## Step 8: Call the API

```bash
# Health + model list
curl http://localhost:8000/v1/models -H "Authorization: Bearer $VLLM_API_KEY"

# Chat completion (model = adapter name for option A, served name for option B)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" -H "Authorization: Bearer $VLLM_API_KEY" \
  -d '{"model": "my-lora", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 200}'

# Python client (pip install openai)
python scripts/07_test_client.py --model my-lora
```

---

## Step 9: Run vLLM as a systemd service

Use this so vLLM starts on boot and restarts if it crashes. Edit the paths and user in
`deploy/vllm.service`, then:

```bash
sudo cp deploy/vllm.service /etc/systemd/system/vllm.service
sudo systemctl daemon-reload
sudo systemctl enable --now vllm
sudo journalctl -u vllm -f          # logs
```

**Security:** always set `--api-key`. Don't expose port 8000 to the internet directly. Put it
behind a reverse proxy (nginx/Caddy) with TLS, or limit access with the firewall
(`sudo ufw allow from <your-ip> to any port 8000`).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` during training | `--batch-size 1 --grad-accum 16`, lower `--max-length`, use `--qlora` |
| vLLM: "not enough KV cache" / "max_model_len too large" | Lower `MAX_MODEL_LEN`, or raise `GPU_UTIL` to 0.95 |
| vLLM: "LoRA rank X greater than max_lora_rank" | Set `MAX_LORA_RANK` ≥ trained rank |
| vLLM fails to start, GPU already in use | A training job or another vLLM is still running. Check `nvidia-smi`. |
| `bitsandbytes` errors | Its CUDA build doesn't match. `pip install -U bitsandbytes`. |
| Gated model 401/403 | Accept the license on the model's HF page and run `hf auth login` |
| Fine-tuned model repeats itself or never stops | Check that training data completions end cleanly. The chat template adds the end token automatically. |

## Directory layout after a full run

```
models/base/                 downloaded base model
models/merged-qwen7b/        merged model (optional)
data/train.jsonl, eval.jsonl prepared data
outputs/qwen7b-lora/         checkpoints + final/ adapter
```
