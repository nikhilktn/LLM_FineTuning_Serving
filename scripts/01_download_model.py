"""Step 2: download a base model from the Hugging Face Hub to a local folder."""
import argparse
import os

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")  # faster downloads

from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="HF repo id")
parser.add_argument("--out", default="models/base", help="local directory")
parser.add_argument("--revision", default=None, help="branch/tag/commit to pin")
args = parser.parse_args()

path = snapshot_download(
    repo_id=args.model,
    revision=args.revision,
    local_dir=args.out,
    # Skip duplicate weight formats; safetensors is all we need.
    ignore_patterns=["*.pth", "*.pt", "original/*", "*.gguf"],
)
print(f"Downloaded {args.model} -> {path}")
