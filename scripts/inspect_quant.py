"""Step 10: look inside quantized checkpoints: what config vLLM will read, how one layer is stored,
and (for 8-bit) how far the dequantized weights are from the original.

SmoothQuant (w8a8) rescales weight columns by s and divides the preceding RMSNorm's gain by s, so
the raw weights no longer match the original at all. What the layer computes is W · diag(gain),
so the error is measured on that product, which SmoothQuant leaves unchanged.

  python scripts/inspect_quant.py models/merged-qwen7b-fp8 models/merged-qwen7b-w8a8 models/merged-qwen7b-w4a16

Run in the quant venv. CPU only, so it works while the GPU is busy.
"""
import argparse
import glob
import json
import os

import torch
from safetensors import safe_open

LAYER = "model.layers.10.mlp.gate_proj"
NORM = "model.layers.10.post_attention_layernorm.weight"  # the RMSNorm that feeds gate_proj


def tensors(model_dir, prefix):
    out = {}
    for f in sorted(glob.glob(f"{model_dir}/*.safetensors")):
        with safe_open(f, "pt") as s:
            out.update({k: s.get_tensor(k) for k in s.keys() if k.startswith(prefix)})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("--original", default="models/merged-qwen7b")
    args = parser.parse_args()
    original = tensors(args.original, f"{LAYER}.weight")[f"{LAYER}.weight"].float()
    gain = tensors(args.original, NORM)[NORM].float()

    for d in args.models:
        size = sum(os.path.getsize(f) for f in glob.glob(f"{d}/*.safetensors")) / 1e9
        qc = json.load(open(f"{d}/config.json")).get("quantization_config", {})
        print(f"== {d}  ({size:.2f} GB of weights)")
        print(f"  format: {qc.get('format')} | ignore: {qc.get('ignore')}")
        for g in qc.get("config_groups", {}).values():
            w, a = g.get("weights") or {}, g.get("input_activations")
            print(f"  weights: {w.get('type')} {w.get('num_bits')}-bit, {w.get('strategy')}"
                  + (f", group {w.get('group_size')}" if w.get("group_size") else "")
                  + f", symmetric={w.get('symmetric')}")
            print("  activations: " + (f"{a.get('type')} {a.get('num_bits')}-bit, {a.get('strategy')}, "
                                       f"{'dynamic' if a.get('dynamic') else 'static'}" if a else "16-bit (not quantized)"))
        t = tensors(d, f"{LAYER}.")
        print(f"  {LAYER} is stored as:")
        for k, v in t.items():
            print(f"    {k.rsplit('.', 1)[-1]:14} {str(v.dtype):22} {tuple(v.shape)}")
        w, scale = t.get(f"{LAYER}.weight"), t.get(f"{LAYER}.weight_scale")
        if w is not None and scale is not None and w.shape == original.shape:  # 8-bit: one value per weight
            deq, gain_q = w.float() * scale.float(), tensors(d, NORM)[NORM].float()
            raw = (deq - original).norm() / original.norm()
            eff = (deq * gain_q - original * gain).norm() / (original * gain).norm()
            print(f"  weight error, as the layer computes it (W·diag(norm gain)): {eff:.2%}")
            if not torch.allclose(gain_q, gain):
                ratio = gain / gain_q
                print(f"    raw weights differ by {raw:.0%}: SmoothQuant rescaled the columns "
                      f"(factor s from {ratio.min():.2f} to {ratio.max():.1f}, median {ratio.median():.2f})")
        lm = tensors(d, "lm_head.")
        print(f"  lm_head: {', '.join(f'{k} {v.dtype}' for k, v in lm.items())}")


if __name__ == "__main__":
    main()
