"""Step 1: exercise the GPU code paths training needs, so CUDA/driver problems show up now.

Runs a bf16 matmul, scaled-dot-product attention, and a bitsandbytes 4-bit (NF4) linear
forward + backward, which is exactly what QLoRA does on every layer.
Run in the train venv: ~/venvs/train/bin/python scripts/smoke_test_gpu.py
"""
import torch
import torch.nn.functional as F

print(f"torch {torch.__version__}, CUDA runtime {torch.version.cuda}, device {torch.cuda.get_device_name(0)}")
print(f"compute capability {torch.cuda.get_device_capability(0)}, bf16 supported: {torch.cuda.is_bf16_supported()}")

a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
print("bf16 matmul:", (a @ a).float().abs().mean().item() > 0)

q = torch.randn(1, 28, 512, 128, device="cuda", dtype=torch.bfloat16)
print("SDPA attention:", F.scaled_dot_product_attention(q, q, q, is_causal=True).shape)

import bitsandbytes as bnb

layer = bnb.nn.Linear4bit(3584, 3584, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4").cuda()
x = torch.randn(8, 3584, device="cuda", dtype=torch.bfloat16, requires_grad=True)
y = layer(x)
y.float().sum().backward()
torch.cuda.synchronize()
print(f"bitsandbytes {bnb.__version__} 4-bit forward/backward:", y.shape, x.grad is not None)
print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print("ALL OK")
