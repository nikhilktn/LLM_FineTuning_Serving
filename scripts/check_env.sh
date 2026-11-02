#!/usr/bin/env bash
# Step 0: report GPU, driver, CUDA, RAM, disk and Python so you can pick a model size.
set -u

echo "=== GPU / driver ==="
if command -v nvidia-smi >/dev/null; then
  nvidia-smi
  nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
else
  echo "nvidia-smi not found: install the NVIDIA driver first."
fi

echo; echo "=== CUDA toolkit (optional; pip wheels bundle their own) ==="
command -v nvcc >/dev/null && nvcc --version | tail -n 1 || echo "nvcc not installed (fine)"

echo; echo "=== RAM / swap ==="
free -h

echo; echo "=== Disk (current dir) ==="
df -h .

echo; echo "=== Python ==="
python3 --version

echo; echo "=== OS ==="
cat /etc/os-release 2>/dev/null | head -n 2
