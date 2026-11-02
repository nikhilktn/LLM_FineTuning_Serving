#!/usr/bin/env bash
# Step 0: report GPU, driver, CUDA, RAM, disk and Python so you can pick a model size.
set -u

echo "=== GPU / driver ==="
if [[ -r /proc/driver/nvidia/version ]]; then
  echo "Kernel module: $(grep -o 'Kernel Module.*' /proc/driver/nvidia/version | head -n 1)"
else
  echo "Kernel module: not loaded (driver missing or not built for kernel $(uname -r))"
fi
if command -v nvidia-smi >/dev/null; then
  if nvidia-smi; then
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
  else
    echo "nvidia-smi failed. 'Driver/library version mismatch' means the driver was upgraded but"
    echo "the old kernel module is still loaded: reboot. If it persists, check 'dkms status'."
  fi
else
  echo "nvidia-smi not found: install the NVIDIA driver first."
fi

echo; echo "=== CUDA toolkit (optional; pip wheels bundle their own) ==="
command -v nvcc >/dev/null && nvcc --version | tail -n 1 || echo "nvcc not installed (fine)"

echo; echo "=== RAM / swap ==="
free -h

echo; echo "=== Disk (current dir) ==="
df -h .

echo; echo "=== Python (need >= 3.10) ==="
for py in $(which -a python3 /usr/bin/python3 2>/dev/null | awk '!seen[$0]++'); do
  ver=$("$py" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')
  ok=$("$py" -c 'import sys; print("ok" if sys.version_info >= (3, 10) else "TOO OLD")')
  echo "$py  $ver  $ok"
done

echo; echo "=== OS ==="
cat /etc/os-release 2>/dev/null | head -n 2
