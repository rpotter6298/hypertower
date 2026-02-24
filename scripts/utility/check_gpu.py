#!/usr/bin/env python3
"""
Quick GPU sanity for PyTorch (CUDA or ROCm).
Prints device visibility, names, memory, and runs a tiny matmul on GPU:0 if available.

Run:
  python3 scripts/check_gpu.py
"""
from __future__ import annotations
import os, time, shutil, platform

def _fmt_gb(b: int) -> str:
    try:
        return f"{b / (1024**3):.2f} GB"
    except Exception:
        return str(b)

def main():
    try:
        import torch
    except Exception as e:
        print(f"torch import failed: {e}")
        return

    print(f"torch: {getattr(torch, '__version__', 'unknown')}")
    print(f"python: {platform.python_version()} on {platform.platform()}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}")
    print(f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', '')!r}")
    print(f"torch.version.cuda={getattr(torch.version, 'cuda', None)}")
    print(f"torch.version.hip={getattr(torch.version, 'hip', None)}")

    # Apple MPS check (macOS)
    if hasattr(torch.backends, 'mps'):
        print(f"mps available={torch.backends.mps.is_available()} built={torch.backends.mps.is_built()}")

    # CUDA/ROCm check
    use_cuda = torch.cuda.is_available()
    print(f"cuda available={use_cuda}")
    if not use_cuda:
        print("No CUDA/ROCm device visible to PyTorch.")
        nvsmi = shutil.which('nvidia-smi')
        rocmsmi = shutil.which('rocm-smi') or shutil.which('rocminfo')
        if nvsmi:
            print("nvidia-smi found; ensure your env uses a CUDA-enabled PyTorch build.")
        if rocmsmi:
            print("ROCm tools found; ensure your env uses a ROCm-enabled PyTorch build.")
        print("Tip: activate your conda env and reinstall the GPU build if needed.")
        return

    # List devices
    try:
        n = torch.cuda.device_count()
    except Exception as e:
        print(f"device_count error: {e}")
        n = 0
    print(f"device_count={n}")
    for i in range(n):
        try:
            name = torch.cuda.get_device_name(i)
        except Exception:
            name = "?"
        try:
            props = torch.cuda.get_device_properties(i)
            mem = _fmt_gb(getattr(props, 'total_memory', 0))
        except Exception:
            mem = "?"
        print(f"  cuda:{i} → {name} | total_memory={mem}")

    # Quick matmul on cuda:0
    import torch
    try:
        dev = torch.device('cuda:0')
        torch.cuda.synchronize()
        a = torch.randn(2048, 2048, device=dev)
        b = torch.randn(2048, 2048, device=dev)
        t0 = time.time()
        c = a @ b
        torch.cuda.synchronize()
        dt = time.time() - t0
        print(f"matmul(2048x2048) on cuda:0 ok in {dt:.3f}s; c.mean={float(c.mean()):.5f}")
        del a, b, c
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"matmul on cuda:0 failed: {e}")

    # cuDNN info (if CUDA build)
    try:
        print(f"cudnn available={torch.backends.cudnn.is_available()} version={torch.backends.cudnn.version()}")
    except Exception:
        pass

if __name__ == "__main__":
    main()

