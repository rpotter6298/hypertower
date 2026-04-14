#!/usr/bin/env bash
# ============================================================
# HyperTower environment setup script.
#
# Installs all required packages for the v3 pipeline.
# PyTorch is installed CPU-only by default.
#
# To upgrade to GPU after running this script:
#   NVIDIA (CUDA 12.8):
#     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
#   AMD (ROCm 6.2):
#     pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
#
# Usage:
#   bash setup_env.sh
# ============================================================

set -e

echo "=== HyperTower environment setup ==="
echo "Installing CPU-only PyTorch (upgrade separately for GPU)"
echo ""

# ── PyTorch (CPU) ─────────────────────────────────────────────
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# ── Core data science ─────────────────────────────────────────
pip install \
    numpy \
    pandas \
    scikit-learn \
    scipy

# ── Image processing ──────────────────────────────────────────
pip install \
    Pillow \
    scikit-image

# ── Visualisation ─────────────────────────────────────────────
pip install \
    matplotlib

# ── Distributed job system ────────────────────────────────────
pip install \
    fastapi \
    "uvicorn[standard]" \
    requests \
    pydantic

# ── Statistics ────────────────────────────────────────────────
pip install \
    statsmodels

# ── Utilities ─────────────────────────────────────────────────
pip install \
    tqdm \
    openpyxl

echo ""
echo "=== Setup complete ==="
echo ""
echo "To enable GPU support:"
echo "  NVIDIA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128"
echo "  AMD:    pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2"
