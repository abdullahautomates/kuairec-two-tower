"""Sanity check: is PyTorch installed, and can it use the Mac's GPU?

MPS ("Metal Performance Shaders") is Apple's GPU backend for PyTorch.
If it's available, we can train on the GPU; if not, the CPU still works,
just slower. For a model this small, either is fine.

Run:  uv run python src/check_torch.py
"""

import sys

import torch

print(f"Python:  {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"MPS built into this PyTorch: {torch.backends.mps.is_built()}")
print(f"MPS available on this Mac:   {torch.backends.mps.is_available()}")

# Do one tiny calculation on the GPU to prove it actually works, not just
# that the flag says so.
if torch.backends.mps.is_available():
    x = torch.ones(3, device="mps")
    print(f"Test tensor on GPU: {x} (device={x.device})")
