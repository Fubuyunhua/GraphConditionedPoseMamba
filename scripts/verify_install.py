#!/usr/bin/env python3
"""Verify dependencies, CUDA extension, parameter count and backward."""

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import selective_scan_cuda_core  # noqa: F401,E402
from lib.utils.learning import load_backbone  # noqa: E402
from lib.utils.tools import get_config  # noqa: E402


config = get_config(
    str(ROOT / "configs/pose3d/graph_posemamba_h36m_w64_d8_0p8m.yaml")
)
model = load_backbone(config)
parameters = sum(parameter.numel() for parameter in model.parameters())
assert parameters == 800_083, parameters
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the selective-scan verification")

model = model.cuda().train()
inputs = torch.randn(1, 243, 17, 3, device="cuda", requires_grad=True)
prediction = model(inputs)
assert prediction.shape == (1, 243, 17, 3), prediction.shape
prediction.square().mean().backward()
assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
print("torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("parameters:", f"{parameters:,}")
print("prediction:", tuple(prediction.shape))
print("forward/backward: OK")
