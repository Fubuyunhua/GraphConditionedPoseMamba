#!/usr/bin/env python3
"""Compare eager and memory-optimized FP32 model math on CUDA."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lib.utils.learning import load_backbone
from lib.utils.tools import get_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/pose3d/graph_posemamba_h36m_w64_d8_0p8m.yaml",
    )
    parser.add_argument("--compile-mode", choices=("default", "reduce-overhead"), default="default")
    parser.add_argument("--activation-checkpoint", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260902)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def disable_drop_path(model):
    # DropPath stochasticity is independently tested with RNG-preserving
    # activation checkpointing.  Disabling it here isolates runtime math.
    for block in model.blocks:
        block.drop_path = nn.Identity()


def gradient_map(model):
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def main():
    options = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    seed_everything(options.seed)
    config = get_config(options.config)

    reference = load_backbone(config).cuda()
    candidate = load_backbone(config).cuda()
    candidate.load_state_dict(reference.state_dict(), strict=True)
    candidate.activation_checkpoint_blocks = bool(options.activation_checkpoint)
    disable_drop_path(reference)
    disable_drop_path(candidate)

    candidate_runtime = torch.compile(
        candidate,
        mode=options.compile_mode,
        fullgraph=False,
    )
    x_reference = torch.randn(
        options.batch_size,
        config.clip_len,
        config.num_joints,
        3,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    x_candidate = x_reference.detach().clone().requires_grad_(True)

    reference.train()
    candidate_runtime.train()
    output_reference = reference(x_reference)
    loss_reference = output_reference.square().mean()
    loss_reference.backward()
    reference_gradients = gradient_map(reference)

    output_candidate = candidate_runtime(x_candidate)
    loss_candidate = output_candidate.square().mean()
    loss_candidate.backward()
    candidate_gradients = gradient_map(candidate)
    torch.cuda.synchronize()

    output_max_abs = float((output_candidate - output_reference).abs().max().item())
    input_grad_max_abs = float((x_candidate.grad - x_reference.grad).abs().max().item())
    parameter_grad_max_abs = 0.0
    parameter_grad_max_name = None
    for name, reference_gradient in reference_gradients.items():
        difference = float(
            (candidate_gradients[name] - reference_gradient).abs().max().item()
        )
        if difference > parameter_grad_max_abs:
            parameter_grad_max_abs = difference
            parameter_grad_max_name = name

    torch.testing.assert_close(output_candidate, output_reference, rtol=1e-4, atol=2e-6)
    torch.testing.assert_close(x_candidate.grad, x_reference.grad, rtol=1e-4, atol=1e-7)
    for name, reference_gradient in reference_gradients.items():
        torch.testing.assert_close(
            candidate_gradients[name],
            reference_gradient,
            rtol=2e-4,
            atol=1e-7,
            msg=lambda message, name=name: f"gradient mismatch for {name}: {message}",
        )

    print(
        json.dumps(
            {
                "status": "PASS",
                "config": options.config,
                "compile_mode": options.compile_mode,
                "activation_checkpoint_blocks": bool(options.activation_checkpoint),
                "batch_size": options.batch_size,
                "seed": options.seed,
                "loss_reference": float(loss_reference.item()),
                "loss_candidate": float(loss_candidate.item()),
                "loss_abs_delta": abs(float(loss_candidate.item() - loss_reference.item())),
                "output_max_abs_delta": output_max_abs,
                "input_gradient_max_abs_delta": input_grad_max_abs,
                "parameter_gradient_max_abs_delta": parameter_grad_max_abs,
                "parameter_gradient_max_name": parameter_grad_max_name,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
