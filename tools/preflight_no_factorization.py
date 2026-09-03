#!/usr/bin/env python3
"""Fail-closed regression preflight for the coupled graph-conditioned ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lib.utils.learning import load_backbone
from lib.utils.tools import get_config


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-config", required=True)
    parser.add_argument("--ablation-config", required=True)
    parser.add_argument("--full-checkpoint", required=True)
    parser.add_argument("--prechange-fixture", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the selective-scan preflight")

    fixture = torch.load(args.prechange_fixture, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.full_checkpoint, map_location="cpu", weights_only=False)
    full_config = get_config(args.full_config)
    ablation_config = get_config(args.ablation_config)

    full = load_backbone(full_config)
    full.load_state_dict(checkpoint["model_pos"], strict=True)
    full_parameters = count_parameters(full)
    full = full.cuda().eval()
    fixture_input = fixture["input"].cuda()
    with torch.no_grad():
        full_output, full_trace = full(fixture_input, return_shape_trace=True)
    reference = fixture["output"].cuda()
    absolute_error = (full_output - reference).abs()
    max_abs_error = float(absolute_error.max().item())
    mean_abs_error = float(absolute_error.mean().item())
    del full, full_output, reference, absolute_error
    torch.cuda.empty_cache()

    ablation = load_backbone(ablation_config)
    ablation_parameters = count_parameters(ablation)
    ablation = ablation.cuda().eval()
    with torch.no_grad():
        candidate_output, candidate_trace = ablation(
            fixture_input, return_shape_trace=True
        )
    finite_output = bool(torch.isfinite(candidate_output).all().item())

    spatial_ssm = ablation.blocks[0].spatial_ssm
    setattr(spatial_ssm, "__DEBUG__", True)
    content = torch.randn(1, 9, 17, ablation.embed_dim, device="cuda")
    with torch.no_grad():
        spatial_ssm(content, context=content)
        first = {key: value.clone() for key, value in spatial_ssm.__data__.items()}
        spatial_ssm(content, context=content + torch.randn_like(content))
        second = spatial_ssm.__data__
    control_decoupling = {
        "u_unchanged": bool(torch.equal(first["us"], second["us"])),
        "delta_changed": bool(not torch.equal(first["dts"], second["dts"])),
        "B_changed": bool(not torch.equal(first["Bs"], second["Bs"])),
        "C_changed": bool(not torch.equal(first["Cs"], second["Cs"])),
    }

    expected_coupled = [1, 243, 17, int(ablation_config.dim_feat)]
    gates = {
        "full_checkpoint_max_abs_error_le_1e-5": max_abs_error <= 1e-5,
        "spatial_input_is_full_btjc": list(candidate_trace["spatial_ssm_input"])
        == expected_coupled,
        "temporal_input_is_full_btjc": list(candidate_trace["temporal_ssm_input"])
        == expected_coupled,
        "graph_feature_is_full_btjc": list(candidate_trace["graph_feature"])
        == expected_coupled,
        "spatial_context_is_full_btjc": list(candidate_trace["spatial_context"])
        == expected_coupled,
        "temporal_context_is_full_btjc": list(candidate_trace["temporal_context"])
        == expected_coupled,
        "candidate_output_is_finite": finite_output,
        "content_control_decoupling": all(control_decoupling.values()),
    }
    result = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "precision": "FP32",
        "full": {
            "parameters": full_parameters,
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "max_abs_error": max_abs_error,
            "mean_abs_error": mean_abs_error,
            "shape_trace": full_trace,
        },
        "ablation": {
            "name": "Graph-Conditioned SSM w/o Factorization",
            "epochs": int(ablation_config.epochs),
            "parameters": ablation_parameters,
            "factorized_spatial_temporal": bool(
                ablation_config.factorized_spatial_temporal
            ),
            "forward_type": str(ablation_config.coupled_ssm_forward_type),
            "spatial_conv_mode": spatial_ssm.conv_mode,
            "spatial_d_conv": int(ablation_config.spatial_ssm_conv),
            "temporal_conv_mode": ablation.blocks[0].temporal_ssm.conv_mode,
            "temporal_d_conv": int(ablation_config.temporal_ssm_conv),
            "shape_trace": candidate_trace,
            "content_control_decoupling": control_decoupling,
        },
        "gates": gates,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
