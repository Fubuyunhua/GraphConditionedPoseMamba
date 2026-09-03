#!/usr/bin/env python3
"""Static architecture and protocol preflight for the 80-epoch scale study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lib.model.mambablocks import FactorizedBiSSM
from lib.utils.learning import load_backbone
from lib.utils.tools import get_config


class PassthroughSSM(nn.Module):
    def forward(self, x, context=None):
        if context is not None and context.shape != x.shape:
            raise RuntimeError("context shape differs from recurrent content")
        return x


FROZEN_FIELDS = (
    "warmup_epochs",
    "batch_size",
    "test_batch_size",
    "learning_rate",
    "weight_decay",
    "lr_decay",
    "checkpoint_frequency",
    "use_ema",
    "ema_decay",
    "backbone",
    "model_type",
    "maxlen",
    "mlp_ratio",
    "ssm_d_state",
    "ssm_ratio",
    "dropout",
    "drop_path_rate",
    "use_graph_mixer",
    "use_symmetry_edges",
    "graph_hidden_ratio",
    "graph_conditioned_ssm",
    "reuse_graph_context",
    "factorized_spatial_temporal",
    "spatial_ssm_conv",
    "temporal_ssm_conv",
    "graph_scale",
    "spatial_res_scale",
    "temporal_res_scale",
    "compile_model",
    "compile_mode",
    "compile_compatible_scan",
    "cuda_graph_model",
    "eager_eval_when_compiled",
    "activation_checkpoint_blocks",
    "data_root",
    "subset_list",
    "dt_file",
    "clip_len",
    "data_stride",
    "sample_stride",
    "num_joints",
    "rootrel",
    "no_conf",
    "gt_2d",
    "train_2d",
    "pretrain_3d_curriculum",
    "no_eval",
    "finetune",
    "partial_train",
    "lambda_3d",
    "lambda_scale",
    "lambda_3d_velocity",
    "lambda_diff",
    "lambda_lv",
    "lambda_lg",
    "lambda_a",
    "lambda_av",
    "lambda_3dw",
    "lambda_attn_diag",
    "lambda_attn_entropy",
    "lambda_tail_aux",
    "lambda_gate_sparsity",
    "synthetic",
    "flip",
    "mask_ratio",
    "mask_T_ratio",
    "noise",
)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        default="configs/pose3d/graph_posemamba_h36m_w64_d8_0p8m_memopt_speed.yaml",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "configs/pose3d/graph_posemamba_h36m_w128_d20_scale_80e.yaml",
            "configs/pose3d/graph_posemamba_h36m_w256_d10_scale_80e.yaml",
        ],
    )
    parser.add_argument(
        "--output",
        default=".experiments/model_scaling_80e_20260903/static_preflight.json",
    )
    args = parser.parse_args()

    baseline = get_config(args.baseline)
    results = []
    for config_path in args.configs:
        config = get_config(config_path)
        mismatches = []
        for field in FROZEN_FIELDS:
            if getattr(config, field) != getattr(baseline, field):
                mismatches.append(
                    {
                        "field": field,
                        "baseline": getattr(baseline, field),
                        "candidate": getattr(config, field),
                    }
                )
        if int(config.epochs) != 80:
            mismatches.append(
                {"field": "epochs", "baseline": 80, "candidate": config.epochs}
            )
        if str(getattr(config, "graph_injection_mode", "control")) != "control":
            mismatches.append(
                {
                    "field": "graph_injection_mode",
                    "baseline": "control",
                    "candidate": getattr(config, "graph_injection_mode", None),
                }
            )
        if mismatches:
            raise RuntimeError(f"{config_path}: frozen protocol mismatch: {mismatches}")

        model = load_backbone(config).eval()
        parameters = count_parameters(model)
        if len(model.blocks) != int(config.depth):
            raise RuntimeError(f"{config_path}: incorrect block count")
        for block in model.blocks:
            if not isinstance(block.spatial_ssm, FactorizedBiSSM):
                raise RuntimeError(f"{config_path}: spatial SSM is not factorized")
            if not isinstance(block.temporal_ssm, FactorizedBiSSM):
                raise RuntimeError(f"{config_path}: temporal SSM is not factorized")
            if block.graph_injection_mode != "control":
                raise RuntimeError(f"{config_path}: graph control is inactive")

        for block in model.blocks:
            block.spatial_ssm = PassthroughSSM()
            block.temporal_ssm = PassthroughSSM()
        with torch.no_grad():
            output, trace = model(
                torch.randn(1, 9, int(config.num_joints), 3),
                return_shape_trace=True,
            )
        expected_output = [1, 9, int(config.num_joints), 3]
        expected_spatial = [9, 1, int(config.num_joints), int(config.dim_feat)]
        expected_temporal = [int(config.num_joints), 1, 9, int(config.dim_feat)]
        if list(output.shape) != expected_output:
            raise RuntimeError(f"{config_path}: output shape mismatch")
        if list(trace["spatial_ssm_input"]) != expected_spatial:
            raise RuntimeError(f"{config_path}: spatial shape mismatch")
        if list(trace["temporal_ssm_input"]) != expected_temporal:
            raise RuntimeError(f"{config_path}: temporal shape mismatch")

        results.append(
            {
                "config": config_path,
                "epochs": int(config.epochs),
                "width": int(config.dim_feat),
                "depth": int(config.depth),
                "ssm_inner": int(float(config.ssm_ratio) * int(config.dim_feat)),
                "mlp_inner": int(float(config.mlp_ratio) * int(config.dim_feat)),
                "parameters": parameters,
                "shape_trace": trace,
                "status": "PASS",
            }
        )

    report = {"status": "PASS", "baseline": args.baseline, "models": results}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
