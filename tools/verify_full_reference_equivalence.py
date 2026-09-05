#!/usr/bin/env python3
"""Compare default Full prediction/loss/gradients against an untouched tree."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch


PROGRAM = r'''
import os
import torch
from lib.model.PoseMamba import GraphConditionedPoseMamba

torch.manual_seed(1701)
model = GraphConditionedPoseMamba(
    num_frame=9,
    in_chans=3,
    embed_dim_ratio=16,
    depth=1,
    mlp_ratio=2.0,
    drop_path_rate=0.2,
).cuda().train()
x = torch.randn(2, 9, 17, 3, device="cuda", requires_grad=True)
upstream = torch.randn(2, 9, 17, 3, device="cuda")
torch.manual_seed(1702)
output = model(x)
loss = (output * upstream).sum()
loss.backward()
torch.save(
    {
        "output": output.detach().cpu(),
        "loss": loss.detach().cpu(),
        "input_grad": x.grad.detach().cpu(),
        "parameter_grads": {
            name: parameter.grad.detach().cpu()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        },
    },
    os.environ["EQUIVALENCE_OUTPUT"],
)
'''


def metric(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    difference = (left - right).abs()
    return {
        "max_abs": float(difference.max()) if difference.numel() else 0.0,
        "mean_abs": float(difference.mean()) if difference.numel() else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--candidate-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reference = Path(args.reference_root).resolve()
    candidate = Path(args.candidate_root).resolve()
    with tempfile.TemporaryDirectory(prefix="full-equivalence-") as directory:
        directory = Path(directory)
        reference_output = directory / "reference_a.pt"
        reference_repeat_output = directory / "reference_b.pt"
        candidate_output = directory / "candidate.pt"
        for root, output in (
            (reference, reference_output),
            (reference, reference_repeat_output),
            (candidate, candidate_output),
        ):
            environment = dict(os.environ)
            environment["EQUIVALENCE_OUTPUT"] = str(output)
            subprocess.run(
                [sys.executable, "-c", PROGRAM],
                cwd=root,
                env=environment,
                check=True,
            )
        expected = torch.load(reference_output, map_location="cpu", weights_only=False)
        repeated = torch.load(
            reference_repeat_output, map_location="cpu", weights_only=False
        )
        actual = torch.load(candidate_output, map_location="cpu", weights_only=False)

    def compare(left, right):
        result = {
            "output": metric(left["output"], right["output"]),
            "loss_abs": float((left["loss"] - right["loss"]).abs()),
            "input_gradient": metric(left["input_grad"], right["input_grad"]),
            "parameter_gradient_max_abs": 0.0,
            "parameter_gradient_mismatch_count": 0,
            "parameter_gradient_tensor_count": len(left["parameter_grads"]),
        }
        if left["parameter_grads"].keys() != right["parameter_grads"].keys():
            raise RuntimeError("parameter gradient key mismatch")
        for name, value in left["parameter_grads"].items():
            difference = metric(value, right["parameter_grads"][name])
            result["parameter_gradient_max_abs"] = max(
                result["parameter_gradient_max_abs"], difference["max_abs"]
            )
            result["parameter_gradient_mismatch_count"] += int(
                difference["max_abs"] != 0.0
            )
        return result

    candidate_comparison = compare(expected, actual)
    reference_repeat = compare(expected, repeated)
    gradient_limits = {
        "input_gradient_max_abs": max(
            1e-6, reference_repeat["input_gradient"]["max_abs"] * 1.25
        ),
        "parameter_gradient_max_abs": max(
            3e-6, reference_repeat["parameter_gradient_max_abs"] * 2.0
        ),
    }
    report = {
        "reference_root": str(reference),
        "candidate_root": str(candidate),
        "candidate_vs_reference": candidate_comparison,
        "reference_repeat_nondeterminism": reference_repeat,
        "gradient_limits": gradient_limits,
    }
    report["status"] = (
        "PASS"
        if candidate_comparison["output"]["max_abs"] == 0.0
        and candidate_comparison["loss_abs"] == 0.0
        and candidate_comparison["input_gradient"]["max_abs"]
        <= gradient_limits["input_gradient_max_abs"]
        and candidate_comparison["parameter_gradient_max_abs"]
        <= gradient_limits["parameter_gradient_max_abs"]
        else "FAIL"
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
