#!/usr/bin/env python3
"""Fail-closed static/protocol preflight for the remaining paper evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.data.dataset_motion_3d import MotionDataset3D
from lib.model.graph_mixer import build_h36m_graph_spec
from lib.utils.learning import load_backbone
from lib.utils.tools import get_config


CONFIGS = {
    "full": "configs/pose3d/repro_full_seed1_80e.yaml",
    "rewired": "configs/pose3d/ablation_full_rewired_graph.yaml",
    "joined": "configs/pose3d/ablation_full_no_recurrence_reset_matched.yaml",
    "a0_legacy": "configs/pose3d/repro_a0_posemamba_seed1_80e.yaml",
    "a0_exact": "configs/pose3d/diagnostic_posemamba_corrected_backward.yaml",
    "a2_seed1": "configs/pose3d/repro_a2_graph_feature_seed1_80e.yaml",
    "a2_seed2": "configs/pose3d/repro_a2_graph_feature_seed2_80e.yaml",
    "full_seed2": "configs/pose3d/repro_full_seed2_80e.yaml",
    "a0_seed2": "configs/pose3d/repro_a0_posemamba_seed2_80e.yaml",
    "mpi_a0": "configs/pose3d_3dhp/posemamba_3dhp_w64_d6_m1_released_protocol_v2.yaml",
    "mpi_full": "configs/pose3d_3dhp/graph_posemamba_3dhp_w64_d8_protocol_v2_memopt.yaml",
}
DATASET_SHA256 = "73b642f2567a8d0b194f88c54a3182c7b635c003c832b48ae6ee559f10232175"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def parameter_signature(model) -> list[tuple[str, tuple[int, ...]]]:
    return [(name, tuple(value.shape)) for name, value in model.named_parameters()]


def checkpoint_state(path: str, model) -> dict[str, Any]:
    if not path:
        return {"status": "SKIPPED", "reason": "checkpoint path not supplied"}
    checkpoint = Path(path)
    if not checkpoint.is_file():
        return {"status": "BLOCKED", "reason": f"checkpoint missing: {checkpoint}"}
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_pos")
    if not isinstance(state, dict):
        return {"status": "FAIL", "reason": "model_pos missing"}
    key_normalization = "none"
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
        key_normalization = "strip_dataparallel_module_prefix"
    model.load_state_dict(state, strict=True)
    return {
        "status": "PASS",
        "path": str(checkpoint.resolve()),
        "sha256": sha256(checkpoint),
        "checkpoint_type": payload.get("checkpoint_type", "unknown"),
        "epoch": payload.get("epoch"),
        "key_normalization": key_normalization,
    }


def add(checks: list[dict[str, Any]], name: str, passed: bool, evidence: Any) -> None:
    checks.append(
        {"name": name, "status": "PASS" if passed else "FAIL", "evidence": evidence}
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=".experiments/paper_remaining_evidence",
    )
    parser.add_argument("--full-checkpoint", default="")
    parser.add_argument("--posemamba-checkpoint", default="")
    args = parser.parse_args()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = {
        name: get_config(str(ROOT / relative))
        for name, relative in CONFIGS.items()
    }
    models = {
        name: load_backbone(configs[name])
        for name in ("full", "rewired", "joined", "a0_legacy", "a0_exact")
    }
    checks: list[dict[str, Any]] = []
    full_signature = parameter_signature(models["full"])
    add(
        checks,
        "rewired_parameter_match",
        parameter_signature(models["rewired"]) == full_signature,
        {"full": sum(p.numel() for p in models["full"].parameters()), "rewired": sum(p.numel() for p in models["rewired"].parameters())},
    )
    add(
        checks,
        "joined_parameter_match",
        parameter_signature(models["joined"]) == full_signature,
        {"full": sum(p.numel() for p in models["full"].parameters()), "joined": sum(p.numel() for p in models["joined"].parameters())},
    )
    a0_signature = parameter_signature(models["a0_legacy"])
    add(
        checks,
        "corrected_a0_parameter_match",
        parameter_signature(models["a0_exact"]) == a0_signature,
        {"legacy": sum(p.numel() for p in models["a0_legacy"].parameters()), "exact": sum(p.numel() for p in models["a0_exact"].parameters())},
    )

    graph_spec = build_h36m_graph_spec(
        "degree_preserving_rewired", seed=3407
    )
    (output_dir / "graph_spec.json").write_text(
        json.dumps(graph_spec, indent=2) + "\n", encoding="utf-8"
    )
    add(
        checks,
        "rewired_graph_constraints",
        bool(graph_spec["bone_connected"])
        and graph_spec["bone_degrees"]
        == build_h36m_graph_spec("anatomical", seed=3407)["bone_degrees"]
        and graph_spec["symmetry_degrees"]
        == build_h36m_graph_spec("anatomical", seed=3407)["symmetry_degrees"],
        graph_spec,
    )
    add(
        checks,
        "all_layers_share_rewired_graph",
        len(
            {
                block.graph_mixer.graph_topology_hash
                for block in models["rewired"].blocks
            }
        )
        == 1,
        models["rewired"].execution_spec(),
    )

    dataset = (
        ROOT
        / configs["full"].data_root
        / configs["full"].dt_file
    ).resolve()
    dataset_status: dict[str, Any]
    if dataset.is_file():
        dataset_hash = sha256(dataset)
        train_dataset = MotionDataset3D(
            configs["full"], configs["full"].subset_list, "train"
        )
        steps_per_epoch = math.ceil(
            len(train_dataset) / int(configs["full"].batch_size)
        )
        dataset_status = {
            "status": "PASS" if dataset_hash == DATASET_SHA256 else "FAIL",
            "path": str(dataset),
            "sha256": dataset_hash,
            "train_samples": len(train_dataset),
            "steps_per_epoch": steps_per_epoch,
        }
    else:
        steps_per_epoch = None
        dataset_status = {
            "status": "BLOCKED",
            "reason": f"dataset missing: {dataset}",
        }
    checks.append(
        {
            "name": "h36m_dataset_identity",
            "status": dataset_status["status"],
            "evidence": dataset_status,
        }
    )

    protocol = {
        "epochs": 80,
        "batch_size": 4,
        "declared_warmup_epochs": 8,
        "effective_linear_warmup": False,
        "learning_rate_epoch1": 0.0005,
        "learning_rate_epoch80": 0.0005 * 0.99**79,
        "checkpoint_lr_after_epoch80": 0.0005 * 0.99**80,
        "steps_per_epoch": steps_per_epoch,
        "optimizer_steps": None if steps_per_epoch is None else 80 * steps_per_epoch,
        "ema_updates": None if steps_per_epoch is None else 80 * steps_per_epoch,
        "ema_decay_per_step": 0.9998,
        "best_metric_rule": "best_ema_test_monitored_first80",
        "fixed_metric_rule": "ema_fixed_epoch80",
    }
    config_summary = {}
    for name, config in configs.items():
        path = ROOT / CONFIGS[name]
        model = load_backbone(config)
        config_summary[name] = {
            "path": CONFIGS[name],
            "sha256": sha256(path),
            "seed": int(config.seed),
            "epochs": int(config.epochs),
            "parameters": sum(p.numel() for p in model.parameters()),
            "execution_spec": model.execution_spec(),
        }

    compatibility = {
        "full": checkpoint_state(args.full_checkpoint, load_backbone(configs["full"])),
        "posemamba_legacy": checkpoint_state(
            args.posemamba_checkpoint, load_backbone(configs["a0_legacy"])
        ),
        "posemamba_exact": checkpoint_state(
            args.posemamba_checkpoint, load_backbone(configs["a0_exact"])
        ),
    }
    for name, result in compatibility.items():
        checks.append(
            {
                "name": f"checkpoint_compatibility_{name}",
                "status": result["status"],
                "evidence": result,
            }
        )

    blocking = [
        check
        for check in checks
        if check["status"] in {"FAIL", "BLOCKED"}
    ]
    payload = {
        "status": "PASS" if not blocking else "BLOCKED",
        "source_head": git_output("rev-parse", "HEAD"),
        "source_status_short": git_output("status", "--short"),
        "checks": checks,
        "dataset": dataset_status,
        "frozen_h36m_protocol": protocol,
        "configs": config_summary,
        "checkpoint_compatibility": compatibility,
        "long_training_authorized": False,
    }
    (output_dir / "preflight_results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    if blocking:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
