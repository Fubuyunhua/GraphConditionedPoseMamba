"""Audited MPI-INF-3DHP training/evaluation for GraphConditionedPoseMamba.

This is derived from the latest repaired RTX5090 protocol-v2 runner.  It uses
the authoritative unclipped dataset, stride-9 tail resampling, one evaluation
per valid centre, mandatory data hashes, separate raw/EMA checkpoints and a
locked epoch-120 test endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# When this file is copied under ``.experiments/<study>/source`` the script
# directory is no longer the repository root.  Locate the first parent that
# owns ``lib`` so the isolated runner can import the frozen model code.
for _candidate in (Path.cwd(), *Path(__file__).resolve().parents):
    if (_candidate / "lib").is_dir():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
else:
    raise RuntimeError("could not locate ReliPose repository root")

from lib.data.dataset_mpi3dhp_protocol_v2 import MotionDatasetMPI3DHPProtocolV2
from lib.model.loss import p_mpjpe
from lib.utils.learning import AverageMeter, load_backbone
from lib.utils.tools import get_config
from lib.utils.utils_data import flip_data
from logger import colorlogger
from train import EMAModel, _unwrap_compiled_model, set_random_seed
from lib.utils.train_epoch_3dhp import train_epoch_3dhp


# Deliberately code-owned so the registered YAML remains byte-for-byte stable.
# This makes the protocol change explicit in source provenance rather than
# silently reinterpreting ``mpi3dhp_checkpoint_selection: final_epoch``.
PER_EPOCH_TEST_MONITORING = False


def _effective_evaluation_policy(declared_policy: str) -> str:
    if PER_EPOCH_TEST_MONITORING:
        return "per_epoch_monitored_test"
    return declared_policy


def _should_evaluate_epoch(declared_policy: str, final_epoch: bool) -> bool:
    return bool(
        PER_EPOCH_TEST_MONITORING
        or declared_policy == "legacy_test_best"
        or final_epoch
    )


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True, help="exact output directory")
    p.add_argument("--resume", default="")
    p.add_argument("--evaluate", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--protocol-only", action="store_true")
    p.add_argument("--smoke-only", action="store_true")
    p.add_argument("--allow-overwrite", action="store_true")
    return p.parse_args()


def apply_runtime_precision_flags(args: Any) -> None:
    """Make the FP32/TF32 contract explicit and reproducible."""

    allow_tf32 = bool(getattr(args, "allow_tf32", False))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32


def model_provenance(args: Any) -> dict[str, Any]:
    files = (
        Path("lib/model/PoseMamba.py"),
        Path("lib/model/mambablocks.py"),
        Path("lib/model/graph_mixer.py"),
    )
    result = {
        "backbone": str(args.backbone),
        "dim_feat": int(args.dim_feat),
        "depth": int(args.depth),
        "mlp_ratio": float(args.mlp_ratio),
        "ssm_ratio": float(args.ssm_ratio),
        "ssm_d_state": int(args.ssm_d_state),
        "source_sha256": {
            str(path): _sha256_file(path.resolve()) for path in files
        },
    }
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_dataset_identity(args: Any) -> tuple[list[Path], dict[str, str]]:
    """Fail closed when the declared authoritative NPZ files are not loaded."""

    data_root = Path(args.data_root).resolve()
    files = [
        data_root / "data_train_3dhp.npz",
        data_root / "data_test_3dhp.npz",
    ]
    expected = [
        str(getattr(args, "mpi3dhp_train_sha256", "")),
        str(getattr(args, "mpi3dhp_test_sha256", "")),
    ]
    if not all(expected):
        raise RuntimeError(
            "official-repaired 3DHP runs require mpi3dhp_train_sha256 and "
            "mpi3dhp_test_sha256"
        )
    actual: dict[str, str] = {}
    for path, wanted in zip(files, expected):
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = _sha256_file(path)
        actual[str(path)] = digest
        if digest.lower() != wanted.lower():
            raise RuntimeError(
                f"3DHP dataset hash mismatch for {path}: "
                f"expected={wanted} actual={digest}"
            )
    return files, actual


def _atomic_write_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _git_output(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return ""


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _save_checkpoint(path: Path, *, epoch: int, lr: float, optimizer: optim.Optimizer,
                     model: nn.Module, ema: EMAModel | None,
                     best_metrics: dict[str, float],
                     kind: str, protocol: dict[str, Any]) -> None:
    checkpoint_model = _unwrap_compiled_model(model)
    raw = {
        key: value.detach().clone()
        for key, value in checkpoint_model.state_dict().items()
    }
    ema_state = None
    if ema is not None:
        # Preserve non-floating buffers (if the model has any) so an EMA
        # checkpoint remains a strict, independently loadable state dict.
        ema_state = {key: value.detach().clone() for key, value in raw.items()}
        ema_state.update(
            {
                key: value.detach().clone()
                for key, value in ema.shadow.items()
            }
        )
    payload = {
        "checkpoint_type": kind,
        "epoch": int(epoch + 1),
        "lr": float(lr),
        "optimizer": optimizer.state_dict(),
        "model_pos": raw if kind == "raw" else (ema_state or raw),
        "model_ema": ema_state,
        "ema_shadow": ema_state,
        "min_loss": float(best_metrics[f"{kind}_p1_mm"]),
        "best_metrics": dict(best_metrics),
        "protocol": protocol,
        "rng_state": _rng_state(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _inverse_normalize(x: torch.Tensor, widths: torch.Tensor, heights: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    w = widths.view(-1, 1, 1).to(out.dtype)
    h = heights.view(-1, 1, 1).to(out.dtype)
    out[..., 0:1] = (out[..., 0:1] + 1.0) * w / 2.0
    out[..., 1:2] = (out[..., 1:2] + h / w) * w / 2.0
    out[..., 2:3] = out[..., 2:3] * w / 2.0
    return out


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    args: Any,
    log,
    *,
    checkpoint_view: str,
) -> dict[str, Any]:
    model.eval()
    p1_values: list[float] = []
    p2_values: list[float] = []
    joint_error_values: list[float] = []
    by_sequence: dict[str, dict[str, Any]] = {}
    center = int(args.clip_len) // 2
    with torch.inference_mode():
        for batch in tqdm(loader, desc="eval", leave=False):
            batch_input, batch_gt, valid, widths, heights, seq_names, _ = batch
            if torch.cuda.is_available():
                batch_input = batch_input.cuda(non_blocking=True)
                batch_gt = batch_gt.cuda(non_blocking=True)
                valid = valid.cuda(non_blocking=True)
                widths = widths.cuda(non_blocking=True)
                heights = heights.cuda(non_blocking=True)
            if getattr(args, "no_conf", False):
                batch_input = batch_input[..., :2]
            if getattr(args, "flip", False):
                pred_a = model(batch_input)
                pred_b = flip_data(model(flip_data(batch_input)))
                pred = (pred_a + pred_b) / 2.0
            else:
                pred = model(batch_input)
            pred = pred[:, center]
            target = batch_gt[:, center]
            pred = _inverse_normalize(pred, widths, heights)
            target = _inverse_normalize(target, widths, heights)
            pred = pred - pred[:, 0:1]
            target = target - target[:, 0:1]
            joint_error = torch.norm(pred - target, dim=-1)
            p1 = joint_error.mean(dim=-1)
            keep = valid > 0.5
            if not bool(keep.any()):
                continue
            pred_np = pred[keep].detach().cpu().numpy()
            target_np = target[keep].detach().cpu().numpy()
            p2 = p_mpjpe(pred_np, target_np)
            names = list(seq_names) if isinstance(seq_names, (list, tuple)) else [str(seq_names)] * len(p1)
            for local_i, global_i in enumerate(torch.where(keep)[0].tolist()):
                name = str(names[global_i])
                row = by_sequence.setdefault(name, {"p1": [], "p2": [], "valid_frame_count": 0})
                row["p1"].append(float(p1[global_i].item()))
                row["p2"].append(float(p2[local_i]))
                row.setdefault("joint_errors", []).extend(
                    joint_error[global_i].detach().cpu().tolist()
                )
                row["valid_frame_count"] += 1
                p1_values.append(float(p1[global_i].item()))
                p2_values.append(float(p2[local_i]))
                joint_error_values.extend(
                    joint_error[global_i].detach().cpu().tolist()
                )
    if not p1_values:
        raise RuntimeError("protocol-v2 evaluator found zero valid centre frames")
    thresholds = np.arange(0.0, 150.0 + 1e-9, 5.0, dtype=np.float64)

    def _pck_auc(errors: list[float]) -> tuple[float, float]:
        values = np.asarray(errors, dtype=np.float64)
        curve = np.asarray([(values <= t).mean() for t in thresholds])
        pck = float(curve[-1] * 100.0)
        auc = float(np.trapz(curve, thresholds) / 150.0 * 100.0)
        return pck, auc

    seq_summary = {}
    for name, row in sorted(by_sequence.items()):
        seq_pck, seq_auc = _pck_auc(row["joint_errors"])
        seq_summary[name] = {
            "p1_mm": float(np.mean(row["p1"])),
            "p2_mm": float(np.mean(row["p2"])),
            "pck_150_percent": seq_pck,
            "auc_0_150_percent": seq_auc,
            "valid_frame_count": int(row["valid_frame_count"]),
        }
    pck_150, auc_0_150 = _pck_auc(joint_error_values)
    result = {
        "protocol": MotionDatasetMPI3DHPProtocolV2.protocol_name,
        "checkpoint_view": checkpoint_view,
        "mpjpe_mm": float(np.mean(p1_values)),
        "p_mpjpe_mm": float(np.mean(p2_values)),
        "pck_150_percent": pck_150,
        "auc_0_150_percent": auc_0_150,
        "pck_auc_aggregation": "valid-frame joint-weighted",
        "valid_frame_count": int(len(p1_values)),
        "sequences": seq_summary,
    }
    log.info(f"protocol-v2 evaluation: {json.dumps(result, sort_keys=True)}")
    return result


def _load_checkpoint(path: Path, model: nn.Module, ema: EMAModel | None,
                     optimizer: optim.Optimizer | None = None) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint_model = _unwrap_compiled_model(model)
    checkpoint_model.load_state_dict(payload["model_pos"], strict=True)
    if ema is not None:
        saved_ema = payload.get("ema_shadow") or payload.get("model_ema")
        if saved_ema:
            ema.shadow = {
                key: value.detach().clone() for key, value in saved_ema.items()
            }
        model_state = checkpoint_model.state_dict()
        ema.shadow = {
            key: value.to(device=model_state[key].device)
            for key, value in ema.shadow.items()
        }
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
        # ``torch.load(..., map_location="cpu")`` is deliberate so large
        # checkpoints can be audited without CUDA.  After the model has been
        # placed on CUDA, however, AdamW's moment/step tensors must follow the
        # corresponding parameter device before the first resumed step.
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                state = optimizer.state.get(parameter, {})
                for key, value in tuple(state.items()):
                    if torch.is_tensor(value):
                        state[key] = value.to(device=parameter.device)
    return payload


def _smoke_forward_backward(
    model: nn.Module,
    loader: DataLoader,
    args: Any,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the protocol-v2 smoke test")
    model.train()
    batch_input, batch_gt = next(iter(loader))
    batch_input = batch_input.cuda(non_blocking=True)
    batch_gt = batch_gt.cuda(non_blocking=True)
    if getattr(args, "no_conf", False):
        batch_input = batch_input[..., :2]
    if getattr(args, "rootrel", False):
        batch_gt = batch_gt - batch_gt[:, :, 0:1, :]
    prediction = model(batch_input)
    loss = torch.mean((prediction - batch_gt) ** 2)
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(loss)):
        raise RuntimeError("non-finite value in CUDA smoke test")
    loss.backward()
    finite_grads = all(
        p.grad is None or bool(torch.isfinite(p.grad).all())
        for p in model.parameters()
    )
    if not finite_grads:
        raise RuntimeError("non-finite gradient in CUDA smoke test")
    model.zero_grad(set_to_none=True)
    return {
        "status": "passed",
        "input_shape": list(batch_input.shape),
        "output_shape": list(prediction.shape),
        "loss": float(loss.detach().item()),
        "finite_gradients": True,
        "max_memory_allocated_mib": float(
            torch.cuda.max_memory_allocated() / (1024 * 1024)
        ),
    }


def main() -> None:
    opts = _args()
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    args.seed = opts.seed
    args.protocol_version = MotionDatasetMPI3DHPProtocolV2.protocol_name
    apply_runtime_precision_flags(args)
    run_dir = Path(opts.checkpoint).resolve()
    if run_dir.exists() and any(run_dir.iterdir()) and not opts.allow_overwrite:
        raise RuntimeError(f"refusing to reuse non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    log = colorlogger(str(run_dir), log_name="train_v2.log")
    resolved_config_path = run_dir / "resolved_config.yaml"
    resolved_config_tmp = resolved_config_path.with_suffix(".yaml.tmp")
    with resolved_config_tmp.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(args), f, sort_keys=False)
    os.replace(resolved_config_tmp, resolved_config_path)

    data_files, data_hashes = _verify_dataset_identity(args)
    train_ds = MotionDatasetMPI3DHPProtocolV2(args, "train")
    test_ds = MotionDatasetMPI3DHPProtocolV2(args, "test")
    protocol = {"train": train_ds.protocol_summary(), "test": test_ds.protocol_summary()}
    expected_valid = int(sum(protocol["test"]["valid_frame_counts"].values()))
    if protocol["test"]["window_count"] != expected_valid:
        raise RuntimeError(
            "test protocol does not evaluate every valid centre exactly once: "
            f"windows={protocol['test']['window_count']} valid={expected_valid}"
        )
    _atomic_write_json(run_dir / "protocol_summary.json", protocol)
    log.info(f"protocol={json.dumps(protocol, sort_keys=True)}")

    dataset_module = Path(
        sys.modules[MotionDatasetMPI3DHPProtocolV2.__module__].__file__
    ).resolve()
    manifest: dict[str, Any] = {
        "command": [sys.executable, *sys.argv],
        "seed": int(opts.seed),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_status_short": _git_output("status", "--short"),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "hashes": {
            "config": _sha256_file(Path(opts.config).resolve()),
            "resolved_config": _sha256_file(resolved_config_path),
            "trainer": _sha256_file(Path(__file__).resolve()),
            "dataset_adapter": _sha256_file(dataset_module),
            **data_hashes,
        },
        "protocol": protocol,
    }
    _atomic_write_json(run_dir / "run_manifest.json", manifest)
    if opts.protocol_only:
        log.info("protocol-only preflight passed")
        return

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(getattr(args, "test_batch_size", args.batch_size)),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    model = load_backbone(args)
    parameter_count = int(sum(p.numel() for p in model.parameters()))
    log.info(f"trainable_parameter_count={parameter_count}")
    if hasattr(model, "execution_spec"):
        log.info(
            "model_execution_path="
            + json.dumps(model.execution_spec(), sort_keys=True)
        )
    manifest["trainable_parameter_count"] = parameter_count
    manifest["model_provenance"] = model_provenance(args)
    _atomic_write_json(run_dir / "run_manifest.json", manifest)
    if torch.cuda.is_available():
        model = model.cuda()
    ema = (
        EMAModel(model, float(getattr(args, "ema_decay", 0.9998)))
        if getattr(args, "use_ema", True)
        else None
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    log.info(
        "effective_optimization_protocol="
        + json.dumps(
            {
                "batch_size": int(args.batch_size),
                "steps_per_epoch": len(train_loader),
                "total_optimizer_steps": len(train_loader) * int(args.epochs),
                "declared_warmup_epochs": int(getattr(args, "warmup_epochs", 0)),
                "effective_linear_warmup": False,
                "initial_lr": float(args.learning_rate),
                "ema_updates": (
                    len(train_loader) * int(args.epochs)
                    if ema is not None
                    else 0
                ),
            },
            sort_keys=True,
        )
    )
    if bool(getattr(args, "compile_model", False)):
        compile_mode = str(getattr(args, "compile_mode", "default"))
        log.info(f"enabling torch.compile(mode={compile_mode!r})")
        model = torch.compile(model, mode=compile_mode, fullgraph=False)
    if opts.smoke_only:
        smoke = _smoke_forward_backward(model, train_loader, args)
        _atomic_write_json(run_dir / "cuda_smoke.json", smoke)
        log.info(f"CUDA smoke passed: {json.dumps(smoke, sort_keys=True)}")
        return

    start_epoch = 0
    best_metrics = {
        "raw_p1_mm": float("inf"),
        "raw_p2_mm": float("inf"),
        "raw_pck_150_percent": float("-inf"),
        "raw_auc_0_150_percent": float("-inf"),
        "raw_epoch": 0,
        "ema_p1_mm": float("inf"),
        "ema_p2_mm": float("inf"),
        "ema_pck_150_percent": float("-inf"),
        "ema_auc_0_150_percent": float("-inf"),
        "ema_epoch": 0,
    }
    resume_path = Path(opts.resume) if opts.resume else None
    if resume_path:
        payload = _load_checkpoint(resume_path, model, ema, optimizer)
        if payload.get("checkpoint_type") != "raw":
            raise RuntimeError("training may resume only from a raw checkpoint")
        start_epoch = int(payload.get("epoch", 0))
        best_metrics.update(payload.get("best_metrics") or {})
        _restore_rng_state(payload.get("rng_state"))
        log.info(f"resumed={resume_path} start_epoch={start_epoch}")
    if opts.evaluate:
        path = Path(opts.evaluate)
        payload = _load_checkpoint(path, model, ema)
        eval_model = (
            _unwrap_compiled_model(model)
            if bool(getattr(args, "eager_eval_when_compiled", False))
            else model
        )
        result = evaluate(
            eval_model,
            test_loader,
            args,
            log,
            checkpoint_view=str(payload.get("checkpoint_type", "loaded")),
        )
        _atomic_write_json(
            run_dir / "evaluation.json",
            {"checkpoint": str(path), **result},
        )
        return

    declared_selection_policy = str(
        getattr(args, "mpi3dhp_checkpoint_selection", "final_epoch")
    )
    if declared_selection_policy not in {"final_epoch", "legacy_test_best"}:
        raise RuntimeError(
            "mpi3dhp_checkpoint_selection must be final_epoch or legacy_test_best"
        )
    effective_evaluation_policy = _effective_evaluation_policy(
        declared_selection_policy
    )
    log.info(
        "3DHP evaluation policy; "
        f"declared_selection_policy={declared_selection_policy} "
        f"effective_evaluation_policy={effective_evaluation_policy}; "
        + (
            "primary=best test-monitored EMA MPJPE; fixed final epoch is secondary"
            if effective_evaluation_policy in {"legacy_test_best", "per_epoch_monitored_test"}
            else "primary=fixed final-epoch EMA MPJPE"
        )
    )
    history: list[dict[str, Any]] = []
    for epoch in range(start_epoch, int(args.epochs)):
        losses = {
            name: AverageMeter()
            for name in (
                "3d_pos",
                "3d_scale",
                "lv",
                "lg",
                "3d_velocity",
                "angle",
                "angle_velocity",
                "total",
            )
        }
        started = time.time()
        train_epoch_3dhp(
            args,
            model,
            train_loader,
            losses,
            optimizer,
            epoch=epoch,
            ema_helper=ema,
        )
        final_epoch = epoch + 1 == int(args.epochs)
        should_evaluate = _should_evaluate_epoch(
            declared_selection_policy, final_epoch
        )
        raw_result = None
        ema_result = None
        if should_evaluate:
            eval_model = (
                _unwrap_compiled_model(model)
                if bool(getattr(args, "eager_eval_when_compiled", False))
                else model
            )
            raw_result = evaluate(
                eval_model,
                test_loader,
                args,
                log,
                checkpoint_view="raw",
            )
            if ema is not None:
                with ema.average_parameters(model):
                    ema_result = evaluate(
                        eval_model,
                        test_loader,
                        args,
                        log,
                        checkpoint_view="ema",
                    )
            else:
                ema_result = dict(raw_result)
                ema_result["checkpoint_view"] = "ema_disabled_raw_alias"
        row = {
            "epoch": epoch + 1,
            "minutes": (time.time() - started) / 60.0,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "raw": raw_result,
            "ema": ema_result,
            "train_losses": {
                name: float(meter.avg) for name, meter in losses.items()
            },
        }
        history.append(row)
        lr = optimizer.param_groups[0]["lr"] * float(getattr(args, "lr_decay", 1.0))
        for group in optimizer.param_groups:
            group["lr"] = lr

        raw_improved = bool(
            raw_result is not None
            and raw_result["mpjpe_mm"] < best_metrics["raw_p1_mm"]
        )
        ema_improved = bool(
            ema_result is not None
            and ema_result["mpjpe_mm"] < best_metrics["ema_p1_mm"]
        )
        if raw_improved:
            best_metrics["raw_p1_mm"] = raw_result["mpjpe_mm"]
            best_metrics["raw_p2_mm"] = raw_result["p_mpjpe_mm"]
            best_metrics["raw_pck_150_percent"] = raw_result["pck_150_percent"]
            best_metrics["raw_auc_0_150_percent"] = raw_result["auc_0_150_percent"]
            best_metrics["raw_epoch"] = epoch + 1
        if ema_improved:
            best_metrics["ema_p1_mm"] = ema_result["mpjpe_mm"]
            best_metrics["ema_p2_mm"] = ema_result["p_mpjpe_mm"]
            best_metrics["ema_pck_150_percent"] = ema_result["pck_150_percent"]
            best_metrics["ema_auc_0_150_percent"] = ema_result["auc_0_150_percent"]
            best_metrics["ema_epoch"] = epoch + 1

        row["best_so_far"] = dict(best_metrics)
        row["declared_selection_policy"] = declared_selection_policy
        row["effective_evaluation_policy"] = effective_evaluation_policy
        _atomic_write_json(run_dir / "metrics.json", history)

        _save_checkpoint(
            run_dir / "latest_epoch.bin",
            epoch=epoch,
            lr=lr,
            optimizer=optimizer,
            model=model,
            ema=ema,
            best_metrics=best_metrics,
            kind="raw",
            protocol=protocol,
        )
        _save_checkpoint(
            run_dir / "latest_ema_epoch.bin",
            epoch=epoch,
            lr=lr,
            optimizer=optimizer,
            model=model,
            ema=ema,
            best_metrics=best_metrics,
            kind="ema",
            protocol=protocol,
        )
        if final_epoch:
            _save_checkpoint(
                run_dir / f"raw_fixed_epoch{epoch + 1}.bin",
                epoch=epoch,
                lr=lr,
                optimizer=optimizer,
                model=model,
                ema=ema,
                best_metrics=best_metrics,
                kind="raw",
                protocol=protocol,
            )
            _save_checkpoint(
                run_dir / f"ema_fixed_epoch{epoch + 1}.bin",
                epoch=epoch,
                lr=lr,
                optimizer=optimizer,
                model=model,
                ema=ema,
                best_metrics=best_metrics,
                kind="ema",
                protocol=protocol,
            )
        if raw_improved:
            _save_checkpoint(
                run_dir / "best_epoch.bin",
                epoch=epoch,
                lr=lr,
                optimizer=optimizer,
                model=model,
                ema=ema,
                best_metrics=best_metrics,
                kind="raw",
                protocol=protocol,
            )
        if ema_improved:
            _save_checkpoint(
                run_dir / "best_ema_epoch.bin",
                epoch=epoch,
                lr=lr,
                optimizer=optimizer,
                model=model,
                ema=ema,
                best_metrics=best_metrics,
                kind="ema",
                protocol=protocol,
            )
        if should_evaluate:
            log.info(
                f"[{epoch + 1}/{int(args.epochs)}] "
                f"raw_p1={raw_result['mpjpe_mm']:.5f} "
                f"raw_p2={raw_result['p_mpjpe_mm']:.5f} "
                f"raw_pck={raw_result['pck_150_percent']:.5f} "
                f"raw_auc={raw_result['auc_0_150_percent']:.5f} "
                f"ema_p1={ema_result['mpjpe_mm']:.5f} "
                f"ema_p2={ema_result['p_mpjpe_mm']:.5f} "
                f"ema_pck={ema_result['pck_150_percent']:.5f} "
                f"ema_auc={ema_result['auc_0_150_percent']:.5f} "
                f"best_raw_epoch={best_metrics['raw_epoch']} "
                f"best_raw_p1={best_metrics['raw_p1_mm']:.5f} "
                f"best_ema_epoch={best_metrics['ema_epoch']} "
                f"best_ema_p1={best_metrics['ema_p1_mm']:.5f} "
                f"lr={row['learning_rate']:.8g} "
                f"train_total={losses['total'].avg:.8f} "
                f"effective_evaluation_policy={effective_evaluation_policy}"
            )
        else:
            log.info(
                f"[{epoch + 1}/{int(args.epochs)}] "
                f"train_total={losses['total'].avg:.8f} "
                "test_locked=true"
            )

    _atomic_write_json(
        run_dir / "complete.json",
        {
            "status": "completed",
            "epochs": len(history),
            "protocol": protocol,
            "best_metrics": best_metrics,
            "declared_checkpoint_selection": declared_selection_policy,
            "effective_evaluation_policy": effective_evaluation_policy,
            "scientific_warning": None,
        },
    )


if __name__ == "__main__":
    main()
