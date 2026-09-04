#!/usr/bin/env python3
"""Benchmark the exact GraphConditionedPoseMamba train.py step on CUDA."""

import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


RELEASE_ROOT = Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) in sys.path:
    sys.path.remove(str(RELEASE_ROOT))
sys.path.insert(0, str(RELEASE_ROOT))

from lib.data.dataset_motion_3d import MotionDataset3D
from lib.utils.learning import AverageMeter, load_backbone
from lib.utils.tools import get_config
from train import (
    CudaGraphTrainModel,
    EMAModel,
    build_lr_schedule,
    train_epoch,
)


LOSS_KEYS = (
    "3d_pos",
    "3d_scale",
    "2d_proj",
    "lg",
    "lv",
    "total",
    "3d_velocity",
    "angle",
    "angle_velocity",
    "grad_norm",
)


def make_meters():
    return {key: AverageMeter() for key in LOSS_KEYS}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=(
            "configs/pose3d/"
            "graph_posemamba_h36m_w64_d8_0p8m.yaml"
        ),
    )
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--real-data", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead"),
        default=None,
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--activation-checkpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=20260902)
    return parser.parse_args()


def main():
    options = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    torch.cuda.manual_seed_all(options.seed)

    config = get_config(options.config)
    if options.batch_size is not None:
        config.batch_size = int(options.batch_size)
    if options.activation_checkpoint:
        config.activation_checkpoint_blocks = True
    config.mask = bool(config.mask_ratio > 0 and config.mask_T_ratio > 0)
    use_compile = bool(getattr(config, "compile_model", False)) and not options.no_compile
    use_cuda_graph = bool(getattr(config, "cuda_graph_model", False))
    compile_mode = "eager"
    if options.no_compile:
        use_compile = False

    if options.real_data:
        dataset = MotionDataset3D(config, config.subset_list, "train")
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )
        iterator = iter(loader)
    else:
        sample = (
            torch.randn(config.batch_size, config.clip_len, config.num_joints, 3),
            torch.randn(config.batch_size, config.clip_len, config.num_joints, 3),
        )
        iterator = iter([sample] * (options.warmup_steps + options.steps))

    base_model = load_backbone(config).cuda()
    if use_compile:
        if hasattr(torch, "_dynamo"):
            torch._dynamo.config.recompile_limit = max(
                int(torch._dynamo.config.recompile_limit),
                64,
            )
        compile_mode = options.compile_mode or str(
            getattr(config, "compile_mode", "reduce-overhead")
        )
        model = torch.compile(base_model, mode=compile_mode, fullgraph=False)
    elif use_cuda_graph:
        model = CudaGraphTrainModel(base_model)
    else:
        model = torch.nn.DataParallel(base_model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    ema = EMAModel(base_model, config.ema_decay)
    steps_per_epoch = len(loader) if options.real_data else (
        options.warmup_steps + options.steps
    )
    lr_schedule = build_lr_schedule(
        config,
        optimizer,
        steps_per_epoch=steps_per_epoch,
        start_step=0,
    )

    torch.cuda.reset_peak_memory_stats()
    warmup_start = time.perf_counter()
    train_epoch(
        config,
        model,
        itertools.islice(iterator, options.warmup_steps),
        make_meters(),
        optimizer,
        has_3d=True,
        has_gt=True,
        ema_helper=ema,
        lr_schedule=lr_schedule,
    )
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - warmup_start
    warmup_peak_allocated = torch.cuda.max_memory_allocated()
    warmup_peak_reserved = torch.cuda.max_memory_reserved()

    losses = make_meters()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    train_epoch(
        config,
        model,
        itertools.islice(iterator, options.steps),
        losses,
        optimizer,
        has_3d=True,
        has_gt=True,
        ema_helper=ema,
        lr_schedule=lr_schedule,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    measured_peak_allocated = torch.cuda.max_memory_allocated()
    measured_peak_reserved = torch.cuda.max_memory_reserved()
    print(
        json.dumps(
            {
                "compiled": use_compile,
                "compile_mode": (
                    compile_mode if use_compile else "eager"
                ),
                "cuda_graph": use_cuda_graph,
                "real_data": options.real_data,
                "batch_size": int(config.batch_size),
                "seed": int(options.seed),
                "activation_checkpoint_blocks": bool(
                    getattr(config, "activation_checkpoint_blocks", False)
                ),
                "linear_warmup_enabled": lr_schedule is not None,
                "warmup_steps": (
                    lr_schedule.warmup_steps if lr_schedule is not None else 0
                ),
                "optimizer_lr_after_steps": float(optimizer.param_groups[0]["lr"]),
                "warmup_seconds": warmup_seconds,
                "warmup_peak_allocated_mib": warmup_peak_allocated / 2**20,
                "warmup_peak_reserved_mib": warmup_peak_reserved / 2**20,
                "steps": options.steps,
                "elapsed_seconds": elapsed,
                "milliseconds_per_step": elapsed * 1000 / options.steps,
                "iterations_per_second": options.steps / elapsed,
                "measured_peak_allocated_mib": measured_peak_allocated / 2**20,
                "measured_peak_reserved_mib": measured_peak_reserved / 2**20,
                "loss": losses["total"].avg,
                "grad_norm_preclip": losses["grad_norm"].avg,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
