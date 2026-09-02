#!/usr/bin/env python3
"""Benchmark the exact GraphConditionedPoseMamba train.py step on CUDA."""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader


RELEASE_ROOT = Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) in sys.path:
    sys.path.remove(str(RELEASE_ROOT))
sys.path.insert(0, str(RELEASE_ROOT))

from lib.data.dataset_motion_3d import MotionDataset3D
from lib.utils.learning import AverageMeter, load_backbone
from lib.utils.tools import get_config
from train import CudaGraphTrainModel, EMAModel, train_epoch


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
    return parser.parse_args()


def main():
    options = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    config = get_config(options.config)
    config.mask = bool(config.mask_ratio > 0 and config.mask_T_ratio > 0)
    use_compile = bool(getattr(config, "compile_model", False)) and not options.no_compile
    use_cuda_graph = bool(getattr(config, "cuda_graph_model", False))
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
        model = torch.compile(base_model, mode="reduce-overhead", fullgraph=False)
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
    )
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - warmup_start

    losses = make_meters()
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
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(
        json.dumps(
            {
                "compiled": use_compile,
                "cuda_graph": use_cuda_graph,
                "real_data": options.real_data,
                "warmup_seconds": warmup_seconds,
                "steps": options.steps,
                "elapsed_seconds": elapsed,
                "milliseconds_per_step": elapsed * 1000 / options.steps,
                "iterations_per_second": options.steps / elapsed,
                "loss": losses["total"].avg,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
