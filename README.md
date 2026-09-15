# GCS-Pose

**Graph-Conditioned Selective State Space Modeling for 3D Human Pose Estimation**

[中文](README_zh.md) · [Detailed reproduction protocol](docs/GCS_POSE_REPRODUCTION.md) · [Data](DATA.md) · [Efficiency](docs/ALL_MODELS_EFFICIENCY_20260915.md)

Research code for video 2D-to-3D human pose lifting. Pose features generate recurrent content U and output gate Z; skeleton-enhanced context generates selective parameters Δ/B/C. Spatial and temporal recurrences are factorized. The implementation class remains `GraphConditionedPoseMamba` for compatibility. This is not SAMA's SSI/MSM.

This repository retains its historical URL and research archives. The six flat YAMLs under `configs/gcs_pose/{detector,gt2d}` are the recommended entry points. Licensed datasets and checkpoints are not included. This source/recipe release does not claim fresh-clone reproduction of every historical score.

## Recorded Human3.6M results

Historical best monitored-test EMA checkpoints, seed 0; P1/P2 in mm.

| Model | True parameters | Detector P1 / P2 | Best epoch | Actual detector run | GT2D P1 / P2 (epoch) |
|---|---:|---:|---:|---|---:|
| W64D8 | 800,083 | 39.845 / 33.232 | 53 | 120 epochs | 16.816 / 13.979 (78) |
| W128D10 | 3,435,395 | 38.484 / 32.383 | 54 | batch 4, 80 epochs | 12.653 / 10.812 (79) |
| W128D20 | 6,836,355 | 37.659 / 31.872 | 45 | stopped at 65/80 | 11.310 / 10.058 (65) |

GT2D runs completed 80 epochs. Their evaluator fixes predicted XY to GT input XY: interpret these as known-XY/depth-lifting results, not detector or unconstrained XYZ results. W128D10 excludes the older batch-8 run. W128D20 is the original S1, not fine-tuned weights. The small model is W64D8, not W64D10. These are recorded results, not newly completed independent replays.

## Installation

Use Linux, an NVIDIA GPU, and a CUDA toolkit/compiler compatible with PyTorch. The recorded RTX 5090 environment used Python 3.10 and PyTorch 2.11.0+cu128. This is an observed environment, not a complete dependency lock. Install matching CUDA-enabled PyTorch and torchvision first using official PyTorch instructions.

```bash
git clone https://github.com/Fubuyunhua/GraphConditionedPoseMamba.git
cd GraphConditionedPoseMamba
python -m venv .venv
source .venv/bin/activate
# Install compatible CUDA PyTorch + torchvision first.
python -m pip install -r requirements.txt
nvcc --version
bash scripts/build_selective_scan.sh
python scripts/verify_install.py
```

The last command checks native CUDA forward/backward and 800,083 parameters; it is not full training. Triton is required by the Linux execution path. CPU-only and native Windows training are not validated. Check toolkit/compiler/PyTorch compatibility and `TORCH_CUDA_ARCH_LIST` if extension building fails; do not substitute another scan implementation silently.

## Data

Obtain Human3.6M under its own license and prepare the historical H36M-SH preprocessing:

```text
DATA_ROOT/
  h36m_sh_conf_cam_source_final.pkl
  H36M-SH/
    train/*.pkl
    test/*.pkl
```

See [DATA.md](DATA.md) and [protocol notes](docs/GCS_POSE_REPRODUCTION.md) for identity checks and limitations. Raw videos alone are insufficient. The complete raw-video-to-identical-preprocessing pipeline is not supplied yet. Both detector and GT2D recipes use this metadata; do not swap detectors or clip boundaries.

## Train from scratch

Choose `w64d8`, `w128d10`, or `w128d20`, and independently `detector` or `gt2d`. All six recipes use 80 epochs, batch 4 and EMA; scientific settings remain size-specific.

```bash
# Prepare config and record hashes; no training. Output must not exist.
python scripts/reproduce_gcspose.py --model w128d20 --protocol detector \
  --data-root /path/to/DATA_ROOT --output runs/inspect_w128d20 --seed 0

# Explicitly launch a fresh run in a different new directory.
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce_gcspose.py \
  --model w128d20 --protocol detector --data-root /path/to/DATA_ROOT \
  --output runs/w128d20_detector_seed0 --seed 0 --run
```

For GT2D change `--protocol gt2d` and use a new output path. The wrapper records `launch.json` and resolved `config.yaml`, accepts no resume/pretrained options, and refuses existing output directories. The trainer adds a timestamp to the checkpoint prefix. Preparation is a layout check, not numerical preflight. The 80-epoch common budget is a new recipe, not an already-completed matched experiment or a convergence guarantee.

## Evaluate a trusted checkpoint

```bash
python -X utf8 train.py --config runs/w128d20_detector_seed0/config.yaml \
  --evaluate /path/to/best_epoch_ema.bin --checkpoint runs/eval_w128d20 --seed 0
sha256sum /path/to/best_epoch_ema.bin
```

Use the actual matching config and raw/EMA checkpoint type. Record its hash, epoch and code commit. The existing trainer uses strict structural loading and requires the prepared train/test layout even for evaluation. `--evaluate` does not train. Only load trusted pickle checkpoints. No public weight URLs are fabricated here.

## Parameters and computation

| Model | Parameters | MAC-equivalent / 243-frame clip |
|---|---:|---:|
| W64D8 | 800,083 | 4.602 G |
| W128D10 | 3,435,395 | 18.739 G |
| W128D20 | 6,836,355 | 37.474 G |
| Local PoseMamba A0 | 790,083 | 4.551 G |

Dense contraction counts plus analytical parallel-scan FLOPs/2; not exact hardware instruction counts. Norm, activations, other pointwise operations, memory and backward are excluded. Vanilla THOP omits functional operations and some parameters. See the [counter audit](docs/COMPLEXITY_REAUDIT_20260915.md); do not compare incompatible counting conventions.

```bash
# Optional profiling dependency; explicit user installation:
python -m pip install thop
python scripts/audit_complexity_conventions.py --root . --output runs/counts.json
```

Profiling uses fresh model instances, requires CUDA and no pretrained weights. Counts alone do not certify third-party checkpoint provenance.

## Status, license and citation

See [release boundaries](docs/GCS_POSE_REPRODUCTION.md) before using archived ablations, repeat experiments or MPI results. Do not automatically execute historical queues. A2/Full repeats are not declared complete by this release; Q/H has no completed native CUDA preflight. Public checkpoints, a full environment lock and clean-environment end-to-end reproduction remain outstanding.

Preserve [LICENSE](LICENSE), [NOTICE](NOTICE) and [third-party acknowledgments](docs/THIRD_PARTY_NOTICES.md). Data and third-party weights have separate licenses. Author list, DOI and publication metadata will be added when supplied; no acceptance or fabricated citation metadata is asserted.
