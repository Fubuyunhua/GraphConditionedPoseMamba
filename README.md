# GCS-Pose

**Graph-Conditioned Selective State Space Modeling for 3D Human Pose Estimation**

[中文](README_zh.md) | [Model and evaluation notes](docs/MODEL_CARD.md) | [Data](docs/DATA.md)

GCS-Pose lifts video 2D keypoints to 3D human poses. Pose features generate recurrent
content U and output gate Z; skeleton-enhanced context generates selective parameters
Δ/B/C. Spatial and temporal recurrences are factorized. The implementation class
remains `GraphConditionedPoseMamba` for checkpoint compatibility.

This public repository contains training/evaluation code, model configurations,
installation checks and checkpoint release metadata. Internal experiment logs,
job queues, research handoffs and development workflows are not part of the release.

## Models

Human3.6M detector input, best EMA checkpoints; P1/P2 in mm.

| Model | Parameters | P1 | P2 | MAC-equivalent / 243-frame clip |
|---|---:|---:|---:|---:|
| W64D8 | 800,083 | 39.845 | 33.232 | 4.602 G |
| W128D10 | 3,435,395 | 38.484 | 32.383 | 18.739 G |
| W128D20 | 6,836,355 | 37.659 | 31.872 | 37.474 G |

See the model card for protocols and selection limits. The small model is W64D8,
not W64D10. Checkpoints are distributed separately from source; see
[weights/README](docs/WEIGHTS.md) for availability and integrity verification.

## Install

Use Linux, Python3.10, NVIDIA CUDA, and compatible PyTorch/torchvision and nvcc.
The reference environment used PyTorch2.11.0+cu128. Other compatible environments
must pass the installation check; native Windows/CPU-only training is not validated.

```bash
git clone https://github.com/Fubuyunhua/GraphConditionedPoseMamba.git
cd GraphConditionedPoseMamba
python -m venv .venv
source .venv/bin/activate
# Install matching CUDA PyTorch and torchvision first.
python -m pip install -r requirements.txt
bash scripts/build_selective_scan.sh
python scripts/verify_install.py
```

Keep Triton matched to the chosen PyTorch distribution. The last command runs a
small native CUDA forward/backward and verifies800083 parameters, not full training.

## Prepare data

Legally acquire Human3.6M and the compatible H36M-SH preprocessed representation.
Datasets and raw videos are not distributed here. See [data requirements](docs/DATA.md).

```text
DATA_ROOT/
  h36m_sh_conf_cam_source_final.pkl
  H36M-SH/train/*.pkl
  H36M-SH/test/*.pkl
```

## Train

Choose `w64d8`, `w128d10`, or `w128d20`. Detector and GT2D configurations are
separate. All public scratch recipes use80 epochs and batch4.

```bash
# Layout check and resolved config only; no training. Output must not exist.
python scripts/reproduce_gcspose.py --model w128d20 --protocol detector \
  --data-root /path/to/DATA_ROOT --output runs/prepare_w128d20 --seed 0

# Explicit scratch training in another new output directory.
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce_gcspose.py \
  --model w128d20 --protocol detector --data-root /path/to/DATA_ROOT \
  --output runs/w128d20_detector --seed 0 --run
```

For GT2D use `--protocol gt2d` and a new output directory. The trainer appends a
timestamp to its checkpoint directory. The wrapper records the resolved config
and data hash; layout checks are not a full numerical preflight. First-batch
compilation can take tens of seconds or longer.

## Evaluate

Prepare a config with the command above (without `--run`), then:

```bash
python scripts/download_weights.py --model w128d20 --protocol detector
CUDA_VISIBLE_DEVICES=0 python -X utf8 train.py \
  --config runs/prepare_w128d20/config.yaml \
  --evaluate weights/gcs_pose_w128d20_detector_ema.bin \
  --checkpoint runs/evaluate_w128d20 --seed 0
```

Use the checkpoint's matching architecture and input protocol, and one visible
GPU. Loading is strict. Verify the release SHA256 before loading; load only trusted
files because the legacy trainer uses pickle-based checkpoint loading. `--evaluate`
does not train. The current entry point expects both prepared train/test directories.

## Complexity

```bash
python -m pip install thop  # optional profiling dependency
python scripts/audit_complexity_conventions.py --root . --output runs/counts.json
```

Counts use dense contraction MACs plus analytical selective-scan FLOPs/2 and the
skip term. Normalization, activations, other pointwise operations, memory traffic
and backward are excluded. This is a MAC-equivalent convention, not exact hardware
instructions; vanilla THOP subtotals are not directly comparable.

## License and citation

Retain [LICENSE](LICENSE), [NOTICE](NOTICE), and
[third-party notices](docs/THIRD_PARTY_NOTICES.md). Data and third-party weights have
separate terms. Author list, DOI and publication metadata will be added when available;
no publication acceptance is implied by this repository.
