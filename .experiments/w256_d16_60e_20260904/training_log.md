# W256/D16 training log

- Last synchronized snapshot (UTC): `2026-09-04T12:14:20+00:00`.
- Status: `R2_RUNNING`.
- Config: `configs/pose3d/graph_posemamba_h36m_w256_d16_scale_60e.yaml`.
- Source commit: `2e4b8040d6f94fd5c7cf330f3411484312fa4144`.
- Width/depth: `256/16`.
- Trainable parameters: `20,192,451`.
- Protocol: H36M-SH, T243/S81, batch 4, FP32 compiled, seed 0, 60 epochs.
- Run directory: `/scratch/home/caiwei/GraphConditionedPoseMamba_W256_D16_60e_20260904/runs/w256_d16_60e/D16_w256_d16_warmup_clip_seed0_2026_09_04_T_17_20_26`.
- Queue/train PID at snapshot: `595849` / `595869`.

## Current summary

- Completed epochs: `6/60`.
- Latest EMA P1/P2: `53.9646/41.1362 mm` at epoch 6.
- Best EMA P1 and paired P2: `53.9646/41.1362 mm` at epoch 6.
- Current iteration trace: unavailable or waiting to start.
- Error matches: `0`.
- Latest pre-clip gradient norm: `8.3657` (configured max norm `1.0`).
- Latest train throughput: `3.198 it/s`.
- Stable mean throughput: `3.198 it/s`.
- Trainer peak reserved VRAM: `21070 MiB`.
- External monitor latest total GPU memory/utilization: `21798 MiB / 95%`.
- External monitor max temperature/power: `65 C / 537.5 W`.
- External total memory may include concurrent GPU processes; use the trainer-reserved value as the per-model metric.

## Registered preflight

- B1: eager, peak reserved `6214 MiB`, loss `1.598296`, throughput `8.406 it/s`.
- B2: eager, peak reserved `12488 MiB`, loss `1.793585`, throughput `5.076 it/s`.
- B4: compiled, peak reserved `20406 MiB`, loss `2.409278`, throughput `0.965 it/s`.

## Completed-epoch history

| Epoch | Train min | LR | Train loss | EMA P1 | Paired P2 | Grad norm | it/s | Reserved MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 23.29 | 0.000106 | 0.085076 | 251.9754 | 140.3122 | 13.4771 | 3.175 | 21070 |
| 2 | 23.15 | 0.000162 | 0.050481 | 141.1140 | 95.5217 | 13.7686 | 3.195 | 21070 |
| 3 | 23.12 | 0.000219 | 0.040036 | 85.9134 | 65.1726 | 12.5011 | 3.198 | 21070 |
| 4 | 23.12 | 0.000275 | 0.032367 | 65.0296 | 51.0919 | 10.8883 | 3.198 | 21070 |
| 5 | 23.12 | 0.000331 | 0.027455 | 55.0944 | 43.5533 | 9.4930 | 3.198 | 21070 |
| 6 | 23.12 | 0.000387 | 0.026702 | 53.9646 | 41.1362 | 8.3657 | 3.198 | 21070 |

## Interpretation guard

Training-time results are provisional. The paper result is the minimum EMA P1 within epochs 1-60 and the P2 from that same checkpoint; raw/EMA checkpoints are strictly replayed after completion.
