# W256/D16 R3 stable optimizer training log

- Last synchronized snapshot (UTC): `2026-09-04T18:53:05+00:00`.
- Status: `R3_RUNNING`.
- Config: `configs/pose3d/graph_posemamba_h36m_w256_d16_stable_r3_60e.yaml`.
- Source commit: `0e23c5d01eb5e6c81c66bc17a621d38787e7c46d`.
- Width/depth: `256/16`.
- Trainable parameters: `20,192,451`.
- Protocol: H36M-SH, T243/S81, batch 4, FP32 compiled, seed 0, 60 epochs.
- Run directory: `runs/w256_d16_r3_60e/D16_w256_d16_stable_r3_seed0_2026_09_05_T_00_48_32`.
- Queue/train PID at snapshot: `713220` / `713242`.

## Current summary

- Completed epochs: `4/60`.
- Latest EMA P1/P2: `67.0120/52.6453 mm` at epoch 4.
- Best EMA P1 and paired P2: `67.0120/52.6453 mm` at epoch 4.
- Current iteration trace: unavailable or waiting to start.
- Error matches: `0`.
- Latest pre-clip gradient norm: `12.1676` (configured max norm `1.0`).
- Latest train throughput: `3.189 it/s`.
- Stable mean throughput: `3.190 it/s`.
- Trainer peak reserved VRAM: `21140 MiB`.
- Latest maximum pre-clip gradient norm: `21.9016`; clipped-step fraction `100.00%`.
- Latest raw parameter movement: relative L2 `1.3821%`, max absolute `0.152415`.
- External monitor latest total GPU memory/utilization: `21866 MiB / 99%`.
- External monitor max temperature/power: `64 C / 532.3 W`.
- External total memory may include concurrent GPU processes; use the trainer-reserved value as the per-model metric.

## Registered preflight

- B1: eager, peak reserved `6212 MiB`, loss `1.165424`, throughput `8.385 it/s`.
- B2: eager, peak reserved `12486 MiB`, loss `1.141652`, throughput `5.075 it/s`.
- B4: compiled, peak reserved `20404 MiB`, loss `1.546054`, throughput `0.963 it/s`.

## Completed-epoch history

| Epoch | Train min | LR | Train loss | EMA P1 | Paired P2 | Grad norm | it/s | Reserved MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 23.31 | 0.000064 | 0.087781 | 253.2292 | 139.7223 | 13.1656 | 3.173 | 21140 |
| 2 | 23.18 | 0.000097 | 0.052941 | 174.9847 | 105.9645 | 13.7333 | 3.190 | 21140 |
| 3 | 23.18 | 0.000131 | 0.043323 | 94.7903 | 70.4865 | 13.7373 | 3.190 | 21140 |
| 4 | 23.19 | 0.000165 | 0.035579 | 67.0120 | 52.6453 | 12.1676 | 3.189 | 21140 |

## Interpretation guard

Training-time results are provisional. The paper result is the minimum EMA P1 within epochs 1-60 and the P2 from that same checkpoint; raw/EMA checkpoints are strictly replayed after completion.
