# W256/D16 R3 stable optimizer training log

- Last synchronized snapshot (UTC): `2026-09-05T00:48:08+00:00`.
- Status: `R3_RUNNING_RECOVERED`.
- Config: `configs/pose3d/graph_posemamba_h36m_w256_d16_stable_r3_60e.yaml`.
- Source commit: `0e23c5d01eb5e6c81c66bc17a621d38787e7c46d`.
- Width/depth: `256/16`.
- Trainable parameters: `20,192,451`.
- Protocol: H36M-SH, T243/S81, batch 4, FP32 compiled, seed 0, 60 epochs.
- Run directory: `runs/w256_d16_r3_60e/D16_w256_d16_stable_r3_seed0_2026_09_05_T_00_48_32`.
- Queue/train PID at snapshot: `713220` / `713242`.

## Current summary

- Completed epochs: `18/60`.
- Latest EMA P1/P2: `39.3476/33.0521 mm` at epoch 18.
- Best EMA P1 and paired P2: `39.3476/33.0521 mm` at epoch 18.
- Current iteration trace: unavailable or waiting to start.
- Error matches: `0`.
- Latest pre-clip gradient norm: `3.1443` (configured max norm `1.0`).
- Latest train throughput: `3.187 it/s`.
- Stable mean throughput: `3.190 it/s`.
- Trainer peak reserved VRAM: `21140 MiB`.
- Latest maximum pre-clip gradient norm: `7.8749`; clipped-step fraction `97.41%`.
- Latest raw parameter movement: relative L2 `2.5844%`, max absolute `0.135993`.
- External monitor latest total GPU memory/utilization: `25497 MiB / 94%`.
- External monitor max temperature/power: `65 C / 536.5 W`.
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
| 5 | 23.19 | 0.000199 | 0.030285 | 55.8359 | 45.2113 | 10.8843 | 3.189 | 21140 |
| 6 | 23.19 | 0.000232 | 0.027638 | 51.8354 | 42.2794 | 9.7250 | 3.189 | 21140 |
| 7 | 23.19 | 0.000266 | 0.032203 | 99.9209 | 62.1321 | 8.8514 | 3.189 | 21140 |
| 8 | 23.19 | 0.000300 | 0.033854 | 154.4010 | 95.7771 | 7.2805 | 3.189 | 21140 |
| 9 | 23.19 | 0.000300 | 0.025066 | 67.6448 | 48.7047 | 6.7729 | 3.189 | 21140 |
| 10 | 23.19 | 0.000299 | 0.021622 | 49.4361 | 39.4220 | 6.1150 | 3.189 | 21140 |
| 11 | 23.19 | 0.000298 | 0.020098 | 45.4801 | 36.9451 | 5.5276 | 3.189 | 21140 |
| 12 | 23.19 | 0.000296 | 0.018407 | 43.3857 | 35.7596 | 5.1876 | 3.189 | 21140 |
| 13 | 23.19 | 0.000294 | 0.017173 | 42.1029 | 34.9815 | 4.7562 | 3.188 | 21140 |
| 14 | 23.19 | 0.000291 | 0.016308 | 41.3167 | 34.4999 | 4.4037 | 3.189 | 21140 |
| 15 | 23.19 | 0.000288 | 0.015260 | 40.5683 | 33.9870 | 4.0969 | 3.189 | 21140 |
| 16 | 23.20 | 0.000285 | 0.014791 | 40.0604 | 33.5978 | 3.7506 | 3.188 | 21140 |
| 17 | 23.21 | 0.000281 | 0.013829 | 39.6964 | 33.3319 | 3.4336 | 3.186 | 21140 |
| 18 | 23.20 | 0.000276 | 0.013467 | 39.3476 | 33.0521 | 3.1443 | 3.187 | 21140 |

## Interpretation guard

Training-time results are provisional. The paper result is the minimum EMA P1 within epochs 1-60 and the P2 from that same checkpoint; raw/EMA checkpoints are strictly replayed after completion.
