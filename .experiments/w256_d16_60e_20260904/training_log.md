# W256/D16 60-epoch training log

- Last synchronized snapshot: `2026-09-04T09:20:30+00:00`.
- Status: `R2_RUNNING`, epoch 1.
- Config: `configs/pose3d/graph_posemamba_h36m_w256_d16_scale_60e.yaml`.
- Parameters: `20,192,451`.
- Budget: `60` epochs.
- Protocol: H36M-SH T243/S81, batch 4, FP32 compiled, seed 0.
- Invalid run directory: `/scratch/home/caiwei/GraphConditionedPoseMamba_W256_D16_60e_20260904/runs/w256_d16_60e/D16_w256_d16_seed0_2026_09_04_T_15_34_18`.
- Invalid run observations: epoch 2 `117.4881/77.3930 mm`; epoch 3
  `221.7309/127.1995 mm`; loss rose 12.62%.
- Numerical audit: all model and optimizer tensors finite; no OOM/CUDA error.
- Root cause finding: declared `warmup_epochs: 8` was inactive in the old
  trainer; no gradient clipping existed.
- Revised run: new prefix, actual per-step warmup and max-grad-norm 1.0;
  re-audit passed and the run started from random initialization.
- R2 source: `2e4b8040d6f94fd5c7cf330f3411484312fa4144`.
- R2 queue/train PID: `595849 / 595869`.
- R2 run directory: `/scratch/home/caiwei/GraphConditionedPoseMamba_W256_D16_60e_20260904/runs/w256_d16_60e/D16_w256_d16_warmup_clip_seed0_2026_09_04_T_17_20_26`.
- Effective scheduler log: start factor 0.1, 8 epochs, 35,496 warmup steps,
  post-warmup decay 0.99.
- R2 latest/best EMA metrics: pending first completed epoch.
- Early live state: 327 iterations at approximately 3.19 it/s, 20,962 MiB
  process memory and 100% GPU utilization; no error.
