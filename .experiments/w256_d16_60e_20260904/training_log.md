# W256/D16 60-epoch training log

- Last synchronized snapshot: `2026-09-04T09:15:00+00:00`.
- Status: `FIRST_RUN_INVALID_AT_EPOCH_3; REVISED_PREFLIGHT_PENDING`.
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
  awaiting re-audit before restart.
