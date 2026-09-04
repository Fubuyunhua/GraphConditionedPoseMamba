# User-priority transition from S1 to S2

- Time: 2026-09-04 15:16 Asia/Shanghai.
- User instruction: terminate the current W128/D20 run and start W256/D10.
- S1 state at termination: 65/80 completed epochs.
- S1 best observed EMA: epoch 45, P1 37.659301 mm and paired P2
  31.871747 mm.
- S1 latest observed EMA: epoch 65, P1 37.846863 mm and P2 31.878519 mm.
- S1 queue PID 395234, monitor PID 395262 and train PID 395265 were verified as
  user `caiwei` with the exact scale-repository cwd, then terminated. No GPU
  compute process remained afterward.
- S1 run directory, raw/EMA latest/best checkpoints and logs were preserved.
- S1 is `CANCELLED`, not `COMPLETED`, and its metric is partial/non-comparable
  under the registered 80-epoch rule.
- S2 uses its original random initialization, seed 0, data, batch 4, optimizer,
  losses, EMA and 80-epoch config; no S1 weights are reused.
- S2 queue PID: 556162; train PID: 556185.
- S2 run directory:
  `/scratch/home/caiwei/GraphConditionedPoseMamba_SCALE_80e_20260903/runs/model_scaling_80e/S2_w256_d10_seed0_2026_09_04_T_15_16_07`.
- Early live state: compiled training entered real iterations at approximately
  5.05 it/s, 13,726 MiB process memory, 97% GPU utilization and no error.
