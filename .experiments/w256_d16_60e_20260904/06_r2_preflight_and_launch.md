# R2 warmup/clipping preflight and launch

- Source: `2e4b8040d6f94fd5c7cf330f3411484312fa4144`.
- Full unit/CUDA suite: PASS.
- Dataset hash: PASS.
- Effective batch-4 warmup: 35,496 optimizer steps.
- LR after two diagnostic steps: `5.00127e-5`.
- Global pre-clip gradient norm: `44.925`; applied max norm: `1.0`.
- Batch1/2/4 reserved memory: 6,214 / 12,488 / 20,406 MiB.
- All diagnostic losses and gradient norms finite; no OOM or CUDA error.
- R2 starts from random initialization with a new output prefix; no R1 state is
  resumed or overwritten.
- Queue PID: 595849.
- Train PID: 595869.
- Launcher log:
  `/scratch/home/caiwei/GraphConditionedPoseMamba_W256_D16_60e_20260904/launch_logs/D16_R2_warmup_clip_092024.log`.
- Run directory:
  `/scratch/home/caiwei/GraphConditionedPoseMamba_W256_D16_60e_20260904/runs/w256_d16_60e/D16_w256_d16_warmup_clip_seed0_2026_09_04_T_17_20_26`.
