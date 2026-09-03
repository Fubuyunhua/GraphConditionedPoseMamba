# Staged preflight and S1 launch

- Source: `d30993d04fa57de57b38b55a69dc3913b90b87b0`.
- Dataset hash: registered H36M-SH hash matched.
- Static shape/protocol/parameter preflight: PASS.
- Full unit/CUDA suite: PASS.
- W128/D20 real-data peak reserved MiB:
  - eager batch 1: 3,734;
  - eager batch 2: 7,254;
  - compiled batch 4: 14,092, finite loss 2.8273.
- W256/D10 real-data peak reserved MiB:
  - eager batch 1: 3,936;
  - eager batch 2: 7,862;
  - compiled batch 4: 13,018, finite loss 5.6382.
- Registered gate: every model-process peak below 24,576 MiB; PASS.
- External process: `wh` PID 367341 retained; no modification or interruption.
- Formal queue PID: 395234.
- S1 train PID: 395265.
- S1 run directory:
  `/scratch/home/caiwei/GraphConditionedPoseMamba_SCALE_80e_20260903/runs/model_scaling_80e/S1_w128_d20_seed0_2026_09_03_T_21_41_37`.
- Early live observation: 261 iterations reached after initial compilation,
  approximately 4.19 instantaneous it/s, 14,972 MiB process memory and no OOM.
- S2 remains serially gated behind valid S1 completion and strict replay.
