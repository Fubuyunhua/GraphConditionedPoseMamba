# A1 Factorized Only — final analysis

## Validity and provenance

- Status: VALID_COMPLETE_TEST_MONITORED.
- Source: remote commit `cba4a2d468388da29aef1fdf39938cc6c95c34f0`.
- Config: `configs/pose3d/ablation_factorized_only.yaml`; seed 0; 80/80
  epochs; H36M-SH T243/S81 dataset hash matches the registered baseline.
- All latest/best raw/EMA checkpoints load. Latest raw/EMA epoch is 80 and
  best raw/EMA epoch is 47. No NaN, Inf, OOM, CUDA error or Traceback occurs.
- Strict post-run evaluation reproduces the selected EMA log metric.

## Result

- Trainable parameters: 749,891.
- Best EMA epoch: 47.
- Primary P1: 40.060496 mm.
- Same-checkpoint P2: 33.356521 mm.
- Raw checkpoint at epoch 47: P1 41.323762 mm, P2 34.500653 mm.
- Final epoch EMA: P1 40.427000 mm, P2 33.269602 mm.

Relative to A0 PoseMamba, A1 improves P1 by 0.165526 mm and P2 by
0.161106 mm. Relative to A3 Full, it is worse by 0.215334 mm P1 and
0.124280 mm P2.

## Dynamics and efficiency

Best P1 occurs at epoch 47; continued optimization lowers training loss but
does not improve P1, indicating test-metric overfitting after the selected
checkpoint. Pure training time sums to 232.93 minutes; full wall time including
compile, evaluation and verification is 4:18:24. Mean stable throughput is
25.82 it/s and trainer peak reserved VRAM is 2,370 MiB. The external monitor's
95th-percentile total GPU usage is 3,080 MiB; its 7,541 MiB maximum is
contaminated by a separately registered preflight and is not attributed to A1.

## Claim and recommendation

The prediction that boundary-preserving factorization improves the matched
PoseMamba reference is SUPPORTED for seed 0. The result does not isolate seed
variance and uses the inherited test-monitored protocol. Recommendation:
KEEP as ablation evidence; do not replace A3 Full with A1.
