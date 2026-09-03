# A2 Graph Feature Fusion — final analysis

## Validity and provenance

- Status: VALID_COMPLETE_TEST_MONITORED.
- Source/config/data/seed/budget match the registered A2 protocol: explicit
  `graph_injection_mode=feature`, seed 0, and 80/80 epochs.
- All latest/best raw/EMA checkpoints load. Latest raw/EMA epoch is 80 and
  best raw/EMA epoch is 52. No NaN, Inf, OOM, CUDA error or Traceback occurs.
- Strict post-run evaluation reproduces the selected EMA result.

## Result

- Trainable parameters: 800,083.
- Best EMA epoch: 52.
- Primary P1: 40.058826 mm.
- Same-checkpoint P2: 33.287296 mm.
- Raw checkpoint at epoch 52: P1 41.006761 mm, P2 34.495551 mm.
- Final epoch EMA: P1 40.571232 mm, P2 33.607390 mm.

A2 improves over A1 by only 0.001670 mm P1, which is not meaningful at one
seed, while paired P2 improves by 0.069224 mm. A3 Full remains better than A2
by 0.213664 mm P1 and 0.055056 mm P2. A2 improves over A0 PoseMamba by
0.167196 mm P1 and 0.230330 mm P2.

## Dynamics and efficiency

Best P1 occurs at epoch 52. Training loss continues to fall, but final P1
regresses by 0.512406 mm, so later optimization overfits the monitored metric.
Pure training time sums to 265.47 minutes; full wall time including compile,
evaluation and verification is 4:53:55. Mean stable throughput is 22.81 it/s
and trainer peak reserved VRAM is 2,640 MiB. External total-GPU memory is
contaminated by another process during much of A2 and is not used as a
per-model peak; direct process observations were approximately 3,340 MiB.

## Claim and recommendation

The registered prediction that topology-conditioned control is better than
ordinary graph feature fusion is SUPPORTED for seed 0: A3 Full improves P1 by
0.213664 mm under equal parameter count and the frozen protocol. Ordinary
feature fusion adds no meaningful P1 gain over A1. Recommendation: KEEP as
causal ablation evidence and REJECT feature fusion as the final design.
