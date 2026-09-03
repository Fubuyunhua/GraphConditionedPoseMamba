# W128/D20 and W256/D10 scaling audit

Verdict: CONDITIONAL — static review passes; RTX5090 compiled real-data smoke is
required after the active A1/A2 queue completes.

First post-ablation attempt at 2026-09-03 20:49 Asia/Shanghai stopped before
tests or smoke: only 28,476 MiB was free versus the registered 28,672 MiB
minimum. No scale training was started and the threshold was not changed.

## Scope and immutable reference

- Reference: GraphConditionedPoseMamba Full W64/D8, 800,083 parameters.
- Reference result: best EMA epoch 53, P1 39.8452 mm and same-checkpoint P2
  33.2322 mm.
- Dataset: H36M-SH xy+confidence, T243/S81, SHA-256
  `73b642f2567a8d0b194f88c54a3182c7b635c003c832b48ae6ee559f10232175`.
- The candidates change only width/depth and the explicitly registered
  80-epoch budget. They keep Full graph-conditioned control and factorization.

## Effective model and tensor flow

Input `[B,243,17,3]` is projected to `[B,243,17,C]`. Each block applies:

1. spatial normalization and joint position;
2. anatomical graph mixing and topology-conditioned Delta/B/C;
3. boundary-preserving spatial BiSSM on `[B*243,1,17,C]`;
4. temporal normalization and temporal position;
5. reused graph context and temporal BiSSM on `[B*17,1,243,C]`;
6. residual MLP and a final LayerNorm/linear 3D prediction head.

W128/D20 uses SSM inner width 240 and MLP inner width 252. W256/D10 uses
SSM inner width 480 and MLP inner width 504. Residual scales initialize to 1;
drop-path ends at 0.2 in both candidates, matching Full.

Static construction reports 6,836,355 trainable parameters for W128/D20 and
12,646,107 for W256/D10. Shape/protocol checks and all 20 non-CUDA unit tests
pass; six CUDA-only tests are intentionally deferred to the post-ablation GPU
gate.

## Frozen optimization, loss, data, and evaluation

- FP32 AdamW, batch 4, LR 5e-4, weight decay 0.012, exponential decay 0.99,
  eight warmup epochs, EMA 0.9998.
- Active losses: 3D position (1.0), scale (0.5), velocity (20.0), and diffusion
  term (0.5). Targets, reductions and gradient paths are unchanged from Full.
- Root-relative H36M-SH input, confidence retained, flip enabled, no masking or
  noise, identical split and evaluator.
- Primary selection: minimum EMA P1 in epochs 1-80. P2 must come from that exact
  checkpoint. Raw and EMA checkpoints are re-evaluated after completion.

## Findings and gates

- P0: none found in the declared configs.
- P1 runtime risk: compiled batch-4 memory and finite gradients are not yet
  measured for either larger model. Required diagnostic: one warmup and one
  timed real-data train step per candidate on the RTX5090. Formal training is
  blocked until both pass with peak reserved VRAM below 28,672 MiB.
- P2 optimization risk: the larger models reuse the W64 learning rate and only
  one seed. This is intentionally controlled, but instability would make the
  run invalid rather than trigger silent LR tuning.
- P2 interpretation risk: W128/D20 and W256/D10 are not parameter-matched; this
  is an accuracy/capacity scaling study, not an architectural ablation.
- P3 cost risk: W256/D10 is expected to be substantially slower. The cheaper
  W128/D20 runs first; metric outcome does not stop W256/D10, but correctness,
  OOM, non-finite values, or checkpoint failure does.

No activation checkpointing is enabled initially because the user prioritizes
training speed. If either smoke exceeds the memory gate, the queue stops for a
new reviewed config instead of changing batch size or checkpointing silently.
