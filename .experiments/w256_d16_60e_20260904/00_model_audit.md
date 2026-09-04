# W256/D16 60-epoch audit

Verdict: PASS — static/CUDA and staged real-data batch1/batch2/compiled-batch4
gates passed before formal training.

## Registered model

- GraphConditionedPoseMamba Full mechanisms, width 256, depth 16.
- 20,192,451 trainable parameters; SSM inner width 480; MLP inner width 504.
- Factorized spatial/temporal K=2 scans, anatomical graph and
  topology-conditioned Delta/B/C remain enabled.
- Input/output contract remains `[B,243,17,3]`.

## Frozen protocol

- H36M-SH xy+confidence T243/S81, seed 0, batch 4, FP32 compiled.
- 60 epochs, warmup 8, AdamW LR 5e-4, WD 0.012, decay 0.99.
- EMA 0.9998 and unchanged position/scale/velocity/diffusion losses.
- Primary result: minimum EMA P1 in epochs 1-60; P2 from that checkpoint.

## Gate

Run eager batch 1, eager batch 2 and compiled batch 4 on real H36M after the
W256/D10 process exits. All losses must be finite, no gradient/runtime error may
occur, and batch-4 model-process peak reserved memory must remain below
28,672 MiB. Failure blocks formal training; no silent batch/LR/checkpointing
change is permitted.

Observed peak reserved memory was 6,214 MiB at eager batch 1, 12,488 MiB at
eager batch 2 and 20,274 MiB at compiled batch 4. All losses were finite and
throughput positive; the formal 60-epoch run was released unchanged.
