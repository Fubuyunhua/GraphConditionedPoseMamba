# W256/D16 60-epoch audit

Verdict: PASS — the first run remains invalid; revised opt-in warmup/clipping
implementation passed unit, CUDA and staged real-data gates before restart.

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

That first run is INVALID for the intended optimizer protocol. The configuration
declared eight warmup epochs but old `train.py` never consumed the field. At
epoch 2 to 3, training loss increased 12.62%, P1 regressed 104.24 mm and P2
regressed 49.81 mm. Model and optimizer tensors remained finite, excluding
NaN/checkpoint corruption.

The revised path is opt-in and does not alter legacy configs: per-step linear
warmup from 0.1x to 1.0x LR across 35,496 steps, followed by the existing
epoch-wise 0.99 decay; pre-clip gradient norm logging; and max-norm 1.0 with
non-finite gradients raising an error. It must use a new run prefix.

R2 preflight confirms the effective batch-4 warmup has 35,496 steps, starts at
approximately 5e-5 LR, reports finite pre-clip gradient norm 44.925, and uses
20,406 MiB peak reserved memory. Eager batch1/batch2 and compiled batch4 all
pass. The corrected run was released under source commit `2e4b804`.
