# Decisions

## D-D16-001

The user requested W256/D16 with a 60-epoch budget and immediate execution.
Interpret immediate execution as replacement of the active W256/D10 run because
their combined FP32 batch-4 memory cannot safely fit on one 32GB GPU. Preserve
and mark W256/D10 as user-cancelled; do not reuse its weights.

## D-D16-002

Keep the original speed-first batch-4 compiled protocol and require staged
eager batch1/batch2 plus compiled batch4 before launch. Do not pre-emptively
enable activation checkpointing or change LR.

## D-D16-003

Invalidate and stop the first D16 run after the configured warmup was proven
inactive and epoch 2-to-3 loss/P1 triggered the instability rule. Preserve the
run, add opt-in per-step warmup and clipping, then return to preflight.

## D-D16-004

Pass the R2 staged preflight and restart from random initialization with an
eight-epoch linear warmup, pre-clip gradient logging, and max norm 1.0. Never
reuse the invalid R1 optimizer state or checkpoint directory.

## D-D16-005

Invalidate and stop R2 at epoch 16. Epoch 15-to-16 loss rose 121.5% and EMA P1
worsened by 92.8847 mm, which triggers the registered fail-closed rule. Preserve
the epoch-15 best and epoch-16 latest raw/EMA checkpoints; all checkpoint tensors
are finite. A `wh` process overlapped part of epoch 16 and reduced throughput,
but this is a confounder rather than an established cause. Do not resume or
automatically launch a replacement experiment.
