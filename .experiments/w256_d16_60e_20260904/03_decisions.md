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
