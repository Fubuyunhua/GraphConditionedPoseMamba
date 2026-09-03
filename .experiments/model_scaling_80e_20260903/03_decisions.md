# Scaling-study decisions

## D-SCALE-001 — register two accuracy-oriented scales

Run W128/D20 before W256/D10, both for 80 epochs. The user explicitly
authorized training after the active A1/A2 ablations complete without
execution or integrity problems. A valid negative metric does not count as an
execution failure and does not stop the second registered scale.

## D-SCALE-002 — freeze the Full protocol

Keep factorized spatial/temporal SSMs, anatomical graph, topology-conditioned
Delta/B/C, batch 4, optimizer, LR, EMA, loss, augmentation and evaluator. Only
width/depth and the registered 80-epoch budget differ from the existing Full
reference. This makes the study a capacity comparison, not a new architecture.

## D-SCALE-003 — speed-first with a fail-closed memory gate

Keep compilation enabled and activation checkpointing disabled initially.
After A1/A2, both candidates must pass a compiled real-H36M batch-4 train step
with finite loss and peak reserved memory below 28 GiB. Any failure stops the
queue for review; batch size, LR and checkpointing must not change silently.
