# S-GT2D-S0 final analysis

## Verdict

`KEEP` as a valid completed small GT2D depth-lifting result. It is not a
detected-input accuracy result and cannot be compared numerically with the
39.845 mm detector-input Full baseline as if only model quality changed.

## Provenance and validity

- W64/D8 GraphConditionedPoseMamba, 800,083 parameters, seed0, 80/80 epochs.
- H36M label-image-xy plus confidence1 input, known input xy preserved by the
  evaluator; 3D target and flip/root conventions passed the registered audit.
- Legacy small-model AdamW recipe: LR5e-4, WD0.012, epoch decay0.99,
  EMA0.9998, DropPath0.20, no warmup or clipping.
- Natural exit code0; 354,960 optimizer/EMA updates; no numerical/runtime error.
- All best/fixed raw and EMA checkpoints are finite and strictly replayed.

## Results

| Checkpoint | Epoch | P1 | Paired P2 |
|---|---:|---:|---:|
| Best EMA, test-monitored | 78 | 16.816311 | 13.979448 |
| Raw at selected best epoch | 78 | 17.594518 | 14.284725 |
| Fixed epoch-80 EMA | 80 | 16.964229 | 14.020911 |
| Fixed epoch-80 raw | 80 | 18.658487 | 14.890873 |

Mean train-only time was5.145 minutes/epoch under user-authorized MPI
concurrency; trainer peak reserved VRAM was3,034 MiB. Final loss was0.004548.
The primary result is test-monitored and single-seed; fixed80 is reported
separately.
