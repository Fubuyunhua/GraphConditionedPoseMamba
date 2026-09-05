# Pre-registered experiment plan

## Structural ablations

### E-RWG-0 — Full with Rewired Graph

- Track: ABLATION; seed 0; graph seed 3407; 80 epochs.
- Delta: anatomical graph to fixed degree-preserving rewired graph only.
- Controls: Full parameters, K=2, independent recurrence, injection, data,
  losses, optimizer, EMA, compilation and evaluator.
- Primary: best EMA P1 in epochs 1-80; secondary: paired P2, fixed@80,
  per-action/joint metrics, memory and throughput.
- Falsification: rewired equals or improves on Full.

### E-NR-0 — Full w/o Recurrence Reset — matched

- Track: ABLATION; seed 0; 80 epochs.
- Delta: recurrence scope `independent -> joined` at the scan boundary only.
- Controls: every parameter and all local projections/convolutions fixed.
- Primary/secondary and interpretation use the same rules as E-RWG-0.
- Falsification: joined equals or improves on Full.

## Implementation diagnostic

### D-PM-BWD-0 — corrected PoseMamba backward

- Track: DIAGNOSTIC; seed 0; 80 epochs; W64/D6/M1.
- Delta: add the missing indexed-path transpose gradient only.
- Acceptance: bit-exact forward, FP64 gradcheck, reference gradient, legacy
  checkpoint inference and real training step all pass.
- Result is separate from released A0 and is never averaged with it.

## Repeats

- R-A0: released PoseMamba seeds 1/2.
- R-A2: Graph Feature Fusion seeds 1/2.
- R-FULL: Full seeds 1/2.
- Report each seed plus paired `A0-Full` and `A2-Full` differences, mean and
  standard deviation. Seed 2 remains behind seed 1 in the execution order.

## MPI-INF-3DHP

- M-A0: predeclared released/legacy PoseMamba W64/D6/M1 seed 0.
- M-FULL: GraphConditionedPoseMamba W64/D8 seed 0.
- Same repaired T81 protocol, joint mapping, data, loss, optimizer and fixed
  epoch-120 selection. This is a second-dataset validation, not zero-shot
  H36M-to-MPI transfer.

All runs remain `PLANNED` until the user selects the next single long run.

