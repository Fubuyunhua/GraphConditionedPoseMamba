# RTX 5090 memory optimization experiment plan

## Common controls

- Source baseline: upstream commit `14216ccc8699839c3a472e32ef95071464732975`.
- Model/data/loss: W64/D8 0.8M, same H36M-SH file and frozen 120-epoch protocol.
- Hardware: RTX 5090 while preserving the existing user's experiment.
- Primary diagnostic metric: per-process peak allocated CUDA MiB.
- Secondary: peak reserved MiB, milliseconds/step, finite loss, output and gradient deltas.
- Formal accuracy metric after selection: EMA P1 MPJPE; P2 is paired secondary.

## Experiments

### M0 — upstream reduce-overhead diagnostic

- Track: DIAGNOSTIC.
- Delta: none.
- Budget: short synthetic and real-data train steps only.
- Purpose: measure the immutable execution baseline; never start a formal run.

### M1 — eager

- Track: DIAGNOSTIC.
- Delta: disable compile only.
- Acceptance: finite batch-4 steps and lower peak than M0.

### M2 — default compile plus eager evaluation

- Track: IMPROVEMENT.
- Delta: compile mode `default`; use unwrapped eager module for evaluation;
  clear gradients before forward.
- Acceptance: at least 20% lower peak reserved memory than M0, no formula or
  protocol change, and throughput no worse than eager.

### M3 — M2 plus activation checkpoint

- Track: IMPROVEMENT.
- Delta: non-reentrant checkpoint for each graph-conditioned block.
- Acceptance: output/gradient equivalence within FP32 reduction tolerance,
  lower peak than M2, and throughput at least 2.0 it/s.

## Selection and stopping rule

Select the fastest candidate whose peak reserved memory leaves at least 2 GiB
of observed RTX 5090 headroom beside the concurrent process.  Prefer M2 over M3
when M2 already satisfies the headroom gate.  Do not trade batch size, T,
precision, loss or model capacity for memory.  After two finite real-data smoke
steps and checkpoint-load validation, launch one seed-0 120-epoch run in a new
directory.  Preserve all failed diagnostics.
