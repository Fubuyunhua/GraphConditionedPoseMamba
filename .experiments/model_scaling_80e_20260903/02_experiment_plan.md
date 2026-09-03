# 80-epoch model-scaling plan

| ID | Track | Width | Depth | Order | Status |
|---|---|---:|---:|---:|---|
| S1-W128-D20 | IMPROVEMENT | 128 | 20 | 1 | PLANNED (6,836,355 params) |
| S2-W256-D10 | IMPROVEMENT | 256 | 10 | 2 | PLANNED (12,646,107 params) |

## Common controls

- Full graph-conditioned factorized architecture; only width/depth differ.
- H36M-SH xy+confidence T243/S81, root-relative, flip, seed 0.
- Batch 4, FP32 AdamW, LR 5e-4, WD 0.012, decay 0.99, warmup 8.
- Identical losses, EMA 0.9998, evaluator, selection and checkpoint rules.
- Budget: 80 epochs each.
- Primary metric: lowest EMA P1 in epochs 1-80.
- Secondary: P2 from the same EMA checkpoint, raw P1/P2, parameters, VRAM,
  throughput and wall time.
- KEEP threshold: P1 <= 39.7452 mm (at least 0.10 mm better than Full). A
  smaller gain is inconclusive at one seed; a worse result rejects that scale.

## Execution gates and stopping

1. A1 and A2 minimal ablations must complete 80 epochs, all four checkpoint
   families must load, strict re-evaluation must pass, and no execution error
   may remain.
2. Static protocol/shape/parameter audit must pass for both scale configs.
3. With at least 24,576 MiB initially free, run eager real-data batch 1 and 2
   steps, then an exact compiled batch-4 step. Every stage must produce finite
   loss with peak reserved VRAM below 24,576 MiB.
4. Run S1, verify and re-evaluate its raw/EMA best checkpoints, then run S2.
5. A valid negative S1 metric does not stop S2. OOM, NaN/Inf, CUDA failure,
   dataset mismatch, dirty source or invalid checkpoint stops the sequence.

The long non-factorized ablation is not a dependency because it studies a
different causal mechanism and would delay this accuracy-oriented scale study
by roughly eight days.
