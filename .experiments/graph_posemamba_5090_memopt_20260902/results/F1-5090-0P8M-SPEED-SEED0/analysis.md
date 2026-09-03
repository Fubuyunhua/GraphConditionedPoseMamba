# F1 RTX5090 speed-first seed-0 analysis

## Validity

- Completed 120/120 epochs from a fresh seed-0 initialization.
- Remote source commit: `b1d9364b3e1f9e0570b4ed37350c4ab11a57b963`.
- Configuration: FP32, W64/D8, 800,083 parameters, batch 4, T=243,
  H36M-SH, AdamW, EMA 0.9998, reduce-overhead training and eager evaluation.
- H36M data identity remains the frozen SHA-256
  `73b642f2567a8d0b194f88c54a3182c7b635c003c832b48ae6ee559f10232175`.
- The log contains 120 metric rows and no NaN/Inf, OOM, traceback or
  non-finite error.  Latest/best raw and EMA checkpoints load strictly.
- The inherited protocol evaluates the H36M test split every epoch; the best
  checkpoint is therefore test-monitored and is not an unbiased locked-test
  selection result.

## Primary result

The best EMA P1 occurs at epoch 53: **39.8452 mm**, with paired P2
**33.2322 mm**.  The independently lowest P2 is 33.1231 mm at epoch 89 and
must not replace the P2 paired with the P1-selected checkpoint.

Strict eager checkpoint re-evaluation:

| Checkpoint | Epoch | P1 | P2 |
|---|---:|---:|---:|
| raw at EMA-best time | 53 | 41.6185 | 34.2645 |
| EMA best | 53 | **39.8452** | **33.2322** |
| raw endpoint | 120 | 42.0001 | 33.9915 |
| EMA endpoint | 120 | 40.9894 | 33.3666 |

At epoch 65 the 5090 EMA result is 39.9584/33.1850 mm.  This is roughly
0.02 mm below the user's rounded RTX5060Ti epoch-65 report of 39.98 mm, but
the external run artifacts are not present here and the difference is too
small for a seed-level superiority claim.

## Dynamics and efficiency

P1 improves rapidly through epoch 53, then regresses while training loss keeps
falling from 0.010465 to 0.008504.  This is evidence of late overfitting or an
objective/evaluation mismatch, not optimizer divergence.  P2 continues to
improve until epoch 89 before also regressing.

Total wall time was 7.6867 hours.  While sharing the GPU, epochs required about
5.77 minutes of training; after the other workload ended, the final 60 epochs
averaged 2.877 minutes.  Training used about 3,734 MiB.  A saved EMA checkpoint
inferred in FP32 eager mode at batch 1 with 610 MiB reserved.

## Action breakdown at best EMA

The lowest P1 actions are WalkTwo 28.7772, Walk 28.4108 and Greet 33.7744 mm.
The hardest are SittingDown 55.4366, Sitting 50.6764 and Photo 48.6686 mm.
This indicates that remaining error is concentrated in seated/occluded-like
poses rather than locomotion.

## Claim evaluation and recommendation

- Complete, memory-safe RTX5090 reproduction: **SUPPORTED**.
- Accuracy preserved relative to the rounded RTX5060Ti epoch-65 snapshot:
  **SUPPORTED directionally**, but the margin is negligible.
- Better final endpoint than best checkpoint: **CONTRADICTED**.
- Multi-seed superiority: **UNRESOLVED**.

Recommendation: `KEEP` epoch-53 EMA as the 0.8M comparison baseline, retain
epoch-120 EMA as the fixed endpoint, and treat the late P1 regression as a
motivation for a validation-based selection/regularization study rather than
silently shortening the training budget.
