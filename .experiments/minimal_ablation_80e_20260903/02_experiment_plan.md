# Minimal 80-epoch ablation plan

## Frozen table

| ID | Factorized | Graph | Feature fusion | Topology-conditioned Delta/B/C | Status |
|---|---|---|---|---|---|
| A0 PoseMamba | no | no | no | no | existing |
| A1 Factorized Only | yes | no | no | no | train 80 epochs |
| A2 Graph Feature Fusion | yes | yes | yes | no | train 80 epochs after A1 |
| A3 Full | yes | yes | no | yes | existing |
| Graph-Conditioned SSM w/o Factorization | no | yes | no | yes | implemented; awaiting preflight approval |

## Common controls

- H36M-SH xy+confidence, data hash
  `73b642f2567a8d0b194f88c54a3182c7b635c003c832b48ae6ee559f10232175`.
- T243/S81, root-relative, flip augmentation, seed 0.
- W64/D8, MLP ratio 1.96875, SSM ratio 1.875, d-state 16.
- AdamW, LR 5e-4, WD 0.012, decay 0.99, warmup 8, batch 4, EMA
  0.9998 and identical active losses.
- Budget: 80 epochs for A1, A2, and Graph-Conditioned SSM w/o Factorization.
  This explicit user override supersedes the pasted draft's 60-epoch text.
- Primary metric: lowest EMA P1 within epochs 1-80.
- Paired secondary metric: P2 from that exact P1-selected EMA checkpoint.
- Raw P1/P2 are strictly re-evaluated from the raw checkpoint saved at the
  same epoch and recorded separately.

## A1 — Factorized Only

- Track: ABLATION.
- Delta from A3: `graph_injection_mode=none`, `use_graph_mixer=false`,
  `graph_conditioned_ssm=false`; no capacity compensation.
- Hypothesis: correct state boundaries improve the coupled PoseMamba baseline.
- Falsification: A1 does not improve over A0 within the registered protocol.

## A2 — Graph Feature Fusion

- Track: ABLATION.
- Delta from A3: `graph_injection_mode=feature` and
  `graph_conditioned_ssm=false`; graph, symmetry and every other setting remain.
- Hypothesis: direct graph replacement of recurrent content is worse than
  topology-only selective control.
- Falsification: A2 equals or improves on A3.

## Graph-Conditioned SSM w/o Factorization

- Track: ABLATION.
- Only scientific delta from A3: `factorized_spatial_temporal=false`.
- The block, anatomical graph, topology-conditioned Delta/B/C, residuals, head,
  data, optimizer, EMA, augmentation and losses remain frozen.
- Both spatial and temporal branches reuse the existing coupled `BiSTSSM` with
  `forward_type=v2_plus_poselimbs`, `conv_mode=2d`, and full `[B,T,J,C]`
  inputs. No frame or joint boundary reset is applied.
- Hypothesis: boundary-preserving factorization improves accuracy over coupled
  PoseMamba-style state propagation under the same graph-conditioned control.
- Falsification: this row equals or improves on A3 under the registered metric.
- Dependency: A1/A2 queue completion, regression preflight, and explicit user
  confirmation. It is intentionally absent from the active A1-to-A2 queue.

## Execution and stopping

A1 runs first.  A2 starts only after A1 exits successfully, all four A1
checkpoint families load, latest epoch equals 80, and raw/EMA best checkpoints
are strictly re-evaluated.  Metric outcome does not stop A2; only correctness,
runtime, data, checkpoint or non-finite failure stops the queue.  Failed runs
remain archived and are never overwritten or resumed automatically.

The new non-factorized row is a separate guarded run. It cannot launch unless
A1/A2 have completed, its dedicated prelaunch record exists, and the explicit
approval marker has been created after user review.
