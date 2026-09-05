# Pre-long-training check report

## 1. Workspace

- Requested branch: `codex/minimal-ablation-80e-20260903`.
- Prompt HEAD: `3fac1a4`; observed HEAD: `a931525`.
- Delta: one R3 training-log synchronization commit only; no model/config drift.
- Implementation worktree: `D:\gpu5090\GraphConditionedPoseMamba-paper-evidence`.
- Local implementation commit: `ad4e2fa`; not pushed.
- Active remote D16 R3 remains untouched and running.

## 2. Experimental deltas

- Rewired: graph edges only; Full model/training path frozen.
- Matched no-reset: state boundary only; local Conv1D/projections/K=2 frozen.
- Corrected A0: indexed backward only; released forward/W64/D6/M1 frozen.
- Seed repeats: seed and output path only.
- MPI: released A0 versus Full under one repaired T81 protocol.

## 3. Fixed rewired graph

- Bone edges: 16; degrees match anatomical; connected; overlap 12.5%.
- Symmetry edges: 6; degrees match anatomical; overlap 0%.
- No self-loops or duplicates; all eight layers share the graph.
- Seed: 3407; SHA256:
  `f9037c7265d94ba73c5941fc3070dec76cd022e8c302d141543e94c85627efad`.
- Exact edges are stored in `graph_spec.json`.

## 4. Matched no-reset mapping

At T243/J17, spatial and temporal joined scan lengths are both 4,131. Local
projection shapes and token values match Full before joining. Direction zero
uses segment order; direction one reverses both tokens and segment order. The
inverse, single-segment equality and cross-batch isolation tests pass.

## 5. Original Full compatibility

- Parameter names/shapes/count: unchanged at 800,083.
- Prediction/loss versus untouched source: max absolute difference 0.
- Input-gradient difference: `3.58e-7`; parameter-gradient difference:
  `9.54e-7`, below untouched CUDA repeat noise (`1.91e-6`).
- Epoch-53 EMA checkpoint strictly loads.
- Full evaluator replay: 39.8450067/33.2318704 mm, within 0.0004 mm of the
  registered 39.8451617/33.2322404.

## 6. Corrected scan

- Historical A0 source was inspected and contains the missing indexed-path
  accumulation.
- Legacy/exact forward is bit-exact; FP64 gradcheck and random-upstream
  reference gradient pass.
- Both modes strictly load the epoch-60 A0 EMA checkpoint.
- Exact-mode evaluator replay is exactly 40.2260220/33.5176263 mm.

## 7. Config activation and parameters

- Rewired execution path: control/degree-preserving-rewired/K2/independent,
  800,083 params.
- No-reset path: control/anatomical/K2/joined, 800,083 params.
- Corrected A0: PoseMamba/K4/exact, 790,083 params.
- Every new config was resolved through `_base_` and instantiated on RTX5090.

## 8. Actual 80-epoch protocol

- Batch 4; 4,437 steps/epoch; 354,960 optimizer and EMA updates.
- AdamW LR 5e-4, WD 0.012, epoch decay 0.99.
- `warmup_epochs: 8` is historical metadata; effective warmup is disabled.
- Epoch-80 train LR is 0.0002260218.
- Report test-monitored best EMA and fixed epoch-80 EMA separately.

## 9. Commands

Unit suite:

```bash
python -m unittest tests.test_graph_conditioned_posemamba
```

Static/provenance preflight:

```bash
python tools/preflight_paper_remaining.py \
  --full-checkpoint /path/to/full/best_ema_epoch.bin \
  --posemamba-checkpoint /path/to/a0/best_ema_epoch.bin
```

Single long runs after confirmation only:

```bash
python train.py --config configs/pose3d/ablation_full_rewired_graph.yaml \
  --checkpoint runs/paper_remaining_evidence/full_rewired_graph_seed0 --seed 0
python train.py --config configs/pose3d/ablation_full_no_recurrence_reset_matched.yaml \
  --checkpoint runs/paper_remaining_evidence/full_no_recurrence_reset_matched_seed0 --seed 0
```

## 10. Gate summary

- PASS: 47 tests, graph constraints/hash, parameter matching, local pre-scan
  equality, direction mapping, batch isolation, corrected gradient, checkpoint
  loads, H36M real steps, MPI protocols and MPI CUDA smokes.
- PASS with disclosed tolerance: Full metric replay and CUDA gradients.
- SKIPPED: exclusive-GPU timing/FLOPs benchmark.
- NOT_RUN: every long training run.
- BLOCKED: historical A0/Full `ema_fixed_epoch80` weights were not saved.

## 11. Cost

The shared-GPU B1 smoke timings are not valid cost estimates. Historical W64
80-epoch A1/A2 runs took 4:18 and 4:54 respectively, but these are context, not
predictions for joined recurrence or MPI. An exclusive-GPU short benchmark is
required before quoting new expected hours.

Long training remains blocked on explicit user selection of one next run.
