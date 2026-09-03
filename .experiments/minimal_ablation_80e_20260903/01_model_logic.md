# Minimal ablation model logic

## Failure mode and claims

Original PoseMamba couples the T-by-J plane in one scan, permitting state
transitions across frame/joint boundaries that are not physical skeleton
edges.  A1 tests whether explicit per-frame spatial and per-joint temporal
state reset is useful by itself.

A2 tests the conventional alternative to Full: fuse graph messages directly
into the recurrent pose feature.  A3 instead keeps pose content and topology
control separate.  A2 versus A3 therefore isolates content-control decoupling.

## Novel and conventional elements

- Boundary-preserving factorized BiSSM and topology-conditioned selective
  dynamics are the claimed model contributions.
- Direct residual graph feature fusion is a conventional control.
- The new mode switch is experimental infrastructure, not a contribution.

## Observable predictions

- If A1 improves over A0, boundary-preserving factorization has useful
  inductive bias at this scale.
- If A3 improves over A2, topology is more useful as selective-dynamics control
  than as recurrent-content replacement.
- If A1 or A2 is best, the paper claim must change accordingly; no result may
  be tuned to force A3 to win.
