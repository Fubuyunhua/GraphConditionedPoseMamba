# Minimal A1/A2 model audit

## Scope

This study adds only the registered `graph_injection_mode` interface and two
training configurations.  It does not change the selective-scan CUDA kernel,
Delta/A/B/C/D equations, K=2 bidirectional scan, state-reset boundaries,
SkeletonGraphMixer edges, losses, data or optimizer.

The user explicitly overrides the attached 60-epoch proposal with an 80-epoch
budget for A1 and A2.  Existing A0 and A3 best checkpoints occur at epochs 60
and 53, respectively, so both are inside the same first-80 selection window.

## Exact data flow

For normalized pose feature `X` and `G=SkeletonGraphMixer(X)`:

- `none`: recurrent content `u=encode(X)`; selective parameters are projected
  from `encode(X)`; no graph tensor is computed or supplied.
- `feature`: `Xg=X+graph_scale*G`; recurrent content and Delta/B/C input both
  come from `Xg`; independent context is `None`.
- `control`: recurrent content comes from `X`; Delta/B/C input comes from
  `X+graph_scale*G`.  This is the frozen A3 Full definition.

Spatial and temporal stages use the same routing rule.  When
`reuse_graph_context=true`, A2 and A3 reuse the identical spatial graph feature
at the temporal stage; therefore their sole routing difference is whether the
graph-enhanced tensor replaces recurrent content.

## Tensor and optimization invariants

- Input/output: `[B,243,17,3]`.
- Hidden: `[B,243,17,64]`.
- Spatial scan: `[B*243,1,17,64]`, reset per frame, conv 1.
- Temporal scan: `[B*17,1,243,64]`, reset per joint, conv 3.
- AdamW, LR 5e-4, WD 0.012, decay 0.99, batch 4, warmup 8, EMA
  0.9998 and all active losses remain frozen.
- FP32 reduce-overhead training and eager evaluation remain the selected
  execution policy.

## Losses

- MPJPE: 1.0.
- scale-normalized MPJPE: 0.5.
- 3D velocity: 20.0.
- prediction-difference regularizer: 0.5.
- All remaining configured losses are zero and short-circuited.

## Risks

- P1: explicit `control` must reproduce the pre-change A3 output at max
  absolute error <=1e-5 using the frozen epoch-53 EMA checkpoint.
- P1: A1 removes graph parameters without capacity compensation; interpret it
  as component removal, not an exact-capacity control.
- P1: per-epoch H36M test monitoring makes best-checkpoint selection
  test-monitored rather than unbiased.
- P2: seed 0 only; conclusions are minimal paper ablation evidence, not
  multi-seed stability.
- P2: A2 must never pass an independent graph context to the SSM.

## Verdict

`CONDITIONAL` until unit tests, CUDA/real-data smoke, parameter/gradient audit
and the pre-change A3 compatibility fixture all pass on RTX5090.  Only then may
the serial A1-to-A2 queue start.
