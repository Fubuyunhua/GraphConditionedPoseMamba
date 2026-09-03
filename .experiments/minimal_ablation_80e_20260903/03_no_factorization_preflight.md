# Graph-Conditioned SSM w/o Factorization — preflight

Status: PASS; AWAITING_USER_CONFIRMATION; LONG_TRAINING_NOT_AUTHORIZED

## Registered delta

The only scientific delta from A3 Full is
`factorized_spatial_temporal: false`. The training budget is 80 epochs per the
user's explicit override. The coupled branches use the existing `BiSTSSM`,
`forward_type=v2_plus_poselimbs`, `conv_mode=2d`, spatial `d_conv=1`, and
temporal `d_conv=3`.

## RTX5090 results

- Python and shell syntax: PASS.
- Unit/CUDA suite: 25/25 PASS.
- Old Full epoch-53 EMA fixture: max/mean absolute error `0.0/0.0`.
- Full parameters: `800,083`.
- Non-factorized ablation parameters: `1,028,563` (not capacity compensated).
- Coupled spatial/temporal SSM input: `[1,243,17,64]` for the FP32 fixture.
- Graph feature, spatial context, and temporal context:
  `[1,243,17,64]`.
- Content-control probe: `u` unchanged; Delta, B, and C all changed when only
  the graph-conditioned context changed.
- Real H36M compiled train-step smoke (`batch_size=4`): PASS with finite loss
  `1.5445`, peak reserved VRAM `3,766 MiB`, and `0.515 it/s` while A1 was also
  using the GPU. The first compile/warmup took `57.42 s`.
- The training split has 17,748 samples (4,437 batch-4 steps/epoch). The
  one-step smoke implies roughly 2.4 hours/epoch or about 8 days for 80 epochs
  before evaluation overhead. This is a provisional lower-confidence estimate
  because it was measured concurrently with A1 and from only one timed step.

## Required gates

1. Full checkpoint output compatibility: max absolute error <= 1e-5.
2. Coupled spatial and temporal SSM inputs remain `[B,T,J,C]`.
3. Spatial/temporal graph contexts remain `[B,T,J,C]`.
4. Debug tensors prove recurrent `u` is invariant when only context changes,
   while Delta/B/C change.
5. Real-data compiled forward/backward/optimizer/EMA smoke passes.
6. A1/A2 sequence completes and final artifacts are verified.
7. User confirms the 13-item inspection report.

The guarded launcher refuses to run unless both `A4_PRELAUNCH_PASS.json` and
`A4_APPROVED` exist. The prelaunch record now exists; the approval marker does
not, so long training cannot start accidentally.
