# W256/D16 60-epoch plan

- ID/track: `D16-W256-D16-60E`, IMPROVEMENT.
- Model: width 256, depth 16, 20,192,451 parameters.
- Exact delta from W256/D10: depth 10 to 16 and user-requested budget 60.
- Controls: graph/factorization, ratios, d-state, residuals, data, batch,
  optimizer, LR, losses, EMA, augmentation and evaluator.
- Seed: 0.
- Primary: lowest EMA P1 within epochs 1-60.
- Secondary: same-checkpoint P2, raw P1/P2, VRAM, throughput and wall time.
- Acceptance: valid complete P1 <= 39.7452 mm.
- Stopping: OOM, NaN/Inf, CUDA/data/checkpoint/source error. Metric outcome
  does not cause an early stop.
- Dependency: staged real-data GPU gate only. The user explicitly replaced and
  cancelled the preceding incomplete W256/D10 run.

## D16-W256-D16-60E-R3 — IMPROVEMENT

- Hypothesis: lower and smoothly decayed optimizer pressure, together with the
  intended no-decay treatment for SSM state parameters, eliminates the R1/R2
  finite parameter excursions without reducing model capacity.
- Exact delta from R2: peak LR `5e-4 -> 3e-4`; eight-epoch linear warmup is
  followed by per-step cosine decay to `3e-5`; parameters explicitly marked
  `_no_weight_decay` receive zero AdamW decay. Global clip 1.0, EMA 0.9998,
  architecture, data, losses, batch 4, seed 0 and 60 epochs remain unchanged.
- Diagnostics: per-epoch maximum pre-clip gradient, clipped-step fraction, and
  raw parameter relative-L2/max-absolute update are logged. These diagnostics
  do not automatically terminate a finite run.
- Primary metric: minimum EMA P1 within epochs 1-60; P2 must come from the same
  checkpoint. Secondary: raw P1/P2, W128 same-epoch curve, update norm, gradient
  statistics, VRAM, throughput and wall time.
- Acceptance: a valid complete run at or below `39.7452 mm`; accuracy target is
  to beat the W128 partial-run best `37.6593 mm`.
- Falsification: another finite loss/P1 collapse, non-finite state, or a valid
  complete P1 above `39.8452 mm`.
- Gate: unit/static checks, real-data eager B1/B2 and compiled B4, correct LR
  endpoints, correct AdamW groups, finite loss/gradient/update telemetry,
  dataset hash match and safe shared-GPU memory. Runtime errors or non-finite
  values block launch. Finite metric movement is reported to the user and does
  not authorize automatic stopping.
