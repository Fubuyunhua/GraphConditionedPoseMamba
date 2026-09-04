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
