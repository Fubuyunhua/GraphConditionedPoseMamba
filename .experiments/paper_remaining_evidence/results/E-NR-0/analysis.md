# E-NR-0 final analysis

## Verdict

`KEEP` as a valid completed single-seed recurrence-boundary ablation. The
parameter-matched joined recurrence is worse than Full, supporting independent
state reset at frame and joint-trajectory boundaries for seed 0. This is not a
multi-seed significance claim.

## Provenance and comparability

- Scientific implementation: `ad4e2fa66492737a3fcd88d9142a88731724f30b`.
- Deployment source: `e537128ffde548b8d2232d4b49165d5095868527`.
- Config SHA256: `86d9d2c3188b1275b1fe9c624984e5402939c485d6ee4faf91ffc67670741af6`.
- H36M dataset SHA256: `73b642f2567a8d0b194f88c54a3182c7b635c003c832b48ae6ee559f10232175`.
- 800,083 parameters, K=2, anatomical graph, joined scan length 4,131,
  seed 0, batch 4, 80/80 epochs and 354,960 optimizer/EMA updates.
- AdamW LR `5e-4`, WD `0.012`, legacy epoch decay `0.99`; no warmup or clipping.
- Natural exit code 0, finite checkpoints, and strict replay of best/fixed raw/EMA.

## Results

| Checkpoint | Epoch | P1 | Paired P2 |
|---|---:|---:|---:|
| Best EMA, test-monitored | 50 | 40.568980 | 33.802451 |
| Raw at selected best epoch | 50 | 41.922773 | 35.218836 |
| Fixed epoch-80 EMA | 80 | 40.909880 | 33.821656 |
| Fixed epoch-80 raw | 80 | 41.784616 | 33.701350 |

The registered Full seed-0 best is `39.845162/33.232240 mm`. Joined no-reset is
worse by `+0.723818 mm` P1 and `+0.570211 mm` paired P2 under the same
test-monitored-best rule. Historical Full lacks a fixed epoch-80 checkpoint, so
no fixed-80 causal comparison is claimed.

## Dynamics and interpretation

- Mean training time: 2.179 minutes/epoch; sustained throughput about 34.0 it/s.
- Peak trainer-reserved VRAM: 3,034 MiB; final loss: 0.009236.
- No NaN, Inf, OOM, CUDA error or traceback occurred.
- Supported for seed 0: keeping recurrence state across unrelated segments
  harms accuracy even when projections, local Conv1D, K, graph and parameter
  count remain fixed.
- Boundary: this isolates recurrence scope under the tested ordering; it does
  not prove that every possible cross-segment state connection is harmful.
