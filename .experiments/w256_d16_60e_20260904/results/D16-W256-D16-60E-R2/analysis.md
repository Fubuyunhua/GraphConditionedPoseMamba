# D16 R2 invalid-run analysis

## Validity

The registered source (`2e4b8040`), D16 configuration, H36M-SH dataset identity,
seed 0, evaluator, FP32 compiled runtime, and batch size remained fixed. R2 was
stopped after epoch 16 because the registered fail-closed condition was met. It
is therefore an `INVALID` partial run and must not be reported as a completed
60-epoch accuracy result.

The best raw/EMA checkpoints are epoch 15 and the latest raw/EMA checkpoints are
epoch 16. All tensors in all four checkpoint files loaded successfully and were
finite, so the event is not evidence of truncated or non-finite checkpoint
corruption.

## Primary result

Before invalidation, the best observed EMA result was **39.8398 mm P1** with
**32.8179 mm paired P2** at epoch 15. This does not satisfy the registered
requirement of a valid complete run at or below 39.7452 mm.

At epoch 16, train loss increased from 0.012514 to 0.027719 (+121.5%) and EMA P1
worsened from 39.8398 to 132.7245 mm (+92.8847 mm). The pre-clip gradient norm
remained finite at 1.5060 and was clipped to the configured maximum of 1.0.

## Diagnostic comparison with W128/D20

This comparison is diagnostic rather than a formal final ranking: W128 did not
use R2 warmup/clipping and was cancelled at epoch 65/80, while D16 is invalid at
epoch 16/60.

| Epoch | W128 P1/P2 | D16 R2 P1/P2 | D16 minus W128 P1/P2 |
|---:|---:|---:|---:|
| 13 | 39.4557 / 33.0084 | 40.5432 / 33.5557 | +1.0875 / +0.5473 |
| 14 | 39.2959 / 32.8150 | 40.2440 / 33.1251 | +0.9481 / +0.3101 |
| 15 | 39.0737 / 32.6902 | 39.8398 / 32.8179 | +0.7661 / +0.1277 |
| 16 | 38.7512 / 32.5491 | 132.7245 / 72.7681 | invalid collapse |

D16 was narrowing the same-epoch gap through epoch 15 but had not surpassed
W128. W128's best observed partial-run result was 37.6593/31.8717 mm at epoch
45, which is 2.1805/0.9462 mm better than D16's preserved pre-failure best.

D16 uses 20.19M versus 6.84M parameters (2.95x), reserves 21,070 versus 14,692
MiB (+43.4%), and before the overlap runs at about 3.198 versus 5.063 it/s. Its
steady epoch time is about 23.13 versus 14.61 minutes (+58.3%). No accuracy gain
was demonstrated for that added cost.

## Failure assessment

`wh` PID 691656 began at 15:50:17 UTC during epoch 16 and used about 3.62 GiB.
The overlap coincides with lower D16 throughput, but shared-GPU contention alone
does not establish the cause of the finite loss/metric excursion. Treat this as
a confounder requiring a controlled diagnostic, not as a causal conclusion.

The R2 stability claim is **CONTRADICTED** by the repeated excursion. The D16
accuracy-capacity claim remains **UNRESOLVED** because the run did not complete.

## Recommendation

`DIAGNOSE`: do not resume or automatically retry R2. The smallest valid next
step is a separately authorized, exclusive-GPU replay/diagnostic around the
epoch-15 state with optimizer-state and raw-versus-EMA movement logging. Any
new recipe must return through audit and staged preflight before a fresh run.
