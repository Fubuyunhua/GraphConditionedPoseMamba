# Early optimizer-instability diagnosis

## Evidence

- Epoch 2: loss 0.034980, EMA P1/P2 117.488052/77.392961 mm.
- Epoch 3: loss 0.039396, EMA P1/P2 221.730884/127.199450 mm.
- Loss increased 12.62%; P1/P2 regressed 104.24/49.81 mm.
- All model and AdamW state tensors are finite; max parameter magnitude and
  global parameter norm decreased rather than exploded.
- Epoch2-to-3 EMA relative L2 change is 3.16%; raw-to-EMA relative L2 is 5.14%.
- Several deep residual gammas moved from 1.0 toward 0.86-0.91 by epoch 3.
- Test loader is deterministic and unshuffled.

## Root finding

`warmup_epochs: 8` was a dead field: the trainer created AdamW at 5e-4 from
the first batch and only applied 0.99 epoch decay. No gradient clipping or norm
telemetry existed. This is a credible mechanism for finite early overshoot in
the 20.19M, 16-block model. A paired corrected run is required for causal
confirmation.

## Minimal bounded modification

- Opt-in `enable_linear_warmup: true`; legacy configs default to old behavior.
- Per-step LR from 0.1x to 1.0x across 8x4437=35,496 steps.
- Existing 0.99 epoch decay begins after warmup.
- Log pre-clip gradient norm; clip global norm at 1.0 and fail on non-finite.
- New run prefix; no resume or overwrite of the invalid run.

## R2 late-instability evidence

R2 improved monotonically through epoch 15, reaching 39.839843/32.817857 mm,
then failed at epoch 16: MPJPE loss rose 121.5%, total loss rose 84.7%, and EMA
P1 regressed 92.8847 mm. All four raw/EMA checkpoints load and all tensors are
finite. LR followed the intended schedule and was 4.66e-4 at epoch 16.

The event is a whole-model optimizer excursion rather than an evaluation-only
artifact. Raw parameters moved 8.72% in relative L2 during epoch 16 and EMA
parameters moved 4.92%. Graph-mixer parameters moved 28.7%, SSM dt-projection
weights 19.7%, temporal positions 34.8%, and the head 12.6%; individual SSM/MLP
matrices moved 30-73%. A-log exponent and dt-bias ranges remained bounded.

The reported gradient norm was only an epoch mean. Global clipping precedes
AdamW and does not directly bound its per-parameter normalized update. In
addition, `A_logs` and `Ds` are marked `_no_weight_decay`, but the trainer's
single AdamW parameter group ignored those markers. Eight-epoch warmup reduced
the start shock but retained the original 5e-4 peak and shallow 0.99 decay.

The `wh` process overlapped part of epoch 16 and explains the throughput drop,
but separate CUDA contexts cannot directly alter this model's weights and no
OOM/CUDA error occurred. Treat concurrency as a confounder, not the root cause.

## R3 bounded modification

Keep the architecture, loss, batch, EMA and clipping fixed. Lower peak LR to
3e-4, replace post-warmup epoch steps with per-step cosine decay to 3e-5, honor
only existing `_no_weight_decay` markers, and log maximum gradient, clipping
fraction, and parameter movement. This directly tests the optimizer-scale
hypothesis without adding a new architectural mechanism.
