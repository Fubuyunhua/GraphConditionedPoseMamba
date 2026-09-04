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
