# Model-scaling results

P2 is always paired with the EMA checkpoint selected by minimum P1 within 80 epochs.

| Variant | Width | Depth | Params | Best epoch | EMA P1 | Paired P2 | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| Full W64/D8 | 64 | 8 | 800083 | 53 | 39.8452 | 33.2322 | COMPLETED_EXISTING |
| S1 W128/D20 | 128 | 20 | 6836355 | — | — | — | RUNNING |
| S2 W256/D10 | 256 | 10 | 12646107 | — | — | — | READY |
