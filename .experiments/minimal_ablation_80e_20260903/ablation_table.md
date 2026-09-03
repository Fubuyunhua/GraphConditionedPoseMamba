# Minimal ablation table

All rows use the best EMA P1 within the registered budget and the P2 from the same checkpoint.

| Variant | Factorized | Graph | Feature Fusion | Topology-conditioned Dynamics | Params | P1 | P2 | Status |
|---|---|---|---|---|---:|---:|---:|---|
| A0 PoseMamba | — | — | — | — | 790083 | 40.2260 | 33.5176 | COMPLETED_EXISTING |
| A1 Factorized Only | ✓ | — | — | — | 749891 | — | — | RUNNING |
| A2 Graph Feature Fusion | ✓ | ✓ | ✓ | — | 800083 | — | — | QUEUED |
| A3 Full | ✓ | ✓ | — | ✓ | 800083 | 39.8452 | 33.2322 | COMPLETED_EXISTING |
