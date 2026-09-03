# Minimal ablation table

All rows use the best EMA P1 within the registered budget and the P2 from the same checkpoint.

| Variant | Factorized | Graph | Feature Fusion | Topology-conditioned Dynamics | Params | P1 | P2 | Status |
|---|---|---|---|---|---:|---:|---:|---|
| A0 PoseMamba | — | — | — | — | 790083 | 40.2260 | 33.5176 | COMPLETED_EXISTING |
| A1 Factorized Only | ✓ | — | — | — | 749891 | 40.0605 | 33.3565 | COMPLETED |
| A2 Graph Feature Fusion | ✓ | ✓ | ✓ | — | 800083 | 40.0588 | 33.2873 | COMPLETED |
| A3 Full | ✓ | ✓ | — | ✓ | 800083 | 39.8452 | 33.2322 | COMPLETED_EXISTING |
| Graph-Conditioned SSM w/o Factorization | — | ✓ | — | ✓ | 1028563 | — | — | AWAITING_USER_CONFIRMATION |
