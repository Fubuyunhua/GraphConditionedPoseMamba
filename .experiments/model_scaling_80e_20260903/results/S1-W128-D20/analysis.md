# S1 W128/D20 partial analysis

Status: USER_CANCELLED_PARTIAL_NOT_COMPARABLE.

S1 completed 65 of the registered 80 epochs before the user explicitly
prioritized W256/D10. It stopped without runtime failure; its run directory,
logs and all latest/best raw/EMA checkpoints remain intact.

The best training-time EMA observation was epoch 45: P1 37.659301 mm and
same-checkpoint P2 31.871747 mm. The latest epoch-65 EMA observation was
37.846863/31.878519 mm. Trainer peak reserved VRAM was 14,692 MiB and stable
throughput was approximately 5.006 it/s.

These numbers are promising but are not the registered paper result: the run
did not complete 80 epochs and strict post-run raw/EMA replay was not executed.
Recommendation: KEEP the partial checkpoint and evidence; do not compare it as
a completed scale experiment or silently resume it.
