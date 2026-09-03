# Scale GPU preflight attempt

- Time: 2026-09-03 20:49 Asia/Shanghai.
- Dependency: A1/A2 sequence completed and strict verification passed.
- GPU state: 3,634 MiB used, 28,476 MiB free, 0% utilization at the snapshot.
- Registered requirement: at least 28,672 MiB free before compiled real-data
  smoke.
- Outcome: BLOCKED_BEFORE_SMOKE by 196 MiB.
- Tests/smokes executed in this attempt: none.
- Formal scale training started: no.
- Changes made after failure: none; batch, LR, activation checkpointing and the
  memory threshold remain locked.
