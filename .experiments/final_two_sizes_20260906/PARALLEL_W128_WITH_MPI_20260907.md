# User-authorized W128 concurrency with MPI Full

The user explicitly requested immediate parallel execution because MPI Full
underutilizes the RTX5090 when running alone. This moves
`FINAL-W128-DET2D-S0` ahead of the previous MPI-completion dependency and
authorizes exactly the pair MPI Full plus W128 Detector. It does not authorize
W256 as a third simultaneous job.

The W128 scientific recipe remains unchanged: W128/D20, 6,836,355 parameters,
DropPath0.25, seed0, batch4, FP32, 80 epochs, AdamW peak LR3e-4 with eight real
warmup epochs, cosine floor3e-5, WD0.012 with SSM exclusions, clip1.0 and
EMA0.9998. H36M detector input and per-epoch monitored-best EMA P1/P2 remain
the registered protocol.

Fresh concurrent B1/B2/B4 real-data gates passed. B4 compiled peak reserved
memory was14,228 MiB and all losses/gradients were finite. MPI used1,570 MiB,
leaving sufficient 32GB capacity. The formal W128 process began at
2026-09-07T10:52:11+08:00 from a fresh initialization. Shared throughput is
engineering telemetry only and must not be used for a publication efficiency
claim. Neither run may be stopped, retuned or restarted merely because the
other slows it.

After one member completes, preserve and audit it while the other continues.
Do not start W256 until the active pair no longer creates an unsafe three-job
combination and its own launch gate passes.
