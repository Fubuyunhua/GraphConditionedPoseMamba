# User-authorized parallel GT launch

2026-09-10 user specifies W128/D10 after requesting immediate parallel launch.
Context is current GT2D; launch BALANCED-GT-S0 only, not detector variant or SAMA.
Keep existing W128/D20 PID2004698 untouched. D10 batch8,80epochs,fresh seed0,
3,435,395parameters,LR5e-4 exponential.99,DP.20,WD.012,clip1,EMA.99960004.
No pretrained/resume. Exact random initialization fingerprint verified by trainer.

Conditional gate: require only identified D20 on GPU and>=16000MiB free; run
real B1/B2/B4/B8 steps,GTxy+confidence assertions,optimizer/reset,EMA checkpoint
roundtrip. B8 measured peak<15800MiB. Failure stops this launch only; no batch
reduction, retry or interruption of D20. Shared timing cannot establish efficiency.
Parallel work necessarily shares compute and may slow D20; cannot promise faster
combined completion. Existing SAMA pipeline still waits for all GPU tasks to end.

This is not a clean depth-only ablation due batch8 vs4 and sample-matched EMA.
Primary best EMA GT P1 over80epochs with same-checkpoint P2; fixed80 secondary.
Single seed/test-monitored result, no target accuracy promise. Stop80/failure/user.
New source and run directory, full config and gate artifacts retained.
