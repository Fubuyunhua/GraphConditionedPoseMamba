# Registered serial expansion

Preserve current corrected A0 -> MPI pair -> six paired H36M repeats. Then S-GT2D80e -> M-DET2D60e -> M-GT2D60e -> L-DET2D60e -> L-GT2D60e -> existing ACC-D16-DP030-S0 detector60e -> XL-GT2D60e. All seed0, separate fresh outputs and per-run gate. Existing XL detector registration is the only XL detector run; do not duplicate it here.

Primary endpoint for each new run is fixed last-epoch EMA P1 with paired P2, using explicitly named detected or GT2D evaluator. No per-epoch test evaluation. New checkpoints save every10 epochs; evaluate only fixed last epoch. Existing S detector is a historical test-monitored reference, not a matching fixed80 comparator. Do not select the better input protocol or best raw/EMA after seeing results.

M/L/XL share batch4,T243,FP32,lr3e-4 with8-epoch real warmup then cosine3e-5,WD0.012 with SSM exclusions,clip1.0,EMA0.9998,DropPath0.30. S-GT uses the legacy small-model80e recipe (lr5e-4,epoch decay0.99,no warmup,DropPath0.20). Do not claim the whole family differs only in parameter count.

Budget: six newly registered runs plus the already-registered XL detector, total440 epochs; CUDA hours must be estimated from each model's real preflight. Do not extrapolate W64 speed to larger models. Follow ledger for falsification/stopping rules; no automatic grid search. After this bounded family, report accuracy versus parameters, input source and training cost, select only with disclosure of test exposure, and stop for user review rather than expanding indefinitely.
