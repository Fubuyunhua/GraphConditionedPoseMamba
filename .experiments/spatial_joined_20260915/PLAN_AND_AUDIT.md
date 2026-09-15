# Spatial-only recurrence-boundary ablation

User confirmed: retain spatial SSM, remove per-frame independence ONLY in spatial
scan; temporal trajectories remain independent. Do not remove spatial branch.

## Model logic
Baseline spatial scan: B*T sequences of length17. Treatment: after identical
local projection/Conv1D, join T frame segments into B sequences of lengthT*17,
with correctly reversed complete backward direction. Temporal branch unchanged:
B*17 sequences of lengthT. U/Z derive from pose features; graph-enhanced context
controls Delta/B/C. No SSI/MSM, new parameters, loss or detector change.

Hypothesis: resetting spatial recurrence at frame boundaries affects P1 and
runtime. Falsification: joined is as accurate/better; do not claim necessity then.
Earlier joined study changes BOTH axes, so is not this single-factor experiment.

## Registered experiment SJOIN-W64D8-S0 (ABLATION)
Baseline: historical Full800083 params,seed0,EMA39.845162/33.232240; matched80e
recipe from conditioning_full_reference_80e.yaml. Treatment is ONLY spatial scope
independent->joined. Temporal scope=independent. Batch4,T243,SH detector/conf,
AdamW5e-4,decay.99,WD.012,EMA.9998,DP.2,no warmup/no clip,original losses.
From random seed0, no pretrained/finetune/resume.80epochs. Primary bestEMA P1;
pairedP2 and fixed80,latency/memory secondary. Single-seed/test-monitored limits.
No success promised; observed differences are not statistical significance.
Stop at80epochs, technical failure or user request, NOT accuracy-based early stop.

## Audit gates
PASS required: same800083params, identical seed0 parameter/buffer fingerprints;
strict state loading; unchanged default forward/gradients vs frozen old fixture;
isolated spatial perturbation crosses frames only when joined, temporal remains
independent; B1/B2/B4 real-data train steps finite, graph gradients nonzero,
EMA roundtrip; B4 torch.compile smoke; datasetSHA73b642f2567a8d0b194f88c54a3182c7b635c003c832b48ae6ee559f10232175.

Risks: P1 prior combined-scope ablation is a confound, now isolated; P2 longer
scan may change memory/latency despite unchanged arithmetic; P2 single seed.
Verdict CONDITIONAL until runtime gates pass. No old run/model files overwritten.

Decision: extend factory with opt-in per-axis overrides, leaving all defaults and
parameter construction unchanged. Record effective scopes in execution_spec.
