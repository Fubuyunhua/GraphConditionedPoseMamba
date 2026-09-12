# Depth-oriented maximum candidate — generated, not launched

2026-09-13 user authorizes training. Start MAX160-DET-S0 first; GT remains planned.
Actual runtime gate additionally compares activation-checkpoint enabled/disabled
outputs and gradients on training DropPath with preserved RNG; actualB4 compiled
steps and EMA roundtrip must pass before formal startup. No old checkpoints loaded.

User requests a reasonable maximum accuracy-oriented model, explicitly excluding
W256D16. Choose W160D32,16,534,531 parameters (CPU counted),rather than claim a
proven optimum. W128D32=10,917,507; W192D24=17,492,883; W192D28=20,399,755;
W256D16=20,192,451. These are count checks,not trained accuracy comparisons.

Evidence: same-width W128 GT D10 best12.652770 vs D20 best11.309951, a1.342819mm
gap. Detector D10 matched-B4 best38.483971 vs historical partial D20 best37.659301,
a.824670mm gap. Clip/optimizer/budget-completion differences and single seeds
prevent proving depth universally dominates width. D20 GT had clip1 while D10GT
had no clipping. Historic D20 detector was incomplete65/80.

W160 keeps widening moderate(+25% vs128) and increases depth32(+60% vs20).
Relative to256D16 it doubles depth with18.12% fewer parameters. No new module,
state dimension or loss; SSM inner300,MLP315,graph hidden80,state16,fullDelta/B/C.
Keep anatomical topology and independent factorized recurrence. Don't removeB/C
solely from a single-seed .052mm Full-vs-Delta-only difference.

Training candidate: separate fresh detector and GT runs,each80epochs,batch4,seed0,
FP32,DP.20,EMA.9998. No pretrained/resume/finetune/freezing. Large-model stability
recipe from R3:peak3e-4,real8e warmup3e-5->3e-4,cosine to3e-5,WD.012 with SSM
exemptions,clip1. This is an accuracy-oriented recipe exploration,not a depth-only
ablation against legacy5e-4. No claim it will beat the previous best. Loss remains
position1,scale.5,GT velocity20,predicted-difference.5.

Blockwise non-reentrant activation checkpointing enabled to retain batch4 at32
layers; extra recomputation trades speed for memory. Model forward preserves RNG
state through checkpointing, but actual compiled/eager gradient parity and speed
must be measured. Do not assume a32GB fit or promise a runtime. No added inference
parameters or checkpointing cost at eval. Tensor widths300/315 are supported by
CPU construction; GPU kernel/stride compatibility remains untested.

Proposed MAX160-DET-S0 and MAX160-GT-S0,IMPROVEMENT,one80e run each if authorized.
Primary best EMA P1 with same-checkpointP2,fixed80 secondary,test-monitored and
single-seed disclosure. GT scores never count toward detector target. Full-budget
validity required; no automatic budget expansion/retry/seed sweep. Keep negative
results. Before launch strict resolved config/hash/GT coordinate checks,random
initializer fingerprint,parameter/optimizer coverage,B1/B2/B4 finite gradients,
checkpointing parity,memory headroom,EMA save/load/lifecycle and evaluator gate.

Static verdict CONDITIONAL for execution: both complete configs CPU verified,
15,881,731 decayed +652,800 exempt parameters cover all16,534,531; fresh/GT flags
and checkpointing confirmed. Exact count verified; no runtime GPU
preflight or training this turn. Missing per-variant fresh-init manifests deliberately
prevent bypassing preflight with an old model's report. Current experiments untouched.
