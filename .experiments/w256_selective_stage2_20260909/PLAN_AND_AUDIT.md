# Stage2 bounded selective fine-tuning

Runtime PASS: B1/B2 finite, frozen raw/EMA exact, selected blocks/head changed,
LR groups and strict roundtrip verified; initial replay37.416286/31.507343.
Formal PID1961722 starts2026-09-09 12:16:25 Asia/Shanghai,2,516,731 trainable parameters.

User authorizes further training on2026-09-09. Source is stage1 best EMA epoch4,
37.416286/31.507343mm, SHA31540f2570a41de072e68456dfe7960a4d6e62e6dcd0d38f3519e00ad006ea54.
Stage1 completed5e with exit0 at08:14; gain.014mm was negligible, not a breakthrough.

IMPROVEMENT R3-SELECTIVE-STAGE2-S0: expand trainable range from last1 to last2
blocks and head. Hypothesis: slightly more adaptable features may improve P1,
while frozen earlier14 blocks limit drift. Conventional fine-tuning, single seed,
test-monitored exploratory selection, no promise of36.x or statistical significance.
Exact identity/strict initial full replay required. All original source weights
retained; export weights-only, reset optimizer/scheduler/EMA count, seed0.

Eight epochs,batch2,blockLR5e-7/head2e-6,one warmup epoch at.1x then cosine to.1x.
Losses and active-block DropPath.20 unchanged,SSM no-decay,WD.012,clip1,FP32,
EMA.9998999949995. Frozen earlier modules eval and frozen EMA never updated.
These joint changes are a recipe exploration, not causal unfreezing ablation.
No extra seeds/extensions. Stop at8, technical failure or user instruction.
Best EMA P1 compared with preserved epoch0; paired P2/fixed8 secondary. >=.2mm
provisional meaningful target; no improvement means retain source. Disclose
R3 training54epochs + stage1 five + stage2 eight; earlier failed trials separately.

Static verdict CONDITIONAL until tests, B1/B2 finite gradients, every selected
block/head update, frozen raw/EMA identity, correct group LR, strict EMA roundtrip
and initial replay within.02mm pass. Report trainable count/config/source/data hashes.
Default one-block helper behavior retains regression test. Existing external GPU
process preserved; require16000MiB free before preflight and peak<14000MiB; shared
timing is not publication efficiency. No periodic automation recreated.
