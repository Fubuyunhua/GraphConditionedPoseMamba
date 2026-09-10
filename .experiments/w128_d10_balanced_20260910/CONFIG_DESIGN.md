# W128/D10 balanced candidate: generated, not launched

User asks for a reasonable0.8M–6.8M configuration informed by SAMA and prior GCPM
experiments. CPU constructor count verifies3,435,395 parameters. No architecture
source change. Both detector and GT configs are strictly fresh seed0; no fine-tuning.

Borrow width128,batch8,80epochs,exponential.99 from SAMA's medium-scale recipe.
Retain GCPM's empirical5e-4 peak,WD.012 all params,DP.20,EMA,clip1 and original
losses(position1,scale.5,velocity20,predicted-difference.5). No warmup, as historical
successful W128 schedule; explicit0 removes the old misleading unused8 field.
SAMA's H36M LR5e-5 is not silently substituted for GCPM's independently supported
scale. No claim that either is universally better.

Batch8 vs4 gives roughly half optimizer updates per epoch; EMA.9998^2=.99960004
preserves sample-domain forgetting. This compensates averaging horizon only, not
all batch/AdamW dynamics or effective cumulative weight decay. Keep LR unchanged
as a conservative choice relative to linear scaling; not a verified optimum.
No assumption of faster batch8 or guaranteed lower memory: actual B8 gate needed.

Historical W128D20 had6,836,355params and detector best37.659301; current GT scratch
W128D20 best11.834459 as of42epochs is promising but incomplete. W64D8 GT best16.816311.
Width128 depth10 tests an intermediate model; changes in batch and training recipe
mean it is a capacity/recipe candidate, NOT a clean depth-only ablation. For causal
depth comparison add a separately registered matched-batch4 control if requested.

Geometry/loss flow unchanged:B,T243,J17,3 -> graph-conditioned independent spatial
and temporal recurrence ->3D. Same anatomical graph,SSM state16,ratio1.875,MLP1.96875.
GT uses labelxy+c1 and knownxy evaluator; never compare its score to detector P1.

Static verdict CONDITIONAL for execution. CPU parameter count and both complete
YAML inheritance checks PASS; fresh flags,GT toggles,batch8 and EMA checked;
no actual B8 GPU/gradient/checkpoint or fresh-init fingerprint gate yet. Each config
requires its own missing preflight manifest, intentionally preventing accidental
use of another model's gate. Do not launch by merely pointing at old W128D20 scripts.
Execution not requested this turn; current W128 GT and independent SAMA queue untouched.

Proposed experiment IDs BALANCED-DET-S0 and BALANCED-GT-S0, IMPROVEMENT,80epochs,
seed0,one run each; primary best EMA P1 with pairedP2,fixed80 secondary; explicitly
exploratory test-monitored. Compare named protocol/capacity, not promised target.
Keep failure evidence; no automatic rerun,extra seed or budget extension.
