# W128 GT2D from scratch: user correction

User explicitly rejects detector-pretrained GT fine-tuning and requests W128
GT2D random initialization. Stop W256 GT fine-tune after9 complete epochs,
interrupt10, preserve weights and markCANCELLED, not a valid completed30e result.
Prior fine-tune was a time-saving hypothesis; it did not establish superiority
over scratch. No causal claim that initialization alone caused worse accuracy.

GT-W128-SCRATCH-S0 / IMPROVEMENT: W128/D20,6,836,355params,T243,batch4,80epochs,
seed0,FP32,GT inputxy from labels+confidence1. All parameters trainable. Finetune,
resume,pretrained,selective mode all off. Exact random seed initialization SHA
must match independently created constructor state at formal startup.

Use historical W128 scratch LR recipe:5e-4 with per-epoch.99 decay, no warmup,
DropPath.20,AdamW WD.012 all groups (historical optimizer behavior),EMA.9998.
Clip1 and finite-error guard retained as declared safety delta. No new losses:
position1,scale.5,velocity20,predicted-difference.5. Not the30e low-LR recipe.
GT full evaluator retains knownxy depth-recovery protocol; report separately
from detector inputs. Same dataset hash/subjects,root/flip. Prior0.8M GT16.816mm
is a reference at different capacity, not a guaranteed target.

Static CONDITIONAL pending runtimePASS: GT coordinate/shape checks,parameter
coverage,deterministic initial SHA,new optimizer/EMA,real B1/B2/B4 finite gradients,
EMA save/load roundtrip and formal fresh-only fingerprint gate.
Primary monitored-best EMA P1 over80epochs, pairedP2;fixed80 secondary. Single seed
test-monitored exploration; no promise of gain. Stop80/technical failure/user;
no automatic retry or additional hyperparameter sweep. New independent run.

SAMA waiting supervisor paused to avoid starting during experiment transition;
no SAMA training was running. Its code remains separate and future work preserved.
Other user's GPU process untouched. Periodic monitoring remains disabled.
