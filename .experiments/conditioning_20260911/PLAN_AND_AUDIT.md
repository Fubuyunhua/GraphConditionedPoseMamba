# Conditional batch4 restart and parallel conditioning ablations

User authorizes: if current D10 best-epoch fine-tune is ineffective, restart D10
from scratch at batch4 using the0.8M recipe, while running the conditioning ablations
in parallel. Interpret ineffective using already proposed0.2mm meaningful threshold:
best FT must be<39.1835665 to avoid restart. Finish all8 FT epochs first; don't stop it.
If FT crashes, diagnose rather than equating failure to an accuracy result.

Batch8 full D10 had177520 updates vs batch4 reference354960 over80epochs. EMA
sample-horizon matching did not equalize AdamW steps or shrinkage. Increased
capacity alone cannot guarantee improvement. Last-two-block fine-tuning is limited
adaptation, not a test of full model capacity; the user prioritizes full retraining.

Registered jobs, all H36M detector T243/S81,seed0,80epochs,fresh constructors:
- delta: W64D8,800083params,only rank-Delta projection from graph context.
- bc: W64D8,800083params,only B/C projection slices from graph context.
- d10_b4: W128D10,3435395params,Full graph control; conditional on FT outcome.
All batch4,LR5e-4 exponential.99,WD.012 one AdamW group,EMA.9998,DP.20,
no actual warmup,no clipping,original losses and flip/root conventions. This
matches recorded0.8M Full effective training recipe, not merely dormant YAML fields.

Route implementation adds no parameters; common projection weights generate both
content/context candidates,select slices BEFORE dt projection/softplus. Unselected
paths remain input-dependent,never zeroed/frozen. u/z/A/D unchanged. Both spatial
and temporal branches use identical target. Full default skips new tensor work.
Partial modes require factorized control route; coupled/feature combinations rejected.

Gate: literal slice/gradient unit tests, actual GPU scan dts/B/C/u equality for
einsum and grouped-conv paths, default Full output AND gradients vs immutable
prior checkout, none vs A1 on identical shared weights, all parameter counts,
real B1/B2/B4 finite loss/gradients and graph-gradient reachability,EMA save/load,
fresh hash formal startup. Runtime failures stop launch; no batch changes or retries.
Frozen prior Full/A1 sources/checkpoints untouched. Full reference39.845162/33.232240
can be reused only with equivalence PASS. A1 no-graph40.0605/33.3565 is functionally
none with749891 active params; disclose its different allocated graph params and
initialization. It is not a strict matched-parameter four-cell interaction study.

Primary per-run best EMA P1 with pairedP2, fixed80 secondary. Tests are monitored,
single seed results exploratory. Preserve negative outcomes. No guarantee of gain.
One controller preflights all,launches delta/bc,waits for FT completion and conditionally
launches d10_b4. Max3 formal jobs after FT exit; SAMA stays paused, existing GT plans
remain later. No app recurring automation. Shared throughput isn't paper efficiency.

Static verdict CONDITIONAL until runtime gates pass. Exact code/source/data hashes
and launch identities stored under verification and launch_logs in independent root.
