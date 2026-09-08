# Selective best-EMA fine-tuning: audit, model logic and registered plan

Runtime verdict PASS: unit tests and actual B1/B2 gates passed; frozen raw/EMA
bitwise identity and final-block/head updates verified. Initial full replay
37.430258/31.520638 mm. Formal PID1888402 started2026-09-09 07:24:05 Asia/Shanghai.

User authorizes2026-09-09 stopping the previous full-model batch2 fine-tune and
launching the proposed constrained adaptation. Previous run completed10/15,
best FT epoch1=37.499918/31.543617; last10=37.904487/31.708000. Original R3
EMA remains best37.430162/31.520564. No checkpoints overwritten.

Hypothesis: restricting updates to final block and head limits destructive
representation drift. Conventional selective fine-tuning, not model innovation.
Use source R3 EMA epoch32, immutable SHA c2175c027329e2d323bba5ceb3e613d2391f8dbf48a9e5cf6477eaf83bce7c7f,
not the failed fine-tune's best/latest. Same W256/D16,T243,detector inputs,root/flip,
losses(position1,scale.5,GT velocity20,predicted difference.5),WD.012 with SSM
exemptions,clip1,FP32. DropPath.20 retained in final block; all frozen modules eval.

R3-SELECTIVE-FT-B2-S0 / IMPROVEMENT: train only blocks.15.* and head.*. Batch2,
five epochs,seed0. Block LR1e-6,head5e-6. One warmup epoch at.1x then cosine to.1x
within5e. Fresh optimizer and EMA count0; EMA.9998999949995 from source weights.
Frozen EMA entries explicitly skip updates to prevent rounding drift. No extra
noise/loss/architecture changes. Parameter groups exclude frozen parameters.

Static verdict CONDITIONAL pending tests. Opt-in helpers leave default training
unchanged. Strict export/full replay, unit tests for default semantics, frozen
train/eval modes, LR ratios, unchanged frozen raw/EMA, changed final/head parameters,
actual compiled B2 finite gradients and EMA save/load/prediction roundtrip required.
Initial full evaluator parity within.02mm required before formal launch.

Primary best EMA P1 vs retained epoch0; pair P2 at same checkpoint, fixed5 secondary.
No gain means keep original. >=.2mm provisional meaningful threshold; single seed
does not establish significance. Historical54e source training plus this5e disclosed,
and prior failed15e attempt consumed10e separately. Test-monitored exploratory
selection, no causal batch-only claim, no automatic extension/seed sweep/early stop.
Only stop at5, technical failure, or user instruction. Separate directory and log.
No periodic automation is recreated. Record initial replay, parameter counts,
effective learning rates, hashes and launch PID after gates pass.
