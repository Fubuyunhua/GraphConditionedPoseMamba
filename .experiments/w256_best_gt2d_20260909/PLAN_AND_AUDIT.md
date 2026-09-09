# W256 corresponding GT2D experiment

User requests immediate GT2D counterpart2026-09-09. Source is same-width best
detector-input EMA, stage1 epoch4,37.416286/31.507343, SHA31540f2570a41de072e68456dfe7960a4d6e62e6dcd0d38f3519e00ad006ea54.
Stage2 completed8 with exit0, best37.418864, so source remains stage1 best.

GT-W256-BEST-S0, IMPROVEMENT: detector-pretrained adaptation, not fresh training.
W256/D16,20,192,451 parameters, all trainable, batch4,30epochs,peakLR3e-5,
3warmup epochs from3e-6,cosine to3e-6,WD.012 with SSM exemption,clip1,EMA.9998,
DropPath.20 matching source. No selective freezing. New optimizer/scheduler/RNG/
EMA count0, weights-only export, strict load. Source never overwritten.

Dataset remains H36M-SH T243/S81, same subjects/root/flip/metric. GT path uses
label-normalized xy and confidence1. Evaluator substitutes known xy before
denormalization/root alignment, estimating missing depth. Scores belong to this
GT protocol and cannot substantiate detector-input improvements. Active losses
unchanged(position1,scale.5,velocity20,predicted-difference.5). No new corruption.

Static audit CONDITIONAL until strict source/export equality, parameter coverage,
optimizer/EMA reset, train/test GT coordinate assertions, real B1/B2/B4 finite
gradients, saving/reloading EMA and finite initial GT evaluation pass. Baseline
GT accuracy measured before optimization; no assumed target from detector37.416.
Source phase1 replay already verified. GPU free>=28000MiB for launch; preserve
external processes. Main source and config hashes in preflight report.

Primary: test-monitored best EMA GT P1 epochs1-30; pairedP2/fixed30 and initialGT
secondary. Report improvement against initialGT without promising gains. Single
seed exploratory; later repeats needed for small differences. Full budget30,
technical failure or explicit stop; no retries/extensions. OriginalR3 consumed54e,
stage1 five, GT adds30; separate failed trials disclosed. No inference-width
mapping, no GT-vs-detector causal claim. Independent directory and log. Periodic
monitoring remains disabled. User explicitly prioritizes W256 GT before old queues.
