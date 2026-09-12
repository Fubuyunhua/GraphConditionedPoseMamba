# D10 GT from scratch, matched batch4 recipe

Runtime PASS: preflight and formal fresh fingerprint verified. PID2118071
started2026-09-12 13:58:31 Asia/Shanghai; initial2epochs finite, actual4437batches/epoch.

User requests starting next GT experiment2026-09-12. Use W128D10 (3,435,395params)
and successful latest batch4 detector recipe, not superseded batch8 or GT fine-tune.
GT-D10-B4-S0 / IMPROVEMENT,random seed0,80epochs,batch4,T243,FP32,all parameters,
LR5e-4 exponential.99,WD.012 single group,no clipping/no actualwarmup,EMA.9998,
DP.20,Full graphDelta/B/C. Original losses position1,scale.5,velocity20,difference.5.
No pretrained,resume,finetune or selective freezing. Initial hash must match fresh
constructor at formal start. Dataset sameH36M subjects/GTxy+c1/root/flip; evaluator
preserves knownxy before metric conversion, so don't compare detector/GT numbers.

Static CONDITIONAL pending realB1/B2/B4 finite gradients,graph reachability,new
optimizer/EMA,parameter count,data SHA,GT train/testxy assertions,checkpoint
roundtrip and formal fresh-hash gate. Other user GPU process preserved. Reuse
existing default Full path with prior equivalence PASS; no model changes.

Primary best EMA GT P1 over80epochs,pairedP2/fixed80 secondary. Compare W64GT16.816
and W128D20GT11.310 as references with capacity/clip differences disclosed. No
accuracy promise, single seed/test-monitored exploratory; no auto retry/extension.
Dedicated directory preserves cancelled prior D10GT attempt. SAMA remains paused.
