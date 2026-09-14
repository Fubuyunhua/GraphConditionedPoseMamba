# Independent fine-tuning sequence: fixed scope

User explicitly requests a sequence of redesigned fine-tuning strategies.
Source for EVERY arm is original S1 W128D20 bestEMA epoch45 SHA6715a014823ac3b7365b8e8d64625486330f706f87191ddcdef2a313f599cbed,
37.659301/31.871747. Never use previous strategy's output as next initializer.
The earlier layerwise-only run completed8 and regressed to38.076095. This motivates
testing objective/regularization changes,not assuming that smaller LR guarantees gain.

Order,all IMPROVEMENT/exploratory: no_droppath (DP.2->0); no_diff (lambda_diff.5->0);
velocity10 (GT velocity coefficient20->10). Each changes ONE factor versus previous
layerwise baseline, NOT cumulative. Each8epochs,batch4,seed0,all6.84M trainable,
input/early1e-6,late3e-6,head1e-5;one warmup then cosine.1;WD.012,clip1,EMA.9998.
Total24additional epochs plus short gates; no automatic additions/extensions.

Predictions: noDropPath may reduce stochastic train/eval discrepancy; noDiff may
avoid suppressing real motion; velocity10 may change pose/velocity tradeoff. These
are hypotheses,not claims current terms caused the regression. Risks include
overfitting and temporal jitter. Neither GT velocity matching nor its coefficient20
alone proves over-smoothing. Motion metrics needed before claiming robustness.

Static CONDITIONAL. Each arm independently strict-loads same source,checks expected
config,all-layer updates,optimizer/EMA resets,B1/B2/B4 finite gradients and save/load,
and full initial evaluator parity. Parameter tensors identical despite DP setting.
No model architecture/loss implementation changes,only tested existing config fields.

Primary per-arm bestEMA P1 with pairedP2 and fixed8. Preserve epoch0 as fallback.
Any gain observed; >=.2mm provisional meaningful single-seed threshold. All3 results
reported,not only winner. Repeated test monitoring makes this exploratory tuning,
not locked-test proof; gains need separate validation/repeats. No seed search.
Technical failure stops sequence for diagnosis. Resource wait max6h per arm; no
other process stopped. One-shot serial controller,not an app recurring automation.
SAMA stays paused. Logs/results/source snapshots in independent root.
