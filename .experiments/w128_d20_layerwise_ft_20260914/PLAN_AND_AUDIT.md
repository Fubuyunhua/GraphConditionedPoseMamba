# W128/D20 best detector EMA: all-parameter layerwise fine-tuning

Runtime PASS2026-09-14: strict initial replay37.659287/31.871699; unit and actual
per-layer update/reset/finite B4/roundtrip gates passed. PID2339948 starts11:23:42.

User explicitly authorizes code and training after requesting a considered plan.
Choose first proposed branch: layerwise learning rates with ORIGINAL losses,
without an uncalibrated parameter-anchor regularizer. No GT transfer or frozen trunk.
Source old S1 W128D20 best EMA epoch45,P1/P2=37.659301/31.871747,hash
6715a014823ac3b7365b8e8d64625486330f706f87191ddcdef2a313f599cbed.
Historical S1 consumed65/80epochs; record that budget plus this additional8epochs.

Hypothesis: slow early-feature changes plus more flexible late/head updates may
improve detection-input generalization without excessive uniform drift. Prior
full-W256 fine-tuning regressed,selective-only gains were tiny; neither proves
this works. No guaranteed36.x. Conventional optimization exploration,not innovation.

W128D20-LAYERWISE-FT-S0,IMPROVEMENT: all6,836,355parameters,seed0,batch4,8epochs,
input/positions+blocks0-9 peak1e-6,blocks10-19 peak3e-6,head1e-5. One warmup epoch
at.1x followed by cosine to.1x. Fresh AdamW moments,scheduler step0,EMA.9998/count0
initialized from sourceEMA weights. WD.012 preserved on all groups;clip1 safety
guard declared versus legacy unclippedsource. DP.20 and original losses retained:
position1,scale.5,GT-supervisedvelocity20,target-free difference.5. No anchor term.
Original detector data,confidence,GT3D targets,root/flip,FP32,T243 unchanged.

Static CONDITIONAL until full layerwise coverage tests,strict source/hash/export,
all20blocks/input/position/head updates,finite B1/B2/B4 gradients,correct LR ratios,
EMA state/reset/roundtrip and full initial replay within.02mm of source pass.
No optimizer or old scheduler/RNG state loaded; CLI pretrained weights-only,not resume.

Primary bestEMA P1 vs retained sourceepoch0,pairedP2/fixed8 secondary. >=.2mm gain
is provisional meaningful threshold,not significance; tiny gains reported honestly.
If no improvement keep source,stop at8; no automatic extensions,branches,seed sweep
or restart. Single seed,test-monitored exploratory adaptation disclosed.
New independent directory/source and checkpoints. SAMA paused,recurring monitoring off.
