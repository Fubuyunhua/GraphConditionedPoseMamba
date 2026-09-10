# Corrected user target: H36M detected input

User clarified parallel W128/D10 means H36M detector input, NOT GT2D. The briefly
started D10 GT PID2025903 was terminated during its first epoch before any completed
epoch. Its directory/logs retained, statusCANCELLED_USER_CORRECTION. Existing
D20 GT PID2004698 remains untouched.

BALANCED-DET-S0:3,435,395params,width128,depth10,80epochs,batch8,seed0,strict
random initialization. LR5e-4*.99/epoch,no warmup,DP.20,WD.012,clip1,EMA.99960004.
gt_2d=false; cache detector xy must differ from labels. No pretrained or resume.
Dedicated source/run directory. Formal constructor fingerprint must match preflight.

Additional boundary gate: dataset17748 yields final batch4 at batch8. Test B1/B2/B4,
then same compiled model B8 followed by B4, capture cumulative peak<15800MiB,
finite gradients and checkpoint roundtrip. Parallel capacity assumes only identified
D20 process and>=16000MiB initially free. Failure stops new launch only, no automatic
batch reduction. SAMA continues waiting; GPU throughput shared, not paper efficiency.

Main decision: monitored-best EMA detectorP1 with pairedP2, fixed80secondary;
single-seed exploratory. Not a causal depth-only comparison because batch differs.
Stop at80, runtime failure or user instruction, no auto retry or sweep.
