# One bounded accuracy experiment

ID ACC-D16-DP030-S0; track IMPROVEMENT; status PLANNED. User authorizes execution after the necessary paper studies and runtime gate.

Model W256/D16,20192451 parameters; seed0;60 epochs; same H36M-SH data hash73b642f2567a8d0b194f88c54a3182c7b635c003c832b48ae6ee559f10232175 and train subjects as R3. Scientific delta: maximum DropPath0.20->0.30 only. All loss/optimizer/batch/precision/EMA settings frozen by the config.

Primary endpoint: EMA P1 at fixed epoch60, with P2 from the same checkpoint. No per-epoch H36M test evaluation or early stopping based on test metrics. Evaluate latest_ema_epoch.bin after verifying epoch60; raw final metrics are secondary, never used to replace the primary endpoint. Save periodic raw/EMA checkpoints for recovery; do not search them on test after seeing final accuracy. No adaptive tuning run is authorized by this registration.

Exploratory target: fixed60 EMA P1<37.430162 mm (historical R3 monitored-best reference, explicitly unmatched). Aspirational target<37mm is not a prediction. Historical R3 fixed60 is unavailable; passing either threshold is not a matched causal comparison or multi-seed claim. If a causal claim about DropPath is needed later, register a full matched control separately.

Budget: one60-epoch run, about23-26 GPU hours using historical R3 step times as an estimate; measure current preflight speed before quoting an updated estimate. Start only after MPI pair, the six paired H36M repeats and any brief necessary evaluation/efficiency checks, on an otherwise idle GPU. Any failure pauses and requires diagnosis; no automatic retry or resume from scientifically changed settings.

Artifacts: source/config/data fingerprints, effective config, gate result, command/environment, loss/runtime telemetry, raw/EMA final and periodic weights, fixed60 evaluation, analysis and GitHub record. Keep data/weights/credentials out of GitHub. Stop this accuracy phase after the single run and report KEEP/REJECT/DIAGNOSE; do not create an unlimited search.
