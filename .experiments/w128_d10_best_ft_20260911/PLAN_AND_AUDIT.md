# Best-epoch W128/D10 detected-input fine-tuning

Runtime PASS: source strict replay39.3835665/32.7786591mm, frozen and LR/gradient/
roundtrip checks passed; actual680835 trainable. Formal PID2058759,09:08:02 China time.

User requests further fine-tuning from best epoch,2026-09-11. The observed39.383528mm
is the best of one recipe, not an established capacity ceiling. No guaranteed gain.
Source: D10 detector best EMA epoch63,pairedP2=32.778690, sourceSHA
cb5a9153f473e9280ba95b78b5276f4d0645336ef9287eed6d8b0b424425ee59.
Baseline original80epochs completed; source checkpoint retained unchanged.

W128-D10-BEST-FT-S0 / IMPROVEMENT. One8epoch run,batch8,seed0,from best EMA.
Train only last2blocks(index8/9) and head,freeze earlier8/input/positions and
keep their modules eval. BlockLR2e-6/head1e-5,one epoch.1x warmup then cosine
to.1x. Same detector input,root/flip/loss,DP.20,WD.012 on trainable params,clip1,
EMA.99960004. New optimizer,EMA count0 from loadedweights; frozen EMA entries skip
updates. No checkpoint optimizer/scheduler/RNG imported. Fresh-only flag disabled
explicitly for this authorized detector fine-tune; GT scratch experiments unchanged.

Hypothesis: limited update scope and smaller steps may improve a useful basin
without destructive full-network drift. Prior W256FT failures motivate caution,
not proof this works. Maintain batch8 to avoid adding a batch-size confounder.
This remains combined recipe adaptation, not causal explanation or innovation.

Static CONDITIONAL until strict source/export equality,unit tests for D10/D16 and
default behavior,real B1/B8+tail4 finite gradients,selected8/9/head changes,frozen
raw/EMA equality,LRratio5,strict checkpoint roundtrip and full initial replay
within.02mm of39.383528/32.778690. Separate output directory and manifest.

Primary best EMA P1 over8epochs vs sourceepoch0; pairedP2/fixed8 secondary.
Any improvement observed; >=.2mm provisional meaningful threshold,not significance.
No improvement means keep original; stop8/failure/user, no automatic extension.
Report source80e plus8e adaptation, test-monitored exploratory single seed.
SAMA remains user-paused. Other registered ablations/D10GT stay planned.
