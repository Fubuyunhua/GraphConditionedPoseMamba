# Remaining paper evidence audit

Verdict: **PASS for implementation and short preflight; BLOCKED for long
training until explicit user confirmation**.

## Provenance

- Requested source branch: `codex/minimal-ablation-80e-20260903`.
- Prompt checkpoint: `3fac1a40792815c92a1edf2eb148eb34bc0efeae`.
- Observed source HEAD: `a9315253a4586dcdb0164798135460eae26095e3`.
- The only observed prompt-HEAD delta was R3 log/state synchronization; model
  source and registered configurations were unchanged.
- Isolated local branch/worktree: `codex/paper-remaining-evidence-20260905` at
  scientific implementation commit `ad4e2fa66492737a3fcd88d9142a88731724f30b`.
- The active remote D16 R3 process was not modified, stopped or reused.

## Model and tensor contracts

- Default Full remains anatomical, factorized, independent-recurrence K=2.
- Rewired changes only non-trainable graph topology and remains 800,083 params.
- Matched no-reset remains factorized K=2 and 800,083 params; local projection
  tensors are unchanged and only scan boundaries join.
- PoseMamba exact mode remains W64/D6/M1 and 790,083 params; forward is identical
  to legacy and only the missing indexed backward accumulation changes.
- Default Full prediction/loss are bit-exact against the untouched source;
  gradients remain inside the untouched CUDA kernel's repeat envelope.

## Loss, data and evaluation

- H36M loss coefficients remain position 1.0, scale 0.5, velocity 20.0 and
  difference 0.5. No new auxiliary objective is active.
- H36M dataset hash is fixed and verified; 17,748 training samples produce
  4,437 optimizer steps per epoch at batch 4.
- Existing and new best results are explicitly test-monitored EMA selections.
  New runs also save `ema_fixed_epoch80.bin`.
- A0 and Full historical epoch-80 weights were not saved; their fixed@80
  checkpoint column is therefore unavailable rather than reconstructed.

## Optimization and runtime

- The actual legacy H36M protocol does **not** enable warmup despite declaring
  `warmup_epochs: 8`. New matched configs explicitly set
  `enable_linear_warmup: false` and preserve the single AdamW WD=0.012 group.
- Effective H36M budget is 354,960 optimizer steps and EMA updates over 80
  epochs. Epoch-80 training LR is 0.0002260218.
- 47 CPU/CUDA tests pass. Compiled H36M B1 real steps and MPI B4 CUDA smokes
  pass with finite outputs, losses and gradients.
- Smoke timing was collected while D16 R3 and another user's process occupied
  the GPU; it is invalid as a publication-quality speed estimate.

## Risks

- P1: H36M test-set monitoring biases best-checkpoint results. Disclose and
  report fixed endpoints separately.
- P1: single-seed differences around 0.2-0.4 mm require seeds 1/2 before strong
  claims.
- P2: joined sequences have length 4,131 at T243; full-batch memory must be
  measured in an exclusive prelaunch gate before a long run.
- P2: the historical A0 source commit is external to this repository, although
  its archived source confirms the backward defect and its checkpoint strictly
  loads in both legacy/exact models.

