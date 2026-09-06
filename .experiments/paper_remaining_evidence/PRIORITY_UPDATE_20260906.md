> Latest run: corrected-A0 is COMPLETED. User-authorized active pair is MPI Full PID1136724 plus S-GT2D-S0 PID1203060, launched19:16:44. GT remote root `/scratch/home/caiwei/GraphConditionedPoseMamba_H36M_GT2D_S_20260906`; log `launch_logs/h36m_gt2d_s_seed0.log`;80e per-epoch best EMA. See capacity_gt2d_20260906/SMALL_GT2D_PARALLEL_20260906.md. Skip its old later queue position; no third run.

> Latest MPI policy amendment: user requests per-epoch test evaluation and best EMA MPJPE selection for Full and released A0. Fixed120 is secondary. Full now resumes epoch1 in `runs/mpi_full_testbest_seed0`, PID1136724, log `launch_logs/mpi_full_testbest_seed0.log`; epoch1 replay participates in best selection. See `.experiments/paper_remaining_evidence/MPI_TESTBEST_POLICY_CHANGE_20260906.md`. Prior fixed-only/current-PID statements below are historical.

> Current concurrency: user explicitly authorized M-FULL alongside corrected-A0. M-FULL PID1131838 at `/scratch/home/caiwei/GraphConditionedPoseMamba_MPI_FULL_20260906`, output `runs/mpi_full_seed0`; original corrected PID1102590 remains. No third formal run. After both finish run M-A0, then existing repeats/capacity queue. See PARALLEL_MPI_FULL_20260906.md and ledger; older serial/M-A0-first wording is superseded.

> Latest scope: user confirmed GT2D means Human3.6M ground-truth2D to3D. After existing required paper studies execute the bounded size/GT2D family in `.experiments/capacity_gt2d_20260906/02_experiment_plan.md` and master ledger execution_order. Reuse existing XL detector once; do not stop the automation before XL-GT2D completes.

# Priority and scope update — 2026-09-06

User instruction: give a simple ablation judgment, prioritize 3DHP next, retain only necessary remaining experiments, then design and start a highest-accuracy candidate using the accumulated evidence. Execution and GitHub synchronization are authorized. This supersedes the earlier order placing H36M seed1 before MPI. Preserve the currently running corrected-A0 diagnostic through its registered endpoint.

## Current judgment

Under the single-seed monitored-best protocol, anatomical Full P1 is39.845162, Rewired40.416912 (+0.571750), matched no-reset40.568980 (+0.723818), and feature fusion A2 40.058826 (+0.213664) mm. Retain anatomical graph, independent recurrence and control injection. These observations support the proposed mechanisms for seed0; they do not establish significance or guarantee a benefit at large width/depth or on MPI.

## Required serial order

1. Finish D-PM-BWD-0 corrected A0 (currently running), strictly verify and analyze it separately from released A0.
2. M-A0: released PoseMamba on MPI-INF-3DHP, seed0,120 epochs.
3. M-FULL: GraphConditionedPoseMamba on MPI-INF-3DHP, seed0,120 epochs.
4. H36M paired repeats A0,A2,Full at seed1, then A0,A2,Full at seed2. Existing seed0 remains unchanged; do not selectively skip seed2 based on seed1.
5. Brief exclusive-GPU matched efficiency measurement and existing-checkpoint evaluation/aggregation; do not add another long model-training study for efficiency. This is required only for a runtime-efficiency claim; otherwise omit that claim.
6. Launch one registered H36M high-accuracy candidate after technical preflight. See ../accuracy_candidate_20260906/. Do not indefinitely add more ablations or hyperparameter searches.

MPI must use the existing repaired T81/stride9 protocol, same data fingerprints, same loss/optimizer/EMA, fixed epoch120 primary endpoint, PCK150/AUC/MPJPE/P-MPJPE, and no per-epoch test selection. Run the two models sequentially. Protocol-only checks and independent B4 CUDA smoke are required before launch, with separate smoke and formal directories. Do not switch released A0 to corrected A0 based on diagnostic accuracy. No per-epoch test numbers is expected under this protocol and is not a monitoring failure.

Why retain repeats: the central A2-Full effect is about0.214 mm and no seed variance is known. Three paired seeds are the minimum already registered evidence needed for a stable-improvement claim. Rewired/no-reset need not be expanded to every seed unless the paper explicitly claims those individual effects are statistically robust. Retain all results regardless of sign.

## High-accuracy scope

Assume the requested high-accuracy target remains Human3.6M, where previous W256/D16 evidence exists. MPI remains the controlled second-dataset experiment; do not silently transplant a tuned H36M configuration into its comparison. The first candidate is W256/D16 R3 with maximum DropPath0.30 instead of0.20, preserving its architecture, optimizer and loss. This is a falsifiable regularization hypothesis, not a known optimal configuration. Technical setup and source must be frozen before launch. If subsequent evidence invalidates this candidate, record a bounded replacement decision before any new training rather than silently stacking changes.

Do not modify or interrupt a healthy running experiment to reorder the queue. Technical failures pause advancement and are reported, without automatic retries or configuration changes. Update this document, ledger and automation together when the current run changes.
