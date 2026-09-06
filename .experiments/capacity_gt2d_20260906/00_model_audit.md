> Latest run: corrected-A0 is COMPLETED. User-authorized active pair is MPI Full PID1136724 plus S-GT2D-S0 PID1203060, launched19:16:44. GT remote root `/scratch/home/caiwei/GraphConditionedPoseMamba_H36M_GT2D_S_20260906`; log `launch_logs/h36m_gt2d_s_seed0.log`;80e per-epoch best EMA. See capacity_gt2d_20260906/SMALL_GT2D_PARALLEL_20260906.md. Skip its old later queue position; no third run.

# Capacity/GT2D audit

BLOCKED for launch pending live per-model gates and predecessors; user authorization is present.

All models keep anatomical/control/independent K2 graph-conditioned blocks, T243/J17, input3/output3. Sizes to CPU-verify: W64/D8 800083; W128/D20 6836355; W256/D10 12646107; W256/D16 20192451. M/L inherit the existing dp030 candidate and R3 AdamW schedule; S-GT retains historical80e optimizer. GT and detected pairs at M/L/XL have identical parameter count and training recipe.

GT2D source: dataset_motion_3d.py constructs input from data_label[..., :2] and confidence1 after train flip; test likewise uses label xy. datareader_h36m.py maps joint3d_image xy using image width/height, not camera X/Y; verify real cached sample provenance before PASS. scale_range_pretrain=null forbids accidental crop/scale branch, noise/masks disabled. train_2d=false: target remains3D, not a detector-training task. No synthetic perspective or camera-XY shortcut is introduced.

Important evaluator contract: train.py evaluate overwrites prediction xy with known GT2D input xy, then denormalizes, multiplies2.5D factor and root-centers. It therefore reports GT2D input-xy-preserving/depth-lifting metrics, not unrestricted raw XYZ prediction. Preserve and disclose this inherited convention; never mix its numbers with detector-input performance or claim universal3D improvement. Confirm root/flip consistency and coordinate provenance using protocol-only samples, not test score selection.

Loss1/0.5/20/0.5 unchanged. no_eval=true freezes endpoint; periodic/latest raw/EMA writes and explicit final --evaluate must pass a throwaway lifecycle smoke. New runs use seed0 fresh initialization. Final test only at registered endpoint, no checkpoint search.

Before launch: record code/config/data hash, resolve inheritance, CPU count and DropPath schedule; train/test sample assertions for GT confidence/xy and finite shapes, forbid hidden scale augmentation; real finite-gradient B1/B2/B4 compiled gate below28672MiB on idle GPU. No model/config mutation in a running directory. P1: GT2D evaluator convention and single-seed uncertainty; P2: larger models may underfit under dp0.30 or overfit; P2: M/L/XL historical runtime estimates are not guarantees. Existing GPU jobs must not be interrupted for checks.

CPU verification completed: all seven configs resolved with expected parameters; first train/test GT sample shape243x17x3, input xy exactly equals label xy and confidence is1. GPU work was disabled; this is not a CUDA launch PASS and does not replace full coordinate-provenance/flip/root checks. See cpu_preflight.json.
