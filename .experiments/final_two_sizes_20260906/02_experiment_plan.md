> Latest amendment: use `.experiments/final_two_sizes_20260906/GT_FINETUNE_AND_SEEDS_LAST.md`. All random-seed repeats move LAST. Detector models train fresh; each GT2D model fine-tunes from its SAME-width best EMA for30 epochs at LR3e-5, not from scratch or across widths. Earlier budgets/order below are historical.

# Remaining bounded plan

Finish active MPI Full and S-GT2D in their already authorized parallel arrangement. MPI-PoseMamba stays CANCELLED. Existing Rewired/no-reset/corrected A0 studies are complete and must not be repeated. Retain the six already-authorized H36M paired repetitions: A0/A2/Full seed1,then seed2, because the injection effect is only~0.214mm and variance unknown.

Then run exactly: FINAL-W128-DET2D-S0(80e),FINAL-W256-DET2D-S0(60e),FINAL-W128-GT2D-S0(80e),FINAL-W256-GT2D-S0(60e). One large run at a time, independent fresh outputs, per-run gate. Old planned M6.8/L12.6/XL20.2 recipes and ACC-D16-DP030 are superseded, not additional jobs. Preserve their files/history but never execute them.

Primary for each is best EMA P1 within its registered budget; paired P2 and fixed final EMA/raw also reported. Only256 detected-input P1<37.0 counts toward36.x. GT is separate task protocol, not evidence of that target. Do not add unregistered sweeps or extend budget until good results appear. After four runs, summarize tradeoff and stop for review. If final paper claims performance for the chosen larger model robust across seeds, further repeats would require a separately scoped plan; these four alone are exploratory.

Optional short work: checkpoint replays, error breakdowns, paired-seed statistics and matched hardware efficiency only if claiming efficiency. No additional long structural ablations or MPI model training. Historical costs suggest W128~20-22h per80e and256~23-26h per60e; refresh estimates from actual gates.
