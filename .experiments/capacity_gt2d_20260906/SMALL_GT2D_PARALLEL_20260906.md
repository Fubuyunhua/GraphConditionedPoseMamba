# Small GT2D immediate parallel launch

User explicitly requests immediate concurrent H36M GT2D training of the0.8M GraphConditionedPoseMamba. This moves S-GT2D-S0 ahead of the remaining paper queue. It overrides this run's predecessor/idle-GPU requirement, not its numerical/protocol gates. The other active run is MPI Full. Corrected A0 has already completed80/80, so only two formal tasks are expected.

Use W64/D8,800083 params,T243,batch4,80 epochs,seed0,legacy small-model AdamW LR5e-4/WD0.012/epoch decay0.99,EMA0.9998,DropPath0.20,no warmup. True2D label xy,confidence1,3D target,flip enabled,no mask/noise/scale augmentation. Training begins from fresh initialization, not the detector checkpoint.

Align this small H36M run with the user's expressed best-EMA preference: no_eval=false, per-epoch EMA evaluation, primary minimum test-monitored EMA P1 over80 epochs with P2 from the same checkpoint; fixed80 EMA/raw secondary. This amends only S-GT2D's previously planned fixed-only observation policy, not the remaining larger candidates. Keep GT known-input-xy/depth-lifting metrics separate from detected-input results.

Fresh protocol checks: first cached train/test labels exactly match normalized raw joint3d_image values; confidence1 and flip consistency pass. Reconstruction of input xy through image denormalization,2.5D factor and root-centering matches target xy within0.000611mm. All312 gradient tensors are finite in the B1 real-data backward; compiled B4 full train-step benchmark has finite loss0.422300,peak reserved2810MiB. Shared-GPU benchmark14.58it/s is engineering data, not efficiency evidence.

Deploy isolated root /scratch/home/caiwei/GraphConditionedPoseMamba_H36M_GT2D_S_20260906 from7c60065 plus explicit config/preflight/launcher additions. The launcher records effective config and hashes, verifies the H36M data hash, uses flock, refuses reuse and allows only the identified MPI Full PID1136724 as a competing compute process. At least16GiB free is required. Do not repeat S-GT2D later when its former queue position is reached.

Monitor both MPI and S-GT2D. Do not launch a third task. After both complete/are verified, resume the remaining MPI baseline, paired repeats and M/L/XL queues according to master ledger, skipping completed S-GT2D. Keep valid unfavorable results and report shared throughput honestly.
