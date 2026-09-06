> Latest MPI policy amendment: user requests per-epoch test evaluation and best EMA MPJPE selection for Full and released A0. Fixed120 is secondary. Full now resumes epoch1 in `runs/mpi_full_testbest_seed0`, PID1136724, log `launch_logs/mpi_full_testbest_seed0.log`; epoch1 replay participates in best selection. See `.experiments/paper_remaining_evidence/MPI_TESTBEST_POLICY_CHANGE_20260906.md`. Prior fixed-only/current-PID statements below are historical.

# User-authorized concurrent MPI Full launch

2026-09-06: User explicitly requested concurrent launch of the approximately0.8M 3DHP Full model while corrected-A0 continues. This overrides exclusive/serial launch requirements only for this named pair. Do not stop corrected-A0 and do not start a third formal run.

Frozen Full MPI config is unchanged: W64/D8,T81,batch4,789715 parameters,seed0,120 epochs,AdamW LR2e-4/WD0.01/epoch decay0.99,EMA0.9998,FP32,compiled default,anatomical/control/independent K2. Use the repaired protocol and original dataset hashes;117024 train windows and2875 valid test centres. Test selection remains fixed epoch120 with no per-epoch test monitoring. M-A0 released baseline will run after both currently active experiments complete; order changes do not alter its registered identity.

Remote root: /scratch/home/caiwei/GraphConditionedPoseMamba_MPI_FULL_20260906. Source archive commit3920d956ba46deb8d02498e4ad6373c7bbba8a39. Config and trainer hashes are checked against the fresh protocol/smoke manifest before launch. Formal run directory runs/mpi_full_seed0 is separate from verification/mpi_full_b4. Launcher scripts/run_mpi_full_parallel_20260906.sh refuses to reuse a started or nonempty formal output and uses flock.

Concurrent B4 smoke passed with input/output[4,81,17,3],finite gradients and loss0.4267011,peak allocated3281.05MiB. Before smoke the GPU had28646MiB free. The launcher allows only the identified corrected-A0 PID1102590 as another compute process, requires at least16000MiB free and smoke allocation<12000MiB. This is a user-accepted shared-GPU runtime gate, not an exclusive efficiency benchmark.

Monitor both PIDs/cwd/logs. Shared timings cannot be used for publication efficiency or to infer an optimizer effect. Record throughput and memory after launch; report material slowdowns and natural failures. On failure do not silently relaunch or stop the other job. After both runs finish and checkpoints/metrics are verified, resume M-A0 followed by the existing repeat and capacity/GT2D queue. A completed or running M-FULL must never be launched again.
