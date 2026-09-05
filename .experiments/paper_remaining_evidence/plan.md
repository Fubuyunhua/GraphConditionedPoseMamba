# Remaining paper evidence plan

Paper structure and claim boundaries are maintained in
`docs/PAPER_OUTLINE_AND_CONTRIBUTIONS.md`; current operational state is in
`RESEARCH_HANDOFF.md`.

The implementation is isolated from the running R3 experiment. No long run and
no remote push are authorized by this preparation task.

## Execution order after explicit confirmation

1. Full with Rewired Graph, seed 0, 80 epochs.
2. Full w/o Recurrence Reset — matched, seed 0, 80 epochs.
3. Corrected-backward PoseMamba diagnostic, seed 0, 80 epochs.
4. A0, A2 and Full seed 1, one run at a time.
5. MPI-INF-3DHP released A0 and Full under the repaired T81 protocol.
6. A0, A2 and Full seed 2, one run at a time.

Every H36M result reports both `best_ema_test_monitored_first80` and
`ema_fixed_epoch80`; these columns are never mixed. MPI uses the locked
epoch-120 endpoint and does not monitor its test set per epoch.

No graph search, seed search, V2 model, extra loss, MoE, frequency module,
additional detector or automatic queue expansion is permitted.
