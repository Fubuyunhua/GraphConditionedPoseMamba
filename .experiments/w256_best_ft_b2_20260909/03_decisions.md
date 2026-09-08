# Decisions

User requested stopping FINAL W256 and immediately fine-tuning old best with batch2.
FINAL PID1813367 received SIGTERM during epoch40, after39 completed epochs.
Best FINAL EMA epoch30=37.994901/31.926142; last39=38.113283/32.010392.
Retain all prior checkpoints/logs. Do not label the39/60 run complete.

Choose the historical W256 R3 best EMA epoch32, not FINAL best, raw companion,
W128 weights or latest R3 epoch54. Verify its immutable SHA before export.
No loss redesign/augmentation stacking; preserve source DropPath.20. Apply the
new small-batch fine-tuning schedule as an independently identified experiment.
