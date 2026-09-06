# D-PM-BWD-0

`RUNNING`. Corrected PoseMamba preserves the released forward, K=4 scan,
CrossMerge, data, loss, optimizer and EMA, changing only the indexed-path
backward from legacy to exact. The real-data B4 gate passed with 790,083
parameters and formal seed-0 training began at 09:11:21 Asia/Shanghai. No
accuracy or backward-effect conclusion is available yet.

At the concurrent snapshot through epoch61, the current best EMA is
`40.660688/33.532125 mm` at epoch52. Mean epoch time increased from 4.384
minutes before MPI concurrency to 8.46 minutes during it. Both processes remain
healthy; shared-GPU timing is not used for efficiency comparison.
