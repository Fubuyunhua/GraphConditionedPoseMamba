# M-FULL — aligned epoch80 result

## Verdict

`KEEP` as a valid user-aligned MPI-INF-3DHP result under the disclosed
test-monitored protocol. It has no matched MPI PoseMamba comparison because the
user cancelled that baseline before launch.

The run preserved epoch1 replay, resumed model/optimizer/EMA/LR/RNG state and
completed metric/checkpoint writes through epoch80. An identity-checked watcher
then sent SIGTERM; launcher exit143 is an authorized stop, not a technical
failure. Partial epoch81 is excluded. All six retained checkpoint states are
finite and the fixed80 copies are byte-identical to the epoch80 latest files.

| View | Selected epoch | MPJPE | P-MPJPE | PCK150 | AUC |
|---|---:|---:|---:|---:|---:|
| Best EMA | 43 | 15.330702 | 8.781069 | 99.625575% | 89.968116% |
| Best raw | 70 | 15.604938 | 9.319566 | 99.656266% | 89.798397% |
| Fixed80 EMA | 80 | 16.205517 | 8.030134 | 99.734015% | 89.303188% |
| Fixed80 raw | 80 | 16.463124 | 8.916045 | 99.598977% | 89.216232% |

Primary selection is lowest EMA MPJPE in epochs1-80; every secondary metric in
that row comes from epoch43. The lower fixed80 P-MPJPE does not replace the
paired P-MPJPE. This protocol uses per-epoch test exposure and must not be
described as a locked-test result. Strict replay is deferred because another
user's GPU process appeared while W128 was running; online evaluations and
checkpoint integrity are already complete.
