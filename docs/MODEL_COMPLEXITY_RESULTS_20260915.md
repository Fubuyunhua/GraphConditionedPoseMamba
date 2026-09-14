# H36M model sizes and computational cost

Measured in evaluation mode on RTX5090, PyTorch2.11.0+cu128. Input B=1,
T=243,17 joints,xy+confidence; output243 frames. No training was performed.

| Model | Parameters | MACs-equivalent / clip | MACs-equivalent / frame | Flip-TTA forward work |
|---|---:|---:|---:|---:|
| PoseMamba A0 W64/D6/M1 (local ~0.8M baseline) | 790,083 | 6.459251G | 26.581280M | 12.918502G |
| W64D8 | 800,083 | 5.799337G | 23.865584M | 11.598674G |
| W128D10 | 3,435,395 | 21.730435G | 89.425657M | 43.460869G |
| W128D20 | 6,836,355 | 43.455053G | 178.827378M | 86.910106G |

The small model is W64D8, not W64D10. Parameter counts were checked against
the actual model constructors. Dense computational cost does not depend on
checkpoint values.

The added PoseMamba A0 row is the historical local W64/D6/MLP-ratio1 baseline,
with xy+confidence and the legacy released forward path. It is not the published
0.9M PoseMamba-S or the9.450M local large checkpoint. It was profiled separately
under the same input/environment/counting convention; all12 selective-scan calls
were counted, all parameters are trainable, output is finite243x17x3, and there
are no unclassified operators. Dense arithmetic profiling needs no trained
checkpoint and does not re-evaluate historical accuracy.

Compared with this A0 baseline, GCPM W64D8 has10,000 more parameters (+1.266%)
but10.217% less estimated forward MAC-equivalent work. This is not a claim of
the same percentage reduction in latency or training time.

Counting convention: fvcore fused multiply-add=1, plus the repository's existing
PoseMamba selective-scan estimate `9*B*L*D*N+B*D*L`. All16/20/40 selective-scan
calls were explicitly counted. These are MAC-equivalent estimates, not literal
hardware FLOPs. Normalization follows fvcore conventions; elementwise
arithmetic/activations and flip/permutation operations are excluded. No
unclassified operators remain. Do not divide these totals by2 to relabel them
MACs. Comparisons with literature require matching counting conventions.

Main values are one forward pass. Flip-TTA values include two model forwards,
not flip or averaging overhead. Training/backward cost is outside this report.

This public update contains measurement results and methodology only. The
profiling implementation and full local diagnostic records have not been
uploaded as part of this update.

