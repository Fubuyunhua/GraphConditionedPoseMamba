# Implementation differences

## Rewired graph

New keys are `graph_topology_mode` (`anatomical` by default) and
`graph_rewire_seed` (3407). A private `random.Random` performs fixed undirected
double-edge swaps independently for bone and symmetry relations. Edge counts,
per-node degree sequences, no-self/no-duplicate constraints and bone
connectivity are enforced without consuming Python global, NumPy or Torch RNG.

The model creates one graph specification before block construction and passes
it to all eight blocks. Targets/sources, adjacency and degree tensors are all
built from that specification. Rewired topology buffers are non-persistent, so
loading parameters cannot overwrite the configured graph; save/load rebuilds
the same graph from mode, seed and hash.

## Matched no-reset

New key `recurrence_scope` defaults to `independent`; `joined` is legal only on
the factorized K=2 path. Local input/context projection, Conv1D, activation,
u/Delta/B/C production and z gating run before joining.

For input `X=[B,T,J,C]`:

- spatial local tensors: `[B*T,2,D,J]` and B/C `[B*T,2,N,J]`;
- spatial joined scan: `[B,2,D,T*J]` and B/C `[B,2,N,T*J]`;
- temporal local tensors: `[B*J,2,D,T]` and B/C `[B*J,2,N,T]`;
- temporal joined scan: `[B,2,D,J*T]` and B/C `[B,2,N,J*T]`.

Forward concatenates segments in original order. Backward already reverses
tokens within each segment, then additionally reverses segment order to form
the exact global reverse. The inverse mapping restores segment order before the
existing per-segment direction alignment and merge. Batch samples remain
separate.

## Corrected PoseMamba backward

New key `posemamba_backward_mode` defaults to `legacy`. `exact` binds a new
autograd function with the released forward unchanged. For the limb direction
`y=x+P(x)`, exact backward returns `g + P^Tg`; transverse and reverse-direction
gradients retain their released mappings. K=4, CrossMerge and the selective
scan CUDA kernel are unchanged.

## Logging and checkpoints

Model construction now logs the actual model, graph mode/hash, recurrence
scope, scan type, K and backward mode. H36M logs effective batch, steps/epoch,
total optimizer steps and EMA updates. New final endpoints are saved explicitly
as `raw_fixed_epoch80.bin` and `ema_fixed_epoch80.bin`; MPI analogously saves
fixed epoch 120.

No existing config was overwritten. Default anatomical/independent/legacy
behavior remains checkpoint compatible.

