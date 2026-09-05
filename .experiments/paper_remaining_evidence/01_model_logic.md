# Model logic and falsifiable claims

## Q1 — injection position

A2 sends `X+G` through the recurrent content path, parameter projections and
ordinary output gate. Full keeps content and the gate on `X`, while `X+G`
controls Delta/B/C. The existing A2-versus-Full comparison tests the complete
injection strategy, not a Delta-only mechanism.

## Q2 — topology content

Full Rewired keeps GraphMixer parameters, aggregation scale, control injection,
factorized K=2 scan and every training control fixed. Only anatomical bone and
symmetry edges are replaced by one seed-3407 degree-preserving graph. Better
Full performance supports anatomical topology; equality supports generic
cross-joint context; a rewired gain contradicts topology indispensability.

## Q3 — recurrence boundary

Matched no-reset retains local Conv1D, input/context projections, u, Delta/B/C,
z, A/D, K=2 and every parameter. It joins already prepared segment tensors only
at the selective-scan boundary. A Full gain supports state resets; equality
reduces factorization to an organization choice; a joined gain contradicts the
claim that cross-segment state is harmful.

## PoseMamba diagnostic

The released A0 forward contains `x + x[..., parent_indices]`, while its custom
backward omits the transpose-index accumulation. Exact mode adds `P^T g` only.
Forward predictions and checkpoint inference must remain unchanged; training
differences diagnose the released derivative rather than invalidate A0.

