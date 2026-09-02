# Model and optimization logic

## Failure mode

The 0.8M model consumes nearly all 16 GiB on RTX 5060 Ti despite its small
parameter count.  The evidence points to saved activations, factorized
selective-scan state and `reduce-overhead` CUDA Graph pools, not parameter size.

## Mechanism

The optimization preserves model math and changes only tensor lifetime and
execution policy:

1. use ordinary Inductor `default` mode to avoid reduce-overhead CUDA Graph pools;
2. evaluate the original eager module to avoid a persistent compiled eval graph;
3. clear previous gradients before the next forward allocation;
4. if needed, use non-reentrant per-block activation checkpointing with DropPath
   RNG preservation, recomputing block activations during backward.

## Novel versus conventional

These are conventional systems optimizations, not model innovations.  They
must not be reported as accuracy contributions.  The research model remains
the graph-conditioned factorized PoseMamba from the frozen upstream commit.

## Assumptions and falsification

- Default compile changes scheduling but not the FP32 computation contract.
- Eager evaluation uses the same parameters and evaluation protocol.
- Non-reentrant checkpointing preserves outputs and gradients when RNG is saved.
- Any non-finite value, material loss/gradient mismatch, checkpoint failure or
  insufficient concurrent headroom falsifies the affected candidate.
