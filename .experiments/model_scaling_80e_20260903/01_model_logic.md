# Model-scaling logic

## Failure mode

The W64/D8 Full model reaches 39.8452 mm, while the user prioritizes absolute
accuracy. Its sub-million-parameter capacity may limit representation of long
temporal dependencies and action-specific pose geometry.

## Mechanism and hypotheses

- W128/D20 tests deeper iterative graph/SSM refinement at moderate width.
- W256/D10 tests greater per-block channel capacity with less depth.
- Both retain boundary-preserving factorization and topology-conditioned state
  control; no new mechanism is introduced.

Observable prediction: at least one candidate reduces best-EMA P1 by 0.10 mm
or more relative to 39.8452 mm without numerical instability. Failure of both
to do so rejects simple capacity scaling as the next accuracy lever under the
locked 80-epoch protocol.

## Novelty and assumptions

Scaling width/depth is conventional and is not a paper innovation. It is an
accuracy-oriented capacity study supporting model-size selection. It assumes
the W64 optimizer schedule remains adequate and 80 epochs are sufficient for
the larger networks; those assumptions are tested, not silently tuned.
