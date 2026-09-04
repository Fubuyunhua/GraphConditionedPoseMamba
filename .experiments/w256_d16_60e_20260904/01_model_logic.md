# W256/D16 model logic

W256/D10 tests wide per-block capacity but may lack iterative graph/SSM
refinement depth. W256/D16 keeps width and every mechanism fixed while adding
six blocks, increasing capacity from 12.65M to 20.19M parameters.

This is conventional depth scaling, not a new paper mechanism. The observable
prediction is lower best-EMA P1 than the W64/D8 Full reference under the
registered 60-epoch budget. A valid result that fails to improve P1 rejects
this depth increase as an accuracy-efficient choice.

## R3 optimization logic

R1 and R2 show that increasing width to 256 changes the trainable optimization
regime: warmup delayed but did not eliminate a finite whole-model parameter
excursion. R3 therefore keeps the 20.19M architecture fixed and changes only
the optimizer schedule and intended SSM decay grouping. The falsifiable claim
is that lower peak LR plus smooth decay prevents large one-epoch parameter
movement while retaining enough capacity to improve the stable W128 curve.
