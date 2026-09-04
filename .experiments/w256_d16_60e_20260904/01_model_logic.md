# W256/D16 model logic

W256/D10 tests wide per-block capacity but may lack iterative graph/SSM
refinement depth. W256/D16 keeps width and every mechanism fixed while adding
six blocks, increasing capacity from 12.65M to 20.19M parameters.

This is conventional depth scaling, not a new paper mechanism. The observable
prediction is lower best-EMA P1 than the W64/D8 Full reference under the
registered 60-epoch budget. A valid result that fails to improve P1 rejects
this depth increase as an accuracy-efficient choice.
