# Model logic

Use existing 0.8M mechanism as the architectural base: anatomical topology encodes context, context controls Delta/B/C, frame/joint trajectories have independent state. Current seed0 Rewired/no-reset results support retaining this mechanism.

M W128/D20 explores a medium-width deep model with previous partial evidence. L W256/D10 increases channel capacity with fewer layers; its previous run was cancelled before a completed epoch, so no accuracy claim exists. XL W256/D16 remains the already-authorized candidate, not a duplicate study. These are capacity/recipe improvement experiments, not clean causal capacity ablations or new algorithm contributions.

True2D input removes detector-coordinate errors and sets confidence1, training each model afresh for that distribution. Expected lower error is a hypothesis; known-input xy passthrough affects interpretation. GT2D does not prove robustness to detected2D. More parameters do not guarantee higher accuracy; R3 overfitting motivates dp0.30 plus its validated optimizer, but the joint recipe is not asserted optimal.
