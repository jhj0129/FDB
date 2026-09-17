# Experiment 0003: sourced Panda morphology and reach baseline

- Status: completed baseline
- Robot: Franka Emika Panda from MuJoCo Menagerie 2026.9.0
- Model object ID: `3d2262eeb81ecec19abfa31dd509e35abbb33e67`
- License: Apache-2.0
- Hypothesis: MuJoCo body Jacobians and joint-limit-clamped DLS can produce a valid
  fixed-arm reach without guessed morphology.
- Target: `[0.45, 0.15, 0.55]` m
- Candidates: damping 0.005, 0.03, and 0.15
- Acceptance: IK convergence, execution error ≤0.02 m, zero joint-limit violations.
- Episode: `episode_20260917T060011847583Z_d98e181c`

All three candidates converged. High damping was selected by the objective score because
its executed endpoint error was smallest: 0.00529 m. It required eight IK iterations,
had no joint-limit violations, and produced no contacts. The result establishes only a
single-target reach baseline; a workspace target suite is required before skill
validation.

