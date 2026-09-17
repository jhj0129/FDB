# Experiment 0008: Panda pick and place baseline

- Status: completed baseline
- Source: `[0.45, 0.15, 0.335]` m
- Target: `[0.58, -0.12, 0.335]` m
- Candidates: place hand z at 0.365, 0.375, and 0.385 m.
- Acceptance: pick lift ≥0.08 m, final XY error ≤0.05 m, final object height within
  0.02 m, stable simulation, and open gripper after release.
- Episode: `episode_20260917T061758856095Z_795c019e`

All three candidate futures completed the task. The selected high candidate lifted the
object 0.14665 m and released it 0.00558 m from the target center at stable table height.
This completes the first reach–grasp–lift–transfer–place–release loop, but it is not yet
generalized beyond one object and scene.

