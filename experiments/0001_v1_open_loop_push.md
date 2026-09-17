# Experiment 0001: v1 open-loop planar push

- Status: completed
- Hypothesis: a small set of force/duration candidates can move a planar MuJoCo puck
  into a target region without feedback.
- Baseline: no force; the puck remains at its initial position and does not complete the
  task.
- Independent variables: force magnitude and force duration.
- Controls: MJCF, initial pose, target pose, timestep, settling duration, and evaluator.
- Objective metrics: target completion, final distance, path length, execution time,
  unintended contacts, boundary exit, and numerical stability.
- Acceptance: distance ≤ 0.14 m, zero unintended contacts, within bounds, stable.
- Episode: `episode_20260917T055041686817Z_255634e7`.

## Iterations

1. A free rectangular body drifted and rotated, so all initial candidates missed. This
   mixed translation and rotation before either behavior had a stable baseline.
2. An incorrect torque-first interpretation of `xfrc_applied` produced extreme rolling.
   Official engine source showed the layout is force first, then torque.
3. The experiment was narrowed to a planar cylindrical puck with x/y translation and
   yaw joints. Candidate magnitudes were retuned without changing the evaluator.

## Result

Only the balanced candidate succeeded. It applied 4 N for 120 steps, then settled for
500 steps. Prediction and independent execution both ended 0.04499 m from the target,
with a 1.25329 m path, 1.24 s simulation time, no unintended contacts, no boundary exit,
and stable numerical state.

## Conclusion

The hypothesis is supported for this deterministic model only. The skill remains a
candidate because the promotion policy requires more independent successes and scope
testing. Next, randomize mass and friction and compare open-loop selection with feedback.

