# Experiment 0007: Panda grasp and lift baseline

- Status: completed baseline
- Task: pre-grasp, descend, close, and lift a 44×44×50 mm box.
- Candidates: hand grasp heights 0.375, 0.385, and 0.395 m.
- Acceptance: lift ≥0.08 m, object-to-hand distance ≤0.16 m, stable, and no table
  recontact after the object has separated by 0.02 m.
- Failure episode: `episode_20260917T061444936705Z_2bbc8a96`
- Corrected episode: `episode_20260917T061516003558Z_30544a4f`

Initial work exposed two issues: an added free joint was zeroed by a keyframe created
before that joint existed, and a sphere slipped from the parallel gripper. The scene now
preserves the object's `qpos0` and copies only the original robot home coordinates; a
box with explicit friction is used for the baseline. The first evaluator also counted
normal table contact at lift onset as forbidden. The corrected metric starts checking
for recontact only after the object rises 0.02 m.

Two height candidates succeeded. The selected low candidate lifted the object 0.1363 m,
retained it 0.0984 m from the hand origin, and had zero forbidden recontact steps. The
high candidate failed to lift. This skill remains a candidate pending object variation
and place/release tests.

