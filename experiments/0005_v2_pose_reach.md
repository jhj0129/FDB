# Experiment 0005: Panda 6D pose reach baseline

- Status: completed baseline
- Goal: reach `[0.45, 0.15, 0.55]` m while preserving the home hand orientation.
- Candidates: DLS damping 0.01, 0.05, 0.15.
- Acceptance: position error ≤0.02 m, orientation error ≤0.03 rad, no limit violation.
- Episode: `episode_20260917T060718354486Z_9a476163`

The initial implementation fed a local quaternion error to a world-frame angular
Jacobian and failed to converge. It was corrected by computing orientation error from
current and desired rotation-matrix columns in world coordinates. All three corrected
candidates converged. The selected high-damping result achieved 0.00532 m position error
and 0.01216 rad orientation error with no joint-limit violation. This remains a
single-pose candidate skill pending workspace and obstacle validation.

