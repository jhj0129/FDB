# Experiment 0006: collision-aware pre-grasp selection

- Status: completed baseline
- Obstacle: sphere centered at `[0.45, 0.15, 0.55]` m, radius 0.06 m.
- Candidates: direct center, centered high, and side high.
- Gate: stable, endpoint error ≤0.02 m, and zero obstacle-contact steps.
- Episode: `episode_20260917T061048103139Z_2c33c4f9`

Direct center contacted the obstacle for 1460 steps and stopped 0.1071 m from its target.
Centered high reached within 0.00543 m but still contacted the obstacle for 12 steps, so
it was rejected. Side high had zero contact steps and 0.00738 m endpoint error and was
selected. Safety feasibility is therefore evaluated before score. The result is a
single-scene baseline; grasp contact and broader obstacle layouts remain unvalidated.

