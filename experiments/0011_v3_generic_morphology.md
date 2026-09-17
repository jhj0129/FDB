# Experiment 0011: generic morphology inspection

- Status: completed baseline
- Models: Franka Panda, Universal Robots UR5e, KUKA iiwa 14.
- Sources: pinned MuJoCo Menagerie artifacts with recorded object IDs and licenses.
- Extracted: body tree, masses, joints, axes, limits, actuators, transmissions, control
  ranges, sites, leaf bodies, gripper candidates, and capability hypotheses.
- Episodes: `episode_20260917T062803472615Z_5cb37749` and corrected
  `episode_20260917T062823086346Z_652207a5`.

One implementation analyzed all three compiled models. UR5e and iiwa expose terminal
`attachment_site` sites and received 0.95 end-effector confidence. Panda has no site in
its default model; its first pass returned two finger leaves. The corrected generic
heuristic detects finger joints and promotes their common parent (`hand`) with 0.9
confidence while retaining the lower-confidence leaves. Morphology parsing is not yet a
validated skill because behavioral joint-motion verification and more morphology types
are required.

