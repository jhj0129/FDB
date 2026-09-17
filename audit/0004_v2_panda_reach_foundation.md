# Audit 0004: Panda reach foundation

- **Model choice:** Franka Panda from MuJoCo Menagerie 2026.9.0, Apache-2.0,
  object ID recorded in the self-model.
- **Reason:** maintained provenance, arm and gripper in one model, explicit actuators,
  and suitable complexity for the first fixed-robot manipulation milestone.
- **No guessed morphology:** bodies, joints, limits, actuators, and end-effector evidence
  are read from the compiled model.
- **Algorithm:** standard damped least squares using MuJoCo's body Jacobian; joint limits
  come from the model.
- **Promotion decision:** reach remains a candidate because only one target is tested.
- **Safety:** simulation only; no hardware authorization.

