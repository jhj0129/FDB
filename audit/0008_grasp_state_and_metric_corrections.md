# Audit 0008: grasp state and metric corrections

- **State correction:** added free-joint coordinates retain model `qpos0`; only original
  Panda coordinates are copied from the older home keyframe.
- **Object correction:** the slipping sphere was replaced by a sized, friction-specified
  box for the first parallel-jaw baseline.
- **Metric correction:** table contact is normal until lift separation; only recontact
  after 0.02 m separation is forbidden.
- **Outcome:** selected candidate lifted 0.1363 m and retained the object without
  forbidden recontact.
- **Promotion:** candidate only; object variation and place/release remain required.

