# Experiment 0012: simulation body-schema motion probe

- Status: completed and validated within the tested serial-arm scope.
- Models: Franka Panda, Universal Robots UR5e, KUKA iiwa 14.
- Intervention: reset the compiled model, perturb each actuator-driven hinge by
  0.02 rad within its declared limit, and compare all body translations and rotations.
- Result: 20/20 probes matched the inferred downstream body tree; no unexpected body
  moved.
- Outputs: `robots/*/motion_probe.json`.
- Episode: `episode_20260917T063201276929Z_622910e5`.

This closes the behavioral-verification gate left open by experiment 0011. It validates
the structural body-schema inference for the three tested open-chain serial arms, not
for mobile bases, closed chains, coupled mechanisms, or deformable bodies.

