# Audit 0006: pose-error frame correction

- **Failure:** 6D IK diverged because local-frame quaternion error was combined with a
  world-frame MuJoCo angular Jacobian.
- **Correction:** orientation error is now expressed in world coordinates using current
  and desired rotation matrices before DLS.
- **Evidence:** all three candidates converged after correction; regression test asserts
  position, orientation, and limit criteria.
- **Promotion:** pose reach remains a candidate because only one pose and no obstacle
  scene have been tested.

