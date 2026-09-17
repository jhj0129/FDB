# Audit 0013: motion-probe promotion

- **Change:** promoted bounded joint motion probing to a validated skill for the tested
  open-chain serial-arm scope.
- **Evidence:** 20/20 actuator-driven hinge probes across Panda, UR5e, and KUKA iiwa
  moved exactly the bodies predicted by the compiled kinematic tree.
- **Boundary:** this does not establish validity for mobile bases, coupled or closed
  chains, soft bodies, or real hardware.
- **Related promotion:** the v3 morphology inspector now has behavioral evidence for
  its body-tree output within the same declared scope.

