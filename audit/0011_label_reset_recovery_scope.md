# Audit 0011: label recovery scope explicitly

- **Capability:** objective metrics diagnose an injected missed grasp and select a
  validated alternative that succeeds after reset.
- **Boundary:** every result records `reset_based=true`; this must not be presented as
  online or real-world recovery.
- **Promotion:** candidate until object pose is observed and recovery executes in the
  same evolving world state.

