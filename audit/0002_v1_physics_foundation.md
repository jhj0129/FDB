# Audit 0002: MuJoCo physics foundation

- **Change:** added an optional MuJoCo dependency, a packaged planar MJCF environment,
  fresh-state candidate rollouts, objective physics metrics, a v1 CLI, and episode
  persistence.
- **Reason:** move from symbolic state mutation to measurable physical interaction while
  retaining the v0 decision and memory boundaries.
- **Failure retained:** initial shape coupling and an incorrect wrench-layout assumption
  are documented in experiment 0001 rather than hidden.
- **Correction evidence:** official MuJoCo Python, simulation, and engine-source records
  are cited in the research log.
- **Safety boundary:** the model is planar, deterministic, and simulated; the skill is
  not approved for hardware.
- **Promotion decision:** `v1.physics_push` remains `candidate` until randomized trials
  and the minimum independent-success gate pass.

