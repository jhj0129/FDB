# Audit 0001: initial FDB v0 architecture

- **Change:** initialized the external-memory directories, machine-readable policies
  and schemas, and a modular symbolic v0 decision loop.
- **Reason:** establish the smallest testable Task → Goal → Decision → Action →
  Evaluation loop before introducing physics, learned critics, retrieval, or robotics.
- **Alternatives considered:** a GUI-first demo and a MuJoCo-first prototype. Both add
  dependencies before the decision and memory contracts are proven.
- **Safety boundary:** v0 changes only an in-memory symbolic world and is not a robot
  controller.
- **Migration impact:** future simulators should implement the same candidate evaluation
  boundary; record revisions must remain backward-readable by schema version.
- **Verification:** four automated tests passed; Korean and English CLI tasks both
  selected the direct plan and reached the target without violations. The skill remains
  a `candidate` because it has not met the promotion policy's independent-evidence gate.

