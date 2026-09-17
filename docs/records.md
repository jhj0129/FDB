# Repository record formats

Canonical JSON Schemas live in `schemas/`. Markdown can accompany a JSON record for
human explanation, but automation should rely on the validated JSON fields.

## Episode

One run of a task or simulation. Episodes are append-only and capture observation,
goal, candidates, decision, execution, objective/user evaluation, sources, skills,
reflection, confidence, and unresolved questions.

## Skill

A reusable behavior with version, status, prerequisites, inputs, outputs, procedure,
objective acceptance criteria, known limits, evidence, and deprecation metadata.
Schema validity does not imply that promotion gates have passed.

## Research log

A sourced investigation. It distinguishes claims made by sources, FDB interpretation,
cross-check results, generated hypotheses, validation protocol, result, confidence,
and open questions. Every externally obtained claim needs a URL or durable citation.

## Robot self-model

Confirmed and inferred morphology are separate. Joint limits and dimensions should
identify their source (URDF, MJCF, manual, sensor, or experiment). Capability claims
must point to evidence and state whether they are simulated or physically verified.

## Experiment

Experiment records should use the episode schema for individual executions and group
them with a human-readable protocol under `experiments/`. A group records hypothesis,
independent variables, controls, metrics, acceptance criteria, episode IDs, conclusion,
and next experiment.

