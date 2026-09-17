# FDB — Fly Decision Brain

FDB is a research project for a connectome-inspired decision system. Its purpose is
not to replace conventional robot control, inverse kinematics, or motion planning.
FDB is the layer that integrates context and memory, selects actions, identifies
missing knowledge, compares candidate strategies in simulation, and improves later
decisions from recorded evidence.

## Research question

Can a connectome-inspired decision system autonomously acquire external knowledge,
understand unfamiliar robot bodies, construct reusable skills, simulate candidate
actions, evaluate its behavior, and improve future decisions from experience?

## Principles

1. Reuse established knowledge, algorithms, models, manuals, and code rather than
   relearning everything from scratch.
2. Treat retrieved information as a hypothesis: interpret, cross-check, simulate,
   evaluate, and only then register it as a reusable skill.
3. Prefer confirmed robot descriptions and sensor data over guessed dimensions or
   limits.
4. Validate in a sandbox or simulator before acting on a real robot.
5. Preserve raw evidence. Summarize for active use, promote repeated findings, and
   deprecate incorrect knowledge with an audit trail rather than silently deleting it.
6. Keep objective task metrics separate from learned or user-preference critics.
7. Optimize first-attempt performance by combining prior knowledge, memory, candidate
   futures, and simulation.

## Current milestone: FDB v1

v0 implements a deterministic 2D decision loop without physical interaction:

```text
Task -> Goal -> Candidate Plans -> Internal Simulation
     -> Objective Evaluation -> Action Selection -> Execution -> Episode Memory
```

The first task is: “move an object of a specified color to a target zone of a
specified color.” The code changes symbolic object state, evaluates multiple plans,
selects the best plan, and writes a structured, append-only episode. This narrow scope
is intentional: later versions can replace the parser, world model, planner, simulator,
critic, and executor independently.

v1 now adds the first MuJoCo physics task. FDB generates three open-loop force
candidates, runs each from a fresh identical model, selects using objective success and
quality metrics, repeats the selected plan as execution, and records prediction versus
actual outcome. This is a planar push experiment, not yet a robot manipulation system.
The robustness suite also compares the nominal open-loop plan with bounded PD feedback
across 30 paired randomized scenarios. Feedback achieved 100% first-attempt success
versus 80% for open-loop and is now the scoped default under dynamics uncertainty.

v2 has begun with a pinned, Apache-2.0 Franka Panda model from MuJoCo Menagerie. FDB
derives a robot self-model from compiled MJCF data and compares joint-limit-aware DLS
reach candidates using MuJoCo body Jacobians. The first position-only reach baseline
executes within 5.3 mm of its target; generalized reach and grasp remain unvalidated.
The follow-up workspace suite reached 12/12 declared targets with a maximum error below
9.7 mm, validating position-only reach within that explicit envelope.
The first 6D pose-reach baseline also preserves hand orientation within 0.0122 rad while
reaching within 5.4 mm, but remains a candidate until multi-pose and obstacle tests pass.
Collision-aware pre-grasp selection now rejects candidates with any obstacle contact
before comparing endpoint score; the first safe candidate reached within 7.4 mm with
zero contact steps.
The first complete `approach → close → lift` baseline raises a box by 0.136 m and retains
it in the gripper. It is intentionally still a candidate pending place/release and object
variation tests.
The first full pick-and-place loop now lifts, transfers, places, releases, and retreats;
its selected candidate finishes 5.6 mm from the target center.

## Quick start

Python 3.10 or newer is required. Runtime code has no third-party dependencies.

From a fresh checkout, install the package and run the example:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m fdb.cli \
  --task "빨간 물체를 파란 영역으로 이동" \
  --world examples/v0_world.json
```

For development tests:

```bash
python -m pip install pytest
pytest
```

Install and run the optional v1 physics environment:

```bash
python -m pip install -e '.[physics]'
python -m fdb.v1.cli --task "빨간 물체를 파란 목표 구역으로 밀어라"
python -m fdb.v1.robustness_cli \
  --output experiments/local_robustness.results.json
python -m pip install -e '.[robot]'
python -m fdb.v2.cli
python -m fdb.v2.reach_suite_cli
python -m fdb.v2.pose_reach_cli
python -m fdb.v2.pregrasp_cli
python -m fdb.v2.grasp_lift_cli
python -m fdb.v2.pick_place_cli
```

Each successful CLI run creates an `episodes/episode_*.json` file. Episode records
include observations, candidate plans, predicted scores, the selected plan, execution
result, reflection, confidence, and unresolved questions. Generated episodes are
research data and may be committed intentionally; they are not automatically promoted
to long-term memory or skills.

## Repository map

| Path | Role |
| --- | --- |
| `src/fdb/` | Decision-system implementation |
| `memory/current/` | Bounded working memory for the active task |
| `memory/long_term/` | Repeatedly validated, generalizable knowledge |
| `memory/summaries/` | Evidence-linked compression of old episodes |
| `memory/archive/` | Inactive but preserved records |
| `episodes/` | Append-only task, simulation, action, and outcome records |
| `skills/` | Validated reusable capabilities with evidence and limits |
| `research/` | Sourced research notes, comparisons, hypotheses, and validation |
| `robots/` | Per-robot self-models and capability evidence |
| `experiments/` | Hypotheses, protocols, metrics, results, and conclusions |
| `state/` | Current goal, hypotheses, open questions, and active stage |
| `audit/` | Reasons for promotions, changes, deprecations, archives, and deletion |
| `schemas/` | Machine-readable record contracts |
| `policies/` | Lifecycle and governance rules |

See [Memory architecture](docs/memory_architecture.md),
[Memory policy](policies/memory_policy.json), and [Repository records](docs/records.md).

## Roadmap

- v1: MuJoCo-based push, move, and rotate interactions (planar push foundation active).
- v2: fixed-robot reach, grasp, lift, place, and release skills.
- v3: unfamiliar URDF/MJCF/mesh/manual morphology analysis.
- v4: simulation-based body-schema and capability discovery.
- v5: autonomous external research loops.
- v6: evidence-based reusable skill creation.
- v7–v8: user preference data, self-critic, and reward modeling while preserving
  separate objective metrics.
- v9: multi-candidate internal futures before execution.
- v10: autonomous research, experiment, evaluation, and improvement loops for
  high-level goals.

## Safety and scope

v0 is symbolic and must not be treated as a controller for physical hardware. A skill
is not considered safe for real-world execution merely because its schema is valid or
because a symbolic test passed. Promotion gates and scope limitations are defined in
the memory policy.
