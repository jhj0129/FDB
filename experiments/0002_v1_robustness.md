# Experiment 0002: open-loop versus feedback robustness

- Status: completed
- Hypothesis: feedback control improves first-attempt success when mass, friction, and
  initial position differ from the nominal model.
- Seed: 1701
- Trials: 30 paired scenarios
- Variation: mass scale 0.4–1.8, sliding friction 0.2–1.4, initial x/y offsets ±0.15 m.
- Controls: every method receives the same scenario, target, timestep, success radius,
  and safety evaluator.
- Acceptance: feedback success rate ≥95% and greater than open-loop.
- Raw results: `0002_v1_robustness.results.json`
- Episode: `episode_20260917T055606642605Z_dff95e12`

## Result

| Method | Success | Rate | Mean final error | Mean score |
| --- | ---: | ---: | ---: | ---: |
| Nominal open-loop plan | 24/30 | 80% | 0.07587 m | 72.46 |
| PD feedback | 30/30 | 100% | 0.03833 m | 93.36 |

Feedback passed the acceptance gate. It improved success by 20 percentage points and
reduced mean final error by about 49.5%. The paired deterministic suite is reproducible
from the saved seed and complete per-scenario results.

## Decision

Use feedback as the default v1 strategy when dynamics or initial state are uncertain.
Keep open-loop force pulses as a baseline and as a candidate only for known deterministic
models. This conclusion is limited to simulation and does not authorize robot hardware.

## Next experiment

Test observation noise and delay later. The immediate milestone is to separate rotation
from translation, then reuse the validated feedback and candidate-evaluation pattern for
a fixed robot reach task.

