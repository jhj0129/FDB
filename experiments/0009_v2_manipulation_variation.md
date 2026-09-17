# Experiment 0009: manipulation variation suite

- Status: completed
- Scenarios: 6
- Varied: box size, density, friction, source XY, target XY.
- Acceptance: ≥80% first-attempt success and ≤0.05 m final XY error.
- Raw results: `0009_v2_manipulation_variation.results.json`
- Episode: `episode_20260917T062218862297Z_e44594ea`

All six paired scenarios succeeded on the first selected full candidate future. Mean
placement error was 0.00914 m and maximum error was 0.02225 m. This validates the
grasp-lift and pick-place skills only over the explicit six-scenario envelope. It does
not cover arbitrary objects, perception error, obstacles during transfer, or hardware.

