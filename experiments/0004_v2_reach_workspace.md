# Experiment 0004: Panda workspace reach suite

- Status: completed
- Targets: 12 Cartesian positions spanning x=0.35–0.65 m, y=-0.20–0.20 m,
  z=0.45–0.75 m.
- Method: compare three DLS damping candidates independently for every target, then
  execute the selected joint target with the model's position actuators.
- Acceptance: ≥95% success, ≤0.02 m execution error, zero joint-limit violations.
- Raw results: `0004_v2_reach_workspace.results.json`
- Episode: `episode_20260917T060344507798Z_cdd2f48e`

All 12 targets succeeded on the first selected execution. Mean endpoint error was
0.00635 m and maximum error was 0.00968 m. There were no joint-limit violations or
contacts. This validates position-only reach within the declared target envelope, not
orientation control, obstacle avoidance, grasping, or hardware execution.

