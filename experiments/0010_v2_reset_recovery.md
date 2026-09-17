# Experiment 0010: reset-based missed-grasp recovery

- Status: completed sandbox baseline
- Injected failure: high grasp pose.
- Detection: lift -0.00364 m, object-to-hand distance 0.25947 m, and table recontact.
- Diagnosis: `missed_grasp`, confidence 0.95.
- Recovery: reset identical sandbox and retry the evidence-backed low grasp.
- Outcome: recovered; retry lifted 0.13630 m and retained the object.
- Episode: `episode_20260917T062511240470Z_d03b9c29`

This proves objective failure classification and strategy switching after a sandbox reset.
It does not prove recovery in the same evolving physical state; that limitation is
encoded in the result and skill status.

