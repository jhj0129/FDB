# Audit 0003: promote feedback-under-uncertainty rule

- **Promotion:** added a scoped long-term control rule and validated
  `v1.feedback_push`.
- **Evidence:** 30 paired randomized scenarios; feedback achieved 100% first-attempt
  success versus 80% for the nominal open-loop plan.
- **Policy check:** reproducible seed and raw results are saved, automated tests cover the
  comparison, scope and exceptions are explicit, the baseline was compared, and failure
  remains bounded to simulation.
- **Limit:** promotion applies only to the declared MuJoCo variation envelope and does
  not authorize physical hardware.

