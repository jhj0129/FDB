# Audit 0007: collision gate precedes score

- **Decision:** collision-free validity is lexicographically prior to endpoint score.
- **Evidence:** the centered-high candidate had smaller endpoint error than the selected
  candidate but made contact for 12 steps and was rejected.
- **Scope:** obstacle contacts are forbidden during pre-grasp motion. Future grasp
  closure requires a separate allowed-contact policy.
- **Promotion:** the skill remains a candidate until multiple obstacle layouts pass.

