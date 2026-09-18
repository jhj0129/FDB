# 실험 0045 — 실행 전 물리 안전 preview

Phase 2 최초 공식 run 뒤 Safety Gate에 각 motion candidate의 IK, Panda joint range,
보수적 joint velocity·acceleration, finite trajectory 검사를 추가했다. 이는 실행 전에
수행되며 World Model 점수로 우회할 수 없다. 실행 중 robot-table contact 0 조건도 유지한다.

같은 seed 0, triangle alignment failure와 circle grasp failure 조건을 새 evidence 경로에서
다시 실행했다.

- 단일 world, reset 1
- 3/3 성공
- actions 24, replans 2
- safety rejection 0
- 총 28.02초

`artifacts/phase2_physical_safety_seed0/summary.json`에 결과를 보존했다. rejection 0은
Gate가 없었다는 뜻이 아니라 모든 선택 후보가 새 IK/kinematic 한계를 통과했다는 뜻이다.
