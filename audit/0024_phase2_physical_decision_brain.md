# 감사 0024 — Phase 2 물리 Decision Brain

Phase 1 논리 환경을 삭제하지 않고 별도 물리 Runtime을 추가했다. 새 환경은 한 MuJoCo
model/data 안에 Panda와 4개 물체·4개 수용구를 유지한다. 원자 Skill 순서는 Task에 없고,
전제조건을 통과한 후보만 예측·안전 검사 후 실행한다.

카메라 RGB 경로는 사용하지만 identity mask는 simulator segmentation이다. 이는 실제
vision 일반화를 증명하지 않는다. Candidate generation과 subgoal은 deterministic rule,
controller는 IK+PD, safety/evaluation은 deterministic이다. CNN과 Memory ablation에서
성공 차이가 없었으므로 학습형 planning 또는 memory-driven improvement로 승격하지 않는다.

seed 1의 2mm perturbation에서 triangle 실패도 보존했다.
