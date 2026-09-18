# FDB Jetson 사전작업 Phase 2 보고서

## 1. 목적과 Architecture

Phase 2는 monolithic `shape_insert`를 보존하면서 하나의 MuJoCo 물리 세계에서 현재 상태에
따라 원자 Skill을 조합하는 Decision Brain을 검증했다.

```text
Camera RGB + simulator segmentation → Agent Observation
Goal → subgoal(rule) → applicable skills(rule) → World Model(mixed)
→ deterministic Safety → execute Panda IK/PD → reobserve
→ objective evaluation → failure-specific replan → task Episode
```

## 2. Physical Persistent World

`PhysicalPersistentShapeEnvironment`는 Panda, table, square/circle/triangle/rectangle,
네 수용구를 하나의 `model/data`에 생성한다. Goal 사이 reset은 없으며 공식 실행의
`reset_count=1`, 최종 `sequence=24`다. object identity, pose, orientation, velocity,
grasped/inserted/released, target, confidence, last_seen을 Observation에 둔다.

## 3. Observation Pipeline과 Goal

Agent 경로는 headless MuJoCo camera RGB → 색상 contour → simulator geom segmentation으로
identity 분리 → 고정 camera calibration으로 XY pose → CNN pair prediction이다. Evaluator는
별도의 MuJoCo body pose를 사용한다. Raw RGB와 evaluator ground truth는 의사결정 입력과
구분한다. simulator segmentation은 실제 카메라에서 제공되지 않는 중요한 한계다.

Goal은 `object_id`, `target_id`, `inside/released` 상태만 제공하며 world를 생성하거나
수정하지 않는다. World가 먼저 생성되고 그 뒤 Goal을 바꾼다.

## 4. Skill Architecture와 Preconditions/Effects

| Skill | 핵심 precondition | effect / 실패 |
| --- | --- | --- |
| observe_scene | object/target not visible | visible / perception failure |
| reach_object | visible, not grasped | ee_near / IK, collision |
| grasp_object | ee_near, open | grasped / grasp failure, slip |
| lift_object | grasped | lifted / slip, collision |
| align_object | grasped, lifted, yaw error | aligned plan / alignment failure |
| recover_alignment | alignment failure, object retained | alternative aligned plan |
| move_to_target | grasped, lifted, aligned | near_target / IK, slip |
| insert_object | near target, aligned, grasped | inserted / alignment, collision |
| release_object | inserted, grasped | released, not grasped |
| retreat | inserted, released | retreated / IK, collision |

`align_object`는 충돌을 줄이기 위한 orientation 계획 Skill이고 실제 회전은 높은 위치의
`move_to_target` 제어에서 적용된다. 각 실행 전 환경도 precondition을 다시 검사하므로
고정 script가 실패 이후 잘못된 후속 동작을 강행할 수 없다.

## 5. Candidate Generation과 Decision

Runtime에 형상별 순서는 없다. 모든 Skill predicate를 현재 Observation에 평가하고,
통과 후보와 거부 이유를 함께 만든다. 예측 성공률 최대·불확실성 최소 후보를 고른 뒤
Safety를 통과해야 실행한다. 현재 subgoal과 후보 생성은 사람이 설계한 state-machine이며
learned planning이 아니다.

## 6. World Model과 Prediction 비교

`AtomicSkillWorldModel`은 선언된 effect를 예측한다. 정렬 후보에서 CNN이 fit을 인식하면
CNN 회전 예측으로 후보 점수를 조정하고, 그렇지 않으면 deterministic geometry fallback을
사용한다. 실행 뒤 predicted effect/actual effect mismatch, predicted/actual success,
predicted/actual collision을 매 step에 저장한다.

공식 run의 26회 prediction 중 neural scoring은 1회, deterministic fallback은 5회였다.
physics fallback은 0회다. Neural이 성공의 주원인이라고 볼 수 없다.

## 7. Safety와 Failure Replanning

Safety는 모델과 독립적으로 entity/pose 유효성, finite 값, Panda 작업영역, 최소 table
clearance, predicted collision을 검사한다. 실행 중 robot-table contact도 계측하며 1회라도
있으면 `COLLISION` 실패다.

- Triangle: `ALIGNMENT_FAILURE` 뒤 grasp/lift 상태를 보존하고 `recover_alignment`
- Circle: `GRASP_FAILURE` 뒤 재관찰하고 `reach_object → grasp_object`
- Slip: grasp/lift/near 상태를 취소하고 현재 물체 위치에서 다시 reach

Task 전체 reset이나 이미 성공한 Goal reset은 하지 않는다.

## 8. Episode Trace와 Replay

Task마다 Episode 하나를 만들고 `decision_steps` 안에 observation, subgoal, 후보·거부
이유, prediction, safety, 선택 이유, execution, actual next state, prediction error를 저장한다.
seed, 초기 object/target pose, Goal, 선택 action, model version, compute device도 기록한다.
같은 소프트웨어와 MuJoCo에서는 재현 가능하지만 GPU renderer와 floating-point 차이까지
bit-exact replay를 보장하지 않는다.

## 9. Persistent Multi-goal과 Perturbation 결과

공식 seed 0은 triangle→circle→rectangle을 한 world에서 3/3 성공했다. 24 actions,
2 replans, 약 24.32초, robot-table contact 0이다. 의도적 orientation 및 grasp 실패를 모두
복구했다.

작은 robustness N=2에서 seed 0은 3/3, seed 1의 object/target XY ±2mm 및 순서
rectangle→square→triangle은 2/3이었다. triangle은 16 actions·4 replans 뒤 timeout이다.
따라서 변경 pose 일반화는 미완료다.

## 10. Legacy / Scripted / Runtime

| 방식 | Nominal | 실패 주입 | 특징 |
| --- | ---: | ---: | --- |
| Legacy monolithic | 4/4 fresh-state | 기존 단일 복구 실험 | 물리 성능, 비지속 |
| Scripted atomic | 3/3 persistent | 0/3 | 고정 순서, replan 없음 |
| FDBRuntime atomic | 3/3 persistent | 3/3 | 매 step 재관찰·실패별 분기 |

Runtime의 장점은 Skill을 쪼갠 것 자체가 아니라 실패 뒤 현재 상태에서 순서를 바꾼 것이다.

## 11. Neural / Memory Ablation

| 구성 | 성공 | actions | replans | 시간 |
| --- | ---: | ---: | ---: | ---: |
| Neural+rules, Memory ON | 3/3 | 24 | 2 | 24.32초 |
| Rules only, Memory ON | 3/3 | 24 | 2 | 24.66초 |
| Neural+rules, Memory OFF | 3/3 | 24 | 2 | 25.30초 |

Neural과 Memory 모두 이 표본에서 성공률이나 planning efficiency를 높이지 않았다. Memory는
23건을 조회했지만 후보 점수에 사용하지 않는다. 온라인 학습도 수행하지 않는다.

## 12. Neural Brain의 실제 역할

| Component | 실제 방식 | Device |
| --- | --- | --- |
| Perception | deterministic color/contour + simulator segmentation + shape-fit CNN | CPU |
| World Model | deterministic skill effects, 일부 CNN alignment score | CPU |
| Candidate generation | deterministic rules | CPU |
| Skill selection | predicted score argmax | CPU |
| Safety | deterministic hard gate | CPU |
| Control | Panda IK + smooth PD trajectory | CPU |
| Evaluation | MuJoCo ground truth objective checks | CPU |
| Memory | Episode 검색·기록, 정책 영향 없음 | CPU |

## 13. Test와 Profile

Fast test는 atomic precondition/effect, candidate, subgoal, failure path, Goal leakage를
포함한다. Integration test는 실제 4-object world와 세 Goal·두 failure recovery를 약
29초에 검사한다. Heavy simulation은 CI fast 목록에서 제외했다.

## 14. 알려진 한계와 Jetson 이전

- simulator segmentation을 실제 camera detector/tracker로 교체해야 한다.
- seed 1 triangle pose perturbation 실패 원인을 교정해야 한다.
- CNN은 일부 target contour를 unknown/오분류하며 neural 기여가 작다.
- 실제 physics candidate clone/fallback이 아직 없다.
- Memory retrieval이 decision value에 연결되지 않았다.
- force/torque sensor, velocity/acceleration limit, 실제 emergency stop이 없다.
- 실제 Panda/RealSense, TensorRT, Jetson 최적화는 하지 않았다.

### A. FDB가 실제로 스스로 결정하는 부분

현재 Observation에서 실행 가능한 Skill을 다시 계산하고, 실패 종류와 보존된 상태에 따라
reach 재시도 또는 alignment recovery를 선택하며, 안전한 후보 중 예측 점수가 높은 것을
고른다.

### B. 사람이 미리 설계한 부분

Skill 종류, precondition/effect, subgoal rule, candidate rule, hard safety, IK/PD controller,
색상과 simulator identity 경계를 사람이 설계했다. 이는 학습형 범용 planner가 아니다.

### C. 지금 Jetson으로 이전해도 되는가

headless CPU 실행 경계와 CLI는 이전 가능하지만 전체 시스템 이전 준비 완료는 아니다.
Phase 3 전에 camera-only tracking, randomized pose robustness, physics fallback, Memory의 실제
의사결정 연결을 반드시 해결해야 한다.
