# FDB Jetson 사전작업 Phase 2.5 보고서

## 1. 목적과 결론 요약

Phase 2.5는 MuJoCo의 segmentation ID와 exact body pose를 FDB 의사결정 경로에서 제거하고,
RGB-D 관측으로 만든 scene state만으로 persistent Panda world의 삽입 작업을 수행하는 단계다.

최종 retained benchmark는 118 task episode, 944 atomic action이며 118/118 성공, replan 0,
evaluator robot-table contact 0이었다. 이 수에는 설정 비교와 경계 재시험이 포함되므로 118개의
독립 분포 표본을 뜻하지 않는다. 전체 level 탐색은 seed 0, 최대 경계는 seed 0/1/2로 수행했다.
31개 persistent benchmark world 모두 마지막 goal 뒤에도 그 world에서 완료한 모든 goal을
objective tolerance 안에 보존했다.

핵심 결과는 다음과 같다.

- Oracle 3/3, Camera-only 3/3: 성공률 저하는 관측되지 않았다(n=3).
- 물체 XY ±20 mm, yaw ±40°, target XY ±10 mm까지 seed 0의 네 형상 20/20,
  20/20, 16/16 성공했다.
- 각 최대 경계를 seed 0/1/2로 재시험했고 shape별 3/3이었다.
- P10/Y20/T10 combined도 shape별 3/3이었다.
- Neural ON/OFF와 Memory ON/OFF는 모두 3/3, 24 actions, 0 replans로 행동 차이가 없었다.
- clean scene의 최초 camera 추정오차는 scene 8개 entity 평균 1.04 mm, 대칭을 고려한
  평균 yaw 절대오차 0.086°였다.

## 2. Architecture

```text
MuJoCoCameraSource(RGB, depth, timestamp, calibration)
  -> color mask / contour classification
  -> RGB-D unprojection / symmetry-aware yaw
  -> semantic + nearest-position temporal tracker
  -> get_agent_observation()
  -> FDBRuntime: candidate -> prediction -> safety -> selection
  -> relative object/target pose IK + PD execution
  -> camera re-observation / effect verification / reselection

get_evaluator_ground_truth()
  -> pose error, objective insertion/release, contact count (logging/scoring only)
```

`CameraSource` protocol의 출력은 RGB, depth, timestamp, intrinsics, camera-to-world transform이다.
상위 Runtime은 MuJoCo renderer를 알지 못하므로 이후 `RealSenseCameraSource`로 교체할 수 있다.

## 3. Privileged Information Audit

| 정보 | 분류 | Camera-only decision 사용 여부 |
|---|---|---:|
| RGB/depth frame, timestamp | Agent-visible | 사용 |
| fx/fy/cx/cy, image size, extrinsic | Agent-visible calibration | 사용 |
| 추정 class/pose/yaw/confidence/track age | Agent-visible derived state | 사용 |
| 명령한 gripper/arm state, robot model joint range | Agent-visible/proprioception 대응 | 사용 |
| MuJoCo body/geom ID 및 segmentation ID | Evaluator/debug-only | 사용 안 함 |
| exact object/target body pose | Evaluator-only | 사용 안 함 |
| exact grasp/insert logical state | Evaluator-only | 사용 안 함 |
| exact geom contact pair/count | Evaluator-only | 사용 안 함 |
| 렌더링용 model/data | sensor backend 내부 | camera frame 생성에만 사용 |

Camera mode의 `get_agent_observation()`은 RGB-D detector/tracker와 agent action belief만 읽는다.
`_execute_physics()`도 action에 포함된 estimated object/target pose를 요구하며 body pose fallback을
허용하지 않는다. exact contact는 camera decision result에서 제거하고 evaluator 누계에만 남긴다.
실기에서는 force/torque sensor가 추가되기 전까지 agent collision observation은 unavailable이다.

## 4. Camera Interface와 Calibration

`CameraIntrinsics`, `CameraCalibration`, `CameraFrame`, `CameraSource` protocol을 추가했다.
MuJoCo 구현은 512×512 top camera를 사용하며 fx/fy/cx/cy와 4×4 camera-to-world transform을
명시적으로 보유한다. 기존의 숨은 meters-per-pixel 환산은 camera-only 경로에서 사용하지 않는다.

## 5. Perception

Perception은 거대 vision model이 아니라 deterministic hybrid다.

- RGB: red object와 blue receptacle color mask
- contour: coarse/fine polygon vertex, aspect ratio, circularity로 4개 shape 분류
- depth: contour 표면 depth median과 calibration으로 3D unprojection
- yaw: square 90°, triangle 120°, rectangle 180° 대칭을 반영
- confidence: class/contour area 기반

Target wall의 pixel centroid를 table depth로 역투영하면 높이 차이 때문에 바깥쪽으로 최대 약
16 mm 편향되었다. blue surface의 median depth를 사용하자 초기 네 target XY error는 약 1 mm로
감소했다. 이는 ground-truth 보정값을 더한 것이 아니라 RGB-D 표면 깊이를 올바르게 사용한 결과다.

## 6. Tracking, Lost, Stale, Occlusion

Object는 semantic class와 0.35 m motion gate로 큰 commanded transfer를 같은 ID에 연결한다.
정적인 target은 0.05 m gate와 nearest existing track을 우선해 조명에 따른 일시적 class flip이
이웃 ID를 바꾸지 못하게 했다. confidence 0.60 미만의 partial detection은 좋은 last pose를
덮어쓰지 않는다. `missed_frames > 8`이면 stale이 되어 manipulation에 사용할 수 없다.

짧은 가림에서는 confidence와 freshness를 분리해 last observation을 유지한다. 장기 가림은
`OBJECT_LOST`, `TARGET_NOT_FOUND`, `LOW_PERCEPTION_CONFIDENCE` 또는 reobserve 경로로 간다.
이번 scene은 shape당 한 개이므로 실제 동일 외형 다중 객체의 `AMBIGUOUS_IDENTITY` 해결은 아직
검증하지 않았다.

## 7. Pose Estimation Error

Evaluator는 매 decision step의 estimated pose와 body truth를 비교해 Episode에
`translation_error_mm`, `yaw_error_deg`를 기록한다. 이 값은 candidate generation이나 skill
selection으로 되돌아가지 않는다.

- clean persistent scene 최초 관측: 평균 1.04 mm / 0.086°, 최대 1.14 mm / 0.292°
- Camera baseline의 episode 전체 scene 평균: triangle 1.66 mm/2.48°,
  circle 1.05 mm/0.30°, rectangle 1.33 mm/0.28°

이 값은 이미 삽입된 물체가 target을 가리는 persistent-world 후반 frame도 포함한다.
Circle yaw는 물리적으로 무의미하여 yaw aggregate에서 제외했다.

## 8. Closed-loop Visual Correction과 검증

Runtime은 매 atomic action 전후 camera observation을 새로 얻는다. reach/grasp/lift/align/
move/insert/release를 한 번에 script하지 않는다. lift는 추정 object z 상승, move는 object-target
XY, insert는 XY와 z를 camera state로 검증한다. 최종 success는 evaluator의 15 mm XY 및 25 mm z
조건과 released state를 동시에 만족해야 한다.

현재 reach, grasp, align, release 자체는 camera만으로 완전히 검증하지 않고 command/proprioceptive
effect belief를 사용한다. 특히 grasp는 lift 후 object z 변화에서 실질적으로 검증된다. 각 goal은
release에서 끝나지 않고 retreat까지 성공해야 완료되며, 이 조건으로 다음 goal의 arm motion이
기존 삽입물을 밀어내는 문제를 제거했다.

## 9. Randomization과 Success Envelope

각 표의 중간 level은 seed 0 pilot 1회/shape다. 최대 level은 seed 0/1/2에서 3회/shape로
재시험했다. 따라서 빈 칸 없는 100%를 모집단 일반화로 해석하면 안 된다.

### Object position

| Perturbation | Square | Circle | Triangle | Rectangle |
|---|---:|---:|---:|---:|
| 0 mm | 1/1 | 1/1 | 1/1 | 1/1 |
| ±2 mm | 1/1 | 1/1 | 1/1 | 1/1 |
| ±5 mm | 1/1 | 1/1 | 1/1 | 1/1 |
| ±10 mm | 1/1 | 1/1 | 1/1 | 1/1 |
| ±20 mm | 3/3 | 3/3 | 3/3 | 3/3 |

### Object yaw

| Yaw | Square | Circle | Triangle | Rectangle |
|---|---:|---:|---:|---:|
| 0° | 1/1 | 1/1 | 1/1 | 1/1 |
| ±5° | 1/1 | 1/1 | 1/1 | 1/1 |
| ±10° | 1/1 | 1/1 | 1/1 | 1/1 |
| ±20° | 1/1 | 1/1 | 1/1 | 1/1 |
| ±40° | 3/3 | 3/3 | 3/3 | 3/3 |

### Target position

| Target perturbation | Square | Circle | Triangle | Rectangle |
|---|---:|---:|---:|---:|
| 0 mm | 1/1 | 1/1 | 1/1 | 1/1 |
| ±2 mm | 1/1 | 1/1 | 1/1 | 1/1 |
| ±5 mm | 1/1 | 1/1 | 1/1 | 1/1 |
| ±10 mm | 3/3 | 3/3 | 3/3 | 3/3 |

### Combined

| Combined level | Square | Circle | Triangle | Rectangle |
|---|---:|---:|---:|---:|
| P2/Y5/T2 | 1/1 | 1/1 | 1/1 | 1/1 |
| P5/Y10/T5 | 1/1 | 1/1 | 1/1 | 1/1 |
| P10/Y20/T10 | 3/3 | 3/3 | 3/3 | 3/3 |

## 10. Shape-specific 결과와 Triangle Failure 분석

최대 네 경계에서 square/circle/triangle/rectangle 모두 12/12(4조건×3 seeds)였다.
Triangle의 경계 평균 pose error는 position max 2.58 mm/0.31°, yaw max 2.90 mm/0.32°,
target max 1.28 mm/0.17°, combined max 1.51 mm/0.19°였다.

Phase 2 seed 1 ±2 mm triangle trace는 insert clearance 실패가 아니라 먼저 `move_to_target`의
`OBJECT_SLIP`이 발생했고, 그 뒤 세 번의 `GRASP_FAILURE`로 recovery가 붕괴해 16-action timeout이
된 cascade였다. 당시 trace만으로 최초 slip의 힘학적 단일 원인을 확정할 force trace는 없다.

Phase 2.5에서 이 문제가 재현되지 않은 관측 가능한 차이는 다음과 같다.

- fixed pixel scale 대신 calibrated RGB-D pose를 사용했다.
- object/target estimated pose에 대한 relative action을 매 step 다시 생성했다.
- triangle vertex 기반 120° yaw estimator를 사용했다.
- move/lift/insert 후 camera effect verification을 수행했다.
- target class flip과 partial occlusion이 track ID/pose를 덮어쓰지 않도록 했다.

따라서 “tolerance를 넓혀 해결”한 것이 아니다. 다만 N=3 경계 시험이므로 triangle의 모든
grasp sensitivity가 해결됐다고 주장하지 않는다.

## 11. Oracle vs Camera

| Mode | 결과 | actions | replans | 평균 task time |
|---|---:|---:|---:|---:|
| Oracle | 3/3 | 8.0 | 0 | 3.41 s |
| Camera-only | 3/3 | 8.0 | 0 | 5.81 s |

Q1에 대한 답은 이 작은 baseline에서 성공률 차이는 0%p이고 camera가 평균 약 2.40 s 느렸다는
것이다. Q2는 end-to-end 실패가 0이라 perception-attributed failure 비율도 0/3이지만, 이는
perception cost/error가 없다는 뜻이 아니다. Camera trajectory에는 위 pose error와 latency가 있다.

## 12. Neural Ablation

P5/Y10/T5 camera 조건에서 Neural ON/OFF 모두 3/3, 24 actions, 0 replans였다.
Camera-only object state에는 기존 segmentation-contour CNN 입력을 넣지 않았으므로 neural call은
실제로 0회였다. Neural ON은 alignment마다 deterministic geometry fallback 3회를 사용했다.

따라서 Q6의 답은 **기여 없음**이다. 기존 shape-fit CNN을 억지로 camera-only 성공의 원인으로
표현하지 않는다. Physics candidate rollout도 0회다.

## 13. Memory Ablation

Memory ON/OFF 모두 3/3, 24 actions, 0 replans였다. 검색된 Episode가 candidate prior나 선택 순서를
바꾸는 연결이 없어 first attempt, actions, replans의 차이는 없었다. Q7의 답은 **측정 가능한
행동 개선 없음**이다. runtime latency 차이는 실행 순서와 host load가 섞인 wall-clock 값이라
Memory benefit으로 해석하지 않는다.

## 14. 실제 신경망 및 사람 설계 기여도

| Component | 실제 방식 |
|---|---|
| Perception | deterministic RGB-D color/contour geometry |
| Tracking | deterministic semantic + nearest association |
| Candidate generation | 사람이 설계한 rule/precondition/effect |
| World Model | neural option 존재, 이번 camera benchmark neural call 0 |
| Skill selection | predicted success/uncertainty 정렬, 현재 대부분 deterministic |
| Safety | deterministic workspace/IK/joint/velocity/acceleration gate |
| IK/control | numerical IK target + MuJoCo actuator/PD |
| Memory | Episode retrieval, 현재 decision prior 미연결 |
| Evaluation | deterministic privileged evaluator |

FDB가 스스로 하는 부분은 현재 관측으로 applicable skill을 다시 계산하고, confidence/staleness와
effect verification 결과에 따라 다음 skill을 고르는 것이다. 사람이 설계한 부분은 shape/color,
skill vocabulary, preconditions/effects, confidence threshold, tracking gate, safety limits, IK waypoint다.
이는 사람 수준 vision understanding이나 범용 learned planner가 아니다.

## 15. Failure Budget

최종 retained 118 episode/944 actions에서:

| 원인 | 횟수 |
|---|---:|
| perception/lost/ambiguous | 0 |
| grasp/slip | 0 |
| alignment/insertion | 0 |
| IK/safety/planning | 0 |
| collision | 0 |
| timeout/unknown | 0 |

개발 중에는 target class flip, occluded low-confidence pose overwrite, stale target 처리 실패가 실제로
발견되었고 수정 전 산출물은 최종 evidence에서 제외했다. 최종 0 failure는 제한된 색상·형상·seed
범위의 결과다.

## 16. Performance와 Benchmark CLI

최종 retained task latency 합은 1118.96 s이며 task당 평균 9.48 s다. Oracle/Camera baseline은 각각
3.41/5.81 s였다. CLI는 observation mode, seed(s), object/target position noise, object/target yaw,
neural, memory, goals, episodes와 output directory를 받는다.

예:

```bash
MUJOCO_GL=egl fdb-physical-persistent --observation-mode camera \
  --position-noise-m 0.01 --target-noise-m 0.01 \
  --object-yaw-noise-deg 20 --target-yaw-noise-deg 0

MUJOCO_GL=egl fdb-phase25-benchmark --suite boundary --seeds 0,1,2
```

각 benchmark는 JSON summary와 CSV records를 자동 생성한다.

## 17. Tests

추가 test는 segmentation 없는 RGB-D shape 분류, triangle orientation, temporal identity,
stale expiry, target class flip 방지, low-confidence overwrite 방지, CameraSource integration,
pose error bound, success-envelope matrix를 포함한다. 최종 전체 fast/integration test 결과는
`103 passed`이다.

## 18. Known Limitations

- 색상과 top-down 고정 camera에 강하게 의존한다.
- 동일 shape/appearance의 다중 객체 identity ambiguity를 풀지 못한다.
- depth noise, motion blur, realistic RealSense artifact의 정량 시험이 없다.
- grasp 자체의 force/contact verification이 없고 lift의 visual motion으로 지연 검증한다.
- camera-only collision sensing은 실제 force/torque sensor가 들어오기 전 unavailable이다.
- 최대 범위는 3 seeds뿐이며 success boundary 밖의 최초 실패점을 찾지 않았다.
- Neural World Model과 Memory는 아직 decision benefit이 없다.
- 실제 Panda/RealSense/Jetson timing과 safety는 검증하지 않았다.

## 19. 연구 질문 답변

- **Q1:** n=3 baseline 성공률 차이 0%p, camera latency +2.40 s/task.
- **Q2:** baseline failure 중 perception 귀속 0/3. 단 trajectory pose error/occlusion cost는 존재.
- **Q3:** 검증된 object XY 범위는 ±20 mm(각 shape 3/3). 그 밖은 미측정.
- **Q4:** 검증된 yaw 범위는 ±40°(각 shape 3/3). 그 밖은 미측정.
- **Q5:** 최종 범위에서 실패 subsystem은 없었다. 개발 중 최초 취약점은 target perception/tracking.
- **Q6:** Neural World Model의 성능/효율 기여 없음(neural call 0).
- **Q7:** Memory의 행동 선택 개선 없음.
- **Q8:** 관측 기반 skill 재선택은 Runtime, 표현/skill/safety/controller는 사람 설계.
- **Q9:** simulation camera-to-deployment interface 단계로 이동할 준비는 됐다. 실기 자율운전 승인은 아님.

## 20. Jetson Readiness Decision

# **GO**

의미는 **Phase 3 Jetson AGX Orin 64GB + RealSense bring-up으로 진행 GO**다.
Camera-only end-to-end, privileged pose/segmentation 제거, relative manipulation, confidence/stale
tracking, headless benchmark, 최소 pose robustness와 objective evaluator가 모두 확인됐다.

이 결정은 실제 로봇 무인 작동 또는 안전 인증 GO가 아니다. Phase 3의 첫 gate는 RealSense
calibration/latency/noise 재검증, force/torque 기반 grasp·collision verification, 동일 외형 다중 객체
tracking이다. Neural/Memory 개선은 권장 연구 항목이지만 이번 Jetson 이전의 blocker로 두지 않는다.

## Evidence

- `artifacts/phase2_5_baseline_verified2/benchmark_summary.json`
- `artifacts/phase2_5_position_retreat/benchmark_summary.json`
- `artifacts/phase2_5_yaw_retreat/benchmark_summary.json`
- `artifacts/phase2_5_target_retreat/benchmark_summary.json`
- `artifacts/phase2_5_combined_retreat/benchmark_summary.json`
- `artifacts/phase2_5_ablation_retreat/benchmark_summary.json`
- `artifacts/phase2_5_boundary_seed1_retreat/benchmark_summary.json`
- `artifacts/phase2_5_boundary_seed2_retreat/benchmark_summary.json`
