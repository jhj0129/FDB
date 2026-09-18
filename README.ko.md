# FDB Fly Decision Brain

FDB는 초파리 connectome에서 영감을 받은 검증 중심 의사결정 연구 프로젝트다. 여러
행동 후보를 만들고, 내부 시뮬레이션과 객관적 기준으로 비교하고, 실행 결과를 에피소드와
재사용 기술로 남긴다.

## 현재 구현

- v0 규칙 기반 목표 해석과 후보 경쟁
- v1 MuJoCo planar push와 feedback 제어
- v2 Franka Panda 도달 파지 상승 배치 해제
- v3 Panda UR5e KUKA iiwa 자기 신체 구조 분석
- v4 20개 관절의 행동 기반 body schema 검증
- v5 5개 MLP 앙상블 세계모델과 안전한 MuJoCo fallback
- v5 Panda 테이블 충돌을 실행 전에 거부하는 학습형 안전 앙상블
- 다중 로봇 기준선: G1, Booster T1, Robotis OP3, Berkeley Humanoid의 5개 과제
- Panda 다중 물체 분류·배치: 세 물체 3/3 성공, 로봇-테이블 접촉 0
- 사람 보행 원리 기반 휴머노이드 보행: G1과 OP3 4초 완주, T1과 Berkeley 실패 보존
- v6 카메라 CNN: 물체·수용구 형상, 삽입 가능성, 필요한 회전 추론
- `네모를 네모칸에 넣어` 한국어 명령에서 Panda 정렬·삽입까지 통합

> 판정 정정: 과거의 4초·10mm 보행 기준은 폐기했다. 최신 보행 합격선은 양발이 실제로
> 번갈아 이륙해 진행 방향 앞에 착지하는 동작이 최소 10걸음 이어지는 것이다. 과거
> `fdb_motion_imitation`은 양발 접촉이 끊기지 않아 보행 실패로 재분류했다.

v5는 3,900개의 MuJoCo rollout으로 학습했다. 독립 시험 500개에서 평균 최종 위치
오차 10.72mm와 성공 판정 정확도 99.4%를 기록했다. 분포 밖 또는 불확실한 입력은
신경망 결과를 신뢰하지 않고 MuJoCo로 되돌린다.

Panda 조작에는 비의도 로봇 테이블 접촉 0과 관통 0을 하드 게이트로 추가했다. 기존
낮은 궤적이 손가락을 최대 2.979mm 관통시키는 문제를 발견하고, 파지 및 배치 높이와
부드러운 단계별 궤적을 교정했다.

추가로 MuJoCo 자세 180개에서 안전 117개와 충돌 63개를 학습한 5개 MLP 안전
앙상블을 만들었다. 독립 시험에서는 위험 자세 11개를 모두 거부했다. 신경망이 허용한
자세도 물리 접촉 하드 게이트를 생략할 수 없으며, 현재 결과는 Panda 작업영역의
`candidate`이다.

한 조작 과제에만 과적합하지 않도록 네 휴머노이드에 서기, 질량 비례 외란, 팔 올리기,
웅크리기 후보, 머리 회전을 적용했다. 적용 가능한 17회 중 9회를 통과했다. 구조상 팔이나
머리가 없는 모델은 실패가 아니라 `해당 없음`으로 분리한다. 이 결과는 보행 정책이 아니라
휴머노이드 챌린지 개발을 위한 정직한 공통 기준선이다.

네 로봇의 무작위 관절 상태와 외란 192회를 추가로 생성해 다음 1초 낙상을 예측하는
5개 MLP 앙상블도 학습했다. 독립 시험에서 낙상을 안정으로 통과시킨 오류는 0이지만,
안정 recall은 41.2%로 보수적이다. 따라서 현재는 물리 rollout 앞의 `candidate` screen이며
처음 보는 로봇에 일반화됐다는 주장은 하지 않는다.

물체 분류에는 RGB와 세 축 크기를 입력으로 받는 5개 MLP 앙상블 후보를 추가했다.
합성 잡음 독립 시험 144개에서는 100%였지만 색으로 크게 분리된 세 클래스 결과이므로
일반 물체 인식 성능이 아니다. 실제 Panda 장면에서는 세 물체를 각 구역에 모두 옮겼고
portable MLP 앙상블의 최소 신뢰도는 0.9998, 최대 XY 오차는 3.15mm, 비의도 테이블
접촉은 0회였다.
source와 target 위치를 바꾼 두 장면까지 포함하면 9/9 배치와 전 장면 접촉 0을 유지했다.
학습 prototype에서 먼 애매한 물체는 `unknown`으로 거부해 자신 있게 오분류하는 경로도
차단했다.

사람 보행의 지지/유각 전환, 유각기 무릎 굽힘, 발목 보상, 반대쪽 팔 흔들기와 hip roll
무게 이동을 네 모델에 적용했다. 발 접촉을 직접 계측한 폐루프 후보에서 G1은 170.1mm와
단일 지지 34.7%, OP3는 25.5mm와 단일 지지 7.0%로 4초 보행 기준을 통과했다. T1은
pitch-발목 되먹임으로 4초 직립했지만 단일 지지 0%, 발 수직 변위 2.4mm라 보행이 아닌
안정화로 기록했다. Berkeley는 0.45초에 낙상했다. 다음 단계는 CoM/ZMP와 capture step을
쓰는 지지면 기반 폐루프 제어다.

개발용 마찰·체중 5% 외란 6조건에서 G1과 OP3가 6/6을 통과한 뒤 더 넓은 마찰과 새로운
외란 7조건을 시험했다. G1은 독립 7/7이었다. OP3는 첫 정책의 3/7 보행·4/7 직립에서
외란 방향별 pitch/roll 운동량 보상을 추가해 6/7 보행·7/7 직립으로 개선됐다. 다만 외란
방향을 시뮬레이터에서 제공받으므로 실제 센서 기반 회복은 아니다. T1은 독립 7/7 직립했지만
단일 지지 0%였다. 다음 우선순위는 외란 방향 추정과 capture step 생성이다.

## 주요 결과물

- `artifacts/fdb_v5_neural_prediction.mp4` 신경망 예측과 실제 물리 결과 비교
- `artifacts/fdb_final_pick_place.mp4` 접촉 0 Panda pick and place
- `artifacts/fdb_humanoid_challenge.mp4` 네 휴머노이드의 연속 과제와 외란 비교
- `artifacts/fdb_multi_object_sorting.mp4` 세 물체 분류·집기·구역 배치
- `artifacts/fdb_human_gait.mp4` 네 휴머노이드의 사람형 보행 후보 비교
- `artifacts/fdb_sort_and_walk_final.mp4` 분류와 보행을 이어 붙인 38.93초 최종본
- `artifacts/fdb_shape_insertion.mp4` 카메라 CNN 판단과 Panda 정사각형 삽입
- `artifacts/fdb_insertion_recovery.mp4` 첫 삽입 실패 뒤 후보 시뮬레이션으로 복구
- `artifacts/fdb_neural_gait_skill.mp4` 사람 보행을 학습한 신경망의 좌우 10걸음(로컬 전용)
- `artifacts/fdb_dynamic_gait_baseline.json` 자유 물리 추종 실패와 다음 정책의 기준선
- `artifacts/fdb_deepmimic_training_1m.json` H1 DeepMimic PPO 100만 step 학습·물리 평가
- `artifacts/fdb_human_cadence_h1_walk.mp4` 실제 접촉 기반 20초·40걸음 H1 동역학 보행
- `artifacts/fdb_unitree_h1_robustness.json` 0–500N 횡외란 복구 경계
- `artifacts/fdb_multi_shape_insertion.mp4` 네 형상 카메라 판단·Panda 정렬 삽입 4/4
- `artifacts/fdb_final_integrated_demo.mp4` 네 형상 조작과 동역학 보행 최종 통합 영상
- `artifacts/fdb_shape_sequence.json` 한국어 3단계 형상 삽입 순서 계획 3/3
- `models/v6/shape_fit_cnn.msgpack` 형상·fit·회전을 예측하는 two-tower CNN
- `models/v5/push_dynamics_ensemble.pt` PyTorch 학습 체크포인트
- `models/v5/push_dynamics_ensemble.npz` MuJoCo 환경용 portable 추론 모델
- `models/v5/panda_table_safety.pt` Panda 자세 안전 PyTorch 체크포인트
- `models/v5/panda_table_safety.npz` PyTorch 없는 환경용 안전 추론 모델
- `models/v5/humanoid_balance_ensemble.npz` 다중 휴머노이드 단기 낙상 screen
- `models/v5/object_sort_ensemble.npz` PyTorch 없이 실행하는 물체 분류 MLP 앙상블
- `experiments/0014_v5_neural_metrics.json` 학습 및 시험 지표
- `experiments/0015_v5_common_sense_safety.results.json` 조작 안전 경계 데이터
- `experiments/0016_v5_safety_classifier.metrics.json` 학습형 안전 정책 시험 지표

## 실행 환경

프로젝트 MuJoCo 검사는 `.venv`에서 실행한다. 신경망 학습은 ROS 시스템 Python을
오염시키지 않도록 `~/venvs/robot_ai`를 사용한다.

```bash
.venv/bin/python -m fdb.v5.data_cli
PYTHONPATH=src ~/venvs/robot_ai/bin/python -m fdb.v5.train_cli
PYTHONPATH=src ~/venvs/robot_ai/bin/python -m fdb.v5.safety_training
.venv/bin/python -m fdb.v5.hybrid_cli
MUJOCO_GL=egl .venv/bin/python -m fdb.v5.render_neural
MUJOCO_GL=egl .venv/bin/python -m fdb.v2.render_final
.venv/bin/python -m fdb.challenge.humanoid_suite
MUJOCO_GL=egl .venv/bin/python -m fdb.challenge.render_humanoids
.venv/bin/python -m fdb.challenge.balance_data
PYTHONPATH=src ~/venvs/robot_ai/bin/python -m fdb.challenge.balance_training
PYTHONPATH=src ~/venvs/robot_ai/bin/python -m fdb.challenge.object_sort_training
MUJOCO_GL=egl .venv/bin/python -m fdb.challenge.multi_object_sorting
MUJOCO_GL=egl .venv/bin/python -m fdb.challenge.human_gait
.venv/bin/python -m fdb.challenge.gait_robustness
.venv/bin/python -m fdb.challenge.sorting_robustness
.venv/bin/python -m fdb.challenge.render_showcase
.venv/bin/python -m fdb.challenge.shape_fit_learning
MUJOCO_GL=egl .venv/bin/python -m fdb.challenge.shape_insertion
.venv/bin/python -m fdb.challenge.shape_command "네모를 네모칸에 넣어"
.venv/bin/python -m fdb.challenge.insertion_recovery
.venv/bin/python -m fdb.challenge.gait_skill_learning
.venv/bin/python -m fdb.challenge.dynamic_gait_baseline
.venv/bin/python -m fdb.challenge.cognitive_curriculum
.venv/bin/python -m fdb.challenge.deepmimic_training --timesteps 1048576 --num-envs 64
PYTHONPATH=src ~/venvs/robot_ai/bin/python -m fdb.challenge.unitree_h1_walk --duration 20 --forward 1.3 --phase-period 1.0
PYTHONPATH=src ~/venvs/robot_ai/bin/python -m fdb.challenge.unitree_h1_robustness
.venv/bin/python -m fdb.challenge.multi_shape_insertion
.venv/bin/python -m fdb.challenge.shape_sequence "먼저 세모를 삼각형 칸에 넣고 다음으로 원을 원형 칸에 넣어"
```

## 실제 접촉 기반 동역학 보행

Unitree의 BSD-3-Clause 공식 H1 신경망 정책을 안정화 기반으로 사용하고, FDB가 사람
보행 연구에 가까운 분당 120걸음 위상과 실제 발-바닥 접촉 판정을 추가했다. 500Hz
MuJoCo 자유 물리에서 20초 동안 좌우 40걸음과 20.47m 이동을 완료했다. 발이 실제로
0.15초 이상 떨어지고 10mm 이상 들린 뒤 80mm 이상 앞에 닿아야 한 걸음으로 인정한다.
350N 횡방향 외란까지 교대 연속 보행을 복구했으며, 400N에서는 연속성 단절, 500N에서는
낙상하는 한계도 보존했다. 자체 LAFAN1 DeepMimic 정책은 접촉 보상 추가 후에도 최장
2.06초·1착지이므로 아직 성공으로 계산하지 않는다.

## 네 형상 물리 삽입

동일한 카메라 CNN과 Panda 행동 경로를 정사각형·원·삼각형·직사각형 물체와 수용구에
연결했다. 4/4 삽입에 성공했고 최종 XY 오차는 2.29–7.55mm, 방향 오차는 최대 1.61도,
비의도 로봇-테이블 접촉은 0회였다. 삼각형의 첫 꼭짓점 파지 실패는 평평한 변 파지로
복구했으며, 형상 불일치 한국어 명령은 실행 전에 거부한다.

한국어 연결어로 묶인 세모→원→직사각형 3단계 명령도 실행 전에 전체 계획을 검증한 뒤
순서대로 3/3 완료했다. 현재 단계별 물리 장면은 fresh-state이며, 다음 목표는 한 장면의
여러 물체를 완료 상태를 기억하면서 처리하는 것이다.

현재 결과는 시뮬레이션 연구 증거이며 실제 로봇 실행 승인이 아니다. 실제 적용 전에는
센서 잡음, 지연, calibration, 비상 정지, 힘 제한과 하드웨어별 안전 검증이 필요하다.
