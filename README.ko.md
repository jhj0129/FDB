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

v5는 3,900개의 MuJoCo rollout으로 학습했다. 독립 시험 500개에서 평균 최종 위치
오차 10.72mm와 성공 판정 정확도 99.4%를 기록했다. 분포 밖 또는 불확실한 입력은
신경망 결과를 신뢰하지 않고 MuJoCo로 되돌린다.

Panda 조작에는 비의도 로봇 테이블 접촉 0과 관통 0을 하드 게이트로 추가했다. 기존
낮은 궤적이 손가락을 최대 2.979mm 관통시키는 문제를 발견하고, 파지 및 배치 높이와
부드러운 단계별 궤적을 교정했다.

## 주요 결과물

- `artifacts/fdb_v5_neural_prediction.mp4` 신경망 예측과 실제 물리 결과 비교
- `artifacts/fdb_final_pick_place.mp4` 접촉 0 Panda pick and place
- `models/v5/push_dynamics_ensemble.pt` PyTorch 학습 체크포인트
- `models/v5/push_dynamics_ensemble.npz` MuJoCo 환경용 portable 추론 모델
- `experiments/0014_v5_neural_metrics.json` 학습 및 시험 지표
- `experiments/0015_v5_common_sense_safety.results.json` 조작 안전 경계 데이터

## 실행 환경

프로젝트 MuJoCo 검사는 `.venv`에서 실행한다. 신경망 학습은 ROS 시스템 Python을
오염시키지 않도록 `~/venvs/robot_ai`를 사용한다.

```bash
.venv/bin/python -m fdb.v5.data_cli
PYTHONPATH=src ~/venvs/robot_ai/bin/python -m fdb.v5.train_cli
.venv/bin/python -m fdb.v5.hybrid_cli
MUJOCO_GL=egl .venv/bin/python -m fdb.v5.render_neural
MUJOCO_GL=egl .venv/bin/python -m fdb.v2.render_final
```

현재 결과는 시뮬레이션 연구 증거이며 실제 로봇 실행 승인이 아니다. 실제 적용 전에는
센서 잡음, 지연, calibration, 비상 정지, 힘 제한과 하드웨어별 안전 검증이 필요하다.

