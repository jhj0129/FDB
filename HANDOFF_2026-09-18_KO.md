# FDB 전체 연구 인계 문서 — 2026-09-18

이 문서는 다른 대화에서 FDB의 현재 상태를 분석하거나 개발을 이어가기 위한 기준 문서다.
성과와 한계를 구분하고, 오픈소스에서 가져온 부분과 FDB에서 직접 구현한 부분을 명시한다.

## 1. 목표

FDB(Fly Decision Brain)는 초파리 connectome에서 영감을 받은 소형 의사결정 시스템에서
출발했다. 현재 목표는 다음 능력을 하나의 시스템으로 연결하는 것이다.

- 카메라에서 물체와 수용구의 형상·크기·방향·삽입 가능성을 판단
- 한국어 명령을 물리 행동과 순서 계획으로 변환
- 바닥·테이블 관통과 위험 동작을 실행 전에 거부
- 실패를 관찰하고 다른 후보 행동으로 재시도
- 실제 접촉이 번갈아 일어나는 휴머노이드 동역학 보행
- 보지 못한 장면과 외란에서의 전이
- 경험 수집→평가→재학습의 지속 학습 폐루프

현재 내부 점수는 25/36(69.4%)이다. 사람의 발달연령으로 환산할 수 없는 좁은 과제형
시스템이며 “10세 지능”이나 일반지능을 달성했다고 볼 수 없다.

## 2. 먼저 답해야 할 질문: 예제인가, 직접 만든 것인가

### 2.1 동적 H1 보행 — 오픈소스 사전학습 정책 기반

20초·40걸음에 성공한 최신 H1 보행의 신경망 가중치는 FDB가 처음부터 학습한 것이
아니다. Unitree 공식 `unitree_rl_gym` 저장소의 사전학습 `motion.pt`를 사용했다.

- 출처: https://github.com/unitreerobotics/unitree_rl_gym
- 라이선스: BSD-3-Clause
- 사용 revision: `276801e46c5d433564f24658bac64f254b7d2d4b`
- 가중치 SHA-256: `44a0fbceb81f3877833ae9a398d039bea1759cb0d3c8188181013885f70589eb`
- 원본 PD 제어기, 관측 구성, 50Hz 정책 호출 구조도 공식 배포 예제를 참고해 재현
- 가중치 파일은 FDB Git 저장소에 복제하지 않고 로컬 캐시에서 읽음

FDB가 직접 추가한 부분은 다음과 같다.

- 500Hz MuJoCo headless 실행·기록·영상화 도구
- 발-바닥 실제 접촉의 `접촉→이륙→재접촉` 기반 걸음 판정기
- 센서 튐 제거, 최소 공중시간, 발 여유 높이, 전진 착지, 좌우 교대 기준
- 사람 연구값을 반영한 위상 주기 조절
- 원본 약 148걸음/분을 120걸음/분으로 바꾸면서 안정성을 유지하는 실험
- 속도 명령 grid와 0–500N 횡외란 경계 평가
- 전진·측면·회전·정지 명령 전이 평가
- 실패를 성공으로 세지 않도록 몸통 높이와 보행 연속성 조건 추가

따라서 최신 보행은 “예제를 그대로 재생”한 것은 아니지만, 균형 능력의 핵심 신경망은
Unitree가 학습한 것이다. FDB의 기여는 정책 통합, 사람 보행률 조정, 엄격한 행동 판정,
외란 평가와 실패 경계 측정이다.

### 2.2 FDB 자체 DeepMimic 보행 — 직접 학습했지만 아직 실패

FDB 자체 보행 정책도 별도로 학습했다. 이 경로는 LocoMuJoCo의 환경과 LAFAN1 사람
동작 데이터를 사용한다.

- LocoMuJoCo: https://github.com/robfiras/loco-mujoco
- 알고리즘: PPOJax
- 로봇: Unitree H1
- 입력/reference: LAFAN1 사람 보행 retargeting
- 제어 비교: direct torque와 PD position control

FDB가 직접 구현한 부분:

- PPO 학습 구성과 체크포인트 재개
- 3초 horizon이 5.8초짜리 10걸음 목표보다 짧다는 오류를 찾아 8초로 수정
- 실제 양발 접촉과 사람 reference 접촉의 일치 보상
- 단일 지지 보너스와 발 높이 추종 보상
- 체크포인트의 결정론적 자유 물리 평가기
- direct torque와 PD control 비교 및 실패 보존

결과:

- 기존 100만 step: 최장 1.39초, touchdown 0회 — 실패
- 접촉 보상 200만 step 추가: 최장 2.06초, 유효 전진 착지 1회 — 개선됐지만 실패
- PD position control 200만 step: 최장 1.62초, 뒤쪽 착지 1회 — 기각

즉 FDB 자체 신경망은 아직 10걸음에 도달하지 않았다. 최신 성공 영상은 이 자체 정책이
아니라 Unitree 사전학습 정책 기반이다.

### 2.3 Panda 로봇팔 — 로봇 모델은 오픈소스, 행동 로직은 FDB 직접 구현

Franka Panda의 기구 모델은 MuJoCo Menagerie 계열 공개 모델을 사용한다. 하지만 물체
장면, 카메라 처리, 경로 단계, 안전 gate, 성공 판정, 실패 복구는 FDB에서 작성했다.

직접 설계한 행동 단계:

1. 물체 위 접근
2. 파지 높이 정렬
3. 그리퍼 닫기
4. 안전 높이까지 들기
5. 수용구로 이동하며 손목 회전
6. 수용구 안으로 하강
7. 해제
8. 후퇴

각 관절 궤적은 Panda inverse-kinematics/pose reach 결과를 부드러운 5차 보간으로 연결한다.
특정 데모의 고정 관절 값을 복사한 것이 아니다. 단, 범용 motion planner나 실제 힘 제어는
아직 아니다.

### 2.4 시각 형상 신경망 — FDB가 직접 설계하고 학습

`models/v6/shape_fit_cnn.msgpack`은 FDB가 직접 만든 two-tower CNN이다.

- 입력: 32×64 RGB, 왼쪽 물체와 오른쪽 수용구 패널
- 공유 3층 convolution encoder
- 물체 형상 head: 정사각형·원·삼각형·직사각형·unknown
- 수용구 형상 head
- 삽입 가능성 binary head
- 물체/수용구 방향 head
- 크기 회귀 head
- 합성 카메라 장면 6,000개 학습, 독립 1,500개 평가
- unknown과 형상 불일치를 안전상 삽입 불가로 강제

합성 독립 시험 safe-fit 정확도는 94.7%였다. 실제 MuJoCo 상단 카메라 픽셀에서는 빨간
물체와 파란 수용구 윤곽을 추출·정규화한 뒤 CNN에 전달한다. 물리 모델의 shape 이름이나
정답 회전을 직접 읽지 않는다.

## 3. 현재 검증된 조작 능력

### 3.1 네 형상 물리 삽입

정사각형·원·삼각형·직사각형을 각각 카메라로 인식하고 Panda가 집어서 대응 수용구에
삽입했다.

| 형상 | 최종 XY 오차 | 높이 오차 | 방향 오차 | 결과 |
|---|---:|---:|---:|---:|
| 정사각형 | 7.55mm | 0.63mm | 0.57도 | 성공 |
| 원 | 2.29mm | 0.43mm | 해당 없음 | 성공 |
| 삼각형 | 3.82mm | 0.75mm | 1.61도 | 성공 |
| 직사각형 | 7.13mm | 0.63mm | 0.62도 | 성공 |

- 전체 4/4 성공
- 비의도 로봇-테이블 접촉 0회
- 삼각형 첫 실행은 꼭짓점 방향 파지가 미끄러져 실패
- 평평한 변을 집는 90도 tool offset으로 수정 후 성공

### 3.2 실패 복구

정사각형 삽입의 카메라 회전에 의도적으로 40도 오류를 넣었다. 첫 실행은 frame에 걸려
실패했다. fresh-state 후보 시뮬레이션에서 여러 회전 보정을 비교하고 성공 후보를 골라
재시도했다. 이는 오류 원인을 완전히 학습한 범용 복구기는 아니며, 회전 오류 한 종류의
후보 탐색이다.

### 3.3 실행 전 상식 안전 gate

사용자가 지적한 “그리퍼를 바닥에 처박으며 잡거나 놓는 문제”를 해결하기 위해 실행 전
손·물체 최저 높이를 계산한다. 바닥/테이블 여유를 위반하는 waypoint는 물리를 실행하기
전에 거부한다. 실제 시뮬레이션에서도 로봇-테이블 접촉 0을 별도 하드 gate로 확인한다.

### 3.4 한국어 명령과 순서 계획

지원 예:

- `네모를 네모칸에 넣어`
- `동그라미를 원형 칸에 넣어`
- `세모를 삼각형 칸에 끼워`
- `직사각형을 직사각형 구멍에 삽입해`
- `먼저 세모를 ... 다음으로 원을 ... 그리고 직사각형을 ...`

3단계 세모→원→직사각형 명령을 순서대로 3/3 완료했다. 실행 전에 모든 절을 먼저
해석하므로 뒤쪽 절이 불완전하면 첫 물체도 움직이지 않는다. 다만 각 단계는 별도의
fresh-state 장면이며 한 작업대에 모든 물체가 동시에 존재하는 지속 장면은 아직 아니다.

## 4. 현재 검증된 보행 능력

### 4.1 성공 기준

높이 변화나 미끄러짐만으로 걷기 성공을 선언하지 않는다. 한 걸음은 다음을 만족해야 한다.

- 발과 바닥의 실제 물리 접촉이 끊어짐
- 공중 상태 0.15초 이상
- 접지 기준보다 발이 최소 10mm 상승
- 진행 방향으로 최소 80mm 앞에 재접촉
- 좌우 발이 번갈아 이어짐
- 몸통 직립과 이동 거리 조건 충족

10mm 기준은 정상 성인 최소 발끝 여유 높이가 대체로 10–20mm라는 연구를 참고했다.
연구: https://pmc.ncbi.nlm.nih.gov/articles/PMC8200481/

### 4.2 최신 H1 결과

- 20초 연속
- 좌우 40걸음, 20보폭
- 120.01걸음/분
- 몸통 순이동 20.47m
- 최소 공중시간 0.25초
- 최소 발 여유 높이 35.4mm
- 최소 전진 착지 139mm
- 최종 몸통 높이 1.014m
- 성공

정상 성인 야외 보행률 평균 116.65걸음/분과 가까운 값이지만, 전신 관절 궤적 전체가
사람과 같다는 의미는 아니다. 연구: https://pmc.ncbi.nlm.nih.gov/articles/PMC7806575/

### 4.3 외란 경계

5초 지점 골반에 0.15초 동안 횡방향 힘을 가했다.

- 350N, 52.5N·s: 29/30걸음 이상 유지, 완전 복구
- 400N, 60N·s: 다시 섰지만 교대 연속성 14걸음으로 단절
- 500N, 75N·s: 낙상

### 4.4 명령 전이와 실패

- 느린 전진 `(0.3, 0, 0)`: 30걸음, 성공
- 측면 이동 `(0, 0.3, 0)`: 29걸음, 성공
- 정지 `(0, 0, 0)`: 직립하지만 15초에 1.18m 표류, 실패
- 좌우 회전: 직립하지만 목표 각도에서 멈추지 않고 여러 바퀴 과회전, 실패
- 위상 고정 정지 후보: 대부분 낙상, 직립한 후보도 13.64m 표류, 기각

다음 보행 과제는 감속→양발 접촉 확인→정적 균형 제어기 전환 상태 머신과 목표 각도
폐루프 회전이다.

## 5. 그 밖의 신경망

### v5 물체 밀기 세계모델

- 5개 MLP 앙상블
- MuJoCo rollout 3,900개 학습
- 독립 시험 500개
- 평균 최종 위치 오차 10.72mm
- 성공 판정 정확도 99.4%
- 분포 밖/불확실 입력은 신경망을 버리고 MuJoCo fallback

### Panda 테이블 안전 앙상블

- MuJoCo 자세 180개: 안전 117, 충돌 63
- 5개 MLP 앙상블
- 독립 시험 위험 자세 11개를 모두 거부
- 신경망 허용 뒤에도 물리 접촉 hard gate 유지

### 다중 휴머노이드 낙상 예측 앙상블

- G1, Booster T1, Robotis OP3, Berkeley Humanoid
- 무작위 관절 상태와 외란 192회
- 다음 1초 낙상 screen
- 위험 상태를 안정으로 통과시킨 오류 0
- 안정 recall 41.2%로 매우 보수적
- 처음 보는 로봇 일반화는 주장하지 않음

### 물체 분류 MLP

- RGB와 세 축 크기 입력
- 5개 MLP portable ensemble
- 합성 독립 시험 144개에서 100%
- 색으로 크게 구분된 3개 클래스이므로 일반 물체 인식이 아님
- Panda 실제 장면 9/9 배치, 최대 XY 오차 3.15mm, 테이블 접촉 0
- prototype에서 먼 입력은 unknown으로 거부

### 키네마틱 사람 보행 MLP

- LAFAN1 10걸음 reference를 위상 조건 MLP로 학습
- 69,338개 매개변수
- 좌우 10회 착지 궤적 생성
- 동역학이 없는 자세 재생이므로 현재 동적 보행 성공과 구분해야 함

## 6. 주요 영상과 결과 파일

- `artifacts/fdb_final_integrated_demo.mp4`
  - 네 형상 Panda 삽입 4회 + H1 20초 보행 최신 통합본
  - SHA-256 `3234934ccfd81348e96f015cdbd969c795f31816f5fe5d492d6b99bf69dd0132`
- `artifacts/fdb_multi_shape_insertion.mp4`
- `artifacts/fdb_human_cadence_h1_walk.mp4`
- `artifacts/fdb_unitree_h1_robustness.json`
- `artifacts/fdb_h1_command_transfer.json`
- `artifacts/fdb_shape_sequence.json`
- `artifacts/fdb_deepmimic_contact_2m_eval.json`
- `experiments/0036_contact_gait_and_h1_policy.md`
- `experiments/0037_multi_shape_physical_insertion.md`
- `experiments/0038_korean_shape_sequence.md`
- `experiments/0039_h1_command_transfer.md`

## 7. 검증과 Git 상태

- 전체 회귀시험: 75개 통과
- 주요 최신 커밋:
  - `90fb9dc 접촉 기반 사람 보행과 네 형상 물리 삽입 완성`
  - `edf1d3f 한국어 다단계 형상 삽입 계획 추가`
  - `4d0ec30 H1 방향 명령 전이와 정지 실패 경계 기록`
  - `a3e3c1b 최종 통합 영상 무결성 값 기록`
- 원격: `https://github.com/jhj0129/FDB.git`
- branch: `main`

## 8. 실행 환경

- Ubuntu 24 계열
- 프로젝트 기본 환경: `.venv`
- PyTorch/Unitree 실행: `~/venvs/robot_ai`
- Unitree 소스 로컬 캐시: `/home/hgui/.cache/fdb/unitree_rl_gym`
- MuJoCo, JAX, Flax, Optax, PyTorch, LocoMuJoCo 사용
- GPU는 Unitree 정책 평가 시 사용하지 않았고 CPU에서 실행

대표 실행 명령:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m fdb.challenge.multi_shape_insertion
.venv/bin/python -m fdb.challenge.shape_sequence "먼저 세모를 삼각형 칸에 넣고 다음으로 원을 원형 칸에 넣어"
PYTHONPATH=src ~/venvs/robot_ai/bin/python -m fdb.challenge.unitree_h1_walk --duration 20 --forward 1.3 --phase-period 1.0
PYTHONPATH=src ~/venvs/robot_ai/bin/python -m fdb.challenge.unitree_h1_robustness
```

## 9. 다음 대화에서 우선 분석할 질문

1. Unitree 사전학습 정책의 안정성을 teacher로 사용해 FDB 자체 DeepMimic 정책에
   distillation할 것인가, 아니면 안정 정책 위에 residual imitation만 학습할 것인가?
2. 보행→감속→양발 접촉→정지 제어기 전환을 어떤 관측과 hysteresis로 안전하게 만들 것인가?
3. 목표 yaw 폐루프 회전에서 과회전을 막기 위해 command taper와 각속도 feedback을
   어떻게 구성할 것인가?
4. 한 작업대의 여러 물체·수용구를 카메라 instance로 분리하고 완료 상태를 기억하는
   scene memory를 어떻게 설계할 것인가?
5. 삽입 접촉력이 커질 때 spiral/edge search를 언제 시작하고 언제 철회할 것인가?
6. 합성 영상 CNN을 실제 카메라로 옮길 때 필요한 domain randomization과 OOD gate는 무엇인가?

## 10. 주장하면 안 되는 것

- FDB가 사람 10세 수준 또는 일반지능에 도달했다는 주장
- 최신 H1 안정 보행 정책을 FDB가 처음부터 학습했다는 주장
- 키네마틱 10걸음을 동역학 보행 성공으로 계산하는 것
- 발 미끄러짐이나 접촉 유지 상태를 걸음으로 계산하는 것
- 시뮬레이션 안전을 실제 로봇 안전으로 간주하는 것
- 원·삼각형·직사각형 4개 성공을 일반 물체 조작 능력으로 확대 해석하는 것

실제 로봇 적용 전에는 센서 잡음, 지연, 카메라 calibration, 힘 제한, 비상 정지,
자기충돌, 사람 접근 감지와 하드웨어별 안전 검증이 별도로 필요하다.
