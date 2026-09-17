# 실험 0028 — 사람 행동 기준 연속 교대 보행

## 기준

걷기는 같은 발의 착지에서 다음 같은 발 착지까지 이어지는 gait cycle이며, 한 발이
지지하는 동안 반대발이 지면을 떠나 앞으로 이동해 착지해야 한다. FDB는 이를 접촉 높이
hysteresis와 전방 발 배치로 판정한다.

- [Coordination of Locomotion with Voluntary Movements in Humans](https://pmc.ncbi.nlm.nih.gov/articles/PMC6725226/)
- [Step-by-step insight into gait analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC13418446/)
- [LocoMuJoCo](https://github.com/robfiras/loco-mujoco)
- [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)

## 결과

LAFAN1 `walk1_subject1`을 LocoMuJoCo로 Unitree H1에 재표적화한 궤적에서 다음을
검증했다.

- 좌우 교대 착지 10회: 왼발 5회, 오른발 5회.
- 완성 stride 5회.
- 연속 동작 시간 5.8초.
- 최소 swing clearance 0.1265m.
- 모든 발의 전방 배치가 양수이며 최소 1.643m.
- 골반 경로 10.492m, 시작-끝 변위 9.545m.

따라서 새 행동 정의는 통과한다. 단, 이는 **사람 궤적의 kinematic playback**이다.
FDB 신경망이 물리 외란 속에서 이 행동을 스스로 생성하거나 회복한 것은 아니다. 다음
게이트는 이 10걸음 궤적을 교사로 한 모방 정책과 실제 접촉 기반 rollout이다.

LAFAN1은 CC BY-NC-ND 4.0이므로 재표적화 영상은 로컬 검증용으로만 보존하고 저장소에
재배포하지 않는다.

