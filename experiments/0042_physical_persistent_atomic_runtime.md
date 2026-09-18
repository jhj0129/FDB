# 실험 0042 — 단일 물리 세계 원자 Skill Runtime

## 조건

- 하나의 Panda MuJoCo `model/data`, reset 1회
- 정사각형·원·삼각형·직사각형과 각 수용구 동시 배치
- Goal 순서: triangle → circle → rectangle
- triangle 정렬 실패 1회, circle 파지 실패 1회 주입
- 카메라 RGB + simulator segmentation + 고정 외부 파라미터 보정
- Goal은 생성 후 world를 변경하지 않음

## 결과

| Goal | 성공 | actions | replans | 시간 |
| --- | ---: | ---: | ---: | ---: |
| triangle | 예 | 8 | 1 | 7.93초 |
| circle | 예 | 9 | 1 | 9.06초 |
| rectangle | 예 | 7 | 0 | 7.34초 |

합계 3/3, 24 actions, 2 replans, safety rejection 0, reset 1회다. 삼각형은 물체를
쥐고 든 상태에서 `ALIGNMENT_FAILURE → recover_alignment`, 원은
`GRASP_FAILURE → reach_object → grasp_object`로 분기했다. 기존 단계 성공을 버리고
Task 전체를 reset하지 않았다.

최종 카메라에는 세 물체가 수용구 안에 남고 정사각형은 원래 위치에 남는다. 상세
Task 단위 step trace는 `artifacts/phase2_physical_seed0/episodes/`, 요약과 최종 프레임은
같은 디렉터리에 있다.

## 정직한 해석

후보 생성과 subgoal은 사람이 설계한 deterministic state-machine이다. CNN은 관측마다
실행되지만 이 run에서 World Model 선택 점수에 실제 사용된 것은 26회 중 1회다. 물리
fallback은 0회다. 성공은 learned planning의 증거가 아니라 **현재 상태를 다시 관찰하는
규칙 기반 Skill 조합과 실패별 재계획**의 증거다.
