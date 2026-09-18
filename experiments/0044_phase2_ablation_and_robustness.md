# 실험 0044 — Phase 2 baseline, ablation, robustness

## 비교

| 방식 | 조건 | 성공 | actions | replans |
| --- | --- | ---: | ---: | ---: |
| Legacy monolithic | fresh-state 물리 | 4/4 | 미기록 | 0 |
| Scripted atomic | nominal, persistent | 3/3 | 21 | 0 |
| Scripted atomic | 정렬·파지 실패 주입 | 0/3 | 21 | 0 |
| FDBRuntime | 같은 실패 주입 | 3/3 | 24 | 2 |

Scripted baseline은 실패해도 고정 순서를 계속 시도해 precondition hard gate에 거부됐다.
Runtime은 실패 종류와 남아 있는 물리 상태에 맞춰 다른 Skill을 선택했다.

## Ablation

| 구성 | 성공 | actions | replans | Neural calls | Memory hits | 시간 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Neural+rules, Memory ON | 3/3 | 24 | 2 | 1 | 23 | 24.32초 |
| Rules only, Memory ON | 3/3 | 24 | 2 | 0 | 23 | 24.66초 |
| Neural+rules, Memory OFF | 3/3 | 24 | 2 | 1 | 0 | 25.30초 |

현재 Neural과 Memory는 성공률, action 수, replan 수를 개선하지 않았다. Neural은 원의
정렬 후보 점수에 한 번 참여했지만 후보 선택을 바꾸지 않았다. Memory는 검색만 수행하며
정책에 반영되지 않는다. 이 결과를 효과가 있는 것처럼 해석하지 않는다.

## 작은 robustness 표본

- seed 0, 무작위 yaw, 기본 순서: 3/3
- seed 1, object/target XY ±2mm, 순서 rectangle→square→triangle: 2/3

seed 1의 triangle은 16 actions와 4 replans 뒤 timeout이었다. 따라서 pose randomization에
일반화됐다고 주장할 수 없다. N=2는 회귀 경계 탐색용 작은 표본이며 통계적 성능 평가가
아니다.
