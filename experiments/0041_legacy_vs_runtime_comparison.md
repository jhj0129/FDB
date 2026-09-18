# 실험 0041 — Legacy 직접 실행과 FDBRuntime 비교

| 측정값 | Legacy | Runtime v1 |
| --- | ---: | ---: |
| 성공 | 4/4 | 3/3 |
| 장면 | 매 형상 fresh-state | 하나의 persistent scene |
| action | 기록 없음 | 4 |
| replan | 0 | 1 |
| 로봇-테이블 접촉 | 0 step | 물리 아님 |
| safety rejection | 기록 없음 | 0 |
| 최대 위치 오차 | 7.55mm | 물리 아님 |
| world-model 호출 | 기록 없음 | 8 |
| 실제 MuJoCo fallback | 기록 없음 | 0 |
| memory retrieval hit 합계 | 0 | 6 |

두 결과는 같은 성능 지표가 아니다. Legacy는 물리 동작 성공을 증명하지만 장면을 매번
새로 만든다. Runtime v1은 상태 지속, 현재 상태 기반 선택, episode, 실패 후 재계획을
증명하지만 물리 정확도를 증명하지 않는다. 다음 gate는 이 두 장점을 한 MuJoCo 지속
장면에서 결합하는 것이다.
