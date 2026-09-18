# 실험 0040 — FDBRuntime 지속 장면 폐루프

## 목적

Goal이 장면을 만들지 않는 상태에서 네 물체와 네 수용구가 항상 존재하는 하나의 장면을
관찰하고, Runtime이 행동을 고르고 실패 후 재계획하는지 확인한다.

## 방법

- 장면: square, circle, triangle, rectangle과 각 수용구
- 순차 Goal: triangle → circle → rectangle
- 삼각형 첫 실행에 40도 정렬 오차를 주입
- 성공 조건: 대상이 지정 수용구 내부이며 release 상태
- 매 실행을 `schemas/episode.schema.json`에 맞춰 기록

## 결과

- 3개 Goal 모두 성공
- 장면 ID 유지, sequence 0 → 4
- 삼각형: 2 actions, 1 replan
- 원·직사각형: 각각 1 action, 0 replan
- 전체 safety rejection 0, 실제 MuJoCo fallback 0
- 네 실행 모두 Episode 생성

상세 수치는 `0040_runtime_e2e.results.json`에 있다.

## 해석과 한계

이 실험은 중앙 의사결정 구조와 상태 지속성을 검증하는 빠른 결정론 환경이다. Panda
MuJoCo 물리 증거가 아니며 기존 0037의 4/4 물리 삽입 결과를 대체하지 않는다. 현재
`shape_insert` 어댑터는 기존 물리 코드의 계약과 증거를 재사용하지만, 한 MuJoCo model에
네 물체를 동시에 유지하는 물리 어댑터는 다음 integration gate다.
