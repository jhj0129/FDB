# FDB Jetson 탑재 전 사전작업 보고서

작성일: 2026-09-19

## 1. 변경·생성·삭제 파일

주요 변경은 `cognitive_curriculum.py`, `memory.py`, `pyproject.toml`, `README.ko.md`,
두 외부 로봇 경로 설정이다. 새 파일은 `src/fdb/core/` 9개 모듈, Goal schema,
Core shape skill, capability evidence, Runtime 실험·Episode, 두 감사 문서, 두 분석 문서,
fast CI와 시험 2개다. 기존 성공·실패 증거는 삭제하거나 deprecated하지 않았다.

## 2. FDBRuntime 구조

```text
Persistent Environment → Observation ─┐
Goal ──────────────────────────────────┤
                                      ↓
Episode retrieval → Skill candidates → World Model
                                      ↓
                         Deterministic Safety Gate
                                      ↓
                    Execute → Observe → Objective evaluation
                                      ↓
                              Episode → Replan
```

`FDBRuntime.run()`은 LLM/VLA 없이 위 루프를 수행한다. 모든 종료 경로는 Episode를
생성하고 latency, world-model 호출, fallback, memory hit, rejection을 계측한다.

## 3. Skill 목록

기존 JSON skill 15개와 새 `core.shape_insert`가 registry에서 검색된다. 실제 Core 호출
어댑터가 완료된 것은 현재 `shape_insert` 하나다. Panda reach/pick-place, v5 push world
model, morphology/probe, H1 walk 등은 기존 실행과 증거는 보존되지만 아직 공통 `execute()`
어댑터가 아니다. 검색 가능과 실행 가능을 혼동하지 않도록 명시적으로 구분했다.

## 4. Memory 구조

- Working: Runtime의 현재 Observation, Goal, failure, candidate
- Episode: 매 판단/실행을 append-only JSON으로 저장
- Long-term: 서로 다른 evidence 2개 이상일 때만 promote
- Archive: 비활성 기록을 삭제 없이 이동
- 수정: 이전 revision을 `memory/audit`에 복사한 뒤 revision 증가
- deprecate: 이유와 replacement ID를 보존

## 5. Goal schema

`schemas/goal.schema.json`은 행동 순서가 아니라 `task`, `object_id`, `target_id`,
`success_conditions`를 요구한다. 현재 조건은 `inside`와 `released`다. Goal은 Observation과
분리되며, Goal이 장면이나 정답 객체만 골라 생성하지 않는다.

## 6. World Model 연결

중앙 `predict(observation, candidate)` 계약과 prediction source, 성공 확률, collision,
uncertainty, predicted next state를 정의했다. 빠른 지속 장면은 신경망 대신 명시적인
결정론 형상 fallback을 사용하며 실제 physics fallback 횟수를 0으로 기록한다. 기존 v5
neural+MuJoCo 구현을 공통 어댑터에 연결하는 일은 남아 있다.

## 7. Safety 구조

Safety Gate는 예측기와 독립되어 NaN/Inf, 최소 테이블 여유, entity 존재, 형상 불일치,
예측 collision을 결정론적으로 거부한다. 모델 confidence로 우회할 수 없다. 실제 Panda
joint/workspace/IK/contact check를 공통 gate로 어댑트하는 작업은 다음 단계다.

## 8. Persistent Scene과 E2E 결과

네 물체와 네 수용구가 처음부터 존재하고 identity와 완료 상태가 유지된다. triangle →
circle → rectangle 세 Goal을 scene reset 없이 3/3 완료했다. 삼각형 첫 행동에는 40도
오차를 주입했고 `ALIGNMENT_FAILURE` 기록 후 5개 대안 생성, 재계획 1회로 성공했다.
scene sequence는 0에서 4가 됐고 네 Episode가 schema 검증을 통과했다.

중요한 한계: 이 장면은 의사결정·상태 지속성용 결정론 환경이다. 하나의 Panda MuJoCo
model에 네 물체를 동시에 둔 물리 장면은 아니다.

## 9. Legacy 비교

Legacy는 fresh-state Panda MuJoCo에서 4/4, 최대 XY 오차 7.55mm, 로봇-테이블 접촉
0이다. Runtime은 지속 장면에서 3/3, action 4, replan 1, world-model call 8, memory hit
6이다. 전자는 물리 성능, 후자는 폐루프 구조의 증거이므로 성공률을 직접 비교하지 않는다.

## 10. Test 결과

- fast 범위: 13 passed, 약 0.10초
- 전체: 89 passed, 2 warnings, 280.54초
- warning은 유지보수가 끝난 JAXopt와 JAX 내부 deprecated API이며 현재 실패는 아니다.
- GitHub Actions는 물리·학습 의존성 없는 fast 범위만 자동 실행한다.

## 11. Jetson AGX Orin 64GB 잔여 문제

- JetPack/Python/CUDA/Torch/MuJoCo ARM64 실제 조합 검증
- JAX/Flax 형상 CNN의 Jetson 추론 전략 또는 portable 변환
- 한 MuJoCo 지속 장면의 Panda 어댑터
- 기존 15개 skill의 공통 실행 어댑터
- v5 neural+physics world model의 중앙 인터페이스 연결
- Panda joint/workspace/IK/contact gate의 중앙 연결
- 실제 camera calibration, 지연, 힘 제한, 비상 정지
- MemoryStore 동시 쓰기 제어
- H1/DROK 외부 자산 배포 방식

사용자명이 박힌 절대 경로는 제거했다. `FDB_UNITREE_RL_GYM`과 `FDB_DROK_ROOT`로 외부
저장소를 지정할 수 있다. TensorRT, 실제 로봇, Local LLM, VLA는 추가하지 않았다.

## 12. 다음 우선순위

1. 기존 `shape_insertion.py`를 한 model·data를 유지하는 MuJoCo persistent adapter로 분리
2. v5 SafeHybridPushPlanner를 중앙 World Model adapter로 연결
3. Panda reach/pick-place와 결정론 접촉 검사를 공통 Skill/Safety 계약으로 연결
4. Integration marker를 기존 물리 시험에 단계적으로 부여
5. 그 뒤 Jetson 호환 환경을 별도 장치에서 재현

## 최종 평가

현재 FDB는 구조화 Goal을 받고, 전체 장면을 관찰하고, Episode를 찾고, 후보를 예측·안전
검사한 뒤 행동하며, 정렬 실패를 분류해 재계획할 수 있다. 아직 이 폐루프 전체가 한 Panda
물리 장면이나 실제 로봇에서 동작하지 않고, 기존 skill 대부분도 중앙 호출 어댑터가 없다.
따라서 **중앙 Runtime 기반은 준비됐지만 Jetson 배포 준비 완료 상태는 아니다.**
