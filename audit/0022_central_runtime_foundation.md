# 감사 0022 — 중앙 FDBRuntime 기반

독립 script를 삭제하지 않고 중앙 폐루프를 새로 추가했다. 구조화 Goal, Goal과 분리된
Observation, Skill registry, World Model interface, 결정론 Safety Gate, Episode 기록,
failure taxonomy, 재계획, profiling을 한 경로에 연결했다.

빠른 지속 장면은 물리 성공을 주장하지 않는다. 기존 MuJoCo 결과는 그대로 보존하며
실행 가능한 Core 어댑터와 검색만 가능한 legacy skill metadata를 구분한다. Memory 수정은
원본 revision을 audit에 보존하고 두 개 이상의 서로 다른 evidence 없이는 long-term으로
승격하지 않는다.
