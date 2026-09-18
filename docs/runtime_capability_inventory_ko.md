# FDB 런타임 통합 전 능력 목록

기준일은 2026-09-19이다. 성공 표시는 저장된 실험과 테스트의 범위 안에서만 유효하며
실제 로봇 성능을 뜻하지 않는다.

| 기능 | 입력 → 출력 | 실행 위치 | 의존성 | 현재 증거 | Core Skill 상태 | Jetson 이전 위험 |
| --- | --- | --- | --- | --- | --- | --- |
| v0 2D 결정 루프 | 물체·목표 → 후보 선택 | `fdb` | 표준 Python | 단위 시험 통과 | 메타데이터만 | 낮음 |
| v1 물리 push | 힘 후보 → MuJoCo rollout | `fdb-v1` | MuJoCo | feedback skill 검증 | 메타데이터만 | ARM64 wheel 확인 |
| Panda reach/pose | 목표 자세 → 관절 목표·오차 | `fdb-v2*` | MuJoCo, Menagerie | reach 검증, pose 후보 | 메타데이터만 | 모델 자산·IK 비용 |
| Panda grasp/pick-place | source/target → 물리 결과 | `fdb-v2-pick-place` | 위와 같음 | 접촉 0, validated | 메타데이터만 | 실제 힘 센서 미검증 |
| 형상 CNN | 카메라 RGB → 형상·fit·회전 | `fdb-shape-fit-train` | JAX/Flax/OpenCV | 독립 safe-fit 94.7% | 런타임 어댑터 입력 근거 | Jetson JAX가 가장 큰 위험 |
| 형상 삽입 | 카메라·장면 → 삽입 결과 | `fdb-shape-insertion` | MuJoCo, OpenCV, CNN | 4형상 4/4 | `shape_insert` 결정 어댑터 | 현재 물리 장면은 fresh-state |
| 삽입 복구 | 실패 각도 → 대안 후보 | `fdb-insertion-recovery` | 형상 삽입 전체 | 40도 오류 복구 | 일반 replan으로 이식 | 후보 비용 큼 |
| 로봇 구조 분석 | MJCF → self model | `fdb-v3-inspect` | MuJoCo/Menagerie | 6개 구조 기록 | 메타데이터만 | 자산 경로 |
| body probing | 관절 동작 → 관측 motion | `fdb-v4-probe` | MuJoCo | Panda/UR5e/iiwa | 메타데이터만 | headless 확인 필요 |
| v5 세계모델 | 상태·행동 → 다음 위치·불확실성 | `fdb-v5-hybrid` | NumPy, 선택적 Torch | 500 시험 10.72mm | 중앙 인터페이스만 연결 | push 전용, 범용 아님 |
| 학습 안전 screen | Panda 자세 → 위험도 | `fdb-v5-safety-*` | NumPy/Torch | 위험 11/11 거부 | 결정론 gate와 분리 | 실제 안전 승인 아님 |
| 물체 분류 | RGB·크기 → 세 클래스 | `fdb-multi-object-sort` | NumPy/MuJoCo | 변형 9/9 | 메타데이터 미등록 | 합성 색상 편향 |
| H1 보행 | 속도·위상 → 관절 action | `fdb-unitree-h1-walk` | 외부 Unitree 저장소, MuJoCo | 20초·40걸음 | 아직 Core 미연결 | 절대 경로, 외부 정책 |
| DeepMimic | 참조 motion → PPO 정책 | `fdb-deepmimic-*` | JAX/Brax 계열 | 현재 실패도 보존 | 아직 Core 미연결 | JAX/CUDA/메모리 |
| Episode | task dict → 불변 JSON | `EpisodeStore` | 표준 Python | 기존 및 Runtime episode | 중앙 Runtime 기본 사용 | 파일 수 증가 |
| 계층 Memory | 지식 → revision/promote/archive | `MemoryStore` | 표준 Python | 단위 시험 | 중앙 조회는 Episode부터 | 동시 쓰기 잠금 미구현 |

## 통합 판단

기존 `skills/*.json` 15개는 검색 가능한 카탈로그로 읽힌다. 현재 중앙 Runtime에서 직접
실행 가능한 어댑터는 `shape_insert` 하나다. 나머지를 실행 가능하다고 표시하지 않은 것은
메타데이터와 호출 가능한 코드 사이를 정직하게 구분하기 위해서다.

현재 Runtime은 `observe → memory retrieval → candidate generation → prediction → hard
safety → selection → execution → objective evaluation → episode → replan`을 수행한다.
Goal은 별도 상태 조건이며 Observation은 네 물체와 네 수용구를 Goal과 무관하게 모두
노출한다.
