from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Capability:
    name: str
    score: int
    target: int
    evidence: str
    next_gate: str


CAPABILITIES = (
    Capability("시각 형상·fit 판단", 3, 4, "독립 합성 시험 safe-fit 94.7%, 카메라 정사각형 4/4", "실제 렌더 원·삼각형·직사각형으로 확장"),
    Capability("한국어 목표 연결", 2, 4, "정해진 형상 별칭과 삽입 동사, 불완전 명령 거부", "관계·순서·수식어를 포함한 조합 명령"),
    Capability("접촉 안전 조작", 3, 4, "삽입 중심 오차 7.76mm 이하, 로봇-테이블 접촉 0", "접촉력 기반 미세 탐색과 실제 센서 잡음"),
    Capability("실패 인식·복구", 2, 4, "40도 오류 실패 뒤 후보 시뮬레이션으로 복구", "관측 기반 원인 분류와 여러 실패 종류 복구"),
    Capability("실행 전 상식 제약", 3, 4, "손·물체 바닥 여유를 실행 전 검사", "충돌·가림·불안정 파지까지 예측"),
    Capability("사람 동작 모방", 3, 4, "신경망 생성 10회 교대 착지, 5 stride", "속도·방향·보폭 명령으로 재조합"),
    Capability("동역학 보행", 0, 4, "PD 궤적 추종 최선도 1.212초에 낙상; 10걸음은 kinematic 생성", "접촉·IMU 입력 정책으로 10걸음 물리 rollout"),
    Capability("새 상황 전이", 1, 4, "각 과제의 제한된 변형만 검증", "보지 못한 배치·물체·마찰에서 사전 고정 시험"),
    Capability("지속 자율 학습", 1, 4, "상태·다음 gate 저장과 예약 실행은 있으나 경험 재학습은 수동", "경험 버퍼→평가→재학습→회귀검사 폐루프"),
)


def build_profile() -> dict[str, object]:
    total = sum(item.score for item in CAPABILITIES)
    target = sum(item.target for item in CAPABILITIES)
    # 가장 낮은 정규화 점수에서, 안전하고 객관적으로 시험 가능한 gate를 우선한다.
    weakest = min(CAPABILITIES, key=lambda item: (item.score / item.target, -item.target))
    return {
        "schema_version": "1.0",
        "operational_score": total,
        "operational_target": target,
        "target_completion_percent": round(100 * total / target, 1),
        "human_age_equivalent": None,
        "age_interpretation": "사람의 발달연령으로 환산할 수 없는 좁은 과제형 시스템",
        "capabilities": [asdict(item) for item in CAPABILITIES],
        "selected_next_gate": {"capability": weakest.name, "task": weakest.next_gate},
        "completion_rule": "모든 항목 4점이며 보지 못한 시험 세트에서 안전 기준을 통과해야 목표 달성",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FDB 능력 간격과 다음 자율 학습 gate 계산")
    parser.add_argument("--output", type=Path, default=Path("state/capability_profile.json"))
    args = parser.parse_args()
    profile = build_profile()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(profile, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
