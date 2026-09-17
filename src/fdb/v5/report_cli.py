from __future__ import annotations

import json
from pathlib import Path

from fdb.memory import EpisodeStore


def main() -> None:
    neural = json.loads(Path("experiments/0014_v5_neural_metrics.json").read_text(encoding="utf-8"))
    safety = json.loads(Path("experiments/0015_v5_common_sense_safety.results.json").read_text(encoding="utf-8"))
    demonstration = json.loads(Path("artifacts/fdb_v5_neural_prediction.json").read_text(encoding="utf-8"))
    panda = json.loads(Path("artifacts/fdb_final_pick_place.json").read_text(encoding="utf-8"))
    episode = EpisodeStore(Path("episodes")).write({
        "stage": "v5",
        "task": "학습형 물리 예측과 인간적인 조작 안전 게이트를 결합한다.",
        "observation": {
            "dataset_samples": neural["train_count"] + neural["validation_count"] + 1000,
            "ensemble_size": neural["ensemble_size"],
            "unsafe_manipulation_examples": 5,
        },
        "goal": {
            "test_mean_position_error_m_max": 0.03,
            "ood_physics_fallback": True,
            "robot_table_contact_steps": 0,
        },
        "candidate_plans": [
            {"plan_id": "neural_ensemble", "test_metrics": neural["splits"]["test"]},
            {"plan_id": "constant_baseline", "test_metrics": neural["constant_baseline_test"]},
        ],
        "decision": {
            "selected_plan_id": "safe_hybrid_neural_physics",
            "reason": "분포 안에서는 앙상블 예측을 사용하고, 분포 밖 또는 불확실한 입력은 MuJoCo로 되돌린다. 조작 충돌은 학습 점수보다 우선하는 하드 게이트로 거절한다.",
            "predicted_evaluation": demonstration,
        },
        "execution": {
            "neural_video": "artifacts/fdb_v5_neural_prediction.mp4",
            "safe_manipulation_video": "artifacts/fdb_final_pick_place.mp4",
        },
        "result": {
            "final_state": {"model": "models/v5/push_dynamics_ensemble.npz"},
            "objective_evaluation": {
                "test": neural["splits"]["test"],
                "ood": neural["splits"]["ood"],
                "demonstration": demonstration,
                "safe_manipulation": panda,
                "minimum_safe_grasp_clearance_m": safety["minimum_observed_safe_grasp_clearance_m"],
                "minimum_safe_place_hand_z_m": safety["minimum_observed_safe_place_hand_z_m"],
            },
            "user_evaluation": None,
        },
        "reflection": {
            "success_factors": ["앙상블 불확실성", "분포 밖 하드 fallback", "접촉 우선 안전 게이트", "부드러운 5차 궤적"],
            "failure_causes": ["기존 궤적은 성공 판정 중에도 양쪽 손가락이 테이블을 관통했다."],
            "learned": "작업 성공과 상식적인 안전 동작은 별도 지표로 검증해야 한다.",
            "next_experiment": "카메라 관측 잡음과 온라인 같은 상태 복구를 추가한다.",
            "confidence": 0.9,
            "unresolved_questions": ["학습형 세계모델이 Panda 접촉 동역학까지 확장될 때 필요한 데이터 규모는 얼마인가?"],
        },
        "sources": ["experiments/0014_v5_neural_world_model.md", "experiments/0015_v5_common_sense_safety.md"],
        "skills_used": ["v5.neural_push_world_model", "v2.panda_pick_place"],
    })
    print(json.dumps({"episode": str(episode)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
