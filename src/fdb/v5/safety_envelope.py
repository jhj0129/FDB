from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from fdb.v2.pick_place import PandaPickPlaceExperiment, PlaceCandidate


def measure_safety_envelope() -> dict[str, object]:
    grasp_trials = []
    for clearance in (0.04, 0.06, 0.08, 0.10):
        result = PandaPickPlaceExperiment(grasp_clearance_m=clearance).run_candidate(
            PlaceCandidate("안전_배치", 0.465)
        )
        grasp_trials.append({"grasp_clearance_m": clearance, **asdict(result)})
        print(
            f"파지 여유 {clearance:.3f}m: 접촉 step {result.robot_table_contact_steps}, "
            f"관통 {1000 * result.deepest_robot_table_penetration_m:.3f}mm, 성공 {result.success}",
            flush=True,
        )
    place_trials = []
    for height in (0.385, 0.405, 0.425, 0.445, 0.465):
        result = PandaPickPlaceExperiment(grasp_clearance_m=0.10).run_candidate(
            PlaceCandidate("배치_높이_탐색", height)
        )
        place_trials.append({"place_hand_z_m": height, **asdict(result)})
        print(
            f"배치 hand z {height:.3f}m: 접촉 step {result.robot_table_contact_steps}, "
            f"관통 {1000 * result.deepest_robot_table_penetration_m:.3f}mm, 성공 {result.success}",
            flush=True,
        )
    safe_grasp = [trial["grasp_clearance_m"] for trial in grasp_trials if trial["success"]]
    safe_place = [trial["place_hand_z_m"] for trial in place_trials if trial["success"]]
    return {
        "label_policy": "비의도 로봇-테이블 접촉 0, 관통 0, 올바른 단계 순서일 때만 안전",
        "grasp_trials": grasp_trials,
        "place_trials": place_trials,
        "minimum_observed_safe_grasp_clearance_m": min(safe_grasp) if safe_grasp else None,
        "minimum_observed_safe_place_hand_z_m": min(safe_place) if safe_place else None,
    }


def main() -> None:
    output = Path("experiments/0015_v5_common_sense_safety.results.json")
    results = measure_safety_envelope()
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"저장: {output}")


if __name__ == "__main__":
    main()
