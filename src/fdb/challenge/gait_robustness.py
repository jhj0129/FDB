from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .human_gait import simulate_gait


@dataclass(frozen=True)
class Scenario:
    name: str
    friction_scale: float = 1.0
    push_xy_fraction: tuple[float, float] = (0.0, 0.0)


SCENARIOS = (
    Scenario("nominal"),
    Scenario("low_friction_50pct", friction_scale=0.5),
    Scenario("low_friction_80pct", friction_scale=0.8),
    Scenario("high_friction_120pct", friction_scale=1.2),
    Scenario("forward_push_5pct_weight", push_xy_fraction=(0.05, 0.0)),
    Scenario("lateral_push_5pct_weight", push_xy_fraction=(0.0, 0.05)),
)

HOLDOUT_SCENARIOS = (
    Scenario("holdout_friction_35pct", friction_scale=0.35),
    Scenario("holdout_friction_65pct", friction_scale=0.65),
    Scenario("holdout_friction_140pct", friction_scale=1.4),
    Scenario("holdout_diagonal_push_3pct", push_xy_fraction=(0.03, 0.03)),
    Scenario("holdout_forward_push_7pct", push_xy_fraction=(0.07, 0.0)),
    Scenario("holdout_lateral_push_7pct", push_xy_fraction=(0.0, 0.07)),
    Scenario("holdout_diagonal_push_5pct", push_xy_fraction=(0.05, 0.05)),
)


def run() -> dict[str, object]:
    runs = []
    for robot in ("unitree_g1", "robotis_op3", "booster_t1"):
        variant = "momentum_feedback" if robot == "robotis_op3" else "pitch_feedback"
        for split, scenarios in (("development", SCENARIOS), ("holdout", HOLDOUT_SCENARIOS)):
            for scenario in scenarios:
                result, _, _ = simulate_gait(
                    robot,
                    variant,
                    floor_friction_scale=scenario.friction_scale,
                    push_xy_fraction=scenario.push_xy_fraction,
                )
                runs.append({"split": split, "scenario": asdict(scenario), "result": asdict(result)})
    return {
        "experiment": "contact-gated gait robustness",
        "robots": ["unitree_g1", "robotis_op3", "booster_t1"],
        "development_scenario_count_per_robot": len(SCENARIOS),
        "holdout_scenario_count_per_robot": len(HOLDOUT_SCENARIOS),
        "success_definition": "human_gait.py의 접촉 기반 보행 gate",
        "summary": {
            robot: {
                split: {
                    "walking_successes": sum(
                        run["result"]["success"] for run in runs
                        if run["result"]["robot"] == robot and run["split"] == split
                    ),
                    "upright_completions": sum(
                        run["result"]["upright_complete"] for run in runs
                        if run["result"]["robot"] == robot and run["split"] == split
                    ),
                    "total": len(scenarios),
                }
                for split, scenarios in (("development", SCENARIOS), ("holdout", HOLDOUT_SCENARIOS))
            }
            for robot in ("unitree_g1", "robotis_op3", "booster_t1")
        },
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="접촉 기반 휴머노이드 보행 강건성 실험")
    parser.add_argument(
        "--output", type=Path,
        default=Path("experiments/0022_gait_robustness.results.json"),
    )
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**report["summary"], "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
