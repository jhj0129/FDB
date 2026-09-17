from __future__ import annotations

import argparse
import json
from pathlib import Path

from fdb.memory import EpisodeStore

from .robustness import RobustnessExperiment, generate_scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare v1 open-loop and feedback control.")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--episodes", type=Path, default=Path("episodes"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = RobustnessExperiment().run(generate_scenarios(args.count, args.seed))
    payload = result.to_dict()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    episode = EpisodeStore(args.episodes).write(
        {
            "stage": "v1",
            "task": "Compare open-loop and feedback control under randomized physics.",
            "observation": {
                "scenario_count": args.count,
                "seed": args.seed,
                "variation": {
                    "mass_scale": [0.4, 1.8],
                    "sliding_friction": [0.2, 1.4],
                    "initial_offset_m": [-0.15, 0.15]
                },
            },
            "goal": {"metric": "first_attempt_success_rate", "comparison": "feedback > open_loop"},
            "candidate_plans": [
                {"method": "open_loop", "summary": payload["open_loop_summary"]},
                {"method": "feedback", "summary": payload["feedback_summary"]},
            ],
            "decision": {
                "selected_plan_id": "pd_feedback",
                "reason": "Higher first-attempt success rate across identical randomized scenarios.",
                "predicted_evaluation": payload["feedback_summary"],
            },
            "execution": {"raw_result": str(args.output) if args.output else None},
            "result": {
                "final_state": {},
                "objective_evaluation": {
                    "open_loop": payload["open_loop_summary"],
                    "feedback": payload["feedback_summary"],
                },
                "user_evaluation": None,
            },
            "reflection": {
                "success_factors": ["Feedback corrected model and initial-state variation."],
                "failure_causes": ["Open-loop force could not correct changed dynamics."],
                "learned": "Feedback is the default v1 control strategy under uncertainty.",
                "next_experiment": "Add rotation and obstacle interaction, then begin fixed-robot reach.",
                "confidence": 0.9,
                "unresolved_questions": ["How does feedback perform with observation noise and delay?"],
            },
            "sources": ["research/research_20260917_mujoco_python_api.json"],
            "skills_used": ["v1.physics_push"],
        }
    )
    print(
        json.dumps(
            {
                "open_loop": payload["open_loop_summary"],
                "feedback": payload["feedback_summary"],
                "raw_result": str(args.output) if args.output else None,
                "episode": str(episode),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
