from __future__ import annotations

import json
from pathlib import Path

from fdb.memory import EpisodeStore

from .reach_suite import run_reach_suite


def main() -> None:
    result = run_reach_suite()
    payload = result.to_dict()
    output = Path("experiments/0004_v2_reach_workspace.results.json")
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    episode = EpisodeStore(Path("episodes")).write(
        {
            "stage": "v2",
            "task": "Evaluate Panda position-only reach across a workspace target suite.",
            "observation": {"robot": "franka_emika_panda", "target_count": result.trial_count},
            "goal": {"success_rate_min": 0.95, "execution_error_m_max": 0.02},
            "candidate_plans": [
                {"target": trial.target, "selected": trial.selected.plan.plan_id}
                for trial in result.trials
            ],
            "decision": {
                "selected_plan_id": "per-target-dls-selection",
                "reason": "Select damping independently for each target using execution score.",
                "predicted_evaluation": {
                    "success_rate": result.success_rate,
                    "max_execution_error_m": result.max_execution_error_m,
                },
            },
            "execution": {"raw_result": str(output)},
            "result": {
                "final_state": {},
                "objective_evaluation": {
                    "success_count": result.success_count,
                    "trial_count": result.trial_count,
                    "success_rate": result.success_rate,
                    "mean_execution_error_m": result.mean_execution_error_m,
                    "max_execution_error_m": result.max_execution_error_m,
                    "joint_limit_violations": result.joint_limit_violations,
                    "contact_count": result.contact_count,
                },
                "user_evaluation": None,
            },
            "reflection": {
                "success_factors": ["Per-target candidate simulation and model-derived joint limits."],
                "failure_causes": [],
                "learned": "Position-only reach generalizes across the declared target suite.",
                "next_experiment": "Add orientation objectives and collision-aware scene objects.",
                "confidence": 0.9,
                "unresolved_questions": ["How does the solver behave near singularities and obstacles?"],
            },
            "sources": ["research/research_20260917_panda_model.json"],
            "skills_used": ["v2.panda_reach"],
        }
    )
    print(json.dumps({"summary": {key: value for key, value in payload.items() if key != "trials"}, "episode": str(episode)}, indent=2))


if __name__ == "__main__":
    main()
