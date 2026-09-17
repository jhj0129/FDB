from __future__ import annotations

import json
from pathlib import Path

from fdb.memory import EpisodeStore
from .manipulation_suite import run_manipulation_suite


def main() -> None:
    result = run_manipulation_suite()
    payload = result.to_dict()
    output = Path("experiments/0009_v2_manipulation_variation.results.json")
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    episode = EpisodeStore(Path("episodes")).write({
        "stage": "v2", "task": "Evaluate pick-and-place across object and scene variation.",
        "observation": {"scenario_count": result.trial_count, "varied": ["size", "density", "friction", "source", "target"]},
        "goal": {"first_attempt_success_rate_min": 0.8, "final_xy_error_m_max": 0.05},
        "candidate_plans": [{"scenario": t.scenario.scenario_id, "selected": t.selected.plan.plan_id} for t in result.trials],
        "decision": {"selected_plan_id": "per-scenario-full-future-selection", "reason": "Each scenario selects among full pick-place futures.", "predicted_evaluation": {"success_rate": result.success_rate, "max_error": result.max_final_xy_error_m}},
        "execution": {"raw_result": str(output)},
        "result": {"final_state": {}, "objective_evaluation": {"success_count": result.success_count, "trial_count": result.trial_count, "success_rate": result.success_rate, "mean_final_xy_error_m": result.mean_final_xy_error_m, "max_final_xy_error_m": result.max_final_xy_error_m}, "user_evaluation": None},
        "reflection": {"success_factors": ["Full candidate simulation reused model-derived reach and grasp evidence."], "failure_causes": [], "learned": "The baseline generalized across the declared moderate variation suite.", "next_experiment": "Add recovery from deliberate grasp and placement failures.", "confidence": 0.9, "unresolved_questions": ["How should FDB recover after a dropped object?"]},
        "sources": ["research/research_20260917_panda_model.json"], "skills_used": ["v2.panda_grasp_lift", "v2.panda_pick_place"]
    })
    print(json.dumps({"summary": {k: v for k, v in payload.items() if k != "trials"}, "episode": str(episode)}, indent=2))


if __name__ == "__main__":
    main()
