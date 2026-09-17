from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from fdb.memory import EpisodeStore
from .pick_place import PandaPickPlaceExperiment


def main() -> None:
    experiment = PandaPickPlaceExperiment()
    selected, candidates = experiment.run()
    episode = EpisodeStore(Path("episodes")).write({
        "stage": "v2", "task": "Pick a box and place it in a target region.",
        "observation": {"source": experiment.source, "target": experiment.target},
        "goal": {"final_xy_error_m_max": 0.05, "released": True},
        "candidate_plans": [asdict(item) for item in candidates],
        "decision": {"selected_plan_id": selected.plan.plan_id, "reason": "Task completion first, then final placement error.", "predicted_evaluation": asdict(selected)},
        "execution": {"sequence": ["pregrasp", "grasp", "lift", "transfer", "lower", "release", "retreat"]},
        "result": {"final_state": {"object_z": selected.final_object_z}, "objective_evaluation": asdict(selected), "user_evaluation": None},
        "reflection": {"success_factors": ["Reused pose reach and grasp-height evidence."], "failure_causes": [], "learned": "Place height can be selected through full candidate futures.", "next_experiment": "Randomize object and target properties.", "confidence": 0.8, "unresolved_questions": ["What object variation envelope preserves first-attempt success?"]},
        "sources": ["research/research_20260917_panda_model.json"], "skills_used": ["v2.panda_pick_place"]
    })
    print(json.dumps({"selected": asdict(selected), "candidates": [asdict(item) for item in candidates], "episode": str(episode)}, indent=2))


if __name__ == "__main__":
    main()
