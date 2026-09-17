from __future__ import annotations

import json
from pathlib import Path

from fdb.memory import EpisodeStore
from .morphology import GenericMorphologyInspector


MODELS = ("franka_emika_panda", "universal_robots_ur5e", "kuka_iiwa_14")


def main() -> None:
    inspector = GenericMorphologyInspector()
    summaries = []
    for name in MODELS:
        self_model = inspector.inspect_menagerie(name)
        path = Path("robots") / name / "auto_self_model.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self_model, indent=2) + "\n", encoding="utf-8")
        summaries.append({"robot": name, "path": str(path), "joints": len(self_model["joints"]), "end_effectors": self_model["end_effector_candidates"]})
    episode = EpisodeStore(Path("episodes")).write({
        "stage": "v3", "task": "Derive self-models for three unfamiliar/open-source robot morphologies.",
        "observation": {"models": list(MODELS)}, "goal": {"self_model_count": 3, "no_guessed_limits": True},
        "candidate_plans": summaries,
        "decision": {"selected_plan_id": "generic_compiled-model-inspection", "reason": "Use one inspector and compiled source data for every robot.", "predicted_evaluation": {"models_written": len(summaries)}},
        "execution": {"outputs": [item["path"] for item in summaries]},
        "result": {"final_state": {}, "objective_evaluation": {"models_written": len(summaries)}, "user_evaluation": None},
        "reflection": {"success_factors": ["Model-independent tree, joint, actuator, and site traversal."], "failure_causes": [], "learned": "Tool sites provide stronger end-effector evidence than leaf-body heuristics.", "next_experiment": "Perform small joint motions to verify inferred end-effectors.", "confidence": 0.85, "unresolved_questions": ["How should branched and mobile morphologies be classified?"]},
        "sources": ["research/research_20260917_panda_model.json"], "skills_used": []
    })
    print(json.dumps({"models": summaries, "episode": str(episode)}, indent=2))


if __name__ == "__main__":
    main()
