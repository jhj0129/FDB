from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from fdb.memory import EpisodeStore
from .recovery import ResetBasedRecoveryExperiment


def main() -> None:
    result = ResetBasedRecoveryExperiment().run()
    episode = EpisodeStore(Path("episodes")).write({
        "stage": "v2", "task": "Detect and recover from a deliberately missed grasp in sandbox.",
        "observation": {"initial_attempt": asdict(result.initial_attempt)},
        "goal": {"diagnose_failure": True, "recovery_success": True},
        "candidate_plans": [{"plan": "retry_validated_low_grasp", "reset_based": True}],
        "decision": {"selected_plan_id": "retry_validated_low_grasp", "reason": "Missed-grasp metrics match prior failure evidence.", "predicted_evaluation": {"diagnosis": asdict(result.diagnosis)}},
        "execution": {"recovery_attempt": asdict(result.recovery_attempt)},
        "result": {"final_state": {}, "objective_evaluation": {"recovered": result.recovered, "reset_based": result.reset_based}, "user_evaluation": None},
        "reflection": {"success_factors": ["Objective failure classification selected a previously validated alternative."], "failure_causes": ["High grasp pose did not retain or lift the object."], "learned": "Reset-based sandbox retry works but does not prove online recovery.", "next_experiment": "Recover in the same world from the observed post-failure object pose.", "confidence": 0.9, "unresolved_questions": ["How should object pose be reacquired after failure?"]},
        "sources": [], "skills_used": ["v2.panda_grasp_lift"]
    })
    print(json.dumps({"result": asdict(result), "episode": str(episode)}, indent=2))


if __name__ == "__main__":
    main()
