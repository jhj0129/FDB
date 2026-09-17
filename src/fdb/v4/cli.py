from __future__ import annotations

import json
from pathlib import Path

from fdb.memory import EpisodeStore
from fdb.v3.cli import MODELS
from .motion_probe import MotionProbe


def main() -> None:
    probe = MotionProbe()
    results = [probe.probe_menagerie(name) for name in MODELS]
    outputs = []
    for result in results:
        path = Path("robots") / result.robot / "motion_probe.json"
        path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
        outputs.append(str(path))
    total = sum(result.probe_count for result in results)
    verified = sum(result.verified_count for result in results)
    episode = EpisodeStore(Path("episodes")).write({
        "stage": "v4", "task": "Verify inferred robot body trees with bounded joint motion probes.",
        "observation": {"robots": list(MODELS), "delta_rad": 0.02},
        "goal": {"unexpected_affected_bodies": 0, "probe_success_rate": 1.0},
        "candidate_plans": [{"robot": r.robot, "probe_count": r.probe_count} for r in results],
        "decision": {"selected_plan_id": "bounded-per-joint-probe", "reason": "Small limit-respecting motions isolate causal downstream effects.", "predicted_evaluation": {"probe_count": total}},
        "execution": {"outputs": outputs},
        "result": {"final_state": {}, "objective_evaluation": {"verified": verified, "total": total, "success_rate": verified / total}, "user_evaluation": None},
        "reflection": {"success_factors": ["Compared both translation and orientation changes against the compiled tree."], "failure_causes": [], "learned": "Small motion probes verify structural downstream predictions for the tested serial arms.", "next_experiment": "Use probes to identify capability and end-effector sensitivity automatically.", "confidence": 0.9, "unresolved_questions": ["How should coupled, closed-chain, and mobile joints be probed?"]},
        "sources": ["experiments/0011_v3_generic_morphology.md"], "skills_used": ["v3.generic_morphology_inspection"]
    })
    print(json.dumps({"verified": verified, "total": total, "results": [r.to_dict() for r in results], "episode": str(episode)}, indent=2))


if __name__ == "__main__":
    main()
