from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from fdb.memory import EpisodeStore

from .pose_reach import PandaPoseReachExperiment


def main() -> None:
    experiment = PandaPoseReachExperiment()
    selected, candidates = experiment.run()
    episode = EpisodeStore(Path("episodes")).write(
        {
            "stage": "v2",
            "task": "Reach a Cartesian target while preserving the Panda hand orientation.",
            "observation": {"robot": "franka_emika_panda", "target_position": experiment.target_position},
            "goal": {"position_error_m_max": 0.02, "orientation_error_rad_max": 0.03},
            "candidate_plans": [asdict(item) for item in candidates],
            "decision": {
                "selected_plan_id": selected.plan.plan_id,
                "reason": "Highest objective 6D execution score.",
                "predicted_evaluation": asdict(selected),
            },
            "execution": {"target_qpos": selected.target_qpos},
            "result": {"final_state": {}, "objective_evaluation": asdict(selected), "user_evaluation": None},
            "reflection": {
                "success_factors": ["Matched world-frame orientation error to MuJoCo angular Jacobian."],
                "failure_causes": [],
                "learned": "Quaternion error must be expressed in the Jacobian frame before DLS.",
                "next_experiment": "Evaluate pose targets and reject obstacle-contacting paths.",
                "confidence": 0.8,
                "unresolved_questions": ["How robust is 6D reach across the workspace?"],
            },
            "sources": ["research/research_20260917_panda_model.json"],
            "skills_used": ["v2.panda_pose_reach"],
        }
    )
    print(json.dumps({"selected": asdict(selected), "episode": str(episode)}, indent=2))


if __name__ == "__main__":
    main()
