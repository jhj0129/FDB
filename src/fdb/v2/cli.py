from __future__ import annotations

import json
from pathlib import Path

from fdb.memory import EpisodeStore

from .inspector import inspect_robot
from .reach import PandaReachExperiment


def main() -> None:
    self_model = inspect_robot()
    robot_path = Path("robots/franka_emika_panda/self_model.json")
    robot_path.parent.mkdir(parents=True, exist_ok=True)
    robot_path.write_text(json.dumps(self_model, indent=2) + "\n", encoding="utf-8")
    experiment = PandaReachExperiment()
    selected, candidates = experiment.run()
    episode = EpisodeStore(Path("episodes")).write(
        {
            "stage": "v2",
            "task": "Reach the Cartesian target with the Franka Panda hand.",
            "observation": {"robot_self_model": str(robot_path), "target_m": experiment.target},
            "goal": {"execution_error_m_max": 0.02, "joint_limit_violations": 0},
            "candidate_plans": [experiment.as_dict(item) for item in candidates],
            "decision": {
                "selected_plan_id": selected.plan.plan_id,
                "reason": "Successful execution first, then objective score and IK iterations.",
                "predicted_evaluation": experiment.as_dict(selected),
            },
            "execution": {"joint_position_target": selected.target_qpos},
            "result": {
                "final_state": {},
                "objective_evaluation": experiment.as_dict(selected),
                "user_evaluation": None,
            },
            "reflection": {
                "success_factors": ["Used model joint limits and MuJoCo body Jacobian."],
                "failure_causes": [],
                "learned": "A sourced robot model can be inspected and controlled without guessed morphology.",
                "next_experiment": "Add orientation-aware reach, collision filtering, and grasp targets.",
                "confidence": 0.8,
                "unresolved_questions": ["How should collision-free pose candidates be generated?"],
            },
            "sources": ["research/research_20260917_panda_model.json"],
            "skills_used": ["v2.panda_reach"],
        }
    )
    print(json.dumps({"selected": experiment.as_dict(selected), "episode": str(episode)}, indent=2))


if __name__ == "__main__":
    main()
