from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from fdb.memory import EpisodeStore

from .decision_loop import PhysicsDecisionLoop


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FDB v1 MuJoCo push task.")
    parser.add_argument("--task", default="Move the red object to the blue target")
    parser.add_argument("--episodes", type=Path, default=Path("episodes"))
    args = parser.parse_args()
    result = PhysicsDecisionLoop(EpisodeStore(args.episodes)).run(args.task)
    print(
        json.dumps(
            {
                "selected_plan": result.selected_prediction.plan.plan_id,
                "prediction": asdict(result.selected_prediction.evaluation),
                "execution": asdict(result.execution.evaluation),
                "final_position": result.execution.final_position,
                "episode": str(result.episode_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

