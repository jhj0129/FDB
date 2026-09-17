from __future__ import annotations

import argparse
import json
from pathlib import Path

from fdb.memory import EpisodeStore
from fdb.v0 import DecisionLoop
from fdb.v0.models import WorldState


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the FDB v0 decision loop.")
    parser.add_argument("--task", required=True, help="Natural-language relocation task")
    parser.add_argument("--world", type=Path, required=True, help="World-state JSON file")
    parser.add_argument("--episodes", type=Path, default=Path("episodes"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    world = WorldState.from_dict(json.loads(args.world.read_text(encoding="utf-8")))
    result = DecisionLoop(EpisodeStore(args.episodes)).run(args.task, world)
    print(
        json.dumps(
            {
                "success": result.evaluation.success,
                "score": result.evaluation.score,
                "selected_plan": result.selected_plan.plan_id,
                "final_state": result.final_state.to_dict(),
                "episode": str(result.episode_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

