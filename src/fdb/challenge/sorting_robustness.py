from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from .multi_object_sorting import OBJECTS, run_sorting


def _scenario(name: str, sources, targets):
    objects = tuple(
        replace(obj, source_xy=source, target_xy=target)
        for obj, source, target in zip(OBJECTS, sources, targets)
    )
    return name, objects


SCENARIOS = (
    ("nominal", OBJECTS),
    _scenario(
        "source_target_jitter_a",
        ((0.42, 0.14), (0.50, 0.18), (0.56, 0.20)),
        ((0.56, -0.17), (0.59, 0.01), (0.54, 0.17)),
    ),
    _scenario(
        "source_target_jitter_b",
        ((0.38, 0.19), (0.47, 0.14), (0.60, 0.16)),
        ((0.54, -0.15), (0.57, -0.01), (0.56, 0.15)),
    ),
)


def run() -> dict[str, object]:
    runs = []
    for name, objects in SCENARIOS:
        metrics = run_sorting(objects=objects)
        metrics.pop("model")
        metrics.pop("data")
        runs.append({
            "scenario": name,
            "objects": [asdict(obj) for obj in objects],
            "metrics": metrics,
        })
    return {
        "experiment": "multi-object sorting position robustness",
        "scenario_count": len(SCENARIOS),
        "all_objects_sorted": all(run["metrics"]["sorting_success_count"] == 3 for run in runs),
        "all_hard_safety_passed": all(run["metrics"]["hard_safety_pass"] for run in runs),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Panda 다중 물체 위치 변화 강건성 실험")
    parser.add_argument(
        "--output", type=Path,
        default=Path("experiments/0023_sorting_robustness.results.json"),
    )
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "all_objects_sorted": report["all_objects_sorted"],
        "all_hard_safety_passed": report["all_hard_safety_passed"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
