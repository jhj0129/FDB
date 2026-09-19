from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .physical_cli import run_tasks
from .physical_scene import PhysicalPersistentShapeEnvironment


POSITION_LEVELS_MM = (0, 2, 5, 10, 20)
YAW_LEVELS_DEG = (0, 5, 10, 20, 40)
TARGET_LEVELS_MM = (0, 2, 5, 10)


def _experiment_matrix(suite: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if suite == "boundary":
        return [
            {"axis": "position_mm", "level": 20, "observation_mode": "camera",
             "position_noise_m": .020, "object_yaw_noise_deg": 0.0, "target_yaw_noise_deg": 0.0},
            {"axis": "yaw_deg", "level": 40, "observation_mode": "camera",
             "object_yaw_noise_deg": 40.0, "target_yaw_noise_deg": 0.0},
            {"axis": "target_mm", "level": 10, "observation_mode": "camera",
             "target_noise_m": .010, "object_yaw_noise_deg": 0.0, "target_yaw_noise_deg": 0.0},
            {"axis": "combined", "level": "P10_Y20_T10", "observation_mode": "camera",
             "position_noise_m": .010, "target_noise_m": .010,
             "object_yaw_noise_deg": 20.0, "target_yaw_noise_deg": 0.0},
        ]
    if suite in {"baseline", "all"}:
        rows.extend({"axis": "baseline", "level": 0, "observation_mode": mode}
                    for mode in ("oracle", "camera"))
    if suite in {"position", "all"}:
        rows.extend({"axis": "position_mm", "level": level, "observation_mode": "camera",
                     "position_noise_m": level / 1000.0,
                     "object_yaw_noise_deg": 0.0, "target_yaw_noise_deg": 0.0}
                    for level in POSITION_LEVELS_MM)
    if suite in {"yaw", "all"}:
        rows.extend({"axis": "yaw_deg", "level": level, "observation_mode": "camera",
                     "object_yaw_noise_deg": float(level), "target_yaw_noise_deg": 0.0}
                    for level in YAW_LEVELS_DEG)
    if suite in {"target", "all"}:
        rows.extend({"axis": "target_mm", "level": level, "observation_mode": "camera",
                     "target_noise_m": level / 1000.0,
                     "object_yaw_noise_deg": 0.0, "target_yaw_noise_deg": 0.0}
                    for level in TARGET_LEVELS_MM)
    if suite in {"combined", "all"}:
        rows.extend([
            {"axis": "combined", "level": "P2_Y5_T2", "observation_mode": "camera",
             "position_noise_m": .002, "target_noise_m": .002,
             "object_yaw_noise_deg": 5.0, "target_yaw_noise_deg": 0.0},
            {"axis": "combined", "level": "P5_Y10_T5", "observation_mode": "camera",
             "position_noise_m": .005, "target_noise_m": .005,
             "object_yaw_noise_deg": 10.0, "target_yaw_noise_deg": 0.0},
            {"axis": "combined", "level": "P10_Y20_T10", "observation_mode": "camera",
             "position_noise_m": .010, "target_noise_m": .010,
             "object_yaw_noise_deg": 20.0, "target_yaw_noise_deg": 0.0},
        ])
    if suite in {"ablation", "all"}:
        for neural in (True, False):
            for memory in (True, False):
                rows.append({
                    "axis": "ablation", "level": f"neural_{'on' if neural else 'off'}_memory_{'on' if memory else 'off'}",
                    "observation_mode": "camera", "use_neural": neural, "use_memory": memory,
                    "position_noise_m": .005, "target_noise_m": .005,
                    "object_yaw_noise_deg": 10.0, "target_yaw_noise_deg": 0.0,
                })
    return rows


def _episode_error(path: str) -> tuple[float | None, float | None, int, Counter[str]]:
    episode = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = [episode.get("initial_perception_error", {})]
    samples.extend(step.get("perception_error", {}) for step in episode.get("decision_steps", []))
    position = [sample.get("mean_translation_error_mm") for sample in samples
                if sample.get("mean_translation_error_mm") is not None]
    yaw = [sample.get("mean_abs_yaw_error_deg") for sample in samples
           if sample.get("mean_abs_yaw_error_deg") is not None]
    collisions = sum(
        int(step.get("execution", {}).get("metrics", {}).get("robot_table_contact_steps", 0) > 0)
        for step in episode.get("decision_steps", [])
    )
    failures: Counter[str] = Counter()
    for step in episode.get("decision_steps", []):
        failure = step.get("execution", {}).get("failure_type")
        if failure:
            failures[str(failure)] += 1
    return (
        statistics.fmean(position) if position else None,
        statistics.fmean(yaw) if yaw else None,
        collisions,
        failures,
    )


def run_benchmark(
    *, suite: str, seeds: tuple[int, ...], goals: tuple[str, ...], output_directory: Path,
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    failure_budget: Counter[str] = Counter()
    for experiment in _experiment_matrix(suite):
        for seed in seeds:
            run_dir = (
                output_directory / str(experiment["axis"]) / str(experiment["level"])
                / f"{experiment['observation_mode']}_seed_{seed}"
            )
            payload = run_tasks(
                goals=goals, seed=seed, output_directory=run_dir,
                use_neural=bool(experiment.get("use_neural", True)),
                use_memory=bool(experiment.get("use_memory", True)),
                position_noise_m=float(experiment.get("position_noise_m", 0.0)),
                target_noise_m=float(experiment.get("target_noise_m", 0.0)),
                observation_mode=str(experiment["observation_mode"]),
                object_yaw_noise_deg=experiment.get("object_yaw_noise_deg"),
                target_yaw_noise_deg=experiment.get("target_yaw_noise_deg"),
            )
            for result in payload["results"]:
                pos_error, yaw_error, collisions, failures = _episode_error(result["episode_paths"][0])
                failure_budget.update(failures)
                if not result["success"] and result["metrics"].get("final_failure_type"):
                    failure_budget[str(result["metrics"]["final_failure_type"])] += 1
                records.append({
                    "axis": experiment["axis"], "level": experiment["level"], "seed": seed,
                    "shape": result["shape"], "observation_mode": experiment["observation_mode"],
                    "neural": bool(experiment.get("use_neural", True)),
                    "memory": bool(experiment.get("use_memory", True)),
                    "success": bool(result["success"]), "actions": result["actions"],
                    "replans": result["replans"], "task_time_s": result["metrics"]["total_task_latency_s"],
                    "collisions": collisions, "position_error_mm": pos_error,
                    "yaw_error_deg": yaw_error,
                    "neural_calls": result["metrics"]["neural_world_model_calls"],
                    "physics_fallback_calls": result["metrics"]["physics_fallback_calls"],
                    "prediction_fallback_calls": result["metrics"]["deterministic_fallback_calls"],
                    "final_failure": result["metrics"].get("final_failure_type"),
                    "goal_preserved_at_end": payload["final_goal_preservation"][result["shape"]],
                    "all_goals_preserved": payload["all_goals_preserved"],
                })
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["axis"]), str(record["level"]), str(record["shape"]))].append(record)
    aggregates = []
    for (axis, level, shape), items in sorted(grouped.items()):
        aggregates.append({
            "axis": axis, "level": level, "shape": shape, "attempts": len(items),
            "successes": sum(item["success"] for item in items),
            "success_rate": sum(item["success"] for item in items) / len(items),
            "mean_actions": statistics.fmean(item["actions"] for item in items),
            "mean_replans": statistics.fmean(item["replans"] for item in items),
            "mean_task_time_s": statistics.fmean(item["task_time_s"] for item in items),
            "mean_position_error_mm": statistics.fmean(
                item["position_error_mm"] for item in items if item["position_error_mm"] is not None
            ),
            "mean_yaw_error_deg": statistics.fmean(
                item["yaw_error_deg"] for item in items if item["yaw_error_deg"] is not None
            ),
            "collisions": sum(item["collisions"] for item in items),
        })
    summary = {
        "schema_version": "1.0", "suite": suite, "seeds": list(seeds), "goals": list(goals),
        "records": records, "aggregates": aggregates, "failure_budget": dict(failure_budget),
    }
    (output_directory / "benchmark_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    if records:
        with (output_directory / "benchmark_records.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="FDB Phase 2.5 카메라·강건성 benchmark")
    parser.add_argument("--suite", choices=("baseline", "position", "yaw", "target", "combined", "ablation", "boundary", "all"), default="baseline")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--goals", default="square,circle,triangle,rectangle")
    parser.add_argument("--episodes", type=int, help="seeds 앞에서 사용할 episode 수")
    parser.add_argument("--output-directory", type=Path, default=Path("artifacts/phase2_5_benchmark"))
    args = parser.parse_args()
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    if args.episodes is not None:
        seeds = seeds[:args.episodes]
    goals = tuple(value.strip() for value in args.goals.split(",") if value.strip())
    invalid = set(goals).difference(PhysicalPersistentShapeEnvironment.SHAPES)
    if invalid or not seeds:
        parser.error(f"invalid goals={sorted(invalid)} or empty seeds")
    summary = run_benchmark(suite=args.suite, seeds=seeds, goals=goals, output_directory=args.output_directory)
    print(json.dumps({"suite": args.suite, "runs": len(summary["records"]),
                      "failure_budget": summary["failure_budget"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
