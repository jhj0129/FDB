"""Reproduce the retained Phase 2.5 matrix without overwriting x86 evidence."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from fdb.core.phase25_benchmark import run_benchmark

    stages = [("baseline", (0,), ("triangle", "circle", "rectangle")),
              ("position", (0,), ("square", "circle", "triangle", "rectangle")),
              ("yaw", (0,), ("square", "circle", "triangle", "rectangle")),
              ("target", (0,), ("square", "circle", "triangle", "rectangle")),
              ("combined", (0,), ("square", "circle", "triangle", "rectangle")),
              ("ablation", (0,), ("triangle", "circle", "rectangle")),
              ("boundary", (1, 2), ("square", "circle", "triangle", "rectangle"))]
    records, worlds, stages_done = [], [], []
    started = time.perf_counter()
    with (args.output_directory / "tegrastats.log").open("w") as log:
        telemetry = subprocess.Popen(["tegrastats", "--interval", "1000"], stdout=log, stderr=subprocess.STDOUT)
        try:
            for suite, seeds, goals in stages:
                directory = args.output_directory / suite
                summary = run_benchmark(suite=suite, seeds=seeds, goals=goals, output_directory=directory)
                records.extend(summary["records"])
                for path in sorted(directory.rglob("summary.json")):
                    payload = json.loads(path.read_text())
                    episodes = [json.loads(Path(r["episode_paths"][0]).read_text()) for r in payload["results"]]
                    worlds.append({"path": str(path), "reset_count": payload["single_world_reset_count"],
                        "all_goals_preserved": payload["all_goals_preserved"],
                        "table_contact_steps": max(e["result"]["objective_evaluation"]["ground_truth"]["robot_table_contact_steps"] for e in episodes)})
                stages_done.append(suite)
                aggregate = {"stages": stages_done, "tasks": len(records),
                    "successes": sum(r["success"] for r in records),
                    "actions": sum(r["actions"] for r in records),
                    "replans": sum(r["replans"] for r in records),
                    "worlds": worlds, "records": records,
                    "elapsed_s": time.perf_counter() - started}
                (args.output_directory / "reproduction.json").write_text(json.dumps(aggregate, indent=2) + "\n")
                print(json.dumps({k: aggregate[k] for k in ("stages", "tasks", "successes", "actions", "elapsed_s")}), flush=True)
                if not all(r["success"] and r["all_goals_preserved"] for r in records):
                    raise RuntimeError("Reproduction task failed; evidence retained")
                if any(w["table_contact_steps"] or w["reset_count"] != 1 for w in worlds):
                    raise RuntimeError("Reproduction safety/persistence gate failed")
            assert len(records) == 118 and len(worlds) == 31
        finally:
            telemetry.terminate()
            telemetry.wait(timeout=10)


if __name__ == "__main__":
    main()
