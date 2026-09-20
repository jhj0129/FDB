"""Produce small, shareable measurement summaries from local raw evidence."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import statistics

import numpy as np


def stats(values):
    if not values:
        return None
    return {"min": min(values), "mean": statistics.fmean(values),
            "p50": float(np.median(values)), "p95": float(np.percentile(values, 95)), "max": max(values)}


def telemetry(path):
    series = defaultdict(list)
    lines = path.read_text().splitlines()
    for line in lines:
        ram = re.search(r"RAM (\d+)/(\d+)MB", line)
        gpu = re.search(r"GR3D_FREQ (\d+)%", line)
        cpu = re.search(r"CPU \[([^]]+)\]", line)
        if ram:
            series["ram_used_mb"].append(int(ram[1]))
        if gpu:
            series["gpu_util_percent"].append(int(gpu[1]))
        if cpu:
            cores = re.findall(r"(\d+)%@(\d+)", cpu[1])
            if cores:
                series["cpu_mean_active_core_percent"].append(statistics.fmean(int(v[0]) for v in cores))
                series["cpu_max_clock_mhz"].append(max(int(v[1]) for v in cores))
        for sensor, value in re.findall(r"([\w]+)@([\d.]+)C", line):
            series[f"temperature_{sensor}_c"].append(float(value))
        for rail, value in re.findall(r"(VDD_\w+|VIN_\w+) (\d+)mW/", line):
            series[f"power_{rail}_mw"].append(int(value))
    result = {"lines": len(lines), "metrics": {k: stats(v) for k, v in series.items()}}
    clocks = path.with_name("clocks.jsonl")
    if clocks.exists():
        samples = [json.loads(line) for line in clocks.read_text().splitlines()]
        result["gpu_clock_hz"] = stats([s["gpu_hz"] for s in samples if s.get("gpu_hz") is not None])
        result["cooling_max_states"] = {
            key: max(s.get(key) or 0 for s in samples)
            for key in samples[0] if key.startswith("cooling_")
        } if samples else {}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.input_directory
    result = {"baseline_commit": "465ac48f706b0093893f5b44b549a90485260c83", "artifacts": str(root)}
    for name in ("smoke", "persistent", "combined_smoke", "recovery"):
        payload = json.loads((root / name / "summary.json").read_text())
        result[name] = {key: payload[key] for key in (
            "successes", "total", "single_world_reset_count", "all_goals_preserved", "observation_mode")}
        result[name]["tasks"] = [{"shape": r["shape"], "success": r["success"], **r["metrics"]} for r in payload["results"]]
        result[name]["table_contact_steps"] = max(
            json.loads(Path(r["episode_paths"][0]).read_text())["result"]["objective_evaluation"]["ground_truth"]["robot_table_contact_steps"]
            for r in payload["results"])
    full = json.loads((root / "full/reproduction.json").read_text())
    result["reproduction"] = {k: full[k] for k in ("tasks", "successes", "actions", "replans", "elapsed_s", "stages")}
    result["reproduction"].update({"worlds": len(full["worlds"]),
        "worlds_preserved": sum(w["all_goals_preserved"] for w in full["worlds"]),
        "table_contact_steps": sum(w["table_contact_steps"] for w in full["worlds"]),
        "neural_calls": sum(r["neural_calls"] for r in full["records"]),
        "task_latency_s": stats([r["task_time_s"] for r in full["records"]])})
    grouped = defaultdict(list)
    for record in full["records"]:
        grouped[(record["axis"], str(record["level"]), record["observation_mode"])].append(record)
    result["reproduction"]["groups"] = [{"axis": k[0], "level": k[1], "mode": k[2],
        "attempts": len(v), "successes": sum(r["success"] for r in v),
        "task_latency_s": stats([r["task_time_s"] for r in v])} for k, v in grouped.items()]
    result["rgbd"] = json.loads((root / "rgbd.json").read_text())
    result["camera_profile"] = json.loads((root / "profile/camera_profile.json").read_text())
    result["neural_profile"] = json.loads((root / "neural/neural_profile.json").read_text())
    result["telemetry"] = {name: telemetry(root / name / "tegrastats.log") for name in ("full", "profile", "neural")}
    idle = root / "idle_after_tegrastats.log"
    if idle.exists():
        result["telemetry"]["idle_after"] = telemetry(idle)
    result["storage_artifacts_bytes"] = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
