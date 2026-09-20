"""Opt-in simulation profiling; never connects to robot or camera hardware."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
import functools
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import threading
import time
from unittest.mock import patch

import numpy as np


def distribution(values):
    return {"count": len(values), "mean_ms": float(np.mean(values) * 1000),
            "p50_ms": float(np.median(values) * 1000),
            "p95_ms": float(np.percentile(values, 95) * 1000),
            "max_ms": float(max(values) * 1000), "total_s": float(sum(values))}


def memory():
    status = Path("/proc/self/status").read_text().splitlines()
    meminfo = Path("/proc/meminfo").read_text().splitlines()
    return {"rss_kib": int(next(x.split()[1] for x in status if x.startswith("VmRSS:"))),
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "system_available_kib": int(next(x.split()[1] for x in meminfo if x.startswith("MemAvailable:"))),
            "fd_count": len(list(Path("/proc/self/fd").iterdir()))}


def sample_clocks(path, stop):
    with path.open("w") as handle:
        while not stop.is_set():
            row = {"time": time.time()}
            for name, source in {
                "gpu_hz": Path("/sys/class/devfreq/17000000.gpu/cur_freq"),
                **{f"cooling_{p.parent.name}": p for p in Path("/sys/class/thermal").glob("cooling_device*/cur_state")},
            }.items():
                try:
                    row[name] = int(source.read_text().strip())
                except (OSError, ValueError):
                    row[name] = None
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            stop.wait(1)


class Timings:
    """Record inclusive and exclusive durations without changing domain methods."""
    def __init__(self):
        self.inclusive = defaultdict(list)
        self.exclusive = defaultdict(list)
        self.stack = []

    def wrap(self, function, name):
        @functools.wraps(function)
        def measured(*args, **kwargs):
            frame = [time.perf_counter(), 0.0]
            self.stack.append(frame)
            try:
                return function(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - frame[0]
                self.stack.pop()
                if self.stack:
                    self.stack[-1][1] += elapsed
                self.inclusive[name].append(elapsed)
                self.exclusive[name].append(elapsed - frame[1])
        return measured

    def summary(self):
        return {key: {"inclusive": distribution(values),
                      "exclusive": distribution(self.exclusive[key])}
                for key, values in self.inclusive.items()}


def profile_camera(output, repeats):
    import mujoco
    from fdb.core import physical_runtime as runtime
    from fdb.core.camera import MuJoCoCameraSource, CameraOnlyShapePerception, SemanticNearestTracker
    from fdb.core.physical_scene import PhysicalPersistentShapeEnvironment as Environment
    from fdb.core.physical_cli import run_tasks
    from fdb.memory import EpisodeStore

    hooks = [(MuJoCoCameraSource, "capture", "camera_render"),
             (CameraOnlyShapePerception, "detect", "perception"),
             (SemanticNearestTracker, "update", "tracking"),
             (runtime.AtomicSkillWorldModel, "predict", "world_model"),
             (runtime, "generate_skill_candidates", "candidate_generation"),
             (runtime.PhysicalSafetyGate, "check", "safety"),
             (Environment, "execute", "skill_execution"),
             (Environment, "evaluate_observation", "evaluation"),
             (Environment, "get_evaluator_ground_truth", "evaluation"),
             (EpisodeStore, "search", "memory"), (EpisodeStore, "write", "episode_io"),
             (mujoco, "mj_step", "physics"),
             (runtime.PhysicalAtomicRuntime, "run", "total_task")]
    runs = []
    for index in range(repeats):
        timing = Timings()
        before = memory()
        started = time.perf_counter()
        with ExitStack() as stack:
            for owner, attr, name in hooks:
                stack.enter_context(patch.object(owner, attr, timing.wrap(getattr(owner, attr), name)))
            result = run_tasks(goals=("triangle", "circle", "rectangle"), seed=0,
                               observation_mode="camera", output_directory=output / f"run_{index:02d}")
        elapsed = time.perf_counter() - started
        contacts = []
        for task in result["results"]:
            episode = json.loads(Path(task["episode_paths"][0]).read_text())
            contacts.append(episode["result"]["objective_evaluation"]["ground_truth"]["robot_table_contact_steps"])
        gc.collect()
        row = {"run": index, "first_process_run": index == 0, "wall_s": elapsed,
               "successes": result["successes"], "all_goals_preserved": result["all_goals_preserved"],
               "table_contact_steps": max(contacts), "before": before, "after": memory(),
               "neural_calls": sum(r["metrics"]["neural_world_model_calls"] for r in result["results"]),
               "timings": timing.summary()}
        runs.append(row)
        (output / "camera_profile.json").write_text(json.dumps(runs, indent=2) + "\n")
        print(json.dumps({k: row[k] for k in ("run", "wall_s", "successes", "after")}), flush=True)
        if result["successes"] != 3 or not result["all_goals_preserved"] or max(contacts):
            raise RuntimeError("Camera profile failed its task/safety gate")


def profile_neural(output, samples):
    import torch
    from fdb.v5.network import NeuralDynamicsEnsemble

    checkpoint = Path("models/v5/push_dynamics_ensemble.pt")
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    results = {"checkpoint": str(checkpoint), "sha256": checkpoint_hash,
               "seed": 0, "samples": samples,
               "scope": "Existing push dynamics model, separate from camera-only runtime (zero neural calls)",
               "torch_threads": torch.get_num_threads(), "devices": {}}
    vectors = None
    predictions = {}
    for device in ("cpu", "cuda"):
        started = time.perf_counter()
        model = NeuralDynamicsEnsemble.load(checkpoint, device=device)
        if vectors is None:
            rng = np.random.default_rng(0)
            vectors = rng.uniform(model.feature_min, model.feature_max, (samples, len(model.feature_mean))).astype(np.float32)
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        load_s = time.perf_counter() - started
        cold_started = time.perf_counter()
        model.predict(vectors[0])
        cold_s = time.perf_counter() - cold_started
        for _ in range(20):
            model.predict(vectors[0])
        timings, outputs, allocations = [], [], []
        for index, vector in enumerate(vectors):
            started = time.perf_counter()
            prediction = model.predict(vector)
            if device == "cuda":
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - started)
            outputs.append(prediction.final_position)
            if device == "cuda" and index % 50 == 0:
                allocations.append(torch.cuda.memory_allocated())
        tensor = torch.from_numpy(((vectors[0] - model.feature_mean) / model.feature_std)[None]).to(device)
        inference = []
        with torch.no_grad():
            for _ in range(samples):
                started = time.perf_counter()
                torch.stack([member(tensor)[0] for member in model.models])
                if device == "cuda":
                    torch.cuda.synchronize()
                inference.append(time.perf_counter() - started)
        predictions[device] = np.asarray(outputs)
        results["devices"][device] = {"load_s": load_s, "cold_predict_s": cold_s,
            "predict_including_transfers_sync": distribution(timings),
            "inference_resident_input_with_sync": distribution(inference),
            "memory": memory(),
            "cuda_allocated_bytes_every_50_predictions": allocations,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0}
        del model
        gc.collect()
    results["max_cpu_cuda_position_difference_m"] = float(np.max(np.abs(predictions["cpu"] - predictions["cuda"])))
    np.testing.assert_allclose(predictions["cpu"], predictions["cuda"], rtol=1e-4, atol=1e-5)
    (output / "neural_profile.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("camera", "neural"), required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.samples < 1:
        parser.error("repeats and samples must be positive")
    args.output_directory.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    with (args.output_directory / "tegrastats.log").open("w") as log:
        telemetry = subprocess.Popen(["tegrastats", "--interval", "1000"], stdout=log, stderr=subprocess.STDOUT)
        stop = threading.Event()
        sampler = threading.Thread(target=sample_clocks, args=(args.output_directory / "clocks.jsonl", stop))
        sampler.start()
        try:
            if args.mode == "camera":
                profile_camera(args.output_directory, args.repeats)
            else:
                profile_neural(args.output_directory, args.samples)
        finally:
            stop.set()
            sampler.join(timeout=5)
            telemetry.terminate()
            telemetry.wait(timeout=10)


if __name__ == "__main__":
    main()
