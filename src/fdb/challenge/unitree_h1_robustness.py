from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from fdb.challenge.contact_gait import evaluate_contact_gait
from fdb.challenge.unitree_h1_walk import DEFAULT_UPSTREAM, simulate_official_policy


def run_suite(upstream: Path, output: Path) -> dict[str, object]:
    cases: list[dict[str, object]] = []
    for force_n in (0, 100, 200, 300, 350, 400, 500):
        pushes = () if force_n == 0 else ((5.0, 0.15, 0.0, float(force_n)),)
        _, qpos, feet, contacts, frequency, _ = simulate_official_policy(
            upstream,
            duration_s=15.0,
            command_m_s=(1.3, 0.0, 0.0),
            phase_period_s=1.0,
            pushes=pushes,
        )
        result, detected, alternating = evaluate_contact_gait(
            feet, contacts, qpos[:, :2], frequency,
        )
        upright = bool(np.min(qpos[:, 2]) >= 0.75 and qpos[-1, 2] >= 0.75)
        # A 120 steps/min gait should produce 30 steps in 15 s. Allow one
        # transition to be disrupted by the push while requiring continuation.
        continuity = bool(len(alternating) >= 29)
        cases.append({
            "lateral_force_n": force_n,
            "duration_s": 0.15 if force_n else 0.0,
            "impulse_ns": force_n * 0.15,
            "upright": upright,
            "continuous_gait": continuity,
            "overall_success": bool(result.success and upright and continuity),
            "minimum_pelvis_height_m": float(np.min(qpos[:, 2])),
            "final_pelvis_height_m": float(qpos[-1, 2]),
            "detected_steps": len(detected),
            "alternating_steps": len(alternating),
            "gait": asdict(result),
        })
    full_recovery = [case for case in cases if case["overall_success"]]
    upright_recovery = [case for case in cases if case["upright"]]
    payload = {
        "policy": "Unitree official H1 motion.pt + FDB human-cadence phase control",
        "command_m_s": [1.3, 0.0, 0.0],
        "phase_period_s": 1.0,
        "expected_cadence_steps_per_min": 120,
        "push_start_s": 5.0,
        "push_duration_s": 0.15,
        "maximum_full_gait_recovery_force_n": max(case["lateral_force_n"] for case in full_recovery),
        "maximum_full_gait_recovery_impulse_ns": max(case["impulse_ns"] for case in full_recovery),
        "maximum_upright_recovery_force_n": max(case["lateral_force_n"] for case in upright_recovery),
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="사람 보행률 H1 정책의 횡방향 외란 복구 평가")
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--output", type=Path, default=Path("artifacts/fdb_unitree_h1_robustness.json"))
    args = parser.parse_args()
    print(json.dumps(run_suite(args.upstream, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
