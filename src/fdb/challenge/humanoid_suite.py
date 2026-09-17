from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np


DEFAULT_ROBOTS = ("unitree_g1", "booster_t1", "robotis_op3", "berkeley_humanoid")


@dataclass(frozen=True)
class TaskResult:
    task: str
    applicable: bool
    success: bool | None
    score: float | None
    metrics: dict[str, Any]
    reason: str


@dataclass(frozen=True)
class HumanoidResult:
    robot: str
    source: dict[str, str]
    morphology: dict[str, Any]
    tasks: tuple[TaskResult, ...]


def _reset(model: Any) -> Any:
    import mujoco

    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    # 모든 대상 모델은 joint position actuator다. 현재 자세를 유지 목표로 삼는다.
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id >= 0:
            qpos_address = int(model.jnt_qposadr[joint_id])
            data.ctrl[actuator_id] = np.clip(
                data.qpos[qpos_address],
                model.actuator_ctrlrange[actuator_id, 0],
                model.actuator_ctrlrange[actuator_id, 1],
            )
    mujoco.mj_forward(model, data)
    return data


def _root_body_id(model: Any) -> int:
    return int(model.jnt_bodyid[0]) if model.njnt else 1


def _posture(model: Any, data: Any, initial_height: float) -> dict[str, float | bool]:
    root_id = _root_body_id(model)
    height = float(data.xpos[root_id, 2])
    upright = float(data.xmat[root_id].reshape(3, 3)[2, 2])
    ratio = height / max(initial_height, 1e-6)
    fallen = bool(ratio < 0.55 or upright < 0.5)
    return {
        "root_height_m": height,
        "height_ratio": ratio,
        "upright_cosine": upright,
        "fallen": fallen,
    }


def _simulate(
    model: Any,
    data: Any,
    seconds: float,
    *,
    callback: Callable[[int, Any], None] | None = None,
) -> None:
    import mujoco

    steps = max(1, int(seconds / model.opt.timestep))
    for step in range(steps):
        if callback is not None:
            callback(step, data)
        mujoco.mj_step(model, data)


def _actuator_match(model: Any, required: tuple[str, ...]) -> list[int]:
    matches = []
    for actuator_id in range(model.nu):
        name = (model.actuator(actuator_id).name or "").lower()
        if all(token in name for token in required):
            matches.append(actuator_id)
    return matches


def _terminal_body(model: Any, side: str) -> int | None:
    side_tokens = ("left", "l_") if side == "left" else ("right", "r_")
    priority = ("hand", "wrist", "elbow", "_el", "arm")
    candidates: list[tuple[int, int]] = []
    for body_id in range(1, model.nbody):
        name = (model.body(body_id).name or "").lower()
        if not any(token in name for token in side_tokens):
            continue
        rank = next((index for index, token in enumerate(priority) if token in name), 99)
        if rank < 99:
            candidates.append((rank, body_id))
    return min(candidates)[1] if candidates else None


def _stand_task(model: Any) -> TaskResult:
    data = _reset(model)
    initial_height = float(data.xpos[_root_body_id(model), 2])
    _simulate(model, data, 2.0)
    posture = _posture(model, data, initial_height)
    success = not bool(posture["fallen"])
    return TaskResult(
        "stand_2s", True, success,
        float(posture["height_ratio"]) * max(0.0, float(posture["upright_cosine"])),
        posture,
        "2초 동안 초기 자세 유지" if success else "높이 또는 직립 기준 이탈",
    )


def _push_task(model: Any) -> TaskResult:
    data = _reset(model)
    root_id = _root_body_id(model)
    initial_height = float(data.xpos[root_id, 2])
    total_mass = float(np.sum(model.body_mass))
    force_n = 0.15 * total_mass * 9.81
    push_start = int(0.5 / model.opt.timestep)
    push_end = int(0.65 / model.opt.timestep)

    def apply_push(step: int, state: Any) -> None:
        state.xfrc_applied[root_id] = 0.0
        if push_start <= step < push_end:
            state.xfrc_applied[root_id, 0] = force_n

    _simulate(model, data, 2.5, callback=apply_push)
    posture = _posture(model, data, initial_height)
    posture["push_force_n"] = force_n
    posture["push_duration_s"] = 0.15
    success = not bool(posture["fallen"])
    return TaskResult(
        "scaled_push_recovery", True, success,
        float(posture["height_ratio"]) * max(0.0, float(posture["upright_cosine"])),
        posture,
        "체중 비례 외란 후 직립 유지" if success else "외란 뒤 쓰러짐",
    )


def _arm_raise_task(model: Any) -> TaskResult:
    endpoint_id = _terminal_body(model, "left")
    actuator_ids = (
        _actuator_match(model, ("left", "shoulder", "pitch"))
        or _actuator_match(model, ("l_sho", "pitch"))
    )
    if endpoint_id is None or not actuator_ids:
        return TaskResult("left_arm_raise", False, None, None, {}, "왼팔 또는 어깨 구동축 없음")
    candidates = []
    for delta in (-0.45, 0.45):
        data = _reset(model)
        root_id = _root_body_id(model)
        initial_height = float(data.xpos[root_id, 2])
        initial_position = data.xpos[endpoint_id].copy()
        actuator_id = actuator_ids[0]
        start = float(data.ctrl[actuator_id])
        target = float(np.clip(
            start + delta,
            model.actuator_ctrlrange[actuator_id, 0],
            model.actuator_ctrlrange[actuator_id, 1],
        ))

        def move(step: int, state: Any) -> None:
            phase = min(1.0, (step + 1) * model.opt.timestep / 1.0)
            blend = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
            state.ctrl[actuator_id] = start + blend * (target - start)

        _simulate(model, data, 1.5, callback=move)
        displacement = data.xpos[endpoint_id] - initial_position
        posture = _posture(model, data, initial_height)
        score = float(displacement[2]) if not posture["fallen"] else -1.0
        candidates.append((score, delta, displacement, posture, target))
    score, delta, displacement, posture, target = max(candidates, key=lambda item: item[0])
    success = bool(score >= 0.03 and not posture["fallen"])
    metrics = {
        **posture,
        "selected_delta_rad": delta,
        "target_rad": target,
        "hand_displacement_m": [float(value) for value in displacement],
        "vertical_gain_m": float(displacement[2]),
        "candidate_scores": [float(candidate[0]) for candidate in candidates],
    }
    return TaskResult(
        "left_arm_raise", True, success, max(0.0, score), metrics,
        "두 방향 후보 중 손을 가장 높인 동작" if success else "안전한 30mm 상승 미달",
    )


def _crouch_task(model: Any) -> TaskResult:
    left = _actuator_match(model, ("left", "knee")) or _actuator_match(model, ("l_knee",)) or _actuator_match(model, ("ll_kfe",))
    right = _actuator_match(model, ("right", "knee")) or _actuator_match(model, ("r_knee",)) or _actuator_match(model, ("lr_kfe",))
    if not left or not right:
        return TaskResult("crouch_candidates", False, None, None, {}, "양쪽 무릎 구동축 없음")
    candidates = []
    for delta in (-0.35, 0.35):
        data = _reset(model)
        root_id = _root_body_id(model)
        initial_height = float(data.xpos[root_id, 2])
        actuator_ids = (left[0], right[0])
        starts = [float(data.ctrl[index]) for index in actuator_ids]
        targets = [
            float(np.clip(
                start + delta,
                model.actuator_ctrlrange[index, 0],
                model.actuator_ctrlrange[index, 1],
            ))
            for start, index in zip(starts, actuator_ids)
        ]

        def bend(step: int, state: Any) -> None:
            phase = min(1.0, (step + 1) * model.opt.timestep / 1.0)
            blend = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
            for index, start, target in zip(actuator_ids, starts, targets):
                state.ctrl[index] = start + blend * (target - start)

        _simulate(model, data, 1.5, callback=bend)
        posture = _posture(model, data, initial_height)
        lowering = 1.0 - float(posture["height_ratio"])
        score = lowering if not posture["fallen"] and posture["upright_cosine"] >= 0.7 else -1.0
        candidates.append((score, delta, posture, targets))
    score, delta, posture, targets = max(candidates, key=lambda item: item[0])
    success = bool(0.02 <= score <= 0.35)
    metrics = {
        **posture,
        "selected_delta_rad": delta,
        "target_rad": targets,
        "height_reduction_ratio": 1.0 - float(posture["height_ratio"]),
        "candidate_scores": [float(candidate[0]) for candidate in candidates],
    }
    return TaskResult(
        "crouch_candidates", True, success, max(0.0, score), metrics,
        "양방향 후보 중 직립하며 몸을 낮춘 동작" if success else "안전한 굽힘 범위 미달",
    )


def _head_turn_task(model: Any) -> TaskResult:
    actuator_ids = _actuator_match(model, ("head", "yaw")) or _actuator_match(model, ("head_pan",))
    if not actuator_ids:
        return TaskResult("head_turn", False, None, None, {}, "머리 yaw 구동축 없음")
    data = _reset(model)
    root_id = _root_body_id(model)
    initial_height = float(data.xpos[root_id, 2])
    actuator_id = actuator_ids[0]
    start = float(data.ctrl[actuator_id])
    target = float(np.clip(
        start + 0.35,
        model.actuator_ctrlrange[actuator_id, 0],
        model.actuator_ctrlrange[actuator_id, 1],
    ))

    def turn(step: int, state: Any) -> None:
        phase = min(1.0, (step + 1) * model.opt.timestep / 0.8)
        blend = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
        state.ctrl[actuator_id] = start + blend * (target - start)

    _simulate(model, data, 1.2, callback=turn)
    joint_id = int(model.actuator_trnid[actuator_id, 0])
    achieved = float(data.qpos[int(model.jnt_qposadr[joint_id])] - start)
    posture = _posture(model, data, initial_height)
    success = bool(abs(achieved) >= 0.25 and not posture["fallen"])
    metrics = {**posture, "command_rad": target - start, "achieved_rad": achieved}
    return TaskResult(
        "head_turn", True, success, min(1.0, abs(achieved) / 0.35), metrics,
        "머리 회전 목표 도달" if success else "회전량 또는 직립 기준 미달",
    )


def evaluate_humanoid(robot_name: str) -> HumanoidResult:
    import mujoco
    import mujoco_menagerie as menagerie

    source = menagerie.get(robot_name)
    model = source.model("scene")
    hinge_count = int(np.sum(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE))
    morphology = {
        "body_count": model.nbody - 1,
        "joint_count": model.njnt,
        "hinge_count": hinge_count,
        "actuator_count": model.nu,
        "total_mass_kg": float(np.sum(model.body_mass)),
        "has_free_base": bool(model.njnt and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE),
        "has_left_arm": _terminal_body(model, "left") is not None,
        "has_head_yaw": bool(_actuator_match(model, ("head", "yaw")) or _actuator_match(model, ("head_pan",))),
    }
    tasks = (
        _stand_task(model),
        _push_task(model),
        _arm_raise_task(model),
        _crouch_task(model),
        _head_turn_task(model),
    )
    return HumanoidResult(
        robot=robot_name,
        source={
            "repository": "https://github.com/google-deepmind/mujoco_menagerie",
            "object_id": source.oid,
            "license": source.license,
            "entry": "scene.xml",
        },
        morphology=morphology,
        tasks=tasks,
    )


def run_suite(robot_names: tuple[str, ...] = DEFAULT_ROBOTS) -> dict[str, Any]:
    results = []
    for index, robot_name in enumerate(robot_names, start=1):
        print(f"휴머노이드 {index}/{len(robot_names)} {robot_name}: 5개 과제 평가", flush=True)
        result = evaluate_humanoid(robot_name)
        for task in result.tasks:
            status = "해당없음" if not task.applicable else ("성공" if task.success else "실패")
            print(f"  {task.task}: {status} - {task.reason}", flush=True)
        results.append(result)
    applicable = sum(task.applicable for result in results for task in result.tasks)
    successes = sum(task.success is True for result in results for task in result.tasks)
    return {
        "schema_version": "1.0",
        "suite": "FDB humanoid challenge baseline",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "robot_count": len(results),
        "task_definition_count": 5,
        "applicable_run_count": applicable,
        "success_count": successes,
        "results": [
            {
                **asdict(result),
                "tasks": [asdict(task) for task in result.tasks],
            }
            for result in results
        ],
    }


def write_self_models(robot_names: tuple[str, ...], directory: Path) -> list[str]:
    from fdb.v3.morphology import GenericMorphologyInspector

    inspector = GenericMorphologyInspector()
    paths = []
    for robot_name in robot_names:
        path = directory / robot_name / "auto_self_model.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(inspector.inspect_menagerie(robot_name), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        paths.append(str(path))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="FDB 다중 휴머노이드 챌린지 기준선")
    parser.add_argument("--robots", nargs="+", default=list(DEFAULT_ROBOTS))
    parser.add_argument(
        "--output", type=Path,
        default=Path("experiments/0017_humanoid_challenge.results.json"),
    )
    parser.add_argument("--self-model-dir", type=Path, default=Path("robots"))
    args = parser.parse_args()
    report = run_suite(tuple(args.robots))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    self_models = write_self_models(tuple(args.robots), args.self_model_dir)
    print(
        f"저장: {args.output} ({report['success_count']}/{report['applicable_run_count']} 성공), "
        f"self-model {len(self_models)}개",
        flush=True,
    )


if __name__ == "__main__":
    main()
