from __future__ import annotations

import math
from importlib.resources import as_file, files
from typing import Any

from .models import PhysicsEvaluation, PhysicsRollout, PushCandidate


def _mujoco() -> Any:
    try:
        import mujoco
    except ImportError as error:
        raise RuntimeError(
            "FDB v1 requires the optional physics dependency. "
            "Install it with: python -m pip install -e '.[physics]'"
        ) from error
    return mujoco


class MujocoPushEnvironment:
    """Fresh-model MuJoCo rollouts for deterministic candidate comparison."""

    def __init__(self, *, success_radius: float = 0.14, arena_limit: float = 1.8) -> None:
        self.success_radius = success_radius
        self.arena_limit = arena_limit

    def simulate(self, plan: PushCandidate) -> PhysicsRollout:
        mujoco = _mujoco()
        asset = files("fdb.v1").joinpath("assets/push.xml")
        with as_file(asset) as model_path:
            model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        puck_id = model.body("puck").id
        puck_geom_id = model.geom("puck_geom").id
        floor_geom_id = model.geom("floor").id
        initial = data.body("puck").xpos[:2].copy()
        target = data.site("target").xpos[:2].copy()
        delta = target - initial
        direction = delta / max(float(math.hypot(*delta)), 1e-12)

        prior = initial.copy()
        path_length = 0.0
        unintended_contacts = 0
        stable = True
        out_of_bounds = False
        total_steps = plan.force_steps + plan.settle_steps
        for step in range(total_steps):
            data.xfrc_applied[puck_id, :] = 0.0
            if step < plan.force_steps:
                # MuJoCo xfrc_applied layout is force[0:3], torque[3:6].
                data.xfrc_applied[puck_id, :2] = direction * plan.force_newtons
            mujoco.mj_step(model, data)
            current = data.body("puck").xpos[:2].copy()
            path_length += float(math.hypot(*(current - prior)))
            prior = current
            if not all(math.isfinite(float(value)) for value in data.qpos):
                stable = False
                break
            if max(abs(float(current[0])), abs(float(current[1]))) > self.arena_limit:
                out_of_bounds = True
            for index in range(data.ncon):
                contact = data.contact[index]
                pair = {int(contact.geom1), int(contact.geom2)}
                if pair != {puck_geom_id, floor_geom_id}:
                    unintended_contacts += 1

        final = data.body("puck").xpos[:2].copy()
        final_distance = float(math.hypot(*(final - target)))
        success = (
            final_distance <= self.success_radius
            and not out_of_bounds
            and stable
            and unintended_contacts == 0
        )
        execution_time = float(data.time)
        # Objective score: prioritize completion, then accuracy, path, time, and safety.
        score = (
            (100.0 if success else 0.0)
            - 50.0 * final_distance
            - 2.0 * path_length
            - execution_time
            - 5.0 * unintended_contacts
            - (100.0 if out_of_bounds or not stable else 0.0)
        )
        return PhysicsRollout(
            plan=plan,
            initial_position=(float(initial[0]), float(initial[1])),
            target_position=(float(target[0]), float(target[1])),
            final_position=(float(final[0]), float(final[1])),
            evaluation=PhysicsEvaluation(
                success=success,
                score=score,
                final_distance=final_distance,
                path_length=path_length,
                execution_time=execution_time,
                unintended_contact_count=unintended_contacts,
                out_of_bounds=out_of_bounds,
                stable=stable,
            ),
        )
