from __future__ import annotations

import math
from dataclasses import dataclass

from .inspector import load_robot
from .reach import PandaReachExperiment


@dataclass(frozen=True)
class PregraspCandidate:
    plan_id: str
    target: tuple[float, float, float]


@dataclass(frozen=True)
class PregraspResult:
    plan: PregraspCandidate
    contact_step_count: int
    position_error_m: float
    stable: bool
    score: float

    @property
    def collision_free(self) -> bool:
        return self.contact_step_count == 0


class CollisionAwarePregraspExperiment:
    obstacle_position = (0.45, 0.15, 0.55)
    obstacle_radius = 0.06

    def candidates(self) -> tuple[PregraspCandidate, ...]:
        x, y, z = self.obstacle_position
        return (
            PregraspCandidate("direct_center", (x, y, z)),
            PregraspCandidate("centered_high", (x, y, z + 0.20)),
            PregraspCandidate("side_high", (x + 0.10, y, z + 0.20)),
        )

    def run_candidate(self, plan: PregraspCandidate) -> PregraspResult:
        import mujoco

        record, _ = load_robot()
        spec = record.spec(record.default_model)
        spec.worldbody.add_geom(
            name="obstacle",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=self.obstacle_position,
            size=[self.obstacle_radius],
            rgba=[0.25, 0.25, 0.25, 1.0],
        )
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        target_qpos = PandaReachExperiment(plan.target).run()[0].target_qpos
        data.ctrl[:7] = target_qpos
        data.ctrl[7] = 255.0
        obstacle_id = model.geom("obstacle").id
        contact_steps = 0
        stable = True
        for _ in range(1500):
            mujoco.mj_step(model, data)
            if any(
                obstacle_id in (int(data.contact[index].geom1), int(data.contact[index].geom2))
                for index in range(data.ncon)
            ):
                contact_steps += 1
            if not all(math.isfinite(float(value)) for value in data.qpos):
                stable = False
                break
        hand = data.body("hand").xpos
        error = math.sqrt(sum((float(hand[i]) - plan.target[i]) ** 2 for i in range(3)))
        valid = stable and contact_steps == 0 and error <= 0.02
        score = (100.0 if valid else 0.0) - 500.0 * error - 2.0 * contact_steps
        return PregraspResult(plan, contact_steps, error, stable, score)

    def run(self) -> tuple[PregraspResult, tuple[PregraspResult, ...]]:
        results = tuple(self.run_candidate(candidate) for candidate in self.candidates())
        selected = max(
            results,
            key=lambda item: (
                item.collision_free and item.stable and item.position_error_m <= 0.02,
                item.score,
            ),
        )
        return selected, results

