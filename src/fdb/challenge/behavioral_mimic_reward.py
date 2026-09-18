from __future__ import annotations

from types import ModuleType
from typing import Any

import mujoco
import numpy as np


def contact_agreement(actual_left, actual_right, desired_left, desired_right, backend):
    """Return dense agreement plus an exact single-support match bonus."""
    left_match = backend.equal(actual_left, desired_left)
    right_match = backend.equal(actual_right, desired_right)
    dense = 0.5 * (left_match.astype(float) + right_match.astype(float))
    desired_single = backend.logical_xor(desired_left, desired_right)
    actual_single = backend.logical_xor(actual_left, actual_right)
    exact = backend.logical_and(
        desired_single,
        backend.logical_and(actual_single, backend.logical_and(left_match, right_match)),
    )
    return dense, exact.astype(float)


def register_behavioral_mimic_reward() -> None:
    """Register lazily so importing FDB does not require the robotics extras."""
    from loco_mujoco.core.reward.base import Reward

    if "BehavioralMimicReward" in Reward.registered:
        return

    from loco_mujoco.core.reward.trajectory_based import MimicReward, check_traj_provided
    from loco_mujoco.core.utils import mj_check_collisions

    class BehavioralMimicReward(MimicReward):
        """DeepMimic reward augmented with explicit human foot-contact timing."""

        def __init__(self, env: Any, **kwargs):
            super().__init__(env, **kwargs)
            model = env._model
            self._floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
            self._left_foot_geom_ids = tuple(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                for name in ("left_foot1", "left_foot2")
            )
            self._right_foot_geom_ids = tuple(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                for name in ("right_foot1", "right_foot2")
            )
            self._left_site_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, "left_foot_mimic"
            )
            self._right_site_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, "right_foot_mimic"
            )
            self._contact_height_m = kwargs.get("contact_height_m", 0.055)
            self._contact_match_weight = kwargs.get("contact_match_weight", 0.35)
            self._single_support_weight = kwargs.get("single_support_weight", 0.25)
            self._foot_height_weight = kwargs.get("foot_height_weight", 0.20)
            self._foot_height_scale = kwargs.get("foot_height_scale", 80.0)

        @staticmethod
        def _any_contact(ids, floor_id, data, backend, collision_fn):
            value = collision_fn(ids[0], floor_id, data, backend)
            for geom_id in ids[1:]:
                value = backend.logical_or(value, collision_fn(geom_id, floor_id, data, backend))
            return value

        @check_traj_provided
        def __call__(self, state, action, next_state, absorbing, info, env, model, data,
                     carry, backend: ModuleType):
            base_reward, carry = super().__call__(
                state, action, next_state, absorbing, info, env, model, data, carry, backend
            )
            target = env.th.traj.data.get(
                carry.traj_state.traj_no, carry.traj_state.subtraj_step_no, backend
            )
            desired_left = target.site_xpos[self._left_site_id, 2] < self._contact_height_m
            desired_right = target.site_xpos[self._right_site_id, 2] < self._contact_height_m
            actual_left = self._any_contact(
                self._left_foot_geom_ids, self._floor_id, data, backend, mj_check_collisions
            )
            actual_right = self._any_contact(
                self._right_foot_geom_ids, self._floor_id, data, backend, mj_check_collisions
            )
            match, single_support = contact_agreement(
                actual_left, actual_right, desired_left, desired_right, backend
            )
            actual_heights = backend.asarray([
                data.site_xpos[self._left_site_id, 2], data.site_xpos[self._right_site_id, 2]
            ])
            target_heights = backend.asarray([
                target.site_xpos[self._left_site_id, 2], target.site_xpos[self._right_site_id, 2]
            ])
            height_reward = backend.exp(
                -self._foot_height_scale * backend.mean(backend.square(actual_heights - target_heights))
            )
            reward = (
                base_reward
                + self._contact_match_weight * match
                + self._single_support_weight * single_support
                + self._foot_height_weight * height_reward
            )
            return backend.nan_to_num(backend.maximum(reward, 0.0), nan=0.0), carry

    BehavioralMimicReward.register()
