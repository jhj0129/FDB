from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import cv2

from fdb.challenge.shape_insertion import (
    DEFAULT_VISION_MODEL,
    PEG_HALF_HEIGHT,
    TABLE_TOP_Z,
    _footprint,
    _hand_quaternion,
    _triangle_mesh,
    _yaw_quaternion,
)
from fdb.challenge.shape_fit_learning import ShapeFitPredictor
from fdb.v2.inspector import load_robot
from fdb.v2.pose_reach import PandaPoseReachExperiment

from .models import CandidateAction, FailureType, Observation, SkillResult
from .camera import MuJoCoCameraSource, CameraOnlyShapePerception, SemanticNearestTracker


@dataclass(frozen=True)
class WorldConfiguration:
    seed: int
    object_poses: dict[str, tuple[float, float, float]]
    target_poses: dict[str, tuple[float, float, float]]


class PhysicalPersistentShapeEnvironment:
    """One Panda MuJoCo model/data shared by every goal and atomic action."""

    SHAPES = ("square", "circle", "triangle", "rectangle")
    IMAGE_SIZE = 512
    CAMERA_CENTER_WORLD = (0.50, 0.0)
    CAMERA_METERS_PER_PIXEL = 0.00141

    def __init__(
        self, *, seed: int = 0, position_noise_m: float = 0.0,
        target_noise_m: float = 0.0, grasp_failure_once: set[str] | None = None,
        alignment_failure_once: set[str] | None = None,
        observation_mode: str = "segmentation",
        object_yaw_noise_deg: float | None = None,
        target_yaw_noise_deg: float | None = None,
    ) -> None:
        if observation_mode not in {"oracle", "segmentation", "camera"}:
            raise ValueError(f"unsupported observation mode: {observation_mode}")
        self.seed = seed
        self.observation_mode = observation_mode
        self.rng = np.random.default_rng(seed)
        self.scene_id = f"physical-persistent-{seed}"
        self.sequence = 0
        self.reset_count = 1
        self.grasp_failure_once = set(grasp_failure_once or ())
        self.alignment_failure_once = set(alignment_failure_once or ())
        self._injected: set[tuple[str, str]] = set()
        self._logical: dict[str, dict[str, Any]] = {}
        self._agent_logical: dict[str, dict[str, Any]] = {}
        self.vision_model = ShapeFitPredictor(DEFAULT_VISION_MODEL)
        self._build(
            position_noise_m, target_noise_m,
            object_yaw_noise_deg=object_yaw_noise_deg,
            target_yaw_noise_deg=target_yaw_noise_deg,
        )

    def _build(
        self, position_noise_m: float, target_noise_m: float, *,
        object_yaw_noise_deg: float | None, target_yaw_noise_deg: float | None,
    ) -> None:
        import mujoco

        record, _ = load_robot()
        spec = record.spec(record.default_model)
        spec.worldbody.add_geom(
            name="persistent_table", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.50, 0.0, 0.27], size=[0.40, 0.40, 0.02],
            friction=[1.1, 0.005, 0.0001], rgba=[0.38, 0.29, 0.21, 1.0], group=5,
        )
        source_y = (-0.24, -0.08, 0.08, 0.24)
        target_y = (-0.24, -0.08, 0.08, 0.24)
        object_poses: dict[str, tuple[float, float, float]] = {}
        target_poses: dict[str, tuple[float, float, float]] = {}
        for index, shape in enumerate(self.SHAPES):
            obj_id = f"{shape}_01"
            target_id = f"{shape}_target"
            ox = 0.38 + float(self.rng.uniform(-position_noise_m, position_noise_m))
            oy = source_y[index] + float(self.rng.uniform(-position_noise_m, position_noise_m))
            object_limit = math.pi / 3 if object_yaw_noise_deg is None else math.radians(object_yaw_noise_deg)
            oyaw = float(self.rng.uniform(-object_limit, object_limit))
            tx = 0.62 + float(self.rng.uniform(-target_noise_m, target_noise_m))
            ty = target_y[index] + float(self.rng.uniform(-target_noise_m, target_noise_m))
            target_limit = math.pi / 6 if target_yaw_noise_deg is None else math.radians(target_yaw_noise_deg)
            tyaw = float(self.rng.uniform(-target_limit, target_limit))
            object_poses[obj_id] = (ox, oy, oyaw)
            target_poses[target_id] = (tx, ty, tyaw)
            self._add_target(spec, shape, target_id, tx, ty, tyaw)
            self._add_object(spec, shape, obj_id, ox, oy, oyaw)
            self._logical[obj_id] = {
                "ee_near": False, "grasped": False, "lifted": False,
                "aligned": False, "near_target": False, "inserted": False,
                "released": False, "retreated": False, "last_failure": None,
                "target_id": target_id,
            }
            self._agent_logical[obj_id] = dict(self._logical[obj_id])
        self.configuration = WorldConfiguration(self.seed, object_poses, target_poses)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:9] = self.model.key(0).qpos[:9]
        self.data.ctrl[:] = self.model.key(0).ctrl
        mujoco.mj_forward(self.model, self.data)
        self.camera_source = MuJoCoCameraSource(self.model, self.data, width=self.IMAGE_SIZE, height=self.IMAGE_SIZE)
        self.camera_perception = CameraOnlyShapePerception()
        self.camera_tracker = SemanticNearestTracker()

    def _add_target(self, spec, shape: str, target_id: str, x: float, y: float, yaw: float) -> None:
        import mujoco
        frame = spec.worldbody.add_body(name=target_id, pos=[x, y, TABLE_TOP_Z], quat=_yaw_quaternion(yaw))
        points = _footprint(shape, receptacle=True)
        wall, height = 0.005, 0.025
        for index, (start, end) in enumerate(zip(points, np.roll(points, -1, axis=0))):
            delta = end - start
            length = float(np.linalg.norm(delta))
            outward = np.asarray([delta[1], -delta[0]]) / length
            center = (start + end) / 2 + outward * wall
            frame.add_geom(
                name=f"{target_id}_frame_{index}", type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[*center, height], size=[length / 2 + wall, wall, height],
                quat=_yaw_quaternion(math.atan2(float(delta[1]), float(delta[0]))),
                rgba=[0.12, 0.52, 0.96, 1], group=5,
            )

    def _add_object(self, spec, shape: str, obj_id: str, x: float, y: float, yaw: float) -> None:
        import mujoco
        body = spec.worldbody.add_body(
            name=obj_id, pos=[x, y, TABLE_TOP_Z + PEG_HALF_HEIGHT], quat=_yaw_quaternion(yaw),
        )
        body.add_freejoint(name=f"{obj_id}_free")
        options = dict(
            name=f"{obj_id}_geom", density=330.0, friction=[2.0, 0.01, 0.001],
            rgba=[0.95, 0.16, 0.08, 1], group=5,
        )
        if shape == "circle":
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.018, PEG_HALF_HEIGHT], **options)
        elif shape == "triangle":
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=_triangle_mesh(spec), **options)
        else:
            extent = np.max(np.abs(_footprint(shape)), axis=0)
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[*extent, PEG_HALF_HEIGHT], **options)

    def _camera_entities(self) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], np.ndarray]:
        """Render agent RGB and derive entity visibility; pose comes from calibrated camera tracking."""
        import mujoco

        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, self.IMAGE_SIZE)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, self.IMAGE_SIZE)
        renderer = mujoco.Renderer(self.model, width=self.IMAGE_SIZE, height=self.IMAGE_SIZE)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0.50, 0.0, TABLE_TOP_Z]
        camera.distance = 0.92
        camera.azimuth = 90
        camera.elevation = -90
        option = mujoco.MjvOption()
        option.geomgroup[:] = 0
        option.geomgroup[5] = 1
        renderer.update_scene(self.data, camera=camera, scene_option=option)
        rgb = renderer.render().copy()
        renderer.close()

        segmentation_renderer = mujoco.Renderer(
            self.model, width=self.IMAGE_SIZE, height=self.IMAGE_SIZE,
        )
        segmentation_renderer.enable_segmentation_rendering()
        segmentation_renderer.update_scene(self.data, camera=camera, scene_option=option)
        segmentation = segmentation_renderer.render().copy()
        segmentation_renderer.close()

        # Color segmentation is the current agent perception boundary. Geometry pose is
        # used only as the calibrated tracker output after visibility has been established.
        red = (rgb[:, :, 0] > 130) & (rgb[:, :, 0] > rgb[:, :, 1] * 1.35)
        blue = (rgb[:, :, 2] > 120) & (rgb[:, :, 2] > rgb[:, :, 0] * 1.25)
        def entity_detection(
            geom_ids: set[int], color_mask: np.ndarray, *, combine: bool = False,
        ) -> dict[str, Any] | None:
            label_mask = np.isin(segmentation[:, :, 0], list(geom_ids)) & color_mask
            if combine:
                points = cv2.findNonZero(label_mask.astype(np.uint8) * 255)
                contours = [cv2.convexHull(points)] if points is not None else []
            else:
                contours, _ = cv2.findContours(
                    label_mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                )
            contours = [contour for contour in contours if cv2.contourArea(contour) >= 40]
            if not contours:
                return None
            contour = max(contours, key=cv2.contourArea)
            moments = cv2.moments(contour)
            rectangle = cv2.minAreaRect(contour)
            return {
                "cx": moments["m10"] / moments["m00"],
                "cy": moments["m01"] / moments["m00"],
                "yaw": math.radians(float(rectangle[2])),
                "area": cv2.contourArea(contour),
                "contour": contour,
            }

        def camera_pose(detection: dict[str, float] | None, z: float) -> list[float] | None:
            if detection is None:
                return None
            return [
                self.CAMERA_CENTER_WORLD[0]
                + (detection["cx"] - self.IMAGE_SIZE / 2) * self.CAMERA_METERS_PER_PIXEL,
                self.CAMERA_CENTER_WORLD[1]
                - (detection["cy"] - self.IMAGE_SIZE / 2) * self.CAMERA_METERS_PER_PIXEL,
                z,
            ]
        objects: dict[str, dict[str, Any]] = {}
        targets: dict[str, dict[str, Any]] = {}
        for index, shape in enumerate(self.SHAPES):
            obj_id, target_id = f"{shape}_01", f"{shape}_target"
            detection = entity_detection({self.model.geom(f"{obj_id}_geom").id}, red)
            target_geom_ids = {
                geom_id for geom_id in range(self.model.ngeom)
                if self.model.geom(geom_id).name.startswith(f"{target_id}_frame_")
            }
            target_detection = entity_detection(target_geom_ids, blue, combine=True)
            if shape in {"square", "rectangle"}:
                yaw = -detection["yaw"] if detection else 0.0
            elif shape == "triangle":
                yaw = math.radians(60.0) - detection["yaw"] if detection else 0.0
            else:
                yaw = 0.0
            neural = self._predict_pair(detection, target_detection)
            # Fixed top-camera contour convention. The neural estimate is retained
            # separately and used by the Phase 2 world model, not relabeled as geometry.
            target_yaw = (
                math.radians(90.0) - target_detection["yaw"]
                if target_detection is not None and shape != "circle" else 0.0
            )
            logical = self._logical[obj_id]
            object_z = 0.58 if logical["lifted"] and not logical["inserted"] else TABLE_TOP_Z + PEG_HALF_HEIGHT
            objects[obj_id] = {
                "object_id": obj_id, "shape": shape,
                "pose": camera_pose(detection, object_z),
                "orientation": float(yaw), "velocity": [0.0, 0.0, 0.0],
                "visible": detection is not None, "confidence": 0.95 if detection else 0.0,
                "last_seen": datetime.now(timezone.utc).isoformat(),
                "yaw_error_deg": math.degrees(yaw - target_yaw), **logical,
                "neural_perception": neural,
                "inside_target": target_id if logical["inserted"] else None,
                "completed": logical["inserted"] and logical["released"],
                "observation_source": "camera_rgb+simulator_segmentation+fixed_extrinsic_calibration",
            }
            targets[target_id] = {
                "target_id": target_id, "shape": shape,
                "pose": camera_pose(target_detection, TABLE_TOP_Z), "orientation": float(target_yaw),
                "visible": target_detection is not None, "confidence": 0.95 if target_detection else 0.0,
                "observation_source": "camera_rgb+simulator_segmentation+fixed_extrinsic_calibration",
            }
        return objects, targets, rgb

    def _predict_pair(
        self, object_detection: dict[str, Any] | None,
        target_detection: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if object_detection is None or target_detection is None:
            return None
        object_contour = object_detection["contour"]
        target_contour = target_detection["contour"]
        target_box = cv2.minAreaRect(target_contour)
        if min(target_box[1]) <= 0:
            return None
        scale = 22.0 / max(target_box[1])

        def panel(contour: np.ndarray, color: tuple[float, float, float], fill: bool) -> np.ndarray:
            center = np.asarray(cv2.minAreaRect(contour)[0])
            points = contour[:, 0, :].astype(np.float32)
            points = np.rint((points - center) * scale + np.asarray([16.0, 16.0])).astype(np.int32)
            output = np.full((32, 32, 3), 0.16, dtype=np.float32)
            if fill:
                cv2.fillPoly(output, [points], color, lineType=cv2.LINE_AA)
            else:
                cv2.fillPoly(output, [points], (0.04, 0.04, 0.05), lineType=cv2.LINE_AA)
                cv2.polylines(output, [points], True, color, 2, lineType=cv2.LINE_AA)
            return output

        image = np.hstack((
            panel(object_contour, (0.92, 0.18, 0.10), True),
            panel(target_contour, (0.18, 0.60, 0.95), False),
        ))
        image[:, 31:33] = 0.32
        prediction = self.vision_model.predict(image)
        return {
            "object_shape": str(prediction["object_shape"]),
            "target_shape": str(prediction["hole_shape"]),
            "fits": bool(prediction["fits"]),
            "rotation_rad": float(prediction["rotation_rad"]),
            "source": "shape_fit_cnn.msgpack",
            "device": "cpu",
        }

    def observe(self) -> Observation:
        if self.observation_mode == "camera":
            return self._camera_only_observation()
        if self.observation_mode == "oracle":
            return self._oracle_observation()
        objects, targets, _ = self._camera_entities()
        grasped = next((name for name, state in self._logical.items() if state["grasped"]), None)
        return Observation(self.scene_id, objects, targets, grasped, self.sequence)

    def _camera_only_observation(self) -> Observation:
        """Agent observation built only from RGB-D plus its own action-state belief."""
        frame = self.camera_source.capture()
        tracks = self.camera_tracker.update(self.camera_perception.detect(frame))
        objects: dict[str, dict[str, Any]] = {}
        targets: dict[str, dict[str, Any]] = {}
        for track_id, track in tracks.items():
            usable = self.camera_tracker.usable(track)
            if track.role == "object":
                belief = self._agent_logical[track_id]
                pose = [track.position[0], track.position[1], track.position[2] - PEG_HALF_HEIGHT]
                target_track = tracks.get(f"{track.semantic_class}_target")
                target_yaw = target_track.yaw if target_track is not None else 0.0
                objects[track_id] = {
                    "object_id": track_id, "shape": track.semantic_class, "pose": pose,
                    "orientation": track.yaw, "velocity": [0.0, 0.0, 0.0],
                    "visible": usable, "currently_detected": track.missed_frames == 0,
                    "confidence": track.confidence, "last_seen": track.last_seen,
                    "tracking_age": track.tracking_age, "missed_frames": track.missed_frames,
                    "stale": not usable, "yaw_error_deg": math.degrees(track.yaw - target_yaw),
                    "neural_perception": None, **belief,
                    "inside_target": belief["target_id"] if belief["inserted"] else None,
                    "completed": belief["inserted"] and belief["released"],
                    "observation_source": "camera_rgbd_color_contour_tracking",
                }
            else:
                targets[track_id] = {
                    "target_id": track_id, "shape": track.semantic_class,
                    "pose": list(track.position), "orientation": track.yaw,
                    "visible": usable, "currently_detected": track.missed_frames == 0,
                    "confidence": track.confidence, "last_seen": track.last_seen,
                    "tracking_age": track.tracking_age, "missed_frames": track.missed_frames,
                    "stale": not usable,
                    "observation_source": "camera_rgbd_color_contour_tracking",
                }
        grasped = next((name for name, state in self._agent_logical.items() if state["grasped"]), None)
        return Observation(self.scene_id, objects, targets, grasped, self.sequence)

    def _oracle_observation(self) -> Observation:
        """Explicit privileged baseline. Never called by camera mode."""
        objects: dict[str, dict[str, Any]] = {}
        targets: dict[str, dict[str, Any]] = {}
        for shape in self.SHAPES:
            obj_id, target_id = f"{shape}_01", f"{shape}_target"
            obj_body, target_body = self.data.body(obj_id), self.data.body(target_id)
            obj_yaw = float(math.atan2(obj_body.xmat[3], obj_body.xmat[0]))
            target_yaw = float(math.atan2(target_body.xmat[3], target_body.xmat[0]))
            logical = self._logical[obj_id]
            objects[obj_id] = {
                "object_id": obj_id, "shape": shape, "pose": obj_body.xpos.tolist(),
                "orientation": obj_yaw, "velocity": [0.0, 0.0, 0.0], "visible": True,
                "confidence": 1.0, "yaw_error_deg": math.degrees(obj_yaw - target_yaw),
                "neural_perception": None, **logical,
                "inside_target": target_id if logical["inserted"] else None,
                "completed": logical["inserted"] and logical["released"],
                "observation_source": "oracle_simulator_body_pose",
            }
            targets[target_id] = {
                "target_id": target_id, "shape": shape, "pose": target_body.xpos.tolist(),
                "orientation": target_yaw, "visible": True, "confidence": 1.0,
                "observation_source": "oracle_simulator_body_pose",
            }
        grasped = next((name for name, state in self._logical.items() if state["grasped"]), None)
        return Observation(self.scene_id, objects, targets, grasped, self.sequence)

    def camera_rgb(self) -> np.ndarray:
        """Return the same RGB sensor view used by the agent pipeline."""
        return self.camera_source.capture().rgb

    def evaluator_ground_truth(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id, "sequence": self.sequence, "reset_count": self.reset_count,
            "objects": {
                name: {"pose": [float(v) for v in self.data.body(name).xpos], **state}
                for name, state in self._logical.items()
            },
        }

    def evaluator_satisfied(self, object_id: str, target_id: str) -> bool:
        state = self._logical[object_id]
        obj = self.data.body(object_id).xpos
        target = self.data.body(target_id).xpos
        return bool(
            state["released"]
            and float(np.linalg.norm(obj[:2] - target[:2])) <= 0.015
            and abs(float(obj[2]) - (TABLE_TOP_Z + PEG_HALF_HEIGHT)) <= 0.025
        )

    def evaluate_observation(self, observation: Observation) -> dict[str, Any]:
        """Evaluator-only pose errors. The returned values are logging-only."""
        entities: dict[str, dict[str, float]] = {}
        for entity_id, estimate in {**observation.objects, **observation.targets}.items():
            if estimate.get("pose") is None:
                continue
            body = self.data.body(entity_id)
            truth_yaw = float(math.atan2(body.xmat[3], body.xmat[0]))
            estimated_yaw = float(estimate.get("orientation", 0.0))
            yaw_error = math.degrees(math.atan2(
                math.sin(estimated_yaw - truth_yaw), math.cos(estimated_yaw - truth_yaw),
            ))
            entities[entity_id] = {
                "translation_error_mm": 1000.0 * float(np.linalg.norm(
                    np.asarray(estimate["pose"][:2]) - body.xpos[:2],
                )),
                "yaw_error_deg": yaw_error,
            }
        translations = [item["translation_error_mm"] for item in entities.values()]
        yaws = [abs(item["yaw_error_deg"]) for name, item in entities.items() if not name.startswith("circle")]
        return {
            "entities": entities,
            "mean_translation_error_mm": float(np.mean(translations)) if translations else None,
            "max_translation_error_mm": float(np.max(translations)) if translations else None,
            "mean_abs_yaw_error_deg": float(np.mean(yaws)) if yaws else None,
            "max_abs_yaw_error_deg": float(np.max(yaws)) if yaws else None,
        }

    def preview_safety(self, action: CandidateAction) -> tuple[str, ...]:
        """Compute IK and conservative joint kinematics before allowing execution."""
        name = action.skill_name
        if name in {"observe_scene", "grasp_object", "release_object", "align_object", "recover_alignment"}:
            return ()
        p = action.parameters
        shape = str(p["object_id"]).split("_", 1)[0]
        offset = math.pi / 2 if shape == "triangle" else 0.0
        obj = p.get("object_pose")
        target = p.get("target_pose")
        if obj is None or target is None:
            return ("INVALID_TRAJECTORY",)
        if name == "reach_object":
            positions = ((float(obj[0]), float(obj[1]), 0.55), (float(obj[0]), float(obj[1]), 0.43))
            yaw, duration = float(p.get("object_yaw", 0.0)) + offset, 1.9
        elif name == "lift_object":
            positions = ((float(obj[0]), float(obj[1]), 0.58),)
            yaw, duration = float(p.get("object_yaw", 0.0)) + offset, 1.6
        elif name == "move_to_target":
            positions = ((float(target[0]), float(target[1]), 0.58),)
            yaw, duration = float(p.get("target_yaw", 0.0)) + offset, 2.0
        elif name == "insert_object":
            positions = ((float(target[0]), float(target[1]), 0.445),)
            yaw, duration = float(p.get("target_yaw", 0.0)) + offset, 1.8
        elif name == "retreat":
            positions = ((float(target[0]), float(target[1]), 0.55),)
            yaw, duration = float(p.get("target_yaw", 0.0)) + offset, 1.3
        else:
            return ("INVALID_TRAJECTORY",)
        current = self.data.ctrl[:7].copy()
        reasons: list[str] = []
        try:
            for position in positions:
                target_qpos = np.asarray(
                    PandaPoseReachExperiment(position, _hand_quaternion(yaw)).run()[0].target_qpos,
                    dtype=float,
                )
                if not np.all(np.isfinite(target_qpos)):
                    reasons.append("INVALID_TRAJECTORY")
                    break
                joint_ranges = self.model.jnt_range[:7]
                if np.any(target_qpos < joint_ranges[:, 0]) or np.any(target_qpos > joint_ranges[:, 1]):
                    reasons.append("JOINT_LIMIT")
                delta = np.abs(target_qpos - current)
                if float(np.max(delta / duration)) > 2.5:
                    reasons.append("VELOCITY_LIMIT")
                if float(np.max(4.0 * delta / (duration * duration))) > 15.0:
                    reasons.append("ACCELERATION_LIMIT")
                current = target_qpos
        except Exception:
            reasons.append("IK_FAILURE")
        return tuple(dict.fromkeys(reasons))

    def execute(self, action: CandidateAction) -> SkillResult:
        self.sequence += 1
        name = action.skill_name
        obj_id = str(action.parameters["object_id"])
        state = self._agent_logical[obj_id] if self.observation_mode == "camera" else self._logical[obj_id]
        if not self._execution_preconditions_met(name, state):
            state["last_failure"] = FailureType.UNSAFE_PLAN
            return SkillResult(False, {"precondition_rejected": True}, FailureType.UNSAFE_PLAN)
        state["last_failure"] = None
        if name == "observe_scene":
            return SkillResult(True, {"camera_reobserved": True})
        if name == "grasp_object" and obj_id in self.grasp_failure_once and (obj_id, name) not in self._injected:
            self._injected.add((obj_id, name))
            state["ee_near"] = False
            state["last_failure"] = FailureType.GRASP_FAILURE
            return SkillResult(False, {"injected": True}, FailureType.GRASP_FAILURE)
        if name in {"align_object", "insert_object"} and obj_id in self.alignment_failure_once and (obj_id, "align") not in self._injected:
            self._injected.add((obj_id, "align"))
            state["last_failure"] = FailureType.ALIGNMENT_FAILURE
            return SkillResult(False, {"injected": True, "object_retained": state["grasped"]}, FailureType.ALIGNMENT_FAILURE)
        try:
            metrics = self._execute_physics(action)
        except Exception as exc:
            state["last_failure"] = FailureType.IK_FAILURE
            return SkillResult(False, {"exception": str(exc)}, FailureType.IK_FAILURE)
        if int(metrics.get("robot_table_contact_steps", 0)) > 0:
            state["last_failure"] = FailureType.COLLISION
            return SkillResult(False, metrics, FailureType.COLLISION)
        target_id = str(action.parameters["target_id"])
        if self.observation_mode == "camera":
            validation_failure = self._validate_camera_effect(name, obj_id, target_id)
            evaluator_failure = self._validate_effect(name, obj_id, target_id)
            if evaluator_failure is None:
                self._logical[obj_id].update(self._effect_for(name))
            else:
                self._logical[obj_id]["last_failure"] = evaluator_failure
        else:
            validation_failure = self._validate_effect(name, obj_id, target_id)
        if validation_failure is not None:
            state["last_failure"] = validation_failure
            if validation_failure == FailureType.ALIGNMENT_FAILURE:
                state["aligned"] = False
            if validation_failure in {FailureType.GRASP_FAILURE, FailureType.OBJECT_SLIP}:
                state.update(grasped=False, lifted=False, ee_near=False)
            return SkillResult(False, {**metrics, "effect_verified": False}, validation_failure)
        state.update(self._effect_for(name))
        return SkillResult(True, metrics)

    def _validate_camera_effect(self, name: str, obj_id: str, target_id: str) -> FailureType | None:
        """Post-action verification without simulator poses or segmentation IDs."""
        if name in {"reach_object", "grasp_object", "align_object", "recover_alignment", "release_object", "retreat"}:
            return None
        observation = self._camera_only_observation()
        obj = observation.objects.get(obj_id)
        target = observation.targets.get(target_id)
        if obj is None or not obj.get("visible"):
            return FailureType.OBJECT_LOST
        if float(obj.get("confidence", 0.0)) < 0.60:
            return FailureType.LOW_PERCEPTION_CONFIDENCE
        if name == "lift_object" and float(obj["pose"][2]) < TABLE_TOP_Z + PEG_HALF_HEIGHT + 0.07:
            return FailureType.GRASP_FAILURE
        if target is None or target.get("stale"):
            return FailureType.TARGET_NOT_FOUND
        distance = float(np.linalg.norm(np.asarray(obj["pose"][:2]) - np.asarray(target["pose"][:2])))
        if name == "move_to_target" and distance > 0.065:
            return FailureType.OBJECT_SLIP
        if name == "insert_object":
            if distance > 0.018:
                return FailureType.ALIGNMENT_FAILURE
            if abs(float(obj["pose"][2]) - (TABLE_TOP_Z + PEG_HALF_HEIGHT)) > 0.030:
                return FailureType.CONTROL_FAILURE
        return None

    def _execute_physics(self, action: CandidateAction) -> dict[str, Any]:
        import mujoco
        name = action.skill_name
        obj_id = str(action.parameters["object_id"])
        target_id = str(action.parameters["target_id"])
        shape = obj_id.split("_", 1)[0]
        obj_yaw = float(action.parameters.get("object_yaw", 0.0))
        target_yaw = float(action.parameters.get("target_yaw", 0.0))
        observed_obj = action.parameters.get("object_pose")
        observed_target = action.parameters.get("target_pose")
        if observed_obj is None or observed_target is None:
            raise ValueError("agent action requires observed object and target poses")
        grasp_offset = math.pi / 2 if shape == "triangle" else 0.0
        position = None
        quaternion = None
        waypoints: list[tuple[tuple[float, float, float], tuple[float, float, float, float], int]] = []
        gripper = float(self.data.ctrl[7])
        steps = 500
        if name == "reach_object":
            quaternion = _hand_quaternion(obj_yaw + grasp_offset)
            waypoints = [
                ((float(observed_obj[0]), float(observed_obj[1]), 0.55), quaternion, 450),
                ((float(observed_obj[0]), float(observed_obj[1]), 0.43), quaternion, 500),
            ]
        elif name == "grasp_object":
            gripper, steps = 0.0, 650
        elif name == "lift_object":
            position = (float(observed_obj[0]), float(observed_obj[1]), 0.58)
            quaternion = _hand_quaternion(obj_yaw + grasp_offset)
            gripper, steps = 0.0, 800
        elif name in {"align_object", "recover_alignment"}:
            # Alignment is a state-estimation/planning action. The actual rotation is
            # applied during the collision-safer elevated transfer below.
            return {"robot_table_contact_steps": 0, "physics_steps": 0,
                    "alignment_plan_only": True}
        elif name == "move_to_target":
            position = (float(observed_target[0]), float(observed_target[1]), 0.58)
            quaternion = _hand_quaternion(target_yaw + grasp_offset)
            gripper, steps = 0.0, 1000
        elif name == "insert_object":
            position = (float(observed_target[0]), float(observed_target[1]), 0.445)
            quaternion = _hand_quaternion(target_yaw + grasp_offset)
            gripper, steps = 0.0, 900
        elif name == "release_object":
            gripper, steps = 255.0, 650
        elif name == "retreat":
            position = (float(observed_target[0]), float(observed_target[1]), 0.55)
            quaternion = _hand_quaternion(target_yaw + grasp_offset)
            gripper, steps = 255.0, 650
        if not waypoints:
            if position is not None and quaternion is not None:
                waypoints = [(position, quaternion, steps)]
            else:
                waypoints = []
        table_id = self.model.geom("persistent_table").id
        robot_geom_ids = {
            geom_id for geom_id in range(self.model.ngeom)
            if 0 < int(self.model.geom_bodyid[geom_id]) <= self.model.body("right_finger").id
        }
        robot_table_contacts = 0
        segments = waypoints or [(None, None, steps)]
        physics_steps = 0
        for waypoint_position, waypoint_quaternion, segment_steps in segments:
            if waypoint_position is not None:
                target_qpos = PandaPoseReachExperiment(
                    waypoint_position, waypoint_quaternion,
                ).run()[0].target_qpos
            else:
                target_qpos = self.data.ctrl[:7].copy()
            start_arm = self.data.ctrl[:7].copy()
            start_gripper = float(self.data.ctrl[7])
            for step in range(segment_steps):
                phase = (step + 1) / segment_steps
                blend = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
                self.data.ctrl[:7] = start_arm + blend * (target_qpos - start_arm)
                self.data.ctrl[7] = start_gripper + blend * (gripper - start_gripper)
                mujoco.mj_step(self.model, self.data)
                physics_steps += 1
                for contact in self.data.contact:
                    pair = {int(contact.geom1), int(contact.geom2)}
                    if table_id in pair and not pair.isdisjoint(robot_geom_ids):
                        robot_table_contacts += 1
                        break
        return {"robot_table_contact_steps": robot_table_contacts, "physics_steps": physics_steps}

    def _validate_effect(self, name: str, obj_id: str, target_id: str) -> FailureType | None:
        obj = self.data.body(obj_id).xpos
        target = self.data.body(target_id).xpos
        if not np.all(np.isfinite(obj)):
            return FailureType.CONTROL_FAILURE
        if name == "lift_object" and float(obj[2]) < TABLE_TOP_Z + PEG_HALF_HEIGHT + 0.07:
            return FailureType.GRASP_FAILURE
        if name == "move_to_target" and float(np.linalg.norm(obj[:2] - target[:2])) > 0.06:
            return FailureType.OBJECT_SLIP
        if name == "insert_object":
            if float(np.linalg.norm(obj[:2] - target[:2])) > 0.015:
                return FailureType.ALIGNMENT_FAILURE
            if abs(float(obj[2]) - (TABLE_TOP_Z + PEG_HALF_HEIGHT)) > 0.025:
                return FailureType.CONTROL_FAILURE
        return None

    @staticmethod
    def _effect_for(name: str) -> dict[str, Any]:
        return {
            "reach_object": {"ee_near": True},
            "grasp_object": {"grasped": True, "released": False},
            "lift_object": {"lifted": True},
            "align_object": {"aligned": True},
            "recover_alignment": {"aligned": True},
            "move_to_target": {"near_target": True},
            "insert_object": {"inserted": True},
            "release_object": {"released": True, "grasped": False},
            "retreat": {"retreated": True},
        }.get(name, {})

    @staticmethod
    def _execution_preconditions_met(name: str, state: dict[str, Any]) -> bool:
        checks = {
            "observe_scene": True,
            "reach_object": not state["grasped"] and not state["inserted"],
            "grasp_object": state["ee_near"] and not state["grasped"] and not state["inserted"],
            "lift_object": state["grasped"] and not state["lifted"],
            "align_object": state["grasped"] and state["lifted"] and not state["aligned"],
            "recover_alignment": state["grasped"] and state["lifted"]
            and state["last_failure"] == FailureType.ALIGNMENT_FAILURE,
            "move_to_target": state["grasped"] and state["lifted"] and state["aligned"],
            "insert_object": state["grasped"] and state["near_target"] and state["aligned"],
            "release_object": state["grasped"] and state["inserted"],
            "retreat": state["inserted"] and state["released"],
        }
        return bool(checks.get(name, False))
