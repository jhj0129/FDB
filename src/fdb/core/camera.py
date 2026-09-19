from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(frozen=True)
class CameraCalibration:
    intrinsics: CameraIntrinsics
    camera_to_world: tuple[tuple[float, float, float, float], ...]
    name: str


@dataclass(frozen=True)
class CameraFrame:
    rgb: np.ndarray
    depth_m: np.ndarray
    timestamp: str
    calibration: CameraCalibration
    source: str


class CameraSource(Protocol):
    def capture(self) -> CameraFrame: ...


class MuJoCoCameraSource:
    """RGB-D source with the same contract intended for a future RealSense adapter."""

    def __init__(self, model, data, *, width: int = 512, height: int = 512) -> None:
        self.model = model
        self.data = data
        self.width = width
        self.height = height
        fovy = math.radians(float(model.vis.global_.fovy))
        fy = height / (2.0 * math.tan(fovy / 2.0))
        fx = fy
        # Free camera: lookat [0.5, 0, 0.29], distance .92, top-down.
        self.calibration = CameraCalibration(
            CameraIntrinsics(fx, fy, width / 2.0, height / 2.0, width, height),
            (
                (1.0, 0.0, 0.0, 0.50),
                (0.0, -1.0, 0.0, 0.0),
                (0.0, 0.0, -1.0, 1.21),
                (0.0, 0.0, 0.0, 1.0),
            ),
            "mujoco_top_camera_v1",
        )

    def capture(self) -> CameraFrame:
        import mujoco

        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, self.width)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, self.height)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0.50, 0.0, 0.29]
        camera.distance = 0.92
        camera.azimuth = 90
        camera.elevation = -90
        option = mujoco.MjvOption()
        option.geomgroup[:] = 0
        option.geomgroup[5] = 1
        renderer = mujoco.Renderer(self.model, width=self.width, height=self.height)
        renderer.update_scene(self.data, camera=camera, scene_option=option)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(self.data, camera=camera, scene_option=option)
        depth = renderer.render().copy()
        renderer.close()
        return CameraFrame(
            rgb, depth, datetime.now(timezone.utc).isoformat(), self.calibration,
            "MuJoCoCameraSource",
        )


@dataclass(frozen=True)
class VisualDetection:
    semantic_class: str
    role: str
    position: tuple[float, float, float]
    yaw: float
    confidence: float
    pixel_centroid: tuple[float, float]
    contour_area: float


class CameraOnlyShapePerception:
    """Deterministic color/contour RGB-D perception; no simulator IDs or body poses."""

    SHAPES = ("square", "circle", "triangle", "rectangle")

    def detect(self, frame: CameraFrame) -> list[VisualDetection]:
        rgb = frame.rgb
        masks = {
            "object": (rgb[:, :, 0] > 130) & (rgb[:, :, 0] > rgb[:, :, 1] * 1.35),
            "target": (rgb[:, :, 2] > 120) & (rgb[:, :, 2] > rgb[:, :, 0] * 1.25),
        }
        detections: list[VisualDetection] = []
        for role, mask in masks.items():
            contours, _ = cv2.findContours(
                mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < 80.0:
                    continue
                shape, class_confidence = self._classify(contour)
                moments = cv2.moments(contour)
                u = float(moments["m10"] / moments["m00"])
                v = float(moments["m01"] / moments["m00"])
                contour_mask = np.zeros(mask.shape, dtype=np.uint8)
                cv2.drawContours(contour_mask, [contour], -1, 1, thickness=-1)
                surface_depths = frame.depth_m[(contour_mask > 0) & mask]
                valid_depths = surface_depths[np.isfinite(surface_depths) & (surface_depths > 0)]
                surface_depth = float(np.median(valid_depths)) if valid_depths.size else None
                position = self._unproject(frame, u, v, depth_m=surface_depth)
                yaw = self._orientation(contour, shape, role)
                confidence = min(0.99, class_confidence * min(1.0, area / 400.0))
                detections.append(VisualDetection(
                    shape, role, position, yaw, confidence, (u, v), area,
                ))
        return detections

    @staticmethod
    def _classify(contour: np.ndarray) -> tuple[str, float]:
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        circularity = 4.0 * math.pi * area / max(perimeter * perimeter, 1e-9)
        vertices = len(cv2.approxPolyDP(contour, 0.06 * perimeter, True))
        width, height = cv2.minAreaRect(contour)[1]
        aspect = max(width, height) / max(min(width, height), 1e-6)
        if circularity > 0.80:
            return "circle", min(0.99, circularity)
        if vertices == 3:
            return "triangle", 0.94
        if aspect > 1.20:
            return "rectangle", min(0.98, 0.75 + (aspect - 1.2) * 0.3)
        return "square", 0.93

    @staticmethod
    def _orientation(contour: np.ndarray, shape: str, role: str) -> float:
        angle = math.radians(float(cv2.minAreaRect(contour)[2]))
        if shape == "circle":
            return 0.0
        if role == "target":
            return math.radians(90.0) - angle
        if shape == "triangle":
            # A triangle's minimum-area rectangle is unstable when a soft shadow
            # changes which edge owns the box.  Its three vertex directions are
            # equivalent modulo 120 degrees, so estimate that symmetry class
            # directly from the polygon instead.
            perimeter = float(cv2.arcLength(contour, True))
            vertices = cv2.approxPolyDP(contour, 0.06 * perimeter, True).reshape(-1, 2)
            if len(vertices) == 3:
                moments = cv2.moments(contour)
                cx = float(moments["m10"] / moments["m00"])
                cy = float(moments["m01"] / moments["m00"])
                estimates = []
                period = 2.0 * math.pi / 3.0
                for x, y in vertices:
                    world_angle = math.atan2(-(float(y) - cy), float(x) - cx)
                    yaw = world_angle + math.pi / 2.0
                    estimates.append((yaw + period / 2.0) % period - period / 2.0)
                return float(np.median(estimates))
            return math.radians(60.0) - angle
        return -angle

    @staticmethod
    def _unproject(
        frame: CameraFrame, u: float, v: float, *, depth_m: float | None = None,
    ) -> tuple[float, float, float]:
        intr = frame.calibration.intrinsics
        row = max(0, min(intr.height - 1, int(round(v))))
        col = max(0, min(intr.width - 1, int(round(u))))
        depth = float(frame.depth_m[row, col]) if depth_m is None else depth_m
        camera_point = np.asarray([
            (u - intr.cx) * depth / intr.fx,
            (v - intr.cy) * depth / intr.fy,
            depth,
            1.0,
        ])
        transform = np.asarray(frame.calibration.camera_to_world)
        world = transform @ camera_point
        # RGB-D observes the top surface. State pose uses the body/table reference plane.
        return float(world[0]), float(world[1]), float(world[2])


@dataclass
class Track:
    track_id: str
    semantic_class: str
    role: str
    position: tuple[float, float, float]
    yaw: float
    confidence: float
    last_seen: str
    tracking_age: int = 1
    missed_frames: int = 0


class SemanticNearestTracker:
    """Semantic + nearest-position tracker without simulator identity labels."""

    def __init__(self, *, distance_gate_m: float = 0.35, stale_after_frames: int = 8) -> None:
        self.distance_gate_m = distance_gate_m
        self.stale_after_frames = stale_after_frames
        self.tracks: dict[str, Track] = {}

    def update(self, detections: list[VisualDetection]) -> dict[str, Track]:
        unmatched = set(self.tracks)
        for detection in detections:
            track_id = f"{detection.semantic_class}_{'01' if detection.role == 'object' else 'target'}"
            existing = self.tracks.get(track_id)
            if existing is not None:
                distance = math.dist(existing.position[:2], detection.position[:2])
                if distance > self.distance_gate_m:
                    existing.missed_frames += 1
                    continue
                existing.position = detection.position
                existing.yaw = detection.yaw
                existing.confidence = detection.confidence
                existing.last_seen = datetime.now(timezone.utc).isoformat()
                existing.tracking_age += 1
                existing.missed_frames = 0
            else:
                self.tracks[track_id] = Track(
                    track_id, detection.semantic_class, detection.role, detection.position,
                    detection.yaw, detection.confidence,
                    datetime.now(timezone.utc).isoformat(),
                )
            unmatched.discard(track_id)
        for track_id in unmatched:
            self.tracks[track_id].missed_frames += 1
            # Preserve the confidence of the last measurement; freshness is a
            # separate signal and the frame-age gate expires the pose.
            pass
        return dict(self.tracks)

    def usable(self, track: Track) -> bool:
        return track.missed_frames <= self.stale_after_frames and track.confidence >= 0.60
