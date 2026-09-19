import math

import cv2
import numpy as np

from fdb.core.camera import CameraCalibration, CameraFrame, CameraIntrinsics, CameraOnlyShapePerception, SemanticNearestTracker


def synthetic_frame() -> CameraFrame:
    rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.rectangle(rgb, (15, 20), (35, 40), (240, 35, 20), -1)
    cv2.circle(rgb, (70, 30), 11, (240, 35, 20), -1)
    cv2.fillPoly(rgb, [np.asarray([[20, 100], [35, 70], [50, 100]])], (240, 35, 20))
    cv2.rectangle(rgb, (70, 75), (108, 95), (240, 35, 20), -1)
    depth = np.ones((128, 128), dtype=np.float32)
    calibration = CameraCalibration(
        CameraIntrinsics(100.0, 100.0, 64.0, 64.0, 128, 128),
        ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
        "test",
    )
    return CameraFrame(rgb, depth, "now", calibration, "test")


def test_rgbd_perception_classifies_without_segmentation_ids() -> None:
    detections = CameraOnlyShapePerception().detect(synthetic_frame())
    assert {item.semantic_class for item in detections} == {"square", "circle", "triangle", "rectangle"}
    assert all(item.role == "object" for item in detections)
    assert all(math.isfinite(value) for item in detections for value in item.position)
    triangle = next(item for item in detections if item.semantic_class == "triangle")
    assert abs(triangle.yaw - math.pi / 3.0) < math.radians(5.0)


def test_tracker_preserves_identity_and_expires_stale_pose() -> None:
    perception = CameraOnlyShapePerception()
    detections = perception.detect(synthetic_frame())
    tracker = SemanticNearestTracker(stale_after_frames=1)
    first = tracker.update(detections)
    first_age = first["triangle_01"].tracking_age
    second = tracker.update(detections)
    assert second["triangle_01"].tracking_age == first_age + 1
    tracker.update([])
    stale = tracker.update([])["triangle_01"]
    assert not tracker.usable(stale)
