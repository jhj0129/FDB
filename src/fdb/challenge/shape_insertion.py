from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess

import numpy as np
import cv2

from fdb.challenge.shape_fit_learning import ShapeFitPredictor, _polygon
from fdb.v2.inspector import load_robot
from fdb.v2.pose_reach import PandaPoseReachExperiment
from fdb.v2.render_final import _ffmpeg_executable


TABLE_TOP_Z = 0.29
PEG_HALF_SIZE = (0.018, 0.018, 0.040)
SOURCE_XY = (0.43, 0.16)
HOLE_XY = (0.56, -0.12)
OBJECT_YAW = math.radians(24.0)
HOLE_YAW = math.radians(-12.0)
DEFAULT_VISION_MODEL = Path(__file__).resolve().parents[3] / "models/v6/shape_fit_cnn.msgpack"


@dataclass(frozen=True)
class InsertionResult:
    perceived_object: str
    perceived_hole: str
    fit_decision: bool
    planned_rotation_deg: float
    final_xy_error_m: float
    final_height_error_m: float
    final_yaw_error_deg: float
    peak_peg_height_m: float
    frame_contact_steps: int
    robot_table_contact_steps: int
    released: bool
    success: bool


def _yaw_quaternion(yaw: float) -> np.ndarray:
    return np.asarray([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def _hand_quaternion(yaw: float) -> tuple[float, float, float, float]:
    import mujoco

    _, model = load_robot()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    output = np.zeros(4)
    mujoco.mju_mulQuat(output, _yaw_quaternion(yaw), data.body("hand").xquat)
    return tuple(float(value) for value in output)


def _build_scene(object_yaw: float = OBJECT_YAW, hole_yaw: float = HOLE_YAW):
    import mujoco

    record, _ = load_robot()
    spec = record.spec(record.default_model)
    spec.worldbody.add_geom(
        name="table", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.5, 0.0, 0.27],
        size=[0.40, 0.40, 0.02], friction=[1.1, 0.005, 0.0001],
        rgba=[0.38, 0.29, 0.21, 1.0], group=5,
    )
    frame = spec.worldbody.add_body(
        name="square_receptacle", pos=[*HOLE_XY, TABLE_TOP_Z], quat=_yaw_quaternion(hole_yaw)
    )
    inner, wall, outer, height = 0.025, 0.010, 0.075, 0.040
    frame.add_geom(name="frame_left", type=mujoco.mjtGeom.mjGEOM_BOX,
                   pos=[0, inner + wall, height], size=[outer, wall, height], rgba=[0.15, 0.55, 0.95, 1], group=5)
    frame.add_geom(name="frame_right", type=mujoco.mjtGeom.mjGEOM_BOX,
                   pos=[0, -inner - wall, height], size=[outer, wall, height], rgba=[0.15, 0.55, 0.95, 1], group=5)
    frame.add_geom(name="frame_front", type=mujoco.mjtGeom.mjGEOM_BOX,
                   pos=[inner + wall, 0, height], size=[wall, inner, height], rgba=[0.15, 0.55, 0.95, 1], group=5)
    frame.add_geom(name="frame_back", type=mujoco.mjtGeom.mjGEOM_BOX,
                   pos=[-inner - wall, 0, height], size=[wall, inner, height], rgba=[0.15, 0.55, 0.95, 1], group=5)
    peg = spec.worldbody.add_body(
        name="square_peg", pos=[*SOURCE_XY, TABLE_TOP_Z + PEG_HALF_SIZE[2]],
        quat=_yaw_quaternion(object_yaw),
    )
    peg.add_freejoint(name="square_peg_free")
    peg.add_geom(
        name="square_peg_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=PEG_HALF_SIZE, density=330.0, friction=[2.0, 0.01, 0.001],
        rgba=[0.95, 0.16, 0.08, 1.0], group=5,
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:9] = model.key(0).qpos[:9]
    data.ctrl[:] = model.key(0).ctrl
    mujoco.mj_forward(model, data)
    return model, data


def _angle_error(angle: float, target: float) -> float:
    period = math.pi / 2
    return abs((angle - target + period / 2) % period - period / 2)


def perception_image() -> np.ndarray:
    image = np.full((32, 64, 3), 0.16, dtype=np.float32)
    image[:, 31:33] = 0.32
    cv2.fillPoly(
        image, [_polygon(0, (16, 16), 8, OBJECT_YAW)],
        (0.92, 0.18, 0.10), lineType=cv2.LINE_AA,
    )
    hole = _polygon(0, (48, 16), 12, HOLE_YAW)
    cv2.fillPoly(image, [hole], (0.04, 0.04, 0.05), lineType=cv2.LINE_AA)
    cv2.polylines(image, [hole], True, (0.18, 0.60, 0.95), 2, lineType=cv2.LINE_AA)
    return image


def perceive_current_scene(
    model_path: Path = DEFAULT_VISION_MODEL,
    object_yaw: float = OBJECT_YAW,
    hole_yaw: float = HOLE_YAW,
):
    model, data = _build_scene(object_yaw, hole_yaw)
    visual_input = camera_perception_image(model, data)
    perception = ShapeFitPredictor(model_path).predict(visual_input)
    return perception, visual_input, model, data


def camera_perception_image(model, data, debug_path: Path | None = None) -> np.ndarray:
    """Build the CNN input from camera pixels; no shape name or object size is read."""
    import mujoco

    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 512)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 512)
    renderer = mujoco.Renderer(model, width=512, height=512)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.50, 0.0, TABLE_TOP_Z]
    camera.distance = 0.72
    camera.azimuth = 90
    camera.elevation = -90
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[5] = 1
    renderer.update_scene(data, camera=camera, scene_option=option)
    rgb = renderer.render().copy()
    renderer.close()
    if debug_path is not None:
        cv2.imwrite(str(debug_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    red = ((rgb[:, :, 0] > 130) & (rgb[:, :, 0] > rgb[:, :, 1] * 1.35)).astype(np.uint8) * 255
    blue = ((rgb[:, :, 2] > 120) & (rgb[:, :, 2] > rgb[:, :, 0] * 1.25)).astype(np.uint8) * 255

    def largest_contour(mask: np.ndarray, use_inner: bool = False) -> np.ndarray:
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise RuntimeError("카메라 영상에서 작업 윤곽을 찾지 못했습니다.")
        if use_inner and hierarchy is not None:
            inner = [contour for index, contour in enumerate(contours) if hierarchy[0][index][3] >= 0]
            if inner:
                return max(inner, key=cv2.contourArea)
        return max(contours, key=cv2.contourArea)

    object_contour = largest_contour(red)
    hole_contour = largest_contour(blue, use_inner=True)

    hole_box = cv2.minAreaRect(hole_contour)
    common_scale = 22.0 / max(hole_box[1])

    def normalized_panel(contour: np.ndarray, color: tuple[float, float, float], fill: bool) -> np.ndarray:
        rectangle = cv2.minAreaRect(contour)
        center = np.asarray(rectangle[0])
        points = contour[:, 0, :].astype(np.float32)
        points = (points - center) * common_scale + np.asarray([16.0, 16.0])
        points = np.rint(points).astype(np.int32)
        panel = np.full((32, 32, 3), 0.16, dtype=np.float32)
        if fill:
            cv2.fillPoly(panel, [points], color, lineType=cv2.LINE_AA)
        else:
            cv2.fillPoly(panel, [points], (0.04, 0.04, 0.05), lineType=cv2.LINE_AA)
            cv2.polylines(panel, [points], True, color, 2, lineType=cv2.LINE_AA)
        return panel

    combined = np.hstack((
        normalized_panel(object_contour, (0.92, 0.18, 0.10), True),
        normalized_panel(hole_contour, (0.18, 0.60, 0.95), False),
    ))
    combined[:, 31:33] = 0.32
    return combined


def run_insertion(
    frame_callback=None,
    model_path: Path = DEFAULT_VISION_MODEL,
    object_yaw: float = OBJECT_YAW,
    hole_yaw: float = HOLE_YAW,
    rotation_bias_rad: float = 0.0,
) -> tuple[InsertionResult, object, object]:
    import mujoco

    perception, _, model, data = perceive_current_scene(model_path, object_yaw, hole_yaw)
    if not perception["fits"]:
        raise RuntimeError(f"시각 안전 게이트가 삽입을 거부했습니다: {perception}")
    # 상단 카메라의 영상 x축은 작업대 월드 x축과 반사 관계다. 이는 물체 정답이 아니라
    # 고정 카메라 외부 파라미터 보정이며, 영상에서 읽은 상대 회전의 부호만 월드로 변환한다.
    planned_rotation = -float(perception["rotation_rad"]) + rotation_bias_rad
    planned_yaw = object_yaw + planned_rotation
    source_quat = _hand_quaternion(object_yaw)
    target_quat = _hand_quaternion(planned_yaw)
    grasp_z = TABLE_TOP_Z + PEG_HALF_SIZE[2] + 0.10
    poses = (
        ((*SOURCE_XY, 0.55), source_quat),
        ((*SOURCE_XY, grasp_z), source_quat),
        ((*SOURCE_XY, 0.58), source_quat),
        ((*HOLE_XY, 0.58), target_quat),
        ((*HOLE_XY, grasp_z + 0.015), target_quat),
        ((*HOLE_XY, 0.55), target_quat),
    )
    targets = [PandaPoseReachExperiment(position, quaternion).run()[0].target_qpos for position, quaternion in poses]
    stages = (
        ("approach", targets[0], 255.0, 700),
        ("align_object", targets[1], 255.0, 700),
        ("grasp", targets[1], 0.0, 650),
        ("lift", targets[2], 0.0, 800),
        ("rotate_and_transfer", targets[3], 0.0, 1100),
        ("insert", targets[4], 0.0, 900),
        ("release", targets[4], 255.0, 650),
        ("retreat", targets[5], 255.0, 650),
    )
    frame_geom_ids = {model.geom(name).id for name in ("frame_left", "frame_right", "frame_front", "frame_back")}
    peg_geom_id = model.geom("square_peg_geom").id
    table_id = model.geom("table").id
    frame_contacts = 0
    robot_table_contacts = 0
    peak_height = float(data.body("square_peg").xpos[2])
    for stage, target, gripper, steps in stages:
        start_arm = data.ctrl[:7].copy()
        start_gripper = float(data.ctrl[7])
        for step in range(steps):
            phase = (step + 1) / steps
            blend = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
            data.ctrl[:7] = start_arm + blend * (target - start_arm)
            data.ctrl[7] = start_gripper + blend * (gripper - start_gripper)
            mujoco.mj_step(model, data)
            peak_height = max(peak_height, float(data.body("square_peg").xpos[2]))
            frame_hit = False
            table_hit = False
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                pair = {int(contact.geom1), int(contact.geom2)}
                frame_hit |= peg_geom_id in pair and not pair.isdisjoint(frame_geom_ids)
                table_hit |= table_id in pair and peg_geom_id not in pair
            frame_contacts += frame_hit
            robot_table_contacts += table_hit
            if frame_callback is not None:
                frame_callback(model, data, stage)
    final = data.body("square_peg").xpos.copy()
    quaternion = data.body("square_peg").xquat.copy()
    yaw = math.atan2(
        2 * (quaternion[0] * quaternion[3] + quaternion[1] * quaternion[2]),
        1 - 2 * (quaternion[2] ** 2 + quaternion[3] ** 2),
    )
    xy_error = float(np.linalg.norm(final[:2] - np.asarray(HOLE_XY)))
    target_z = TABLE_TOP_Z + PEG_HALF_SIZE[2]
    height_error = abs(float(final[2]) - target_z)
    yaw_error = _angle_error(yaw, hole_yaw)
    released = float(data.joint("finger_joint1").qpos[0]) >= 0.035
    success = (
        xy_error <= 0.009 and height_error <= 0.012 and yaw_error <= math.radians(8)
        and peak_height - target_z >= 0.08 and released and robot_table_contacts == 0
    )
    result = InsertionResult(
        perceived_object=str(perception["object_shape"]),
        perceived_hole=str(perception["hole_shape"]),
        fit_decision=bool(perception["fits"]),
        planned_rotation_deg=math.degrees(planned_rotation),
        final_xy_error_m=xy_error,
        final_height_error_m=height_error,
        final_yaw_error_deg=math.degrees(yaw_error),
        peak_peg_height_m=peak_height,
        frame_contact_steps=frame_contacts,
        robot_table_contact_steps=robot_table_contacts,
        released=released,
        success=success,
    )
    return result, model, data


def render(
    output: Path, width: int = 960, height: int = 540, fps: int = 30,
    rotation_bias_rad: float = 0.0,
) -> InsertionResult:
    import mujoco

    process = subprocess.Popen([
        _ffmpeg_executable(), "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
        "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ], stdin=subprocess.PIPE)
    assert process.stdin is not None
    renderer = None
    next_frame = 0.0

    def callback(model, data, stage):
        nonlocal renderer, next_frame
        if renderer is None:
            model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
            model.vis.global_.offheight = max(model.vis.global_.offheight, height)
            renderer = mujoco.Renderer(model, width=width, height=height)
        if data.time + 1e-9 < next_frame:
            return
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0.50, 0.0, 0.34]
        camera.distance = 1.25
        camera.azimuth = 145
        camera.elevation = -28
        renderer.update_scene(data, camera=camera)
        frame = renderer.render().copy()
        frame[:8, :, :] = (50, 210, 105)
        process.stdin.write(frame.tobytes())
        next_frame += 1.0 / fps

    result, _, _ = run_insertion(callback, rotation_bias_rad=rotation_bias_rad)
    if renderer is not None:
        renderer.close()
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("형상 삽입 영상 인코딩 실패")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="정사각형 인식-정렬-삽입 물리 실험")
    parser.add_argument("--output", type=Path, default=Path("artifacts/fdb_shape_insertion.mp4"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = render(args.output)
    payload = {"task": "perceive square, match square receptacle, align, insert", "result": asdict(result), "video": str(args.output)}
    args.output.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    model, data = _build_scene()
    visual_input = camera_perception_image(
        model, data, args.output.with_name(args.output.stem + "_camera.png")
    )
    cv2.imwrite(str(args.output.with_name(args.output.stem + "_vision.png")),
                cv2.cvtColor(np.asarray(visual_input * 255, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    subprocess.run([_ffmpeg_executable(), "-loglevel", "error", "-y", "-ss", "12", "-i", str(args.output),
                    "-frames:v", "1", str(args.output.with_suffix(".png"))], check=True)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
