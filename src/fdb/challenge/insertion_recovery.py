from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess

from fdb.challenge.shape_insertion import InsertionResult, render, run_insertion
from fdb.v2.render_final import _ffmpeg_executable


@dataclass(frozen=True)
class RecoveryResult:
    injected_initial_error_deg: float
    initial_success: bool
    attempted_biases_deg: tuple[float, ...]
    selected_bias_deg: float | None
    recovered_success: bool
    final_result: InsertionResult | None


def recover_from_rotation_error(initial_bias_rad: float = math.radians(40.0)) -> RecoveryResult:
    initial, _, _ = run_insertion(rotation_bias_rad=initial_bias_rad)
    if initial.success:
        return RecoveryResult(
            math.degrees(initial_bias_rad), True, (math.degrees(initial_bias_rad),),
            math.degrees(initial_bias_rad), True, initial,
        )
    # 실패 원인이 방향 오차와 frame 접촉으로 관찰됐을 때 기준 예측 주변의 후보를
    # fresh-state 내부 시뮬레이션에서 비교한다. 실제 로봇 실행 전 후보 검증 단계다.
    candidate_biases = (0.0, math.radians(-10), math.radians(10), math.radians(-20), math.radians(20))
    candidates: list[tuple[float, InsertionResult]] = []
    attempted = [initial_bias_rad]
    for bias in candidate_biases:
        result, _, _ = run_insertion(rotation_bias_rad=bias)
        attempted.append(bias)
        if result.success:
            candidates.append((bias, result))
    if not candidates:
        return RecoveryResult(
            math.degrees(initial_bias_rad), False,
            tuple(math.degrees(value) for value in attempted), None, False, None,
        )
    bias, selected = min(
        candidates,
        key=lambda item: (
            item[1].final_xy_error_m + math.radians(item[1].final_yaw_error_deg),
            item[1].frame_contact_steps,
            abs(item[0]),
        ),
    )
    return RecoveryResult(
        math.degrees(initial_bias_rad), False,
        tuple(math.degrees(value) for value in attempted),
        math.degrees(bias), selected.success, selected,
    )


def render_recovery(output: Path, initial_bias_rad: float = math.radians(40.0)) -> RecoveryResult:
    output.parent.mkdir(parents=True, exist_ok=True)
    failed_video = output.with_name(output.stem + "_failed.mp4")
    recovered_video = output.with_name(output.stem + "_recovered.mp4")
    render(failed_video, rotation_bias_rad=initial_bias_rad)
    recovery = recover_from_rotation_error(initial_bias_rad)
    if not recovery.recovered_success or recovery.selected_bias_deg is None:
        raise RuntimeError("삽입 방향 오류 복구 후보를 찾지 못했습니다.")
    render(recovered_video, rotation_bias_rad=math.radians(recovery.selected_bias_deg))
    subprocess.run([
        _ffmpeg_executable(), "-loglevel", "error", "-y", "-i", str(failed_video),
        "-i", str(recovered_video), "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
        "-map", "[v]", "-c:v", "libx264", "-crf", "21", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ], check=True)
    failed_video.unlink()
    recovered_video.unlink()
    return recovery


def main() -> None:
    parser = argparse.ArgumentParser(description="삽입 방향 오류 탐지와 후보 재시도")
    parser.add_argument("--output", type=Path, default=Path("artifacts/fdb_insertion_recovery.mp4"))
    args = parser.parse_args()
    result = render_recovery(args.output)
    payload = {"task": "detect failed insertion and recover by candidate simulation", "result": asdict(result), "video": str(args.output)}
    args.output.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
