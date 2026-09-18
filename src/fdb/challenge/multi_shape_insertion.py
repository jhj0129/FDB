from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path

from fdb.challenge.shape_insertion import render, run_insertion
from fdb.v2.render_final import _ffmpeg_executable


SHAPES = ("square", "circle", "triangle", "rectangle")
KOREAN_COMMANDS = {
    "square": "네모를 네모칸에 넣어",
    "circle": "동그라미를 원형 칸에 넣어",
    "triangle": "세모를 삼각형 칸에 끼워",
    "rectangle": "직사각형을 직사각형 구멍에 삽입해",
}


def evaluate_suite() -> dict[str, object]:
    trials = []
    for shape in SHAPES:
        result, _, _ = run_insertion(shape=shape)
        trials.append({"shape": shape, "command": KOREAN_COMMANDS[shape], "result": asdict(result)})
    return {
        "task": "카메라 형상 판단과 네 종류 물체-수용구 정렬 삽입",
        "shapes": list(SHAPES),
        "successes": sum(bool(trial["result"]["success"]) for trial in trials),
        "total": len(trials),
        "robot_table_contact_steps": sum(
            int(trial["result"]["robot_table_contact_steps"]) for trial in trials
        ),
        "trials": trials,
    }


def render_suite(output: Path) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    trials = []
    with tempfile.TemporaryDirectory(prefix="fdb_multi_shape_") as directory:
        temporary = Path(directory)
        videos = []
        for shape in SHAPES:
            video = temporary / f"{shape}.mp4"
            result = render(video, shape=shape)
            videos.append(video)
            trials.append({"shape": shape, "command": KOREAN_COMMANDS[shape], "result": asdict(result)})
        inputs: list[str] = []
        for video in videos:
            inputs.extend(("-i", str(video)))
        streams = "".join(f"[{index}:v]" for index in range(len(videos)))
        subprocess.run([
            _ffmpeg_executable(), "-loglevel", "error", "-y", *inputs,
            "-filter_complex", f"{streams}concat=n={len(videos)}:v=1:a=0[v]",
            "-map", "[v]", "-c:v", "libx264", "-crf", "21", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(output),
        ], check=True)
    payload = {
        "task": "카메라 형상 판단과 네 종류 물체-수용구 정렬 삽입",
        "shapes": list(SHAPES),
        "successes": sum(bool(trial["result"]["success"]) for trial in trials),
        "total": len(trials),
        "robot_table_contact_steps": sum(
            int(trial["result"]["robot_table_contact_steps"]) for trial in trials
        ),
        "trials": trials,
        "video": str(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    subprocess.run([
        _ffmpeg_executable(), "-loglevel", "error", "-y", "-ss", "27", "-i", str(output),
        "-frames:v", "1", str(output.with_suffix(".png")),
    ], check=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="네 형상 카메라 판단·정렬·물리 삽입")
    parser.add_argument("--output", type=Path, default=Path("artifacts/fdb_multi_shape_insertion.mp4"))
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    payload = evaluate_suite() if args.no_video else render_suite(args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
