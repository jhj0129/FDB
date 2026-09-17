from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from fdb.v2.render_final import _ffmpeg_executable


def combine(sorting: Path, gait: Path, output: Path) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        _ffmpeg_executable(), "-loglevel", "error", "-y",
        "-i", str(sorting), "-i", str(gait),
        "-filter_complex",
        "[0:v]scale=720:540:force_original_aspect_ratio=decrease,"
        "pad=960:540:(ow-iw)/2:(oh-ih)/2:black,setsar=1[v0];"
        "[1:v]scale=960:540,setsar=1[v1];"
        "[v0][v1]concat=n=2:v=1:a=0[v]",
        "-map", "[v]", "-r", "30", "-c:v", "libx264", "-crf", "21",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ], check=True)
    return {
        "video": str(output),
        "resolution": [960, 540],
        "fps": 30,
        "timeline": {
            "0_to_34_93s": "Panda 3-object neural classification and sorting",
            "34_93_to_38_93s": "four-humanoid contact-gated gait comparison",
        },
        "sources": [str(sorting), str(gait)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FDB 분류·보행 최종 통합 영상")
    parser.add_argument("--sorting", type=Path, default=Path("artifacts/fdb_multi_object_sorting.mp4"))
    parser.add_argument("--gait", type=Path, default=Path("artifacts/fdb_human_gait.mp4"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/fdb_sort_and_walk_final.mp4"))
    args = parser.parse_args()
    metrics = combine(args.sorting, args.gait, args.output)
    metrics_path = args.output.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**metrics, "metrics": str(metrics_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
