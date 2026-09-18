from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass

from fdb.challenge.shape_insertion import perceive_current_scene, run_insertion


SHAPE_ALIASES = {
    "정사각형": "square", "네모": "square", "사각형": "square", "square": "square",
    "원형": "circle", "동그라미": "circle", "원": "circle", "circle": "circle",
    "삼각형": "triangle", "세모": "triangle", "triangle": "triangle",
    "직사각형": "rectangle", "rectangle": "rectangle",
}


@dataclass(frozen=True)
class ShapeInsertionGoal:
    action: str
    object_shape: str
    receptacle_shape: str


def interpret_shape_command(command: str) -> ShapeInsertionGoal:
    lowered = command.lower()
    if not any(token in lowered for token in ("넣", "끼", "삽입", "insert", "fit")):
        raise ValueError("삽입 행동이 명확하지 않습니다.")
    mentions: list[tuple[int, str]] = []
    # 긴 별칭을 먼저 찾아 '정사각형' 안의 '사각형'을 중복 인식하지 않는다.
    occupied: list[tuple[int, int]] = []
    for token in sorted(SHAPE_ALIASES, key=len, reverse=True):
        for match in re.finditer(re.escape(token), lowered):
            span = match.span()
            if any(span[0] < end and start < span[1] for start, end in occupied):
                continue
            occupied.append(span)
            mentions.append((span[0], SHAPE_ALIASES[token]))
    mentions.sort()
    if len(mentions) < 2:
        raise ValueError("물체 형상과 구멍 형상을 모두 말해야 합니다.")
    return ShapeInsertionGoal("insert", mentions[0][1], mentions[1][1])


def execute_shape_command(command: str):
    goal = interpret_shape_command(command)
    if goal.object_shape != goal.receptacle_shape:
        raise RuntimeError("명령 물체와 수용구 형상이 달라 안전하게 삽입할 수 없습니다.")
    perception, _, _, _ = perceive_current_scene(shape=goal.object_shape)
    if perception["object_shape"] != goal.object_shape:
        raise RuntimeError(
            f"명령 물체는 {goal.object_shape}이지만 카메라는 {perception['object_shape']}로 인식했습니다."
        )
    if perception["hole_shape"] != goal.receptacle_shape:
        raise RuntimeError(
            f"명령 구멍은 {goal.receptacle_shape}이지만 카메라는 {perception['hole_shape']}로 인식했습니다."
        )
    if not perception["fits"]:
        raise RuntimeError("신경망과 형상 일치 안전 게이트가 삽입 불가로 판단했습니다.")
    result, model, data = run_insertion(shape=goal.object_shape)
    return goal, result, model, data


def main() -> None:
    parser = argparse.ArgumentParser(description="한국어 형상 삽입 명령 실행")
    parser.add_argument("command", nargs="?", default="네모를 네모칸에 넣어")
    args = parser.parse_args()
    goal, result, _, _ = execute_shape_command(args.command)
    print(json.dumps({"command": args.command, "goal": asdict(goal), "result": asdict(result)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
