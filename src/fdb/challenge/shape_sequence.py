from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass

from fdb.challenge.shape_command import ShapeInsertionGoal, execute_shape_command, interpret_shape_command


@dataclass(frozen=True)
class SequenceResult:
    commands: tuple[str, ...]
    goals: tuple[ShapeInsertionGoal, ...]
    completed: int
    success: bool
    results: tuple[object, ...]


def interpret_shape_sequence(command: str) -> tuple[str, ...]:
    normalized = re.sub(r"^(먼저|우선)\s*", "", command.strip())
    clauses = [
        clause.strip(" ,.")
        for clause in re.split(r"\s*(?:그\s*다음(?:으로)?|다음(?:으로)?|그리고|마지막으로)\s*", normalized)
        if clause.strip(" ,.")
    ]
    if len(clauses) < 2:
        raise ValueError("순차 과제에는 두 개 이상의 삽입 명령이 필요합니다.")
    # Parse every clause before executing anything. An ambiguous later clause
    # therefore cannot leave the world half-mutated.
    for clause in clauses:
        interpret_shape_command(clause)
    return tuple(clauses)


def execute_shape_sequence(command: str) -> SequenceResult:
    clauses = interpret_shape_sequence(command)
    goals: list[ShapeInsertionGoal] = []
    results: list[object] = []
    for clause in clauses:
        goal, result, _, _ = execute_shape_command(clause)
        goals.append(goal)
        results.append(result)
        if not result.success:
            break
    return SequenceResult(
        commands=clauses,
        goals=tuple(goals),
        completed=len(results),
        success=len(results) == len(clauses) and all(result.success for result in results),
        results=tuple(results),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="한국어 다단계 형상 삽입 계획")
    parser.add_argument(
        "command", nargs="?",
        default="먼저 세모를 삼각형 칸에 넣고 다음으로 원을 원형 칸에 넣어",
    )
    args = parser.parse_args()
    result = execute_shape_sequence(args.command)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
