from __future__ import annotations

import re

from .models import Goal


COLOR_ALIASES = {
    "red": "red",
    "blue": "blue",
    "green": "green",
    "yellow": "yellow",
    "빨간": "red",
    "빨강": "red",
    "파란": "blue",
    "파랑": "blue",
    "초록": "green",
    "녹색": "green",
    "노란": "yellow",
    "노랑": "yellow",
}


class TaskInterpreter:
    """A deliberately small v0 parser with explicit failure on ambiguity."""

    def interpret(self, task: str) -> Goal:
        lowered = task.lower()
        hits: list[tuple[int, str]] = []
        for token, normalized in COLOR_ALIASES.items():
            for match in re.finditer(re.escape(token), lowered):
                hits.append((match.start(), normalized))
        hits.sort()

        ordered_colors: list[str] = []
        for _, color in hits:
            if not ordered_colors or ordered_colors[-1] != color:
                ordered_colors.append(color)
        if len(ordered_colors) < 2:
            raise ValueError(
                "Task must identify an object color and a target-zone color. "
                "Example: 'Move the red object to the blue zone'."
            )
        return Goal(object_color=ordered_colors[0], target_color=ordered_colors[1])

