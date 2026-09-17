from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Position:
    x: int
    y: int

    def distance_to(self, other: "Position") -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)


@dataclass(frozen=True)
class SceneObject:
    object_id: str
    color: str
    position: Position


@dataclass(frozen=True)
class TargetZone:
    zone_id: str
    color: str
    position: Position


@dataclass(frozen=True)
class Goal:
    object_color: str
    target_color: str


@dataclass(frozen=True)
class MoveAction:
    object_id: str
    destination: Position

    @property
    def action_type(self) -> str:
        return "move"


@dataclass(frozen=True)
class CandidatePlan:
    plan_id: str
    actions: tuple[MoveAction, ...]
    rationale: str


@dataclass(frozen=True)
class Evaluation:
    success: bool
    score: float
    path_length: int
    action_count: int
    violations: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorldState:
    width: int
    height: int
    objects: tuple[SceneObject, ...]
    zones: tuple[TargetZone, ...]

    def validate_position(self, position: Position) -> None:
        if not (0 <= position.x < self.width and 0 <= position.y < self.height):
            raise ValueError(f"Position outside world bounds: {position}")

    def object_by_color(self, color: str) -> SceneObject:
        matches = [item for item in self.objects if item.color == color]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {color!r} object, found {len(matches)}")
        return matches[0]

    def zone_by_color(self, color: str) -> TargetZone:
        matches = [item for item in self.zones if item.color == color]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {color!r} zone, found {len(matches)}")
        return matches[0]

    def move(self, action: MoveAction) -> "WorldState":
        self.validate_position(action.destination)
        if not any(item.object_id == action.object_id for item in self.objects):
            raise ValueError(f"Unknown object: {action.object_id}")
        moved = tuple(
            SceneObject(item.object_id, item.color, action.destination)
            if item.object_id == action.object_id
            else item
            for item in self.objects
        )
        return WorldState(self.width, self.height, moved, self.zones)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorldState":
        return cls(
            width=int(data["width"]),
            height=int(data["height"]),
            objects=tuple(
                SceneObject(
                    object_id=item["object_id"],
                    color=item["color"],
                    position=Position(**item["position"]),
                )
                for item in data["objects"]
            ),
            zones=tuple(
                TargetZone(
                    zone_id=item["zone_id"],
                    color=item["color"],
                    position=Position(**item["position"]),
                )
                for item in data["zones"]
            ),
        )

