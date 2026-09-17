from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


class EpisodeStore:
    """Append-only JSON episode storage; existing episode files are never overwritten."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, episode: dict[str, Any]) -> Path:
        now = datetime.now(timezone.utc)
        episode_id = f"episode_{now.strftime('%Y%m%dT%H%M%S%fZ')}_{uuid4().hex[:8]}"
        payload = {
            "schema_version": "1.0",
            "episode_id": episode_id,
            "created_at": now.isoformat(),
            **episode,
        }
        destination = self.root / f"{episode_id}.json"
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return destination

    def search(self, *, task_contains: str | None = None) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for path in sorted(self.root.glob("episode_*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if task_contains and task_contains.lower() not in payload.get("task", "").lower():
                continue
            results.append(payload)
        return results

