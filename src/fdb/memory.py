from __future__ import annotations

import json
import shutil
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


class MemoryStore:
    """Revision-preserving memory operations for current, long-term, and archive tiers."""

    TIERS = ("current", "long_term", "archive")

    def __init__(self, root: Path) -> None:
        self.root = root
        for tier in self.TIERS:
            (root / tier).mkdir(parents=True, exist_ok=True)
        (root / "audit").mkdir(parents=True, exist_ok=True)

    def write(self, payload: dict[str, Any], *, tier: str = "current") -> str:
        self._require_tier(tier)
        memory_id = payload.get("memory_id") or f"memory_{uuid4().hex}"
        if self._find(memory_id) is not None:
            raise FileExistsError(memory_id)
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "schema_version": "1.0", "memory_id": memory_id, "revision": 1,
            "status": "active", "created_at": now, "updated_at": now, **payload,
        }
        self._path(tier, memory_id).write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        return memory_id

    def read(self, memory_id: str) -> dict[str, Any]:
        found = self._find(memory_id)
        if found is None:
            raise KeyError(memory_id)
        return json.loads(found.read_text(encoding="utf-8"))

    def search(self, *, query: str | None = None, tiers: tuple[str, ...] = ("current", "long_term")) -> list[dict[str, Any]]:
        for tier in tiers:
            self._require_tier(tier)
        results = []
        needle = query.lower() if query else None
        for tier in tiers:
            for path in sorted((self.root / tier).glob("memory_*.json")):
                payload = json.loads(path.read_text(encoding="utf-8"))
                if needle and needle not in json.dumps(payload, ensure_ascii=False).lower():
                    continue
                results.append({"tier": tier, **payload})
        return results

    def update(self, memory_id: str, changes: dict[str, Any], *, reason: str) -> dict[str, Any]:
        path = self._find_required(memory_id)
        previous = json.loads(path.read_text(encoding="utf-8"))
        revision = int(previous["revision"]) + 1
        history = self.root / "audit" / f"{memory_id}.r{previous['revision']}.json"
        shutil.copy2(path, history)
        updated = {
            **previous, **changes, "memory_id": memory_id, "revision": revision,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self._audit(memory_id, "update", reason, revision)
        return updated

    def promote(self, memory_id: str, *, evidence_ids: list[str], minimum_evidence: int = 2) -> None:
        if len(set(evidence_ids)) < minimum_evidence:
            raise ValueError("promotion requires repeated independent evidence")
        self._move(memory_id, "long_term", "promote", f"evidence={evidence_ids}")

    def archive(self, memory_id: str, *, reason: str) -> None:
        self._move(memory_id, "archive", "archive", reason)

    def deprecate(self, memory_id: str, *, reason: str, replacement_id: str | None = None) -> None:
        changes = {"status": "deprecated", "deprecation_reason": reason,
                   "replacement_id": replacement_id}
        self.update(memory_id, changes, reason=reason)

    def _move(self, memory_id: str, tier: str, operation: str, reason: str) -> None:
        source = self._find_required(memory_id)
        destination = self._path(tier, memory_id)
        if source != destination:
            shutil.move(str(source), str(destination))
        self._audit(memory_id, operation, reason, self.read(memory_id)["revision"])

    def _audit(self, memory_id: str, operation: str, reason: str, revision: int) -> None:
        now = datetime.now(timezone.utc)
        path = self.root / "audit" / f"event_{now.strftime('%Y%m%dT%H%M%S%fZ')}_{uuid4().hex[:8]}.json"
        path.write_text(json.dumps({
            "memory_id": memory_id, "operation": operation, "reason": reason,
            "revision": revision, "created_at": now.isoformat(),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _find_required(self, memory_id: str) -> Path:
        found = self._find(memory_id)
        if found is None:
            raise KeyError(memory_id)
        return found

    def _find(self, memory_id: str) -> Path | None:
        for tier in self.TIERS:
            path = self._path(tier, memory_id)
            if path.exists():
                return path
        return None

    def _path(self, tier: str, memory_id: str) -> Path:
        if Path(memory_id).name != memory_id or not memory_id.startswith("memory_"):
            raise ValueError("invalid memory id")
        return self.root / tier / f"{memory_id}.json"

    def _require_tier(self, tier: str) -> None:
        if tier not in self.TIERS:
            raise ValueError(f"unknown memory tier: {tier}")
