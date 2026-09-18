from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EVIDENCE = REPOSITORY_ROOT / "experiments" / "capability_evidence.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "state" / "capability_profile.json"


def load_evidence(path: Path = DEFAULT_EVIDENCE) -> dict[str, Any]:
    """Load and validate the reviewed evidence used to derive capability state."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    capabilities = payload.get("capabilities")
    if not isinstance(capabilities, list) or not capabilities:
        raise ValueError("capability evidence must contain a non-empty capabilities list")
    names: set[str] = set()
    for item in capabilities:
        required = {"name", "score", "target", "evidence", "next_gate", "evidence_refs"}
        missing = required.difference(item)
        if missing:
            raise ValueError(f"capability evidence is missing fields: {sorted(missing)}")
        if item["name"] in names:
            raise ValueError(f"duplicate capability: {item['name']}")
        names.add(item["name"])
        if not isinstance(item["score"], int) or not isinstance(item["target"], int):
            raise ValueError(f"score and target must be integers: {item['name']}")
        if item["target"] <= 0 or not 0 <= item["score"] <= item["target"]:
            raise ValueError(f"invalid score range: {item['name']}")
        if not isinstance(item["evidence_refs"], list) or not item["evidence_refs"]:
            raise ValueError(f"at least one evidence reference is required: {item['name']}")
        for reference in item["evidence_refs"]:
            if not (REPOSITORY_ROOT / reference).exists():
                raise ValueError(f"missing evidence reference: {reference}")
    return payload


def build_profile(evidence_path: Path = DEFAULT_EVIDENCE) -> dict[str, object]:
    """Derive a profile; historical scores are deliberately not embedded in code."""
    evidence = load_evidence(evidence_path)
    capabilities = evidence["capabilities"]
    total = sum(item["score"] for item in capabilities)
    target = sum(item["target"] for item in capabilities)
    try:
        source = str(evidence_path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        source = str(evidence_path)
    return {
        "schema_version": "1.1",
        "derived_from": source,
        "evidence_revision": evidence["evidence_revision"],
        "operational_score": total,
        "operational_target": target,
        "target_completion_percent": round(100 * total / target, 1),
        "human_age_equivalent": None,
        "age_interpretation": "사람의 발달연령으로 환산할 수 없는 좁은 과제형 시스템",
        "capabilities": capabilities,
        "selected_next_gate": evidence["selected_next_gate"],
        "completion_rule": "모든 항목 4점이며 보지 못한 시험 세트에서 안전 기준을 통과해야 목표 달성",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FDB 증거에서 능력 상태를 계산")
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    profile = build_profile(args.evidence.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(profile, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
