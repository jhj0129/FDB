import json

import pytest

from fdb.challenge.cognitive_curriculum import DEFAULT_EVIDENCE, build_profile, load_evidence


def test_profile_is_derived_from_latest_evidence() -> None:
    profile = build_profile()
    assert profile["operational_score"] == 25
    assert profile["operational_target"] == 36
    assert profile["human_age_equivalent"] is None
    dynamic_gait = next(item for item in profile["capabilities"] if item["name"] == "동역학 보행")
    assert dynamic_gait["score"] == 4
    assert profile["selected_next_gate"]["capability"] == "시각 형상·fit 판단과 보행 통합"


def test_score_changes_only_when_evidence_changes(tmp_path) -> None:
    evidence = json.loads(DEFAULT_EVIDENCE.read_text(encoding="utf-8"))
    evidence["capabilities"][0]["score"] = 3
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence, ensure_ascii=False), encoding="utf-8")
    assert build_profile(path)["operational_score"] == 24


def test_untraceable_evidence_is_rejected(tmp_path) -> None:
    evidence = json.loads(DEFAULT_EVIDENCE.read_text(encoding="utf-8"))
    evidence["capabilities"][0]["evidence_refs"] = ["missing/result.json"]
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="missing evidence reference"):
        load_evidence(path)
