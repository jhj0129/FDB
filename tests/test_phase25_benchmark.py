from fdb.core.phase25_benchmark import _experiment_matrix


def test_success_envelope_matrix_has_required_levels() -> None:
    rows = _experiment_matrix("all")
    levels = {}
    for row in rows:
        levels.setdefault(row["axis"], set()).add(row["level"])
    assert levels["position_mm"] == {0, 2, 5, 10, 20}
    assert levels["yaw_deg"] == {0, 5, 10, 20, 40}
    assert levels["target_mm"] == {0, 2, 5, 10}
    assert {row["observation_mode"] for row in rows if row["axis"] == "baseline"} == {
        "oracle", "camera",
    }
    assert len([row for row in rows if row["axis"] == "ablation"]) == 4
