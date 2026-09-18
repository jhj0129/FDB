from fdb.core.phase2_evaluation import SCRIPTED_SEQUENCE, summarize_runtime_file


def test_scripted_baseline_is_explicitly_not_autonomous() -> None:
    assert SCRIPTED_SEQUENCE == (
        "reach_object", "grasp_object", "lift_object", "align_object",
        "move_to_target", "insert_object", "release_object",
    )


def test_runtime_summary_keeps_neural_and_memory_contribution_separate(tmp_path) -> None:
    path = tmp_path / "summary.json"
    path.write_text('''{
      "mode":"rules_only", "memory":false, "successes":1, "total":1,
      "results":[{"actions":7,"replans":0,"metrics":{"total_task_latency_s":2.0,
      "world_model_calls":7,"neural_world_model_calls":0,"memory_retrieval_count":0}}]
    }''')
    summary = summarize_runtime_file(path)
    assert summary["neural_calls"] == 0
    assert summary["memory_hits"] == 0
    assert summary["actions"] == 7
