import pytest

from fdb.memory import MemoryStore


def test_memory_revision_promotion_archive_and_search(tmp_path) -> None:
    store = MemoryStore(tmp_path)
    memory_id = store.write({"summary": "triangle insertion alignment", "tags": ["shape"]})
    assert store.search(query="triangle")[0]["memory_id"] == memory_id
    updated = store.update(memory_id, {"summary": "verified alignment"}, reason="second trial")
    assert updated["revision"] == 2
    with pytest.raises(ValueError, match="repeated independent evidence"):
        store.promote(memory_id, evidence_ids=["episode_1"])
    store.promote(memory_id, evidence_ids=["episode_1", "episode_2"])
    assert store.search(query="verified", tiers=("long_term",))[0]["memory_id"] == memory_id
    store.deprecate(memory_id, reason="new collision evidence", replacement_id="memory_replacement")
    assert store.read(memory_id)["status"] == "deprecated"
    store.archive(memory_id, reason="inactive")
    assert store.search(query="verified", tiers=("current", "long_term")) == []
    assert store.search(query="verified", tiers=("archive",))[0]["memory_id"] == memory_id
