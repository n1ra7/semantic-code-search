from semantic_index.state import StateStore


def test_incremental_reindex_logic(tmp_path):
    store = StateStore(str(tmp_path / "state.sqlite"))

    sha_v1 = StateStore.content_sha("def foo(): pass")
    assert store.needs_reindex("a.py", sha_v1)  # unseen -> index

    store.record("a.py", sha_v1, chunk_count=1)
    assert not store.needs_reindex("a.py", sha_v1)  # unchanged -> skip

    sha_v2 = StateStore.content_sha("def foo(): return 1")
    assert store.needs_reindex("a.py", sha_v2)  # changed -> re-index


def test_prune_tracking(tmp_path):
    store = StateStore(str(tmp_path / "state.sqlite"))
    store.record("a.py", "sha", 1)
    store.record("b.py", "sha", 1)
    assert store.all_paths() == {"a.py", "b.py"}
    store.remove("a.py")
    assert store.all_paths() == {"b.py"}
