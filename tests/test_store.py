from vproc.store.lancedb_store import Store

def _row(id, project, vec, text="x"):
    return {"id": id, "project_id": project, "memory_id": "m", "speaker": "SPEAKER_0",
            "start_ts": 0.0, "end_ts": 1.0, "said_text": text, "on_screen_text": "",
            "embed_text": text, "source_video": "/v.mp4", "frame_path": "", "vector": vec}

def test_vector_search_orders_by_similarity(tmp_path):
    store = Store(str(tmp_path / "db.lance"))
    store.add([_row("near", "p", [1.0, 0.0]), _row("far", "p", [0.0, 1.0])])
    hits = store.vector_search([1.0, 0.0], k=2)
    assert hits[0]["id"] == "near"
    assert "_distance" in hits[0]

def test_metadata_prefilter(tmp_path):
    store = Store(str(tmp_path / "db.lance"))
    store.add([_row("a", "p1", [1.0, 0.0]), _row("b", "p2", [1.0, 0.0])])
    hits = store.vector_search([1.0, 0.0], k=5, where="project_id = 'p2'")
    assert [h["id"] for h in hits] == ["b"]

def test_search_on_empty_store_returns_empty(tmp_path):
    store = Store(str(tmp_path / "db.lance"))
    assert store.vector_search([1.0, 0.0], k=3) == []
    assert store.fts_search("anything", k=3) == []

def test_delete_memory_removes_only_that_memory(tmp_path):
    store = Store(str(tmp_path / "db.lance"))
    a = _row("a", "p", [1.0, 0.0]); a["memory_id"] = "keep"
    b = _row("b", "p", [1.0, 0.0]); b["memory_id"] = "drop"
    store.add([a, b])
    store.delete_memory("drop")
    assert [h["id"] for h in store.vector_search([1.0, 0.0], k=5)] == ["a"]

def test_delete_memory_on_empty_store_is_noop(tmp_path):
    store = Store(str(tmp_path / "db.lance"))
    store.delete_memory("anything")  # must not raise on a not-yet-created table
