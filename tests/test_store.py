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

def test_delete_memory_scoped_to_project(tmp_path):
    store = Store(str(tmp_path / "db.lance"))
    a = _row("a", "alpha", [1.0, 0.0]); a["memory_id"] = "standup"
    b = _row("b", "beta", [1.0, 0.0]); b["memory_id"] = "standup"
    store.add([a, b])
    store.delete_memory("standup", project_id="beta")
    assert [h["id"] for h in store.vector_search([1.0, 0.0], k=5)] == ["a"]

def test_delete_memory_without_project_removes_all(tmp_path):
    store = Store(str(tmp_path / "db.lance"))
    a = _row("a", "alpha", [1.0, 0.0]); a["memory_id"] = "standup"
    b = _row("b", "beta", [1.0, 0.0]); b["memory_id"] = "standup"
    store.add([a, b])
    store.delete_memory("standup")
    assert store.vector_search([1.0, 0.0], k=5) == []

def test_table_visible_beyond_table_names_pagination(tmp_path):
    store = Store(str(tmp_path / "db.lance"))
    store.add([_row("seg", "p", [1.0, 0.0])])
    # 12 tables sorting before 'segments' push it past table_names()' default limit of 10.
    for i in range(12):
        store.db.create_table("a%02d" % i, data=[{"id": "x", "vector": [1.0, 0.0]}])
    assert "segments" not in store.db.table_names()  # paginated listing hides it
    assert [h["id"] for h in store.vector_search([1.0, 0.0], k=5)] == ["seg"]

def test_table_lookup_does_not_call_table_names(tmp_path):
    store = Store(str(tmp_path / "db.lance"))
    store.add([_row("seg", "p", [1.0, 0.0])])
    def boom(*a, **k):
        raise AssertionError("table_names must not be used to locate the table")
    store.db.table_names = boom
    assert store._table() is not None
    assert [h["id"] for h in store.vector_search([1.0, 0.0], k=5)] == ["seg"]

def test_add_tolerates_concurrent_create(tmp_path):
    path = str(tmp_path / "db.lance")
    store = Store(path)
    store.add([_row("a", "p", [1.0, 0.0])])  # table now exists on disk
    # Simulate the race: our _table() check misses it, then create_table reports it exists.
    store._table = lambda: None
    def already_exists(*a, **k):
        raise ValueError("Table 'segments' already exists")
    store.db.create_table = already_exists
    store.add([_row("b", "p", [1.0, 0.0])])  # must fall back to open_table + add
    ids = {h["id"] for h in Store(path).vector_search([1.0, 0.0], k=5)}
    assert ids == {"a", "b"}

def test_delete_memory_keep_ids_preserves_reingested_rows(tmp_path):
    store = Store(str(tmp_path / "db.lance"))
    old_a = _row("old_a", "p", [1.0, 0.0]); old_a["memory_id"] = "standup"
    old_b = _row("old_b", "p", [1.0, 0.0]); old_b["memory_id"] = "standup"
    other = _row("other", "p", [1.0, 0.0]); other["memory_id"] = "keep"
    store.add([old_a, old_b, other])
    # re-ingest: add fresh rows first, then delete only the stale ones of this memory.
    new_a = _row("new_a", "p", [1.0, 0.0]); new_a["memory_id"] = "standup"
    new_b = _row("new_b", "p", [1.0, 0.0]); new_b["memory_id"] = "standup"
    store.add([new_a, new_b])
    store.delete_memory("standup", project_id="p", keep_ids=["new_a", "new_b"])
    ids = {h["id"] for h in store.vector_search([1.0, 0.0], k=10)}
    assert ids == {"new_a", "new_b", "other"}

def test_delete_memory_keep_ids_escapes_quotes(tmp_path):
    store = Store(str(tmp_path / "db.lance"))
    keep = _row("o'brien", "p", [1.0, 0.0]); keep["memory_id"] = "m1"
    drop = _row("drop", "p", [1.0, 0.0]); drop["memory_id"] = "m1"
    store.add([keep, drop])
    # unescaped, id NOT IN ('o'brien') is a SQL syntax error; escaping keeps o'brien.
    store.delete_memory("m1", keep_ids=["o'brien"])
    assert [h["id"] for h in store.vector_search([1.0, 0.0], k=5)] == ["o'brien"]

def test_delete_memory_empty_keep_ids_deletes_all(tmp_path):
    store = Store(str(tmp_path / "db.lance"))
    a = _row("a", "p", [1.0, 0.0]); a["memory_id"] = "m1"
    b = _row("b", "p", [1.0, 0.0]); b["memory_id"] = "m1"
    store.add([a, b])
    store.delete_memory("m1", keep_ids=[])  # empty behaves as today: delete all
    assert store.vector_search([1.0, 0.0], k=5) == []
