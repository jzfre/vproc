from vproc.store.lancedb_store import Store
import pytest

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

def test_update_speaker_on_empty_store_returns_zero(tmp_path):
    store = Store(str(tmp_path / "db.lance"))
    assert store.update_speaker("m1", "default", "SPEAKER_00", "X") == 0  # no table yet, no raise


def test_update_speaker_renames_only_matching_rows(tmp_path):
    store = Store(str(tmp_path / "db"))
    store.add([
        {"id": "a", "memory_id": "m1", "project_id": "default", "speaker": "SPEAKER_02",
         "start_ts": 0.0, "end_ts": 1.0, "said_text": "x", "on_screen_text": "",
         "embed_text": "x", "source_video": "v", "frame_path": "", "vector": [1.0, 0.0]},
        {"id": "b", "memory_id": "m1", "project_id": "default", "speaker": "SPEAKER_01",
         "start_ts": 1.0, "end_ts": 2.0, "said_text": "y", "on_screen_text": "",
         "embed_text": "y", "source_video": "v", "frame_path": "", "vector": [0.0, 1.0]},
        {"id": "c", "memory_id": "m2", "project_id": "default", "speaker": "SPEAKER_02",
         "start_ts": 0.0, "end_ts": 1.0, "said_text": "z", "on_screen_text": "",
         "embed_text": "z", "source_video": "v", "frame_path": "", "vector": [1.0, 1.0]},
    ])
    n = store.update_speaker("m1", "default", "SPEAKER_02", "PATINO, DANIEL")
    assert n == 1  # only row "a" matched (memory_id + speaker + project_id)
    rows = {r["id"]: r["speaker"] for r in store._table().to_arrow().to_pylist()}
    assert rows == {"a": "PATINO, DANIEL", "b": "SPEAKER_01", "c": "SPEAKER_02"}
    assert store.update_speaker("m1", "default", "SPEAKER_02", "X") == 0  # no longer present

def test_memory_rows_filters_and_strips_vector(tmp_path):
    store = Store(str(tmp_path / "db"))
    store.add([
        {"id": "a", "memory_id": "m1", "project_id": "default", "speaker": "S1",
         "start_ts": 5.0, "end_ts": 9.0, "said_text": "x", "on_screen_text": "",
         "embed_text": "x", "source_video": "v.mp4", "frame_path": "", "vector": [1.0, 0.0]},
        {"id": "b", "memory_id": "m2", "project_id": "default", "speaker": "S2",
         "start_ts": 0.0, "end_ts": 1.0, "said_text": "y", "on_screen_text": "",
         "embed_text": "y", "source_video": "w.mp4", "frame_path": "", "vector": [0.0, 1.0]},
    ])
    rows = store.memory_rows("m1", "default")
    assert [r["id"] for r in rows] == ["a"]
    assert "vector" not in rows[0]
    assert store.memory_rows("m1", "other-project") == []
    assert store.memory_rows("nope") == []
    assert Store(str(tmp_path / "empty")).memory_rows("m1") == []  # no table yet

def test_all_rows_strips_vector(tmp_path):
    store = Store(str(tmp_path / "db"))
    store.add([{"id": "a", "memory_id": "m1", "project_id": "default", "speaker": "S1",
                "start_ts": 0.0, "end_ts": 1.0, "said_text": "x", "on_screen_text": "",
                "embed_text": "x", "source_video": "v.mp4", "frame_path": "",
                "vector": [1.0, 0.0]}])
    rows = store.all_rows()
    assert len(rows) == 1 and "vector" not in rows[0]
    assert Store(str(tmp_path / "empty")).all_rows() == []


def test_metadata_queries_return_all_matching_rows_and_escape_filters(tmp_path):
    store = Store(str(tmp_path / "db"))
    rows = [
        {**_row(str(i), "owner's project", [1.0, 0.0]), "memory_id": "Tim's meeting"}
        for i in range(25)
    ]
    rows.append(_row("other", "other project", [0.0, 1.0]))
    store.add(rows)
    selected = store.memory_rows("Tim's meeting", "owner's project")
    assert {r["id"] for r in selected} == {str(i) for i in range(25)}
    assert all("vector" not in r for r in selected)
    assert len(store.all_rows()) == 26
    assert store.memory_rows("' OR true OR memory_id = '") == []
    assert store.memory_rows("Tim's meeting", "' OR true OR project_id = '") == []


def test_embedding_identity_persists_and_rejects_same_dimension_model_change(tmp_path):
    path = str(tmp_path / "db")
    store = Store(path)
    store.add([_row("old", "p", [1.0, 0.0])], embedding_model="model-a")
    reopened = Store(path)
    reopened.validate_embedding_model("model-a", dimension=2)
    with pytest.raises(ValueError, match="VPROC_INDEX_PATH"):
        reopened.validate_embedding_model("model-b", dimension=2)
    with pytest.raises(ValueError, match="VPROC_INDEX_PATH"):
        reopened.add([_row("new", "p", [1.0, 0.0])], embedding_model="model-b")
    assert [row["id"] for row in reopened.all_rows()] == ["old"]


def test_legacy_index_allows_metadata_but_rejects_model_dependent_work(tmp_path):
    store = Store(str(tmp_path / "db"))
    store.add([_row("legacy", "p", [1.0, 0.0])])
    with pytest.raises(ValueError, match="re-ingest"):
        store.validate_embedding_model("model-a")
    with pytest.raises(ValueError, match="re-ingest"):
        store.vector_search([1.0, 0.0], 1, embedding_model="model-a")
    with pytest.raises(ValueError, match="re-ingest"):
        store.add([_row("new", "p", [1.0, 0.0])], embedding_model="model-a")
    assert [row["id"] for row in store.memory_rows("m", "p")] == ["legacy"]
    assert len(store.all_rows()) == 1


def test_known_index_rejects_unidentified_writes_and_wrong_dimensions(tmp_path):
    store = Store(str(tmp_path / "db"))
    store.add([_row("old", "p", [1.0, 0.0])], embedding_model="model-a")
    with pytest.raises(ValueError, match="model"):
        store.add([_row("unknown", "p", [1.0, 0.0])])
    with pytest.raises(ValueError, match="dimension"):
        store.add([_row("wide", "p", [1.0, 0.0, 0.0])], embedding_model="model-a")
    with pytest.raises(ValueError, match="dimension"):
        store.vector_search([1.0, 0.0, 0.0], 1, embedding_model="model-a")
    assert [row["id"] for row in store.all_rows()] == ["old"]


@pytest.mark.parametrize("metadata", [
    {b"vproc:embedding_model": b"model-a"},
    {b"vproc:embedding_dimension": b"2"},
    {b"vproc:embedding_model": b"", b"vproc:embedding_dimension": b"2"},
    {b"vproc:embedding_model": b" ", b"vproc:embedding_dimension": b"2"},
    {b"vproc:embedding_model": b"model-a", b"vproc:embedding_dimension": b"-2"},
    {b"vproc:embedding_model": b"model-a", b"vproc:embedding_dimension": b"3"},
])
def test_malformed_embedding_metadata_does_not_get_adopted(tmp_path, metadata):
    import pyarrow as pa

    store = Store(str(tmp_path / "db"))
    schema = pa.Table.from_pylist([_row("old", "p", [1.0, 0.0])]).schema
    i = schema.get_field_index("vector")
    schema = schema.set(i, pa.field("vector", pa.list_(pa.float32(), 2)))
    store.db.create_table(store.TABLE, [_row("old", "p", [1.0, 0.0])],
                          schema=schema.with_metadata(metadata))
    with pytest.raises(ValueError, match="metadata"):
        store.validate_embedding_model("model-a")
    assert [row["id"] for row in store.all_rows()] == ["old"]


def test_replace_memory_commits_one_scoped_row_change(tmp_path, monkeypatch):
    from lancedb.table import LanceTable

    # Isolate the row transaction from optional FTS index commits.
    monkeypatch.setattr(LanceTable, "create_fts_index", lambda *args, **kwargs: None)
    store = Store(str(tmp_path / "db"))
    store.add([_row("same", "owner's project", [1.0, 0.0], "old"),
               _row("stale", "owner's project", [1.0, 0.0]),
               _row("other-project", "other", [1.0, 0.0]),
               {**_row("other-memory", "owner's project", [1.0, 0.0]), "memory_id": "keep"}],
              embedding_model="model-a")
    before = store._table().version
    store.replace_memory("m", "owner's project", [
        _row("same", "owner's project", [0.0, 1.0], "updated"),
        _row("new", "owner's project", [0.0, 1.0]),
    ], embedding_model="model-a")
    assert store._table().version == before + 1
    rows = {row["id"]: row for row in store.all_rows()}
    assert set(rows) == {"same", "new", "other-project", "other-memory"}
    assert rows["same"]["said_text"] == "updated"
    store.validate_embedding_model("model-a", dimension=2)


def test_failed_lance_replacement_preserves_old_rows_and_version(tmp_path):
    store = Store(str(tmp_path / "db"))
    store.add([_row("old", "p", [1.0, 0.0])], embedding_model="model-a")
    before = store._table().version
    # This reaches Lance's real schema conversion; no mutation may precede it.
    bad = {**_row("new", "p", [1.0, 0.0]), "start_ts": "not a number"}
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        store.replace_memory("m", "p", [bad], embedding_model="model-a")
    reopened = Store(str(tmp_path / "db"))
    assert reopened._table().version == before
    assert [row["id"] for row in reopened.all_rows()] == ["old"]


@pytest.mark.parametrize("rows", [
    [],
    [_row("new", "wrong-project", [1.0, 0.0])],
    [{**_row("new", "p", [1.0, 0.0]), "memory_id": "wrong-memory"}],
    [_row("new", "p", [1.0, 0.0]), _row("new", "p", [1.0, 0.0])],
    [_row("", "p", [1.0, 0.0])],
    [_row("new", "p", [])],
    [_row("new", "p", [float("nan"), 0.0])],
    [_row("new", "p", [float("inf"), 0.0])],
])
def test_replace_memory_rejects_invalid_rows_without_touching_previous_data(tmp_path, rows):
    store = Store(str(tmp_path / "db"))
    store.add([_row("old", "p", [1.0, 0.0])], embedding_model="model-a")
    before = store._table().version
    with pytest.raises(ValueError):
        store.replace_memory("m", "p", rows, embedding_model="model-a")
    assert store._table().version == before
    assert [row["id"] for row in store.all_rows()] == ["old"]


def test_replace_memory_rejects_cross_scope_id_collision(tmp_path):
    store = Store(str(tmp_path / "db"))
    store.add([_row("old", "p", [1.0, 0.0]), _row("taken", "other", [1.0, 0.0])],
              embedding_model="model-a")
    before = store._table().version
    with pytest.raises(ValueError, match="id"):
        store.replace_memory("m", "p", [_row("taken", "p", [1.0, 0.0])], "model-a")
    assert store._table().version == before
    assert {row["id"] for row in store.all_rows()} == {"old", "taken"}


def test_replace_memory_on_new_store_records_identity(tmp_path):
    store = Store(str(tmp_path / "db"))
    store.replace_memory("m", "p", [_row("new", "p", [1.0, 0.0])], "model-a")
    store.validate_embedding_model("model-a", dimension=2)
    assert [row["id"] for row in store.all_rows()] == ["new"]


def test_fts_rebuild_failure_does_not_report_committed_replacement_as_failed(tmp_path, monkeypatch, caplog):
    from lancedb.table import LanceTable

    store = Store(str(tmp_path / "db"))
    store.add([_row("old", "p", [1.0, 0.0])], embedding_model="model-a")
    def fail_fts(*args, **kwargs):
        raise RuntimeError("could contain transcript content")
    monkeypatch.setattr(LanceTable, "create_fts_index", fail_fts)
    store.replace_memory("m", "p", [_row("new", "p", [1.0, 0.0])], "model-a")
    assert [row["id"] for row in store.all_rows()] == ["new"]
    assert "FTS" in caplog.text
    assert "could contain transcript content" not in caplog.text


@pytest.mark.parametrize("operation", ["add", "replace", "delete", "rename"])
def test_all_store_mutations_respect_another_writer(tmp_path, operation):
    from vproc.errors import IndexBusyError

    path = str(tmp_path / "db")
    owner, contender = Store(path), Store(path)
    owner.add([_row("old", "p", [1.0, 0.0])], embedding_model="model-a")
    actions = {
        "add": lambda: contender.add([_row("new", "p", [1.0, 0.0])], embedding_model="model-a"),
        "replace": lambda: contender.replace_memory("m", "p", [_row("new", "p", [1.0, 0.0])], "model-a"),
        "delete": lambda: contender.delete_memory("m", "p"),
        "rename": lambda: contender.update_speaker("m", "p", "SPEAKER_0", "Renamed"),
    }
    before = owner._table().version
    with owner.write_lock():
        with pytest.raises(IndexBusyError):
            actions[operation]()
        # Readers remain usable while a writer stages replacement data.
        assert [row["id"] for row in contender.memory_rows("m", "p")] == ["old"]
    assert owner._table().version == before
    assert owner.memory_rows("m", "p")[0]["speaker"] == "SPEAKER_0"


def test_replace_memory_rejects_duplicate_existing_ids(tmp_path):
    store = Store(str(tmp_path / "db"))
    store.add([_row("same", "p", [1.0, 0.0]), _row("same", "p", [1.0, 0.0])],
              embedding_model="model-a")
    before = store._table().version
    with pytest.raises(ValueError, match="id"):
        store.replace_memory("m", "p", [_row("same", "p", [0.0, 1.0])], "model-a")
    assert store._table().version == before
    assert len(store.all_rows()) == 2


def test_embedding_identity_rejects_nonfloating_vector_storage(tmp_path):
    import pyarrow as pa

    store = Store(str(tmp_path / "db"))
    row = _row("old", "p", [1, 0])
    schema = pa.Table.from_pylist([row]).schema
    schema = schema.set(schema.get_field_index("vector"), pa.field("vector", pa.list_(pa.int32(), 2)))
    schema = schema.with_metadata({b"vproc:embedding_model": b"model-a",
                                   b"vproc:embedding_dimension": b"2"})
    store.db.create_table(store.TABLE, [row], schema=schema)
    with pytest.raises(ValueError, match="metadata"):
        store.validate_embedding_model("model-a")


def test_fts_search_failure_warns_without_logging_query_or_transcript(tmp_path, monkeypatch, caplog):
    from lancedb.table import LanceTable

    store = Store(str(tmp_path / "db"))
    store.add([_row("old", "p", [1.0, 0.0])], embedding_model="model-a")
    original = LanceTable.search
    def fail_fts(self, *args, **kwargs):
        if kwargs.get("query_type") == "fts":
            raise RuntimeError("private transcript text")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(LanceTable, "search", fail_fts)
    assert store.fts_search("private query text", 1) == []
    assert store.vector_search([1.0, 0.0], 1, embedding_model="model-a")[0]["id"] == "old"
    assert "FTS" in caplog.text
    assert "private query text" not in caplog.text
    assert "private transcript text" not in caplog.text
