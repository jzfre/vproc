from vproc.config import Config, Endpoint
from vproc.models import Segment
from vproc.store.lancedb_store import Store
from vproc.ingest.embed_index import embed_and_store, embed_rows
import pytest

def _cfg():
    ep = Endpoint("u", "m")
    return Config(ep, ep, ep, "x", "0.0.0.0", 8765, 0.25, 0.5, None)

def _seg(i):
    return Segment(id=f"s{i}", project_id="p", memory_id="m", start_ts=0.0, end_ts=1.0,
                   speaker="SPEAKER_0", said_text="t", source_video="/v.mp4", embed_text=f"text {i}")

def test_embed_and_store_writes_rows(tmp_path):
    store = Store(str(tmp_path / "db.lance"))
    def fake_embed(base_url, model, texts):
        return [[float(len(t)), 1.0] for t in texts]
    embed_and_store(_cfg(), store, [_seg(1), _seg(2)], embed=fake_embed)
    hits = store.vector_search([6.0, 1.0], k=2)
    assert {h["id"] for h in hits} == {"s1", "s2"}

def test_embed_and_store_noop_when_empty(tmp_path):
    store = Store(str(tmp_path / "db.lance"))
    embed_and_store(_cfg(), store, [], embed=lambda *a: [])  # must not raise
    assert store.vector_search([0.0, 0.0], k=1) == []

def test_embed_rows_returns_rows_without_storing():
    calls = {"n": 0}
    def fake_embed(base_url, model, texts):
        calls["n"] += 1
        return [[float(len(t)), 1.0] for t in texts]
    rows = embed_rows(_cfg(), [_seg(1), _seg(2)], embed=fake_embed)
    assert [r["id"] for r in rows] == ["s1", "s2"]
    assert all("vector" in r for r in rows)
    assert calls["n"] == 1

def test_embed_rows_empty_does_not_call_embed():
    called = {"n": 0}
    assert embed_rows(_cfg(), [], embed=lambda *a: called.__setitem__("n", called["n"] + 1)) == []
    assert called["n"] == 0


@pytest.mark.parametrize("vectors", [[], [[1.0, 0.0]], [[1.0, 0.0]] * 3])
def test_embed_rows_rejects_missing_or_extra_embeddings(vectors):
    with pytest.raises(ValueError, match="embedding"):
        embed_rows(_cfg(), [_seg(1), _seg(2)], embed=lambda *args: vectors)


def test_embed_and_store_records_configured_embedding_model(tmp_path):
    store = Store(str(tmp_path / "db"))
    embed_and_store(_cfg(), store, [_seg(1)], embed=lambda *args: [[1.0, 0.0]])
    store.validate_embedding_model("m", dimension=2)
    with pytest.raises(ValueError, match="model"):
        store.validate_embedding_model("different-model")


@pytest.mark.parametrize("legacy", [False, True])
def test_embed_and_store_rejects_incompatible_index_before_embedding(tmp_path, legacy):
    store = Store(str(tmp_path / "db"))
    rows = embed_rows(_cfg(), [_seg(1)], embed=lambda *args: [[1.0, 0.0]])
    store.add(rows, embedding_model=None if legacy else "different-model")
    def must_not_embed(*args):
        raise AssertionError("Incompatible indexes must fail before calling the embedding model")
    with pytest.raises(ValueError, match="VPROC_INDEX_PATH"):
        embed_and_store(_cfg(), store, [_seg(2)], embed=must_not_embed)
    assert [row["id"] for row in store.all_rows()] == ["s1"]


def test_embed_and_store_rejects_dimension_change_without_adding_rows(tmp_path):
    store = Store(str(tmp_path / "db"))
    embed_and_store(_cfg(), store, [_seg(1)], embed=lambda *args: [[1.0, 0.0]])
    with pytest.raises(ValueError, match="dimension"):
        embed_and_store(_cfg(), store, [_seg(2)], embed=lambda *args: [[1.0, 0.0, 0.0]])
    assert [row["id"] for row in store.all_rows()] == ["s1"]
