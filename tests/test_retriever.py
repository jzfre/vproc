from vproc.config import Config, Endpoint
from vproc.store.lancedb_store import Store
from vproc.retrieve.retriever import retrieve, rrf

def _cfg():
    ep = Endpoint("u", "m")
    return Config(ep, ep, ep, "x", "0.0.0.0", 8765, 0.25, 0.5, None)

def _row(id, vec, text):
    return {"id": id, "project_id": "p", "memory_id": "m", "speaker": "SPEAKER_0",
            "start_ts": 0.0, "end_ts": 1.0, "said_text": text, "on_screen_text": "",
            "embed_text": text, "source_video": "/v.mp4", "frame_path": "", "vector": vec}

def test_rrf_merges_rankings():
    scores = rrf([["a", "b"], ["b", "c"]])
    assert scores["b"] > scores["a"]  # b appears in both lists

def test_retrieve_returns_hits_and_top_sim(tmp_path):
    store = Store(str(tmp_path / "db.lance"))
    store.add([_row("near", [1.0, 0.0], "petrol engine cycle"),
               _row("far", [0.0, 1.0], "hiring budget")])
    def fake_embed(base_url, model, texts):
        return [[1.0, 0.0]]  # query embeds near "near"
    hits, top_sim = retrieve(store, _cfg(), "engine", k=2, embed=fake_embed)
    assert hits[0]["id"] == "near"
    assert top_sim > 0.9  # cosine sim of identical direction ≈ 1

def test_retrieve_empty_store(tmp_path):
    store = Store(str(tmp_path / "db.lance"))
    hits, top_sim = retrieve(store, _cfg(), "anything", embed=lambda *a: [[1.0, 0.0]])
    assert hits == [] and top_sim == 0.0

def test_retrieve_floors_top_sim_on_fts_hit(tmp_path):
    store = Store(str(tmp_path / "db.lance"))
    store.add([_row("code", [1.0, 0.0], "quarterly rollout blocked"),
               _row("other", [1.0, 0.1], "unrelated budget talk")])
    # Query vector is near-orthogonal to every stored vector (vector sim < sim_floor),
    # but the token matches exactly via FTS: the fts leg must lift top_sim to sim_floor
    # so ask_memory's abstention gate cannot throw away the exact match.
    hits, top_sim = retrieve(store, _cfg(), "rollout", k=2,
                             embed=lambda *a: [[0.1, 1.0]])
    assert top_sim >= _cfg().sim_floor
    assert any(h["id"] == "code" for h in hits)

def test_retrieve_generic_question_incidental_fts_overlap_does_not_lift_floor(tmp_path):
    store = Store(str(tmp_path / "db.lance"))
    store.add([_row("chunk", [1.0, 0.0], "budget meeting on tuesday")])
    # Generic question that only incidentally shares the single token 'budget' via FTS's
    # default OR match — no hit contains ALL query tokens, so the abstention floor must
    # NOT be lifted (top_sim stays at the weak vector similarity, below sim_floor).
    assert store.fts_search("budget forecast next quarter", 2)  # OR overlap does produce an fts hit
    hits, top_sim = retrieve(store, _cfg(), "budget forecast next quarter", k=2,
                             embed=lambda *a: [[0.1, 1.0]])
    assert top_sim < _cfg().sim_floor
