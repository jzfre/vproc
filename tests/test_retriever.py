from vproc.config import Config, Endpoint
from vproc.store.lancedb_store import Store
from vproc.retrieve.retriever import retrieve, rrf
import pytest

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
               _row("far", [0.0, 1.0], "hiring budget")], embedding_model="m")
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
               _row("other", [1.0, 0.1], "unrelated budget talk")], embedding_model="m")
    # Query vector is near-orthogonal to every stored vector (vector sim < sim_floor),
    # but the token matches exactly via FTS: the fts leg must lift top_sim to sim_floor
    # so ask_memory's abstention gate cannot throw away the exact match.
    hits, top_sim = retrieve(store, _cfg(), "rollout", k=2,
                             embed=lambda *a: [[0.1, 1.0]])
    assert top_sim >= _cfg().sim_floor
    assert any(h["id"] == "code" for h in hits)

def test_retrieve_generic_question_incidental_fts_overlap_does_not_lift_floor(tmp_path):
    store = Store(str(tmp_path / "db.lance"))
    store.add([_row("chunk", [1.0, 0.0], "budget meeting on tuesday")], embedding_model="m")
    # Generic question that only incidentally shares the single token 'budget' via FTS's
    # default OR match — no hit contains ALL query tokens, so the abstention floor must
    # NOT be lifted (top_sim stays at the weak vector similarity, below sim_floor).
    assert store.fts_search("budget forecast next quarter", 2)  # OR overlap does produce an fts hit
    hits, top_sim = retrieve(store, _cfg(), "budget forecast next quarter", k=2,
                             embed=lambda *a: [[0.1, 1.0]])
    assert top_sim < _cfg().sim_floor


def test_retrieve_different_short_error_code_does_not_lift_floor(tmp_path):
    store = Store(str(tmp_path / "db.lance"))
    store.add([_row("chunk", [1.0, 0.0], "Deployment failed with ERR-99")], embedding_model="m")
    assert store.fts_search("ERR-12", 2)
    hits, top_sim = retrieve(store, _cfg(), "ERR-12", k=2, embed=lambda *a: [[0.0, 1.0]])
    assert hits and top_sim < _cfg().sim_floor


def test_retrieve_substring_overlap_does_not_lift_floor(tmp_path):
    store = Store(str(tmp_path / "db.lance"))
    store.add([_row("chunk", [1.0, 0.0], "discard engine prototype")], embedding_model="m")
    assert store.fts_search("car engine", 2)
    hits, top_sim = retrieve(store, _cfg(), "car engine", k=2, embed=lambda *a: [[0.0, 1.0]])
    assert hits and top_sim < _cfg().sim_floor


@pytest.mark.parametrize("legacy", [False, True])
def test_retrieve_rejects_incompatible_index_before_embedding(tmp_path, legacy):
    store = Store(str(tmp_path / "db"))
    store.add([_row("old", [1.0, 0.0], "budget")],
              embedding_model=None if legacy else "other-model")
    def must_not_embed(*args):
        raise AssertionError("Incompatible indexes must fail before calling the embedding model")
    with pytest.raises(ValueError, match="VPROC_INDEX_PATH"):
        retrieve(store, _cfg(), "budget", embed=must_not_embed)


def test_retrieve_rejects_changed_query_embedding_dimension(tmp_path):
    store = Store(str(tmp_path / "db"))
    store.add([_row("old", [1.0, 0.0], "budget")], embedding_model="m")
    with pytest.raises(ValueError, match="dimension"):
        retrieve(store, _cfg(), "budget", embed=lambda *args: [[1.0, 0.0, 0.0]])


@pytest.mark.parametrize("vectors", [[], [[1.0, 0.0], [1.0, 0.0]]])
def test_retrieve_rejects_missing_or_extra_query_embeddings(tmp_path, vectors):
    store = Store(str(tmp_path / "db"))
    with pytest.raises(ValueError, match="embedding"):
        retrieve(store, _cfg(), "budget", embed=lambda *args: vectors)


def test_query_prefix_reaches_the_embedder_but_not_keyword_search(tmp_path):
    # Instruction-tuned embedders (Qwen3-Embedding: "Instruct: ...\nQuery: ") want a prefix on
    # queries only. Stored passages and the literal FTS/keyword leg keep the raw text.
    from dataclasses import replace

    store = Store(str(tmp_path / "db.lance"))
    store.add([_row("code", [0.0, 1.0], "error ERR-4127 on rollout")], embedding_model="m")
    seen = []

    def fake_embed(base_url, model, texts):
        seen.extend(texts)
        return [[1.0, 0.0]]  # orthogonal: only the keyword leg can lift the floor

    cfg = replace(_cfg(), embed_query_prefix="Instruct: find passages\nQuery: ")
    hits, top_sim = retrieve(store, cfg, "ERR-4127", k=2, embed=fake_embed)
    assert seen == ["Instruct: find passages\nQuery: ERR-4127"]
    assert hits[0]["id"] == "code" and top_sim == cfg.sim_floor


def test_hits_carry_similarity_and_true_keyword_match(tmp_path):
    # Search always returns the k nearest vectors, so callers need to tell "about this" from
    # "merely nearest". keyword_match means EVERY query token is in the hit (FTS alone is OR).
    store = Store(str(tmp_path / "db.lance"))
    store.add([_row("both", [1.0, 0.0], "error ERR-4127 on rollout"),
               _row("vector_only", [0.9, 0.1], "deployment failed yesterday"),
               _row("partial", [0.0, 1.0], "rollout budget")], embedding_model="m")
    hits, _ = retrieve(store, _cfg(), "ERR-4127 rollout", k=3, embed=lambda *a: [[1.0, 0.0]])
    by_id = {h["id"]: h for h in hits}
    assert by_id["both"]["_keyword_match"] is True
    assert by_id["both"]["_similarity"] == pytest.approx(1.0, abs=1e-5)
    assert by_id["vector_only"]["_keyword_match"] is False
    assert 0.9 < by_id["vector_only"]["_similarity"] < 1.0
    assert by_id["partial"]["_keyword_match"] is False  # shares one token only


def test_keyword_match_ignores_the_structural_markers_of_indexed_text(tmp_path):
    # Review finding: indexed text carries "[SCREEN]", "[SPOKEN]" and "LABEL (mm:ss):" scaffolding,
    # so the query "screen" was an "exact" match for every chunk and lifted the similarity floor.
    from vproc.ingest.align import make_embed_text
    from vproc.ingest.transcribe import TranscriptSegment

    text = make_embed_text("", [TranscriptSegment(615.0, 620.0, "we talked about the hiring budget",
                                                  speaker="SPEAKER_01")])
    store = Store(str(tmp_path / "db.lance"))
    store.add([_row("chunk", [0.0, 1.0], text)], embedding_model="m")
    orthogonal = lambda *a: [[1.0, 0.0]]  # noqa: E731
    for query in ("screen", "spoken", "10", "15"):
        hits, top_sim = retrieve(store, _cfg(), query, k=2, embed=orthogonal)
        assert not any(h["_keyword_match"] for h in hits), query
        assert top_sim < _cfg().sim_floor, query          # no floor lift from scaffolding
    hits, top_sim = retrieve(store, _cfg(), "hiring budget", k=2, embed=orthogonal)
    assert hits[0]["_keyword_match"] is True and top_sim == _cfg().sim_floor
    hits, _ = retrieve(store, _cfg(), "SPEAKER_01 budget", k=2, embed=orthogonal)
    assert hits[0]["_keyword_match"] is True              # the speaker label is real content
