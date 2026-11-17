import re

from vproc.llm import client


# Scaffolding that ingest adds to indexed text ("[SCREEN]", "[SPOKEN]", "LABEL (mm:ss):"). It is
# not meeting content, so it must not count as a literal match. The speaker label stays: it is.
_SCAFFOLD = re.compile(r"^\[(?:SCREEN|SPOKEN)\]$|\(\d{2,}:\d{2}\):", re.M)


def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, sid in enumerate(ranking):
            scores[sid] = scores.get(sid, 0.0) + 1.0 / (k + rank + 1)
    return scores


def retrieve(store, cfg, query: str, k: int = 8, where: str | None = None,
             embed=client.embed_texts):
    store.validate_embedding_model(cfg.embed.model)
    # The prefix is for the embedder only; FTS and the keyword rule below use the raw query.
    vectors = embed(cfg.embed.base_url, cfg.embed.model, [cfg.embed_query_prefix + query])
    if len(vectors) != 1:
        raise ValueError(f"Expected one query embedding, received {len(vectors)}")
    qvec = vectors[0]
    vhits = store.vector_search(qvec, k, where, embedding_model=cfg.embed.model)
    fhits = store.fts_search(query, k, where)
    if not vhits and not fhits:
        return [], 0.0
    # LanceDB 0.33 FTS defaults to operator=OR with no stop-word removal, so almost any
    # question shares a token with some chunk. Only lift the floor on a true keyword match
    # (one hit's text contains ALL query tokens) so an exact match like 'ERR-4127' survives
    # ask_memory's top_sim < sim_floor gate without generic questions bypassing it.
    qtokens = set(re.findall(r"\w+", query.lower()))

    def keyword_match(hit: dict) -> bool:
        content = _SCAFFOLD.sub(" ", hit.get("embed_text", ""))
        return bool(qtokens) and qtokens <= set(re.findall(r"\w+", content.lower()))

    similarity = {h["id"]: 1.0 - float(h["_distance"]) for h in vhits}
    by_id = {h["id"]: h for h in (vhits + fhits)}
    for sid, hit in by_id.items():
        # Search returns the k nearest vectors whatever the query, so expose how near each hit
        # is and whether it literally contains the query. Cosine values are model-specific.
        hit["_similarity"] = similarity.get(sid)
        hit["_keyword_match"] = keyword_match(hit)
    ranked = sorted(
        rrf([[h["id"] for h in vhits], [h["id"] for h in fhits]]).items(),
        key=lambda kv: kv[1], reverse=True,
    )
    merged = [by_id[sid] for sid, _ in ranked if sid in by_id][:k]
    # Only evidence actually passed to generation may open the relevance gate.
    # Fusion can discard a literal FTS match in favor of hits shared by both legs.
    top_sim = max((h["_similarity"] for h in merged if h["_similarity"] is not None), default=0.0)
    if any(h["_keyword_match"] for h in merged):
        top_sim = max(top_sim, cfg.sim_floor)
    return merged, top_sim
