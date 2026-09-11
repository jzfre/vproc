import re

from vproc.llm import client


def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, sid in enumerate(ranking):
            scores[sid] = scores.get(sid, 0.0) + 1.0 / (k + rank + 1)
    return scores


def retrieve(store, cfg, query: str, k: int = 8, where: str | None = None,
             embed=client.embed_texts):
    store.validate_embedding_model(cfg.embed.model)
    vectors = embed(cfg.embed.base_url, cfg.embed.model, [query])
    if len(vectors) != 1:
        raise ValueError(f"Expected one query embedding, received {len(vectors)}")
    qvec = vectors[0]
    vhits = store.vector_search(qvec, k, where, embedding_model=cfg.embed.model)
    fhits = store.fts_search(query, k, where)
    if not vhits and not fhits:
        return [], 0.0
    top_sim = (1.0 - float(vhits[0]["_distance"])) if vhits else 0.0
    # LanceDB 0.33 FTS defaults to operator=OR with no stop-word removal, so almost any
    # question shares a token with some chunk. Only lift the floor on a true keyword match
    # (one hit's text contains ALL query tokens) so an exact match like 'ERR-4127' survives
    # ask_memory's top_sim < sim_floor gate without generic questions bypassing it.
    qtokens = set(re.findall(r"\w+", query.lower()))
    if qtokens and any(qtokens <= set(re.findall(r"\w+", h.get("embed_text", "").lower()))
                       for h in fhits):
        top_sim = max(top_sim, cfg.sim_floor)
    by_id = {h["id"]: h for h in (vhits + fhits)}
    ranked = sorted(
        rrf([[h["id"] for h in vhits], [h["id"] for h in fhits]]).items(),
        key=lambda kv: kv[1], reverse=True,
    )
    merged = [by_id[sid] for sid, _ in ranked if sid in by_id][:k]
    return merged, top_sim
