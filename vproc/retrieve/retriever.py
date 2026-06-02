from vproc.llm import client


def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, sid in enumerate(ranking):
            scores[sid] = scores.get(sid, 0.0) + 1.0 / (k + rank + 1)
    return scores


def retrieve(store, cfg, query: str, k: int = 8, where: str | None = None,
             embed=client.embed_texts):
    qvec = embed(cfg.embed.base_url, cfg.embed.model, [query])[0]
    vhits = store.vector_search(qvec, k, where)
    fhits = store.fts_search(query, k, where)
    if not vhits and not fhits:
        return [], 0.0
    top_sim = (1.0 - float(vhits[0]["_distance"])) if vhits else 0.0
    by_id = {h["id"]: h for h in (vhits + fhits)}
    ranked = sorted(
        rrf([[h["id"] for h in vhits], [h["id"] for h in fhits]]).items(),
        key=lambda kv: kv[1], reverse=True,
    )
    merged = [by_id[sid] for sid, _ in ranked if sid in by_id][:k]
    return merged, top_sim
