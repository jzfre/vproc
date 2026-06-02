from vproc.answer import generate as gen
from vproc.answer.evidence import build_evidence, evidence_block
from vproc.answer.faithfulness import filter_claims
from vproc.models import Answer, Citation, Claim, Evidence, mmss
from vproc.retrieve.retriever import retrieve

ABSTAIN = "Not discussed in these meetings."


def _abstain() -> Answer:
    return Answer(answered=False, abstained=True, text=ABSTAIN, claims=[], evidence=[])


def _cite_tag(c: Citation) -> str:
    return f"[{c.memory_title} · {mmss(c.start_ts)} · {c.speaker}]"


def ask_memory(store, cfg, question: str, scorer, where: str | None = None, k: int = 8,
               _retrieve=retrieve, _generate=gen.generate) -> Answer:
    hits, top_sim = _retrieve(store, cfg, question, k=k, where=where)
    if not hits or top_sim < cfg.sim_floor:
        return _abstain()

    evidence = build_evidence(hits)
    by_key: dict[str, Evidence] = {e.key: e for e in evidence}
    result = _generate(cfg, question, evidence_block(evidence))
    if not result["answered"]:
        return _abstain()

    kept = filter_claims(result["claims"], by_key, scorer, cfg.hhem_threshold)
    if not kept:
        return _abstain()

    out_claims: list[Claim] = []
    lines: list[str] = []
    for c in kept:
        cites = [
            Citation(memory_title=by_key[k].memory_title, start_ts=by_key[k].start_ts,
                     end_ts=by_key[k].end_ts, speaker=by_key[k].speaker)
            for k in c["evidence_ids"] if k in by_key
        ]
        out_claims.append(Claim(text=c["text"], evidence_ids=c["evidence_ids"], citations=cites))
        tags = " ".join(_cite_tag(cit) for cit in cites)
        lines.append(f"{c['text']} {tags}".strip())

    return Answer(answered=True, abstained=False, text="\n".join(lines),
                  claims=out_claims, evidence=evidence)


def search_memory(store, cfg, query: str, where: str | None = None, k: int = 8,
                  _retrieve=retrieve) -> list[Evidence]:
    hits, _ = _retrieve(store, cfg, query, k=k, where=where)
    return build_evidence(hits)
