from vproc.models import Evidence


def _text(hit: dict) -> str:
    said = hit.get("said_text", "") or ""
    screen = (hit.get("on_screen_text") or "").strip()
    return said + (f"\n[screen] {screen}" if screen else "")


def build_evidence(hits: list[dict]) -> list[Evidence]:
    out: list[Evidence] = []
    for i, h in enumerate(hits, start=1):
        out.append(Evidence(
            key=f"E{i}", segment_id=h["id"], memory_title=h.get("memory_id", ""),
            start_ts=float(h["start_ts"]), end_ts=float(h["end_ts"]),
            speaker=h.get("speaker", "SPEAKER_0"), text=_text(h),
        ))
    return out


def evidence_block(evidence: list[Evidence]) -> str:
    return "\n".join(f'[{e.key}] "{e.text}"' for e in evidence)
