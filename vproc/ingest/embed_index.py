from vproc.llm import client
from vproc.models import Segment


def _row(seg: Segment, vector: list[float]) -> dict:
    return {
        "id": seg.id, "project_id": seg.project_id, "memory_id": seg.memory_id,
        "speaker": seg.speaker, "start_ts": seg.start_ts, "end_ts": seg.end_ts,
        "said_text": seg.said_text, "on_screen_text": seg.on_screen_text,
        "embed_text": seg.embed_text, "source_video": seg.source_video,
        "frame_path": seg.frame_path or "", "vector": vector,
    }


def embed_rows(cfg, segments: list[Segment], embed=client.embed_texts) -> list[dict]:
    """Embed the segments and build store rows WITHOUT writing. Kept separate from the
    store write so ingest can embed before deleting prior rows (a failed embed must not
    leave the memory wiped)."""
    if not segments:
        return []
    vectors = embed(cfg.embed.base_url, cfg.embed.model, [s.embed_text for s in segments])
    return [_row(seg, vec) for seg, vec in zip(segments, vectors)]


def embed_and_store(cfg, store, segments: list[Segment], embed=client.embed_texts) -> None:
    store.add(embed_rows(cfg, segments, embed))
