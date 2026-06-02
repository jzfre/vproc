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


def embed_and_store(cfg, store, segments: list[Segment], embed=client.embed_texts) -> None:
    if not segments:
        return
    vectors = embed(cfg.embed.base_url, cfg.embed.model, [s.embed_text for s in segments])
    store.add([_row(seg, vec) for seg, vec in zip(segments, vectors)])
