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


def preflight(cfg, store, embed=client.embed_texts) -> None:
    """Prove the embedding backend works and matches the index BEFORE minutes of ASR and
    diarization: it is the one ingest dependency that otherwise fails last."""
    vectors = embed(cfg.embed.base_url, cfg.embed.model, ["vproc ingest preflight"])
    if len(vectors) != 1:
        raise ValueError(f"Expected one preflight embedding, received {len(vectors)}")
    store.validate_embedding_model(cfg.embed.model, len(vectors[0]))


def embed_rows(cfg, segments: list[Segment], embed=client.embed_texts) -> list[dict]:
    """Build rows without writing, so failed embedding leaves prior index data intact."""
    if not segments:
        return []
    vectors = embed(cfg.embed.base_url, cfg.embed.model, [s.embed_text for s in segments])
    if len(vectors) != len(segments):
        raise ValueError(f"Expected {len(segments)} embeddings, received {len(vectors)}")
    return [_row(seg, vec) for seg, vec in zip(segments, vectors)]


def embed_and_store(cfg, store, segments: list[Segment], embed=client.embed_texts) -> None:
    if not segments:
        return
    with store.write_lock():
        store.validate_embedding_model(cfg.embed.model)
        store.add(embed_rows(cfg, segments, embed), embedding_model=cfg.embed.model)
