import vproc.ingest.pipeline as P
from vproc.config import Config, Endpoint
from vproc.store.lancedb_store import Store
from vproc.ingest.frames import RawFrame
from vproc.ingest.transcribe import TranscriptSegment

def _cfg():
    ep = Endpoint("u", "m")
    return Config(ep, ep, ep, "x", "0.0.0.0", 8765, 0.25, 0.5, None)

def _patch(monkeypatch):
    monkeypatch.setattr(P.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(P, "_list_frames", lambda d: ["/f/0.png", "/f/1.png"])
    monkeypatch.setattr(P.F, "parse_frames_log",
                        lambda text, paths: [RawFrame("/f/0.png", 0.0), RawFrame("/f/1.png", 10.0)])
    monkeypatch.setattr(P.F, "phash_dedup", lambda frames, **k: frames)
    monkeypatch.setattr(P, "_read_log", lambda p: "")
    monkeypatch.setattr(P.T, "transcribe",
                        lambda wav: [TranscriptSegment(1.0, 3.0, "we ship in July")])
    monkeypatch.setattr(P.O, "ocr_frame", lambda ocr, path: "Roadmap Q3")
    monkeypatch.setattr(P.EI, "embed_and_store",
                        lambda cfg, store, segs, **k: store.add(
                            [{"id": s.id, "project_id": s.project_id, "memory_id": s.memory_id,
                              "speaker": s.speaker, "start_ts": s.start_ts, "end_ts": s.end_ts,
                              "said_text": s.said_text, "on_screen_text": s.on_screen_text,
                              "embed_text": s.embed_text, "source_video": s.source_video,
                              "frame_path": s.frame_path or "", "vector": [1.0, 0.0]} for s in segs]))

def test_ingest_video_builds_segments(tmp_path, monkeypatch):
    _patch(monkeypatch)
    store = Store(str(tmp_path / "db.lance"))
    n = P.ingest_video("/videos/standup.mp4", cfg=_cfg(), store=store)
    assert n == 2  # two screen states, both have content
    hits = store.vector_search([1.0, 0.0], k=5)
    assert any("we ship in July" in h["said_text"] for h in hits)
    assert all(h["memory_id"] == "standup" for h in hits)

def test_reingest_replaces_instead_of_duplicating(tmp_path, monkeypatch):
    _patch(monkeypatch)
    store = Store(str(tmp_path / "db.lance"))
    P.ingest_video("/videos/standup.mp4", cfg=_cfg(), store=store)
    P.ingest_video("/videos/standup.mp4", cfg=_cfg(), store=store)  # same memory again
    hits = store.vector_search([1.0, 0.0], k=50)
    assert len(hits) == 2  # prior rows replaced, not appended (would be 4)
