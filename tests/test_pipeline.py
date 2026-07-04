import os
import types

import pytest

import vproc.ingest.pipeline as P
from vproc.config import Endpoint
from vproc.ingest.frames import RawFrame
from vproc.ingest.transcribe import TranscriptSegment


class FakeStore:
    """Records add/delete instead of hitting LanceDB, so these tests don't depend on the
    store's (concurrently-changing) delete_memory signature."""
    def __init__(self):
        self.rows = []
        self.deletes = []
        self.calls = []  # ordered log of add/delete so tests can assert the tail ordering

    def add(self, rows):
        self.calls.append("add")
        self.rows.extend(rows)

    def delete_memory(self, memory_id, project_id=None, keep_ids=None):
        self.calls.append("delete")
        self.deletes.append((memory_id, project_id, keep_ids))
        keep = set(keep_ids or [])
        self.rows = [r for r in self.rows
                     if r["memory_id"] != memory_id or r["id"] in keep]


def _cfg(tmp_path):
    ep = Endpoint("u", "m")
    return types.SimpleNamespace(
        ocr=ep, embed=ep, transcribe=ep,
        index_path=str(tmp_path / "db.lance"),
        frames_dir=str(tmp_path / "frames"),
    )


def _src_frame(tmp_path, name):
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    p = src / name
    p.write_bytes(b"png")
    return str(p)


def _row_of(s):
    return {"id": s.id, "project_id": s.project_id, "memory_id": s.memory_id,
            "speaker": s.speaker, "start_ts": s.start_ts, "end_ts": s.end_ts,
            "said_text": s.said_text, "on_screen_text": s.on_screen_text,
            "embed_text": s.embed_text, "source_video": s.source_video,
            "frame_path": s.frame_path or "", "vector": [1.0, 0.0]}


def _patch(monkeypatch, tmp_path, frames=None, transcript=None, has_audio=True):
    frames = frames if frames is not None else [
        RawFrame(_src_frame(tmp_path, "00000001.png"), 0.0),
        RawFrame(_src_frame(tmp_path, "00000002.png"), 10.0),
    ]
    ts = transcript if transcript is not None else [TranscriptSegment(1.0, 3.0, "we ship in July")]
    monkeypatch.setattr(P.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(P, "_has_audio_stream", lambda v: has_audio)
    monkeypatch.setattr(P, "_list_frames", lambda d: [f.path for f in frames])
    monkeypatch.setattr(P.F, "parse_frames_log", lambda text, paths: list(frames))
    monkeypatch.setattr(P.F, "phash_dedup", lambda fr, **k: list(fr))
    monkeypatch.setattr(P, "_read_log", lambda p: "")
    monkeypatch.setattr(P.T, "transcribe", lambda wav, ep=None: list(ts))
    monkeypatch.setattr(P.O, "ocr_frame", lambda ocr, path: "Roadmap Q3")
    monkeypatch.setattr(P.EI, "embed_rows", lambda cfg, segs, **k: [_row_of(s) for s in segs])


def test_ingest_video_builds_segments(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    store = FakeStore()
    n = P.ingest_video("/videos/standup.mp4", cfg=_cfg(tmp_path), store=store)
    assert n == 2  # two screen states, both have content
    assert any("we ship in July" in r["said_text"] for r in store.rows)
    assert all(r["memory_id"] == "standup" for r in store.rows)


def test_reingest_replaces_instead_of_duplicating(tmp_path, monkeypatch):
    store = FakeStore()
    cfg = _cfg(tmp_path)
    _patch(monkeypatch, tmp_path)
    P.ingest_video("/videos/standup.mp4", cfg=cfg, store=store)
    _patch(monkeypatch, tmp_path)  # recreate source frames for the second pass
    P.ingest_video("/videos/standup.mp4", cfg=cfg, store=store)  # same memory again
    assert len(store.rows) == 2  # prior rows replaced, not appended (would be 4)


def test_embed_happens_before_delete(tmp_path, monkeypatch):
    # A failed embed must NOT delete prior rows (embed-before-delete ordering).
    _patch(monkeypatch, tmp_path)
    store = FakeStore()
    store.rows = [{"memory_id": "standup", "id": "old"}]

    def boom(cfg, segs, **k):
        raise RuntimeError("embed endpoint down")
    monkeypatch.setattr(P.EI, "embed_rows", boom)
    try:
        P.ingest_video("/videos/standup.mp4", cfg=_cfg(tmp_path), store=store)
    except RuntimeError:
        pass
    assert store.deletes == []  # never reached the delete
    assert store.rows == [{"memory_id": "standup", "id": "old"}]  # prior rows intact


def test_zero_frames_still_indexes_transcript(tmp_path, monkeypatch):
    # Static/webcam recording: no scene-change frames, but the transcript must survive.
    _patch(monkeypatch, tmp_path)
    monkeypatch.setattr(P.F, "phash_dedup", lambda fr, **k: [])
    store = FakeStore()
    n = P.ingest_video("/videos/webcam.mp4", cfg=_cfg(tmp_path), store=store)
    assert n >= 1
    assert any("we ship in July" in r["said_text"] for r in store.rows)


def test_trailing_frame_after_speech_not_inverted(tmp_path, monkeypatch):
    # Closing slide at t=150 after speech ends at t=100: the OCR-only state must not invert.
    frames = [RawFrame(_src_frame(tmp_path, "a.png"), 0.0),
              RawFrame(_src_frame(tmp_path, "b.png"), 150.0)]
    _patch(monkeypatch, tmp_path, frames=frames,
           transcript=[TranscriptSegment(50.0, 100.0, "closing remarks")])
    store = FakeStore()
    P.ingest_video("/videos/talk.mp4", cfg=_cfg(tmp_path), store=store)
    for r in store.rows:
        assert r["end_ts"] >= r["start_ts"], (r["start_ts"], r["end_ts"])
    assert any(r["start_ts"] == 150.0 for r in store.rows)  # closing-slide segment present


def test_no_audio_stream_skips_transcription(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path, has_audio=False)

    def boom(*a, **k):
        raise AssertionError("must not transcribe a video with no audio stream")
    monkeypatch.setattr(P.T, "transcribe", boom)
    store = FakeStore()
    n = P.ingest_video("/videos/silent.mp4", cfg=_cfg(tmp_path), store=store)
    assert n == 2  # OCR-only segments for both frames
    assert all(r["said_text"] == "" for r in store.rows)


def test_frames_relocated_and_workdir_cleaned(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    created = {}
    real_mkdtemp = P.tempfile.mkdtemp

    def spy(*a, **k):
        d = real_mkdtemp(*a, **k)
        created["workdir"] = d
        return d
    monkeypatch.setattr(P.tempfile, "mkdtemp", spy)
    store = FakeStore()
    cfg = _cfg(tmp_path)
    P.ingest_video("/videos/standup.mp4", cfg=cfg, store=store)
    assert not os.path.exists(created["workdir"])  # temp workdir removed
    fps = [r["frame_path"] for r in store.rows if r["frame_path"]]
    assert fps
    for p in fps:
        assert p.startswith(os.path.join(cfg.frames_dir, "default", "standup"))  # project-scoped
        assert os.path.exists(p)  # persisted frame path is durable, not a dangling temp file


def test_has_audio_stream_probes_with_ffprobe(monkeypatch):
    cap = {}

    class R:
        def __init__(self, out, code=0):
            self.stdout = out
            self.stderr = ""
            self.returncode = code

    def run(out):
        def _run(cmd, **k):
            cap["cmd"] = cmd
            return R(out)
        return _run

    monkeypatch.setattr(P.subprocess, "run", run("0\n"))
    assert P._has_audio_stream("/v.mp4") is True
    assert cap["cmd"][0] == "ffprobe" and cap["cmd"][-1] == "/v.mp4" and "a" in cap["cmd"]

    monkeypatch.setattr(P.subprocess, "run", run("  \n"))
    assert P._has_audio_stream("/v.mp4") is False


def test_has_audio_stream_raises_on_ffprobe_failure(monkeypatch):
    # A nonzero ffprobe exit must be loud, not silently treated as "no audio".
    class R:
        stdout = ""
        stderr = "moov atom not found"
        returncode = 1
    monkeypatch.setattr(P.subprocess, "run", lambda *a, **k: R())
    with pytest.raises(RuntimeError) as e:
        P._has_audio_stream("/bad.mp4")
    assert "/bad.mp4" in str(e.value) and "moov atom not found" in str(e.value)


def test_add_before_delete_carries_new_row_ids(tmp_path, monkeypatch):
    # store.add must run before delete_memory (a crash between them leaves recoverable
    # duplicates, not a wiped memory), and keep_ids must be exactly the new row ids.
    _patch(monkeypatch, tmp_path)
    store = FakeStore()
    P.ingest_video("/videos/standup.mp4", cfg=_cfg(tmp_path), store=store)
    assert store.calls == ["add", "delete"]
    memory_id, project_id, keep_ids = store.deletes[-1]
    assert memory_id == "standup" and project_id == "default"
    assert set(keep_ids) == {r["id"] for r in store.rows}
    assert len(keep_ids) == 2  # exactly the two new rows, nothing extra kept
