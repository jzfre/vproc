import os
import pathlib
import shutil
import types
from contextlib import nullcontext

import pytest

import vproc.ingest.pipeline as P
from vproc.config import Endpoint
from vproc.ingest.frames import RawFrame
from vproc.ingest.transcribe import TranscriptSegment


class FakeStore:
    """Exercise pipeline effects while real Store transactions are tested separately."""
    def __init__(self):
        self.rows = []
        self.calls = []

    def write_lock(self):
        return nullcontext()

    def validate_embedding_model(self, model, dimension=None):
        pass

    def memory_rows(self, memory_id, project_id=None):
        return [r for r in self.rows if r["memory_id"] == memory_id
                and r.get("project_id", "default") == (project_id or "default")]

    def replace_memory(self, memory_id, project_id, rows, embedding_model):
        self.calls.append("replace")
        self.rows = [r for r in self.rows if r["memory_id"] != memory_id
                     or r.get("project_id", "default") != project_id] + list(rows)


def _cfg(tmp_path):
    ep = Endpoint("u", "m")
    return types.SimpleNamespace(
        ocr=ep, embed=ep, transcribe=ep,
        index_path=str(tmp_path / "db.lance"),
        frames_dir=str(tmp_path / "frames"),
        ocr_timeout=60.0,
        ocr_max_tokens=1024,
        diarize_model="",
        speaker_naming=False,
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
    monkeypatch.setattr(P, "_video_duration", lambda v: 30.0)
    monkeypatch.setattr(P, "_list_frames", lambda d: [f.path for f in frames])
    monkeypatch.setattr(P.F, "parse_frames_log", lambda text, paths: list(frames))
    monkeypatch.setattr(P.F, "phash_dedup", lambda fr, **k: list(fr))
    monkeypatch.setattr(P, "_read_log", lambda p: "")
    monkeypatch.setattr(P.T, "transcribe", lambda wav, ep=None: list(ts))
    monkeypatch.setattr(P.O, "ocr_frame", lambda ocr, path, timeout=None, max_tokens=None: "Roadmap Q3")
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


def test_embed_failure_preserves_existing_rows(tmp_path, monkeypatch):
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
    assert store.calls == []
    assert store.rows == [{"memory_id": "standup", "id": "old"}]  # prior rows intact


def test_zero_frames_still_indexes_transcript(tmp_path, monkeypatch):
    # Static/webcam recording: no scene-change frames, but the transcript must survive.
    _patch(monkeypatch, tmp_path)
    monkeypatch.setattr(P.F, "phash_dedup", lambda fr, **k: [])
    store = FakeStore()
    n = P.ingest_video("/videos/webcam.mp4", cfg=_cfg(tmp_path), store=store)
    assert n >= 1
    assert any("we ship in July" in r["said_text"] for r in store.rows)


def test_ocr_failure_degrades_to_transcript(tmp_path, monkeypatch):
    # A slow/broken vision backend must not discard the transcript: OCR is best-effort per frame.
    _patch(monkeypatch, tmp_path)
    def boom(ocr, path, timeout=None, max_tokens=None):
        raise TimeoutError("vision backend too slow")
    monkeypatch.setattr(P.O, "ocr_frame", boom)
    store = FakeStore()
    n = P.ingest_video("/videos/standup.mp4", cfg=_cfg(tmp_path), store=store)
    assert n >= 1  # transcript still indexed despite OCR failing on every frame
    assert all(r["on_screen_text"] == "" for r in store.rows)


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


def test_ingest_replaces_memory_in_one_store_operation(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    store = FakeStore()
    P.ingest_video("/videos/standup.mp4", cfg=_cfg(tmp_path), store=store)
    assert store.calls == ["replace"]
    assert len(store.rows) == 2
    assert {(r["memory_id"], r["project_id"]) for r in store.rows} == {("standup", "default")}


def test_diarization_labels_reach_store(tmp_path, monkeypatch):
    from vproc.ingest.diarize import SpeakerTurn
    _patch(monkeypatch, tmp_path,
           transcript=[TranscriptSegment(0.0, 4.0, "hello there"),
                       TranscriptSegment(4.0, 9.0, "hi back")])
    monkeypatch.setattr(P.D, "diarize",
                        lambda wav, cfg: [SpeakerTurn(0.0, 4.0, "SPEAKER_00"),
                                          SpeakerTurn(4.0, 9.0, "SPEAKER_01")])
    cfg = _cfg(tmp_path)
    cfg.diarize_model = "pyannote/fake"
    store = FakeStore()
    P.ingest_video("/videos/standup.mp4", cfg=cfg, store=store)
    speakers = {r["speaker"] for r in store.rows}
    assert {"SPEAKER_00", "SPEAKER_01"} <= speakers  # align sub-splits on speaker change


def test_diarization_disabled_when_model_empty(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    def _boom(wav, cfg):
        raise AssertionError("diarize must not be called when disabled")
    monkeypatch.setattr(P.D, "diarize", _boom)
    store = FakeStore()
    n = P.ingest_video("/videos/standup.mp4", cfg=_cfg(tmp_path), store=store)
    assert n >= 1  # ingest proceeded, diarize never invoked


def test_diarization_failure_degrades_to_unlabeled(tmp_path, monkeypatch, capsys):
    _patch(monkeypatch, tmp_path)
    def _boom(wav, cfg):
        raise RuntimeError("bad token")
    monkeypatch.setattr(P.D, "diarize", _boom)
    cfg = _cfg(tmp_path)
    cfg.diarize_model = "pyannote/fake"
    store = FakeStore()
    n = P.ingest_video("/videos/standup.mp4", cfg=cfg, store=store)
    assert n >= 1
    assert all(r["speaker"] == "SPEAKER_0" for r in store.rows)  # transcript survives unlabeled
    err = capsys.readouterr().err
    assert "diarization failed" in err
    assert "pyannote/fake" in err  # model named for diagnosability


def test_speaker_naming_applies_names_to_store(tmp_path, monkeypatch):
    from vproc.ingest.diarize import SpeakerTurn
    _patch(monkeypatch, tmp_path,
           transcript=[TranscriptSegment(0.0, 4.0, "hello there"),
                       TranscriptSegment(4.0, 9.0, "hi back")])
    monkeypatch.setattr(P.D, "diarize",
                        lambda wav, cfg: [SpeakerTurn(0.0, 4.0, "SPEAKER_00"),
                                          SpeakerTurn(4.0, 9.0, "SPEAKER_01")])
    monkeypatch.setattr(P.N, "sample_name_votes", lambda video, turns, cfg: [])
    monkeypatch.setattr(P.N, "resolve_names",
                        lambda votes: ({"SPEAKER_00": "Repan, Jozef"},
                                       {"SPEAKER_01": {"votes": {}, "evidence": []}}))
    cfg = _cfg(tmp_path)
    cfg.diarize_model = "pyannote/fake"
    cfg.speaker_naming = True
    store = FakeStore()
    P.ingest_video("/videos/standup.mp4", cfg=cfg, store=store)
    speakers = {r["speaker"] for r in store.rows}
    assert "Repan, Jozef" in speakers and "SPEAKER_01" in speakers
    import json as _json
    m = _json.load(open(os.path.join(cfg.frames_dir, "default", "standup", "speakers.json")))
    assert m["mapping"] == {"SPEAKER_00": "Repan, Jozef"}
    assert "SPEAKER_01" in m["suggestions"]


def test_naming_failure_keeps_diarized_labels(tmp_path, monkeypatch, capsys):
    from vproc.ingest.diarize import SpeakerTurn
    _patch(monkeypatch, tmp_path,
           transcript=[TranscriptSegment(0.0, 4.0, "hello"), TranscriptSegment(4.0, 9.0, "hi")])
    monkeypatch.setattr(P.D, "diarize",
                        lambda wav, cfg: [SpeakerTurn(0.0, 4.0, "SPEAKER_00"),
                                          SpeakerTurn(4.0, 9.0, "SPEAKER_01")])
    def boom(video, turns, cfg):
        raise RuntimeError("vision endpoint down")
    monkeypatch.setattr(P.N, "sample_name_votes", boom)
    cfg = _cfg(tmp_path)
    cfg.diarize_model = "pyannote/fake"
    cfg.speaker_naming = True
    store = FakeStore()
    n = P.ingest_video("/videos/standup.mp4", cfg=cfg, store=store)
    assert n >= 1
    assert {r["speaker"] for r in store.rows} >= {"SPEAKER_00", "SPEAKER_01"}  # labels survive
    assert "speaker naming failed" in capsys.readouterr().err


def test_speaker_map_write_failure_does_not_abort_ingest(tmp_path, monkeypatch, capsys):
    from vproc.ingest.diarize import SpeakerTurn
    _patch(monkeypatch, tmp_path,
           transcript=[TranscriptSegment(0.0, 4.0, "hello there"),
                       TranscriptSegment(4.0, 9.0, "hi back")])
    monkeypatch.setattr(P.D, "diarize",
                        lambda wav, cfg: [SpeakerTurn(0.0, 4.0, "SPEAKER_00"),
                                          SpeakerTurn(4.0, 9.0, "SPEAKER_01")])
    monkeypatch.setattr(P.N, "sample_name_votes", lambda video, turns, cfg: [])
    monkeypatch.setattr(P.N, "resolve_names",
                        lambda votes: ({"SPEAKER_00": "Repan, Jozef"}, {}))
    def boom(mem_dir, data):
        raise OSError("disk full")
    monkeypatch.setattr(P.N, "save_speaker_map", boom)
    cfg = _cfg(tmp_path)
    cfg.diarize_model = "pyannote/fake"
    cfg.speaker_naming = True
    store = FakeStore()
    n = P.ingest_video("/videos/standup.mp4", cfg=cfg, store=store)
    assert n >= 1  # the sidecar write failed, but the ingest still succeeded
    assert "Repan, Jozef" in {r["speaker"] for r in store.rows}  # store already has the rows
    assert "warning: failed to write speakers.json" in capsys.readouterr().err


def test_naming_disabled_skips_probes(tmp_path, monkeypatch):
    from vproc.ingest.diarize import SpeakerTurn
    _patch(monkeypatch, tmp_path)
    monkeypatch.setattr(P.D, "diarize", lambda wav, cfg: [SpeakerTurn(0.0, 3.0, "SPEAKER_00")])
    def boom(video, turns, cfg):
        raise AssertionError("must not probe when speaker_naming is off")
    monkeypatch.setattr(P.N, "sample_name_votes", boom)
    cfg = _cfg(tmp_path)          # speaker_naming=False by default
    cfg.diarize_model = "pyannote/fake"
    store = FakeStore()
    assert P.ingest_video("/videos/standup.mp4", cfg=cfg, store=store) >= 1


def test_failed_store_write_preserves_previous_frames(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    old_frame = tmp_path / "frames" / "default" / "standup" / "00000001.png"
    old_frame.parent.mkdir(parents=True)
    old_frame.write_bytes(b"previous screenshot")
    store = FakeStore()
    store.rows = [{"memory_id": "standup", "id": "old", "frame_path": str(old_frame)}]

    def fail_replace(*args, **kwargs):
        raise RuntimeError("index unavailable")

    monkeypatch.setattr(store, "replace_memory", fail_replace)
    with pytest.raises(RuntimeError, match="index unavailable"):
        P.ingest_video("/videos/standup.mp4", cfg=cfg, store=store)

    assert old_frame.read_bytes() == b"previous screenshot"
    assert store.rows[0]["id"] == "old"


def test_ambiguous_replacement_preserves_all_referenced_frames(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    old_frame = tmp_path / "frames" / "default" / "standup" / "00000001.png"
    old_frame.parent.mkdir(parents=True)
    old_frame.write_bytes(b"previous screenshot")
    store = FakeStore()
    store.rows = [{"memory_id": "standup", "id": "old", "frame_path": str(old_frame)}]

    replace = store.replace_memory

    def ambiguous_replace(*args, **kwargs):
        replace(*args, **kwargs)
        raise RuntimeError("commit acknowledgement lost")

    monkeypatch.setattr(store, "replace_memory", ambiguous_replace)
    with pytest.raises(RuntimeError, match="acknowledgement lost"):
        P.ingest_video("/videos/standup.mp4", cfg=cfg, store=store)

    assert old_frame.read_bytes() == b"previous screenshot"
    assert all(os.path.isfile(row["frame_path"]) for row in store.rows)
    assert len({row["frame_path"] for row in store.rows}) == len(store.rows)


def test_reingest_keeps_unrelated_frames_in_shared_storage(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    unrelated = tmp_path / "frames" / "default" / "standup" / "other-index.png"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_bytes(b"another index still references this")

    P.ingest_video("/videos/standup.mp4", cfg=cfg, store=FakeStore())

    assert unrelated.read_bytes() == b"another index still references this"


def test_busy_frame_storage_stops_ingest_before_model_calls(tmp_path, monkeypatch):
    from vproc.errors import IndexBusyError
    from vproc.locking import IndexWriteLock

    _patch(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    store = FakeStore()

    def unexpected_probe(*args):
        pytest.fail("an overlapping ingest must be rejected before probing or inference")

    monkeypatch.setattr(P, "_video_duration", unexpected_probe)
    with IndexWriteLock(cfg.frames_dir, name=".vproc-frames.lock"):
        with pytest.raises(IndexBusyError):
            P.ingest_video("/videos/standup.mp4", cfg=cfg, store=store)
    assert store.calls == []


def test_index_and_frames_can_share_a_directory(tmp_path, monkeypatch):
    from vproc.store.lancedb_store import Store

    _patch(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    cfg.index_path = cfg.frames_dir
    store = Store(cfg.index_path)
    assert P.ingest_video("/videos/standup.mp4", cfg=cfg, store=store) == 2
    assert len(store.memory_rows("standup", "default")) == 2


def test_incompatible_model_stops_ingest_before_extraction(tmp_path, monkeypatch):
    from vproc.errors import IndexCompatibilityError

    _patch(monkeypatch, tmp_path)
    store = FakeStore()

    def incompatible(*args):
        raise IndexCompatibilityError("embedding model mismatch")

    monkeypatch.setattr(store, "validate_embedding_model", incompatible)
    with pytest.raises(IndexCompatibilityError, match="mismatch"):
        P.ingest_video("/videos/standup.mp4", cfg=_cfg(tmp_path), store=store)
    assert store.calls == []


def test_symlinked_memory_directory_cannot_redirect_ingest_writes(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    project = tmp_path / "frames" / "default"
    project.mkdir(parents=True)
    (project / "standup").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="directory"):
        P.ingest_video("/videos/standup.mp4", cfg=cfg, store=FakeStore())
    assert list(outside.iterdir()) == []


def test_empty_reingest_preserves_previous_memory(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path, transcript=[], has_audio=False)
    monkeypatch.setattr(P.O, "ocr_frame", lambda *args: "")
    cfg = _cfg(tmp_path)
    old_frame = tmp_path / "frames" / "default" / "standup" / "previous.png"
    old_frame.parent.mkdir(parents=True)
    old_frame.write_bytes(b"previous screenshot")
    store = FakeStore()
    store.rows = [{"memory_id": "standup", "id": "old", "frame_path": str(old_frame)}]

    assert P.ingest_video("/videos/standup.mp4", cfg=cfg, store=store) == 0

    assert store.calls == []
    assert store.rows[0]["id"] == "old"
    assert old_frame.read_bytes() == b"previous screenshot"


def test_relative_media_paths_are_persisted_as_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _patch(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    cfg.frames_dir = "frames"
    store = FakeStore()

    P.ingest_video("videos/standup.mp4", cfg=cfg, store=store)

    assert {row["source_video"] for row in store.rows} == {str(tmp_path / "videos" / "standup.mp4")}
    assert all(os.path.isabs(row["frame_path"]) for row in store.rows)


@pytest.mark.parametrize("project_id", ["", ".", "..", "../other", "/tmp/elsewhere", "a/b", "a\\b"])
def test_invalid_project_id_cannot_write_outside_memory_dir(tmp_path, monkeypatch, project_id):
    _patch(monkeypatch, tmp_path)
    store = FakeStore()
    if os.path.isabs(project_id):
        project_id = str(tmp_path / "outside-project")

    with pytest.raises(ValueError, match="project"):
        P.ingest_video("/videos/standup.mp4", cfg=_cfg(tmp_path), store=store,
                       project_id=project_id)

    assert store.calls == []


@pytest.mark.parametrize("video_path", ["/videos/..", "/"])
def test_invalid_memory_title_is_rejected(tmp_path, monkeypatch, video_path):
    _patch(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="memory"):
        P.ingest_video(video_path, cfg=_cfg(tmp_path), store=FakeStore())


def test_static_silent_video_covers_full_media_duration(tmp_path, monkeypatch):
    frames = [RawFrame(_src_frame(tmp_path, "opening.png"), 0.0)]
    _patch(monkeypatch, tmp_path, frames=frames, transcript=[], has_audio=False)
    monkeypatch.setattr(P, "_video_duration", lambda v: 120.5)
    store = FakeStore()

    P.ingest_video("/videos/slide.mp4", cfg=_cfg(tmp_path), store=store)

    assert [(row["start_ts"], row["end_ts"]) for row in store.rows] == [(0.0, 120.5)]


def test_successful_reingest_removes_previous_screenshots(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    store = FakeStore()
    cfg = _cfg(tmp_path)
    P.ingest_video("/videos/standup.mp4", cfg=cfg, store=store)
    previous_paths = {row["frame_path"] for row in store.rows}

    _patch(monkeypatch, tmp_path)
    P.ingest_video("/videos/standup.mp4", cfg=cfg, store=store)

    assert all(os.path.exists(row["frame_path"]) for row in store.rows)
    assert all(not os.path.exists(path) for path in previous_paths)


@pytest.mark.parametrize("output", ["N/A", "nan", "inf", "-1"])
def test_video_duration_rejects_unknown_or_invalid_duration(monkeypatch, output):
    monkeypatch.setattr(P.subprocess, "run", lambda *args, **kwargs:
                        types.SimpleNamespace(returncode=0, stdout=output, stderr=""))
    with pytest.raises(RuntimeError, match="duration"):
        P._video_duration("/videos/clip.mp4")


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
                    reason="ffmpeg and ffprobe are required for media integration")
def test_real_silent_video_ingest_and_replacement(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from PIL import Image
    from vproc.ingest.embed_index import embed_rows
    from vproc.service import create_app
    from vproc.store.lancedb_store import Store

    video = tmp_path / "slides.mp4"
    P.subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
         "-f", "lavfi", "-i", "color=c=blue:s=64x64:r=10:d=2.5",
         "-c:v", "mpeg4", str(video)], check=True,
    )
    monkeypatch.setattr(P.O, "ocr_frame", lambda *args: "Agenda")
    monkeypatch.setattr(P.EI, "embed_rows", lambda cfg, segments:
                        embed_rows(cfg, segments, embed=lambda *args: [[1.0, 0.0]] * len(segments)))
    cfg = _cfg(tmp_path)
    store = Store(cfg.index_path)

    assert P.ingest_video(str(video), cfg=cfg, store=store) == 1
    first = store.memory_rows("slides", "default")[0]
    assert first["start_ts"] == 0.0
    assert first["end_ts"] == pytest.approx(2.5)
    assert first["source_video"] == str(video)
    assert first["said_text"] == ""
    assert first["on_screen_text"] == "Agenda"
    with Image.open(first["frame_path"]) as frame:
        assert frame.size == (64, 64)

    client = TestClient(create_app(store=store, cfg=cfg, scorer=lambda p, h: 1.0))
    assert client.get("/api/memories").json()[0]["duration_s"] == pytest.approx(2.5)
    old_name = client.get("/api/memories/slides/segments").json()[0]["frame_name"]
    assert client.get(f"/api/frames/slides/{old_name}").content == pathlib.Path(first["frame_path"]).read_bytes()
    partial = client.get("/api/media/slides", headers={"Range": "bytes=0-7"})
    assert partial.status_code == 206
    assert partial.content == video.read_bytes()[:8]

    # A same-stem re-ingest from a new source directory must update this running app.
    replacement = tmp_path / "replacement" / video.name
    replacement.parent.mkdir()
    shutil.copyfile(video, replacement)
    assert P.ingest_video(str(replacement), cfg=cfg, store=store) == 1
    video.unlink()
    rows = store.memory_rows("slides", "default")
    assert len(rows) == 1
    assert rows[0]["frame_path"] != first["frame_path"]
    assert os.path.isfile(rows[0]["frame_path"])
    assert not os.path.exists(first["frame_path"])
    assert client.get("/api/media/slides").content == replacement.read_bytes()
    new_name = client.get("/api/memories/slides/segments").json()[0]["frame_name"]
    assert client.get(f"/api/frames/slides/{new_name}").status_code == 200
    assert client.get(f"/api/frames/slides/{old_name}").status_code == 404
