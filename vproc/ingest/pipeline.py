import math
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import uuid

from vproc.config import load_config
from vproc.ingest import align as A
from vproc.ingest import diarize as D
from vproc.ingest import embed_index as EI
from vproc.ingest import frames as F
from vproc.ingest import naming as N
from vproc.ingest import ocr as O
from vproc.ingest import transcribe as T
from vproc.locking import IndexWriteLock
from vproc.paths import memory_directory
from vproc.store.lancedb_store import Store


def _list_frames(frames_dir: str) -> list[str]:
    return sorted(str(p) for p in pathlib.Path(frames_dir).glob("*.png"))


def _read_log(path: str) -> str:
    return pathlib.Path(path).read_text() if os.path.exists(path) else ""


def _has_audio_stream(video_path: str) -> bool:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", video_path],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        # A probe error (corrupt/unreadable file) must not masquerade as "no audio" and
        # silently drop the transcript — fail loudly with enough context to diagnose.
        raise RuntimeError(f"ffprobe failed for {video_path}: {r.stderr.strip()}")
    return bool(r.stdout.strip())


def _video_duration(video_path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}: {r.stderr.strip()}")
    try:
        duration = float(r.stdout.strip())
    except ValueError:
        raise RuntimeError(f"Could not determine video duration for {video_path}") from None
    if not math.isfinite(duration) or duration < 0:
        raise RuntimeError(f"Invalid video duration for {video_path}: {r.stdout.strip()}")
    return duration


def ingest_video(video_path: str, cfg=None, store=None, project_id: str = "default") -> int:
    cfg = cfg or load_config()
    title = pathlib.Path(video_path).stem  # doubles as memory_id / citation title in v1
    memory_directory(cfg.frames_dir, project_id, title)
    video_path = str(pathlib.Path(video_path).expanduser().resolve())
    store = Store(cfg.index_path) if store is None else store
    with store.write_lock(), IndexWriteLock(cfg.frames_dir, name=".vproc-frames.lock"):
        mem_dir = str(memory_directory(cfg.frames_dir, project_id, title))
        store.validate_embedding_model(cfg.embed.model)
        return _ingest_locked(video_path, cfg, store, project_id, title, mem_dir)


def _ingest_locked(video_path, cfg, store, project_id, title, mem_dir) -> int:
    workdir = tempfile.mkdtemp(prefix="vproc-")
    frames_dir = os.path.join(workdir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    log_path = os.path.join(workdir, "frames.log")
    wav_path = os.path.join(workdir, "audio.wav")
    try:
        duration = _video_duration(video_path)
        subprocess.run(F.ffmpeg_sample_cmd(video_path, frames_dir, log_path), check=True)
        raw = F.parse_frames_log(_read_log(log_path), _list_frames(frames_dir))
        kept = F.phash_dedup(raw)

        if _has_audio_stream(video_path):
            subprocess.run(T.extract_audio_cmd(video_path, wav_path), check=True)
            transcript = T.transcribe(wav_path, cfg.transcribe)
        else:
            transcript = []  # no audio track: OCR-only ingest

        speaker_map = None
        if transcript and cfg.diarize_model:
            try:
                turns = D.diarize(wav_path, cfg)
                D.assign_speakers(transcript, turns)
                if cfg.speaker_naming and turns:
                    try:
                        votes = N.sample_name_votes(video_path, turns, cfg)
                        mapping, suggestions = N.resolve_names(votes)
                        N.apply_names(transcript, mapping)
                        speaker_map = {
                            "mapping": mapping, "suggestions": suggestions,
                            "votes": [{"speaker": v.speaker, "t": round(v.t), "name": v.name}
                                      for v in votes],
                        }
                        for spk, name in sorted(mapping.items()):
                            print(f"named: {spk} -> {name}")
                        for spk in sorted(suggestions):
                            print(f"unresolved speaker: {spk} (review with 'vproc speakers {title}')")
                    except Exception as e:
                        # Naming must never undo diarization labels or block the ingest.
                        print(f"warning: speaker naming failed: {e}", file=sys.stderr)
            except Exception as e:
                # Best-effort like OCR: speaker labels are never worth losing the transcript.
                print(f"warning: diarization failed ({cfg.diarize_model}): {e}", file=sys.stderr)

        end_time = max(duration, max((f.t for f in kept), default=0.0),
                       max((seg.end for seg in transcript), default=0.0))
        states = A.build_screen_states(kept, end_time)
        if not states and transcript:
            # No scene-change frames (static/webcam recording): pair the transcript with one
            # frameless screen state so a valid ASR transcript is never silently dropped.
            states = [A.ScreenState("ss0", 0.0, end_time, "")]
        for state in states:
            if state.frame_path:
                try:
                    state.on_screen_text = O.ocr_frame(
                        cfg.ocr, state.frame_path, cfg.ocr_timeout, cfg.ocr_max_tokens)
                except Exception as e:
                    # OCR is best-effort: a slow/broken vision backend must not discard the
                    # (valuable) transcript. Degrade this frame to no on-screen text and warn.
                    print(f"warning: OCR failed for {state.frame_path}: {e}", file=sys.stderr)
                    state.on_screen_text = ""

        # A new generation must never overwrite frames referenced by the previous rows.
        # Preserve both generations if a store write fails or its outcome is uncertain.
        generation = uuid.uuid4().hex
        moves = {}
        for f in kept:
            dest = os.path.join(mem_dir, f"{generation}-{os.path.basename(f.path)}")
            moves[f.path] = dest
            f.path = dest
        for state in states:
            if state.frame_path:
                state.frame_path = moves[state.frame_path]

        segments = A.build_segments(project_id, title, video_path, states, transcript)
        if not segments:
            return 0  # a blank/failed extraction must not erase an existing memory
        rows = EI.embed_rows(cfg, segments)  # embed BEFORE any destructive write

        # Shared frame storage may hold files referenced by another index. Only
        # remove this memory's previously indexed files, never every PNG by glob.
        previous_frames = {
            pathlib.Path(r["frame_path"]) for r in store.memory_rows(title, project_id)
            if r.get("frame_path")
            and pathlib.Path(r["frame_path"]).parent.resolve() == pathlib.Path(mem_dir)
        }
        os.makedirs(mem_dir, exist_ok=True)
        for src, dest in moves.items():
            shutil.move(src, dest)
        store.replace_memory(title, project_id, rows, embedding_model=cfg.embed.model)
        # Old files are safe to remove only after their rows have been replaced.
        for path in [*previous_frames, pathlib.Path(mem_dir) / "speakers.json"]:
            try:
                path.unlink(missing_ok=True)
            except OSError as e:
                print(f"warning: failed to remove old ingest file {path}: {e}", file=sys.stderr)
        if speaker_map is not None:
            try:
                N.save_speaker_map(mem_dir, speaker_map)
            except Exception as e:
                # The sidecar is a convenience for `vproc speakers`; it must never
                # abort an otherwise-successful ingest.
                print(f"warning: failed to write speakers.json: {e}", file=sys.stderr)
        return len(segments)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
