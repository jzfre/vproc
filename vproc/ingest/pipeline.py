import math
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

from openai import OpenAIError

from vproc.config import load_config
from vproc.errors import IngestError
from vproc.ingest import align as A
from vproc.ingest import diarize as D
from vproc.ingest import embed_index as EI
from vproc.ingest import frames as F
from vproc.ingest import naming as N
from vproc.ingest import ocr as O
from vproc.ingest import transcribe as T
from vproc.llm import client
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
        raise IngestError(f"ffprobe failed for {video_path}: {r.stderr.strip()}")
    return bool(r.stdout.strip())


def _video_duration(video_path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise IngestError(f"ffprobe failed for {video_path}: {r.stderr.strip()}")
    try:
        duration = float(r.stdout.strip())
    except ValueError:
        raise IngestError(f"Could not determine video duration for {video_path}") from None
    if not math.isfinite(duration) or duration < 0:
        raise IngestError(f"Invalid video duration for {video_path}: {r.stdout.strip()}")
    return duration


def _run_ffmpeg(cmd: list[str], what: str) -> None:
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        # ffmpeg already printed its own reason to stderr; add which step and stop cleanly.
        raise IngestError(f"ffmpeg failed while {what} (exit {e.returncode}); "
                          "see the ffmpeg message above") from None


def _progress(stage: str, detail: str, since: float) -> float:
    """One line per finished stage. stderr keeps stdout for the CLI's result; flush so a
    redirected or backgrounded ingest shows where it is instead of minutes of silence."""
    now = time.monotonic()
    print(f"[ingest] {stage}: {detail} ({now - since:.0f}s)", file=sys.stderr, flush=True)
    return now


def ingest_video(video_path: str, cfg=None, store=None, project_id: str = "default") -> int:
    cfg = cfg or load_config()
    client.configure(cfg)
    title = pathlib.Path(video_path).stem  # doubles as memory_id / citation title in v1
    memory_directory(cfg.frames_dir, project_id, title)
    video_path = str(pathlib.Path(video_path).expanduser().resolve())
    store = Store(cfg.index_path) if store is None else store
    with store.write_lock(), IndexWriteLock(cfg.frames_dir, name=".vproc-frames.lock"):
        mem_dir = str(memory_directory(cfg.frames_dir, project_id, title))
        try:
            EI.preflight(cfg, store)  # embedding backend + index compatibility, before any media work
        except OpenAIError as e:
            # Class name only: SDK messages can carry URLs or response bodies.
            raise IngestError(f"the embedding backend check failed ({type(e).__name__}). Check "
                              "VPROC_EMBED_BASE_URL, VPROC_EMBED_MODEL and its API key, or run "
                              "'vproc doctor --online'.") from None
        return _ingest_locked(video_path, cfg, store, project_id, title, mem_dir)


def _ingest_locked(video_path, cfg, store, project_id, title, mem_dir) -> int:
    workdir = tempfile.mkdtemp(prefix="vproc-")
    frames_dir = os.path.join(workdir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    log_path = os.path.join(workdir, "frames.log")
    wav_path = os.path.join(workdir, "audio.wav")
    try:
        clock = time.monotonic()
        duration = _video_duration(video_path)
        _run_ffmpeg(F.ffmpeg_sample_cmd(video_path, frames_dir, log_path), "sampling frames")
        raw = F.parse_frames_log(_read_log(log_path), _list_frames(frames_dir))
        kept = F.phash_dedup(raw)
        clock = _progress("frames", f"{len(kept)} kept of {len(raw)} scene changes", clock)

        if _has_audio_stream(video_path):
            _run_ffmpeg(T.extract_audio_cmd(video_path, wav_path), "extracting audio")
            transcript = T.transcribe(wav_path, cfg.transcribe)
            clock = _progress("transcript", f"{len(transcript)} segments", clock)
        else:
            transcript = []  # no audio track: OCR-only ingest
            clock = _progress("transcript", "no audio track", clock)

        speaker_map = None
        degraded: list[str] = []  # best-effort stages that failed; summarised once at the end
        if transcript and cfg.diarize_model:
            try:
                turns = D.diarize(wav_path, cfg)
                transcript = D.assign_speakers(transcript, turns)
                clock = _progress(
                    "speakers", f"{len({t.speaker for t in turns})} speakers in {len(turns)} turns", clock)
                # Nameplate naming reads frames with the same vision endpoint as OCR.
                if cfg.speaker_naming and cfg.ocr.base_url and turns:
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
                        degraded.append("speaker naming failed (labels stay SPEAKER_xx)")
            except Exception as e:
                # Best-effort like OCR: speaker labels are never worth losing the transcript.
                print(f"warning: diarization failed ({cfg.diarize_model}): {e}", file=sys.stderr)
                # ...but never at the price of a memory that already has them: an unlabelled
                # re-ingest would silently replace good speaker attribution.
                if any(r.get("speaker", "SPEAKER_0") != "SPEAKER_0"
                       for r in store.memory_rows(title, project_id)):
                    raise ValueError(
                        f"diarization failed and '{title}' already has speaker labels; refusing to "
                        "replace it with an unlabelled transcript. Fix diarization (HF_TOKEN, gated "
                        "model access, the diarization extra) or set VPROC_DIARIZE_MODEL= to ingest "
                        "without speakers.") from None
                degraded.append("diarization failed, so there are no speaker labels (check HF_TOKEN, "
                                "gated model access and the diarization extra, or set "
                                "VPROC_DIARIZE_MODEL= to silence this)")

        end_time = max(duration, max((f.t for f in kept), default=0.0),
                       max((seg.end for seg in transcript), default=0.0))
        states = A.build_screen_states(kept, end_time)
        if not states and transcript:
            # No scene-change frames (static/webcam recording): pair the transcript with one
            # frameless screen state so a valid ASR transcript is never silently dropped.
            states = [A.ScreenState("ss0", 0.0, end_time, "")]
        ocr_failures = 0
        for state in states:
            if state.frame_path and cfg.ocr.base_url:
                try:
                    state.on_screen_text = O.ocr_frame(
                        cfg.ocr, state.frame_path, cfg.ocr_timeout, cfg.ocr_max_tokens)
                except Exception as e:
                    # OCR is best-effort: a slow/broken vision backend must not discard the
                    # (valuable) transcript. Degrade this frame to no on-screen text and warn.
                    print(f"warning: OCR failed for {state.frame_path}: {e}", file=sys.stderr)
                    state.on_screen_text = ""
                    ocr_failures += 1
        if ocr_failures:
            degraded.append(f"OCR failed for {ocr_failures} of "
                            f"{sum(bool(s.frame_path) for s in states)} frames")
        clock = _progress("ocr", f"{sum(bool(s.on_screen_text) for s in states)} of "
                          f"{sum(bool(s.frame_path) for s in states)} frames have text"
                          if cfg.ocr.base_url else "disabled (VPROC_OCR_BASE_URL is empty)", clock)

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

        def summarise_degradation() -> None:
            if degraded:
                print(f"[ingest] DEGRADED: {'; '.join(degraded)}", file=sys.stderr, flush=True)

        segments = A.build_segments(project_id, title, video_path, states, transcript)
        if not segments:
            summarise_degradation()  # here it is the explanation for the empty result
            return 0  # a blank/failed extraction must not erase an existing memory
        # Cleanup follows index rows, so a stored frame no row points at could never be found
        # again. Unreferenced frames stay in the workdir and go with it.
        used = {seg.frame_path for seg in segments if seg.frame_path}
        moves = {src: dest for src, dest in moves.items() if dest in used}
        rows = EI.embed_rows(cfg, segments)  # embed BEFORE any destructive write

        # Shared frame storage may hold files referenced by another index. Only
        # remove this memory's previously indexed files, never every PNG by glob.
        previous_rows = store.memory_rows(title, project_id)
        if speaker_map is None and any(not str(r.get("speaker", "SPEAKER_0")).startswith("SPEAKER_")
                                       for r in previous_rows):
            # Diarization labels are not stable across runs, so old names cannot be carried over.
            print(f"[ingest] note: previous speaker names in '{title}' are replaced by SPEAKER_xx "
                  f"labels; re-apply them with: vproc speakers \"{title}\" --set 'SPEAKER_XX=Name'",
                  file=sys.stderr, flush=True)
        previous_frames = {
            pathlib.Path(r["frame_path"]) for r in previous_rows
            if r.get("frame_path")
            and pathlib.Path(r["frame_path"]).parent.resolve() == pathlib.Path(mem_dir)
        }
        os.makedirs(mem_dir, exist_ok=True)
        for src, dest in moves.items():
            shutil.move(src, dest)
        store.replace_memory(title, project_id, rows, embedding_model=cfg.embed.model)
        _progress("index", f"{len(segments)} chunks embedded and stored", clock)
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
        summarise_degradation()
        return len(segments)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
