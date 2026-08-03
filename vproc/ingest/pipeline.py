import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

from vproc.config import load_config
from vproc.ingest import align as A
from vproc.ingest import diarize as D
from vproc.ingest import embed_index as EI
from vproc.ingest import frames as F
from vproc.ingest import naming as N
from vproc.ingest import ocr as O
from vproc.ingest import transcribe as T
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


def ingest_video(video_path: str, cfg=None, store=None, project_id: str = "default") -> int:
    cfg = cfg or load_config()
    store = store or Store(cfg.index_path)
    title = pathlib.Path(video_path).stem  # doubles as memory_id / citation title in v1
    mem_dir = os.path.join(cfg.frames_dir, project_id, title)  # durable, project-scoped

    workdir = tempfile.mkdtemp(prefix="vproc-")
    frames_dir = os.path.join(workdir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    log_path = os.path.join(workdir, "frames.log")
    wav_path = os.path.join(workdir, "audio.wav")
    try:
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

        end_time = max(kept[-1].t if kept else 0.0, transcript[-1].end if transcript else 0.0)
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

        # Bake the FINAL durable paths into rows now, but defer the physical move until the
        # embed succeeds: any failure before the swap must leave the prior frames intact.
        moves = {}
        for f in kept:
            dest = os.path.join(mem_dir, os.path.basename(f.path))
            moves[f.path] = dest
            f.path = dest
        for state in states:
            if state.frame_path:
                state.frame_path = moves[state.frame_path]

        segments = A.build_segments(project_id, title, video_path, states, transcript)
        rows = EI.embed_rows(cfg, segments)  # embed BEFORE any destructive write

        # Swap now that rows exist: replace the previous ingest's frames, then write rows
        # add-before-delete so a crash between the two leaves dupes (re-ingestable), not a wipe.
        if os.path.isdir(mem_dir):
            shutil.rmtree(mem_dir)
        os.makedirs(mem_dir, exist_ok=True)
        for src, dest in moves.items():
            shutil.move(src, dest)
        if speaker_map is not None:
            N.save_speaker_map(mem_dir, speaker_map)
        store.add(rows)
        store.delete_memory(title, project_id, keep_ids=[r["id"] for r in rows])
        return len(segments)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
