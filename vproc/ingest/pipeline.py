import os
import pathlib
import subprocess
import tempfile

from vproc.config import load_config
from vproc.ingest import align as A
from vproc.ingest import embed_index as EI
from vproc.ingest import frames as F
from vproc.ingest import ocr as O
from vproc.ingest import transcribe as T
from vproc.store.lancedb_store import Store


def _list_frames(frames_dir: str) -> list[str]:
    return sorted(str(p) for p in pathlib.Path(frames_dir).glob("*.png"))


def _read_log(path: str) -> str:
    return pathlib.Path(path).read_text() if os.path.exists(path) else ""


def ingest_video(video_path: str, cfg=None, store=None, project_id: str = "default") -> int:
    cfg = cfg or load_config()
    store = store or Store(cfg.index_path)
    title = pathlib.Path(video_path).stem  # doubles as memory_id / citation title in v1

    workdir = tempfile.mkdtemp(prefix="vproc-")
    frames_dir = os.path.join(workdir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    log_path = os.path.join(workdir, "frames.log")
    wav_path = os.path.join(workdir, "audio.wav")

    subprocess.run(F.ffmpeg_sample_cmd(video_path, frames_dir, log_path), check=True)
    raw = F.parse_frames_log(_read_log(log_path), _list_frames(frames_dir))
    kept = F.phash_dedup(raw)

    subprocess.run(T.extract_audio_cmd(video_path, wav_path), check=True)
    transcript = T.transcribe(wav_path, cfg.transcribe)

    end_time = transcript[-1].end if transcript else (kept[-1].t if kept else 0.0)
    states = A.build_screen_states(kept, end_time)
    for state in states:
        state.on_screen_text = O.ocr_frame(cfg.ocr, state.frame_path)

    segments = A.build_segments(project_id, title, video_path, states, transcript)
    store.delete_memory(title)  # idempotent re-ingest: drop prior rows for this memory
    EI.embed_and_store(cfg, store, segments)
    return len(segments)
