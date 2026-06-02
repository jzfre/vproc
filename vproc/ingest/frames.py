import re
from dataclasses import dataclass

import imagehash
from PIL import Image


@dataclass
class RawFrame:
    path: str
    t: float


def ffmpeg_sample_cmd(video: str, out_dir: str, log_path: str, scene: float = 0.08) -> list[str]:
    vf = f"mpdecimate,select='gt(scene,{scene})',metadata=print:file={log_path}"
    return ["ffmpeg", "-hide_banner", "-i", video, "-vf", vf,
            "-fps_mode", "vfr", "-frame_pts", "1", f"{out_dir}/%08d.png"]


def parse_frames_log(log_text: str, frame_paths: list[str]) -> list[RawFrame]:
    times = [float(x) for x in re.findall(r"pts_time:([0-9.]+)", log_text)]
    paths = sorted(frame_paths)
    return [RawFrame(path=p, t=t) for p, t in zip(paths, times)]


def phash_dedup(frames: list[RawFrame], threshold: int = 6, floor_s: float = 30.0) -> list[RawFrame]:
    kept: list[RawFrame] = []
    last_hash = None
    last_t = None
    for f in frames:
        h = imagehash.phash(Image.open(f.path))
        changed = last_hash is None or (h - last_hash) > threshold
        anchor = last_t is None or (f.t - last_t) >= floor_s
        if changed or anchor:
            kept.append(f)
            last_hash, last_t = h, f.t
    return kept
