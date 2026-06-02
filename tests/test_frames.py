from PIL import Image
from vproc.ingest.frames import ffmpeg_sample_cmd, parse_frames_log, phash_dedup, RawFrame

def _img(tmp_path, name, color):
    p = tmp_path / name
    Image.new("RGB", (64, 64), color).save(p)
    return str(p)

def test_cmd_has_filters():
    cmd = ffmpeg_sample_cmd("in.mp4", "/out", "/out/frames.log", scene=0.08)
    joined = " ".join(cmd)
    assert "mpdecimate" in joined and "gt(scene,0.08)" in joined
    assert "metadata=print:file=/out/frames.log" in joined

def test_parse_frames_log_pairs_paths_to_times():
    log = "frame:0 pts_time:0.000000\nframe:1 pts_time:12.500000\n"
    frames = parse_frames_log(log, ["/o/00000002.png", "/o/00000001.png"])
    assert [f.t for f in frames] == [0.0, 12.5]
    # paths are sorted before pairing
    assert frames[0].path == "/o/00000001.png"

def test_phash_dedup_drops_near_duplicates(tmp_path):
    a = _img(tmp_path, "a.png", (0, 0, 0))
    a2 = _img(tmp_path, "a2.png", (0, 0, 0))      # identical → dropped
    b = _img(tmp_path, "b.png", (255, 255, 255))  # different → kept
    frames = [RawFrame(a, 0.0), RawFrame(a2, 1.0), RawFrame(b, 2.0)]
    kept = phash_dedup(frames, threshold=0, floor_s=999)
    assert [f.path for f in kept] == [a, b]

def test_phash_dedup_floor_forces_anchor(tmp_path):
    a = _img(tmp_path, "a.png", (0, 0, 0))
    a2 = _img(tmp_path, "a2.png", (0, 0, 0))  # identical
    frames = [RawFrame(a, 0.0), RawFrame(a2, 40.0)]  # identical but 40s apart
    kept = phash_dedup(frames, threshold=6, floor_s=30)
    assert [f.path for f in kept] == [a, a2]
