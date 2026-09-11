from PIL import Image, ImageDraw
from vproc.ingest.frames import ffmpeg_sample_cmd, parse_frames_log, phash_dedup, RawFrame

def _img(tmp_path, name, color):
    p = tmp_path / name
    Image.new("RGB", (64, 64), color).save(p)
    return str(p)

def _img_checker(tmp_path, name):
    """Create a black-and-white checkerboard image (perceptually distinct from solid colors)."""
    p = tmp_path / name
    img = Image.new("RGB", (64, 64), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                draw.rectangle([i * 8, j * 8, i * 8 + 7, j * 8 + 7], fill=(255, 255, 255))
    img.save(p)
    return str(p)

def test_cmd_has_filters():
    cmd = ffmpeg_sample_cmd("in.mp4", "/out", "/out/frames.log", scene=0.08)
    joined = " ".join(cmd)
    assert "mpdecimate" in joined and "gt(scene,0.08)" in joined
    # eq(n,0) keeps the first frame so the opening screen / a scene-change-free video is captured
    assert "eq(n,0)" in joined
    assert "metadata=print:file=/out/frames.log" in joined
    assert "-nostdin" in cmd

def test_parse_frames_log_pairs_paths_to_times():
    log = "frame:0 pts_time:0.000000\nframe:1 pts_time:12.500000\n"
    frames = parse_frames_log(log, ["/o/00000002.png", "/o/00000001.png"])
    assert [f.t for f in frames] == [0.0, 12.5]
    # paths are sorted before pairing
    assert frames[0].path == "/o/00000001.png"

def test_phash_dedup_drops_near_duplicates(tmp_path):
    a = _img(tmp_path, "a.png", (0, 0, 0))
    a2 = _img(tmp_path, "a2.png", (0, 0, 0))      # identical → dropped
    b = _img_checker(tmp_path, "b.png")           # perceptually distinct → kept
    frames = [RawFrame(a, 0.0), RawFrame(a2, 1.0), RawFrame(b, 2.0)]
    kept = phash_dedup(frames, threshold=6, floor_s=999)
    assert [f.path for f in kept] == [a, b]

def test_phash_dedup_floor_forces_anchor(tmp_path):
    a = _img(tmp_path, "a.png", (0, 0, 0))
    a2 = _img(tmp_path, "a2.png", (0, 0, 0))  # identical
    frames = [RawFrame(a, 0.0), RawFrame(a2, 40.0)]  # identical but 40s apart
    kept = phash_dedup(frames, threshold=6, floor_s=30)
    assert [f.path for f in kept] == [a, a2]


def test_ffmpeg_writes_metadata_in_directory_with_filter_special_characters(tmp_path):
    import os
    import shutil
    import subprocess

    import pytest

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is required for the path-escaping integration check")
    # An apostrophe is legal on Windows; on POSIX also exercise colon (drive separator),
    # backslash and filtergraph delimiters using a real filesystem path.
    dirname = "owner's recordings [1],part;2" if os.name == "nt" else "C:\\owner's [1],part;2"
    work = tmp_path / dirname
    work.mkdir()
    source = work / "source.png"
    Image.new("RGB", (64, 64), "blue").save(source)
    out = work / "frames"
    out.mkdir()
    log = work / "frames.log"
    result = subprocess.run(ffmpeg_sample_cmd(str(source), str(out), str(log)),
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    frames = parse_frames_log(log.read_text(), [str(p) for p in out.glob("*.png")])
    assert len(frames) == 1
    assert frames[0].t == 0.0
