from vproc.ingest.frames import RawFrame
from vproc.ingest.transcribe import TranscriptSegment as TS
from vproc.ingest.align import build_screen_states, assign_segments, build_segments, make_embed_text

def test_build_screen_states_intervals():
    frames = [RawFrame("a.png", 0.0), RawFrame("b.png", 10.0)]
    states = build_screen_states(frames, end_time=25.0)
    assert (states[0].t_start, states[0].t_end) == (0.0, 10.0)
    assert (states[1].t_start, states[1].t_end) == (10.0, 25.0)

def test_assign_by_midpoint_when_straddling():
    frames = [RawFrame("a.png", 0.0), RawFrame("b.png", 10.0)]
    states = build_screen_states(frames, end_time=20.0)
    # segment 8.0-12.0 straddles the 10.0 boundary; midpoint 10.0 → state b
    seg = TS(8.0, 12.0, "boundary talk")
    buckets = assign_segments(states, [seg])
    assert buckets["ss0"] == []
    assert buckets["ss1"] == [seg]

def test_build_segments_attaches_screen_and_spoken():
    frames = [RawFrame("a.png", 0.0)]
    states = build_screen_states(frames, end_time=10.0)
    states[0].on_screen_text = "Roadmap Q3"
    transcript = [TS(1.0, 3.0, "we ship in July", speaker="SPEAKER_0")]
    segs = build_segments("p", "meeting-1", "/v.mp4", states, transcript)
    assert len(segs) == 1
    s = segs[0]
    assert s.said_text == "we ship in July"
    assert s.on_screen_text == "Roadmap Q3"
    assert "[SCREEN]\nRoadmap Q3" in s.embed_text
    assert "[SPOKEN]\nSPEAKER_0 (00:01): we ship in July" in s.embed_text
    assert s.start_ts == 1.0 and s.end_ts == 3.0

def test_build_segments_skips_empty_states():
    frames = [RawFrame("a.png", 0.0)]
    states = build_screen_states(frames, end_time=10.0)  # no OCR text, no transcript
    assert build_segments("p", "m", "/v.mp4", states, []) == []
