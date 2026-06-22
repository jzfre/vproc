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

def test_build_segments_subsplits_long_monologue_for_citation_precision():
    # One near-static screen state spanning 0-300s with a 5-minute monologue.
    frames = [RawFrame("a.png", 0.0)]
    states = build_screen_states(frames, end_time=300.0)
    states[0].on_screen_text = "Architecture Diagram"
    transcript = [TS(i * 60.0, i * 60.0 + 60.0, f"point {i}", speaker="SPEAKER_0") for i in range(5)]
    segs = build_segments("p", "m", "/v.mp4", states, transcript)
    # must NOT be a single 5-minute blob — split so citations point at a real moment
    assert len(segs) > 1
    # each sub-segment stays within the ~90s window
    assert all(s.end_ts - s.start_ts <= 90.0 + 1e-6 for s in segs), [s.end_ts - s.start_ts for s in segs]
    # visual context preserved: every sub-segment keeps the same screen state + OCR
    assert all(s.screen_state_id == "ss0" for s in segs)
    assert all(s.on_screen_text == "Architecture Diagram" for s in segs)
    # full temporal coverage, in chronological order
    assert segs[0].start_ts == 0.0
    assert segs[-1].end_ts == 300.0
    assert [s.start_ts for s in segs] == sorted(s.start_ts for s in segs)

def test_build_segments_splits_on_speaker_change_within_screen_state():
    # Same screen, two speakers back-to-back, well under the time window.
    frames = [RawFrame("a.png", 0.0)]
    states = build_screen_states(frames, end_time=60.0)
    states[0].on_screen_text = "Agenda"
    transcript = [
        TS(1.0, 5.0, "alice speaking", speaker="SPEAKER_0"),
        TS(6.0, 10.0, "bob replying", speaker="SPEAKER_1"),
    ]
    segs = build_segments("p", "m", "/v.mp4", states, transcript)
    assert len(segs) == 2
    assert segs[0].speaker == "SPEAKER_0" and segs[0].said_text == "alice speaking"
    assert segs[1].speaker == "SPEAKER_1" and segs[1].said_text == "bob replying"
