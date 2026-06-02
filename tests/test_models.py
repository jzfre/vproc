from vproc.models import Segment, Evidence, Citation, Claim, Answer, mmss

def test_mmss():
    assert mmss(0) == "00:00"
    assert mmss(75.4) == "01:15"
    assert mmss(3661) == "61:01"

def test_segment_roundtrips():
    s = Segment(id="s1", project_id="p", memory_id="m", screen_state_id="ss0",
                start_ts=1.0, end_ts=2.0, speaker="SPEAKER_0", said_text="hi",
                on_screen_text="slide", on_screen_confidence=None,
                frame_path="/f.png", source_video="/v.mp4", embed_text="[SCREEN]\nslide")
    assert Segment(**s.model_dump()) == s

def test_answer_defaults():
    a = Answer(answered=False, abstained=True, text="Not discussed in these meetings.",
               claims=[], evidence=[])
    assert a.answered is False and a.claims == []
