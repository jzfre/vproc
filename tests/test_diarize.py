from vproc.ingest.diarize import SpeakerTurn, assign_speakers, diarize
from vproc.ingest.transcribe import TranscriptSegment


def _seg(s, e):
    return TranscriptSegment(start=s, end=e, text="x")


def test_majority_overlap_wins():
    t = [_seg(0.0, 10.0)]
    assign_speakers(t, [SpeakerTurn(0.0, 3.0, "SPEAKER_00"),
                        SpeakerTurn(3.0, 10.0, "SPEAKER_01")])
    assert t[0].speaker == "SPEAKER_01"


def test_overlap_sums_across_turns():
    # 00 speaks 0-3 and 7-10 (6s total) vs 01 speaks 3-7 (4s): summed overlap wins
    t = [_seg(0.0, 10.0)]
    assign_speakers(t, [SpeakerTurn(0.0, 3.0, "SPEAKER_00"),
                        SpeakerTurn(3.0, 7.0, "SPEAKER_01"),
                        SpeakerTurn(7.0, 10.0, "SPEAKER_00")])
    assert t[0].speaker == "SPEAKER_00"


def test_tie_breaks_to_earliest_overlapping_turn():
    t = [_seg(0.0, 8.0)]
    assign_speakers(t, [SpeakerTurn(4.0, 8.0, "SPEAKER_00"),   # listed first, starts later
                        SpeakerTurn(0.0, 4.0, "SPEAKER_01")])  # equal 4s overlap, starts at 0
    assert t[0].speaker == "SPEAKER_01"


def test_zero_overlap_takes_nearest_turn_by_midpoint():
    t = [_seg(10.0, 12.0)]  # midpoint 11
    assign_speakers(t, [SpeakerTurn(0.0, 2.0, "SPEAKER_00"),     # mid 1  -> dist 10
                        SpeakerTurn(13.0, 15.0, "SPEAKER_01")])  # mid 14 -> dist 3
    assert t[0].speaker == "SPEAKER_01"


def test_empty_turns_is_noop():
    t = [_seg(0.0, 5.0)]
    assign_speakers(t, [])
    assert t[0].speaker == "SPEAKER_0"


def test_segments_labeled_independently():
    t = [_seg(0.0, 4.0), _seg(4.0, 8.0)]
    assign_speakers(t, [SpeakerTurn(0.0, 4.0, "SPEAKER_00"),
                        SpeakerTurn(4.0, 8.0, "SPEAKER_01")])
    assert [s.speaker for s in t] == ["SPEAKER_00", "SPEAKER_01"]


class _Span:
    def __init__(self, s, e):
        self.start, self.end = s, e


class _FakeAnnotation:
    def itertracks(self, yield_label=False):
        yield _Span(0.0, 2.5), None, "SPEAKER_00"
        yield _Span(2.5, 5.0), None, "SPEAKER_01"


def test_diarize_uses_injected_diarizer_and_converts_turns():
    calls = {}
    def fake_pipeline(wav):
        calls["wav"] = wav
        return _FakeAnnotation()
    turns = diarize("/tmp/a.wav", cfg=None, diarizer=fake_pipeline)
    assert calls["wav"] == "/tmp/a.wav"
    assert turns == [SpeakerTurn(0.0, 2.5, "SPEAKER_00"),
                     SpeakerTurn(2.5, 5.0, "SPEAKER_01")]


def test_diarize_unwraps_pyannote4_output_wrapper():
    class _Wrapped:
        speaker_diarization = _FakeAnnotation()
    turns = diarize("/tmp/a.wav", cfg=None, diarizer=lambda wav: _Wrapped())
    assert [t.speaker for t in turns] == ["SPEAKER_00", "SPEAKER_01"]
