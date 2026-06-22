from vproc.ingest.transcribe import (
    segments_from_whisper, extract_audio_cmd, transcribe, TranscriptSegment,
)

def test_extract_audio_cmd():
    cmd = extract_audio_cmd("in.mp4", "/o/a.wav")
    assert cmd[:1] == ["ffmpeg"]
    assert "-ar" in cmd and "16000" in cmd and cmd[-1] == "/o/a.wav"

def test_segments_from_whisper_filters_empty_and_strips():
    result = {"segments": [
        {"start": 0.0, "end": 1.0, "text": " hello "},
        {"start": 1.0, "end": 2.0, "text": "   "},
        {"start": 2.0, "end": 3.0, "text": "world"},
    ]}
    segs = segments_from_whisper(result)
    assert segs == [
        TranscriptSegment(0.0, 1.0, "hello"),
        TranscriptSegment(2.0, 3.0, "world"),
    ]
    assert segs[0].speaker == "SPEAKER_0"

def test_segments_from_whisper_collapses_repetition_loop():
    # Whisper silence-hallucination: a content word looped 20+ times.
    result = {"segments": [
        {"start": 0.0, "end": 5.0, "text": "and that's hard " + "struggle " * 20},
    ]}
    segs = segments_from_whisper(result)
    assert segs[0].text.lower().split().count("struggle") == 1
    assert segs[0].text.startswith("and that's hard")

def test_segments_from_whisper_drops_segment_that_is_pure_repetition():
    result = {"segments": [{"start": 0.0, "end": 5.0, "text": "the the the the the the"}]}
    segs = segments_from_whisper(result)
    # collapses to a single "the" — not dropped, but no longer a loop
    assert segs[0].text == "the"

def test_segments_from_whisper_keeps_natural_short_repetition():
    result = {"segments": [{"start": 0.0, "end": 1.0, "text": "no no no"}]}
    segs = segments_from_whisper(result)
    assert segs[0].text == "no no no"

def test_transcribe_disables_conditioning_to_curb_hallucination():
    captured = {}
    def fake(audio, path_or_hf_repo, **kw):
        captured["audio"] = audio
        captured["model"] = path_or_hf_repo
        captured.update(kw)
        return {"segments": [{"start": 0.0, "end": 1.0, "text": "hi"}]}
    out = transcribe("a.wav", transcriber=fake)
    # the load-bearing anti-hallucination flag (stops the "Ministry Ministry" repetition loop)
    assert captured["condition_on_previous_text"] is False
    assert captured["audio"] == "a.wav"
    assert out == [TranscriptSegment(0.0, 1.0, "hi")]
