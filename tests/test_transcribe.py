from vproc.ingest.transcribe import segments_from_whisper, extract_audio_cmd, TranscriptSegment

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
