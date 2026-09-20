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


def _wseg(start, end, words):
    """A segment carrying word timestamps: words = [(start, end, text), ...]."""
    return TranscriptSegment(start=start, end=end, text=" ".join(w[2] for w in words), words=words)


def test_assign_speakers_returns_segments_and_still_labels_in_place():
    t = [_seg(0.0, 4.0)]
    out = assign_speakers(t, [SpeakerTurn(0.0, 4.0, "SPEAKER_00")])
    assert out == t and out[0] is t[0] and t[0].speaker == "SPEAKER_00"


def test_words_split_a_segment_at_the_speaker_change():
    # One ASR segment straddling a turn change used to put B's words under A's name, and the
    # faithfulness gate then confirmed the false "A said ..." claim. Words decide instead.
    seg = _wseg(0.0, 6.0, [(0.0, 1.0, "deploy"), (1.0, 2.0, "everything"), (2.0, 2.8, "daily."),
                           (3.2, 4.0, "That's"), (4.0, 5.0, "too"), (5.0, 6.0, "slow.")])
    out = assign_speakers([seg], [SpeakerTurn(0.0, 3.0, "SPEAKER_00"),
                                  SpeakerTurn(3.0, 6.0, "SPEAKER_01")])
    assert [(s.speaker, s.text, s.start, s.end) for s in out] == [
        ("SPEAKER_00", "deploy everything daily.", 0.0, 2.8),
        ("SPEAKER_01", "That's too slow.", 3.2, 6.0),
    ]
    assert out[1].words == [(3.2, 4.0, "That's"), (4.0, 5.0, "too"), (5.0, 6.0, "slow.")]


def test_backchannel_sentence_inside_a_segment_becomes_its_own_turn():
    seg = _wseg(0.0, 5.0, [(0.0, 1.0, "So"), (1.0, 2.0, "batched."), (2.1, 2.4, "Right."),
                           (2.6, 3.5, "Exactly,"), (3.5, 5.0, "yes.")])
    out = assign_speakers([seg], [SpeakerTurn(0.0, 2.05, "SPEAKER_00"),
                                  SpeakerTurn(2.05, 2.5, "SPEAKER_01"),
                                  SpeakerTurn(2.5, 5.0, "SPEAKER_00")])
    assert [(s.speaker, s.text) for s in out] == [
        ("SPEAKER_00", "So batched."), ("SPEAKER_01", "Right."), ("SPEAKER_00", "Exactly, yes.")]


def test_diarization_noise_inside_a_sentence_does_not_split_it():
    # Observed on a real meeting: 2.8 s in the middle of one person's sentence were diarized as
    # someone else (an overlapping "mm-hm"), and word-by-word labels cut the sentence in three.
    # People change mid-sentence far less often than diarization is wrong for a moment.
    words = [(0.0, 1.0, "we"), (1.0, 2.0, "don't"), (2.0, 3.0, "need"), (3.0, 4.0, "to"),
             (4.0, 5.0, "deploy"), (5.0, 6.0, "everything"), (6.0, 7.0, "every"), (7.0, 8.0, "single"),
             (8.0, 9.0, "day.")]
    seg = _wseg(0.0, 9.0, words)
    out = assign_speakers([seg], [SpeakerTurn(0.0, 4.0, "SPEAKER_00"),
                                  SpeakerTurn(4.0, 6.0, "SPEAKER_03"),   # noise
                                  SpeakerTurn(6.0, 9.0, "SPEAKER_00")])
    assert out == [seg] and seg.speaker == "SPEAKER_00"


def test_a_sentence_goes_to_whoever_spoke_most_of_it():
    # Word times jitter by a few hundred ms, so the last word of a sentence often lands in the
    # next speaker's turn. The sentence stays whole; the reply starts at its own first word.
    seg = _wseg(0.0, 6.0, [(0.0, 1.0, "Too"), (1.0, 2.0, "much"), (2.0, 2.9, "time"),
                           (2.9, 3.3, "consuming."), (3.4, 4.5, "Right,"), (4.5, 6.0, "agreed.")])
    out = assign_speakers([seg], [SpeakerTurn(0.0, 2.95, "SPEAKER_00"),
                                  SpeakerTurn(2.95, 6.0, "SPEAKER_01")])
    assert [(s.speaker, s.text, s.start, s.end) for s in out] == [
        ("SPEAKER_00", "Too much time consuming.", 0.0, 3.3),
        ("SPEAKER_01", "Right, agreed.", 3.4, 6.0)]


def test_unpunctuated_words_fall_back_to_one_vote_for_the_whole_segment():
    # No sentence ends to anchor on (an ASR backend without punctuation): same outcome as
    # whole-segment attribution, never a word-by-word split.
    seg = _wseg(0.0, 6.0, [(0.0, 1.0, "deploy"), (1.0, 2.0, "everything"), (2.0, 3.0, "daily"),
                           (3.0, 4.0, "thats"), (4.0, 5.0, "too"), (5.0, 5.5, "slow")])
    out = assign_speakers([seg], [SpeakerTurn(0.0, 3.0, "SPEAKER_00"),
                                  SpeakerTurn(3.0, 6.0, "SPEAKER_01")])
    assert out == [seg] and seg.speaker == "SPEAKER_00"  # 3.0 s of words vs 2.5 s


def test_word_outside_every_turn_joins_the_nearest_turn_edge():
    # Distance to the turn INTERVAL, not its midpoint: a word just after a long turn belongs
    # to it, even though a short later turn has the nearer midpoint.
    seg = _wseg(0.0, 21.0, [(1.0, 2.0, "long"), (20.2, 20.6, "tail")])
    out = assign_speakers([seg], [SpeakerTurn(0.0, 20.0, "SPEAKER_00"),
                                  SpeakerTurn(22.0, 23.0, "SPEAKER_01")])
    assert [(s.speaker, s.text) for s in out] == [("SPEAKER_00", "long tail")]


def test_single_speaker_segment_with_words_is_kept_whole():
    seg = _wseg(0.0, 2.0, [(0.0, 1.0, "hello"), (1.0, 2.0, "there")])
    out = assign_speakers([seg], [SpeakerTurn(0.0, 5.0, "SPEAKER_02")])
    assert out == [seg] and seg.speaker == "SPEAKER_02" and (seg.start, seg.end) == (0.0, 2.0)


def test_split_runs_collapse_asr_repetition_loops():
    words = ([(float(i), float(i) + 1, "struggle") for i in range(6)]
             + [(6.0, 7.0, "done."), (7.0, 8.0, "Ok.")])
    out = assign_speakers([_wseg(0.0, 8.0, words)], [SpeakerTurn(0.0, 6.9, "SPEAKER_00"),
                                                     SpeakerTurn(6.9, 8.0, "SPEAKER_01")])
    assert [(s.speaker, s.text) for s in out] == [("SPEAKER_00", "struggle done."), ("SPEAKER_01", "Ok.")]


def test_diarize_prefers_exclusive_turns_for_transcript_alignment():
    # pyannote 4 also returns a non-overlapping variant built for reconciling with a
    # transcript: every instant has at most one speaker, so a word's turn is unambiguous.
    class _Exclusive:
        def itertracks(self, yield_label=False):
            yield _Span(0.0, 5.0), None, "SPEAKER_07"

    class _Wrapped:
        speaker_diarization = _FakeAnnotation()
        exclusive_speaker_diarization = _Exclusive()

    turns = diarize("/tmp/a.wav", cfg=None, diarizer=lambda wav: _Wrapped())
    assert turns == [SpeakerTurn(0.0, 5.0, "SPEAKER_07")]


def _pcm_wav(path, samples, rate=16000, channels=1, width=2):
    import struct
    import wave

    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels); w.setsampwidth(width); w.setframerate(rate)
        w.writeframes(struct.pack(f"<{len(samples)}h", *samples) if width == 2 else bytes(samples))


def test_read_wav_gives_unit_range_float32_for_the_ingest_wav(tmp_path):
    # vproc writes this wav itself (16 kHz mono s16). Handing pyannote samples instead of a
    # path keeps diarization working when torchcodec (its file decoder) cannot load.
    from vproc.ingest.diarize import read_wav

    wav = tmp_path / "a.wav"
    _pcm_wav(wav, [0, 16384, -32768])
    samples, rate = read_wav(str(wav))
    assert rate == 16000 and str(samples.dtype) == "float32"
    assert samples.tolist() == [0.0, 0.5, -1.0]


def test_read_wav_declines_formats_it_does_not_understand(tmp_path):
    from vproc.ingest.diarize import read_wav

    stereo = tmp_path / "stereo.wav"
    _pcm_wav(stereo, [0, 0, 1, 1], channels=2)
    eight_bit = tmp_path / "u8.wav"
    _pcm_wav(eight_bit, [128, 129], width=1)
    assert read_wav(str(stereo)) is None and read_wav(str(eight_bit)) is None
    assert read_wav(str(tmp_path / "missing.wav")) is None  # caller falls back to the path


def test_diarization_progress_prints_each_new_ten_percent_once(capsys):
    from vproc.ingest.diarize import progress_hook

    hook = progress_hook()
    hook("segmentation", object())                       # no totals: nothing to report
    for done in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 24):  # pyannote can overshoot its total
        hook("embeddings", None, total=20, completed=done)
    lines = capsys.readouterr().err.splitlines()
    assert lines == ["[ingest] speakers: embeddings 10%", "[ingest] speakers: embeddings 20%",
                     "[ingest] speakers: embeddings 30%", "[ingest] speakers: embeddings 40%",
                     "[ingest] speakers: embeddings 50%", "[ingest] speakers: embeddings 100%"]


def test_a_sentence_continues_across_an_asr_segment_boundary():
    # Whisper cuts segments mid-sentence. Observed: a sentence split over two segments, with
    # 2.8 s of the second half diarized as someone else. Voted per segment, the halves went to
    # different people; voted as one sentence, it is one person.
    first = _wseg(0.0, 6.0, [(0.0, 2.0, "because"), (2.0, 4.0, "we"), (4.0, 6.0, "needn't")])
    second = _wseg(6.0, 12.0, [(6.0, 7.5, "deploy"), (7.5, 9.0, "everything."), (9.5, 12.0, "Right.")])
    out = assign_speakers([first, second], [SpeakerTurn(0.0, 6.0, "SPEAKER_00"),
                                            SpeakerTurn(6.0, 9.2, "SPEAKER_03"),    # noise
                                            SpeakerTurn(9.2, 12.0, "SPEAKER_01")])
    assert [(s.speaker, s.text, s.start, s.end) for s in out] == [
        ("SPEAKER_00", "because we needn't deploy everything.", 0.0, 9.0),
        ("SPEAKER_01", "Right.", 9.5, 12.0)]


def test_unpunctuated_segments_are_never_merged_into_one_vote():
    # Without sentence ends nothing says where a thought stops, so an unfinished tail is only
    # carried into a segment that actually closes it. Otherwise: one vote per segment, as before.
    first = _wseg(0.0, 4.0, [(0.0, 2.0, "so"), (2.0, 4.0, "batched")])
    second = _wseg(4.0, 8.0, [(4.0, 6.0, "yes"), (6.0, 8.0, "exactly")])
    out = assign_speakers([first, second], [SpeakerTurn(0.0, 4.0, "SPEAKER_00"),
                                            SpeakerTurn(4.0, 8.0, "SPEAKER_01")])
    assert out == [first, second]
    assert [s.speaker for s in out] == ["SPEAKER_00", "SPEAKER_01"]


def test_long_monologue_is_emitted_in_bounded_runs_that_end_on_sentences():
    # The chunker groups whole segments into <=90 s windows and cannot cut inside one, so a
    # same-speaker run stops growing at about a Whisper segment's length.
    words = []
    for i in range(8):  # eight 10-second sentences by one speaker
        words += [(i * 10.0, i * 10.0 + 5, f"part{i}"), (i * 10.0 + 5, i * 10.0 + 10, "done.")]
    out = assign_speakers([_wseg(0.0, 80.0, words)], [SpeakerTurn(0.0, 80.0, "SPEAKER_00")])
    assert [(s.start, s.end) for s in out] == [(0.0, 30.0), (30.0, 60.0), (60.0, 80.0)]
    assert all(s.text.endswith("done.") and s.speaker == "SPEAKER_00" for s in out)


def test_segments_without_words_keep_their_place_between_worded_ones():
    a = _wseg(0.0, 2.0, [(0.0, 2.0, "Hello.")])
    b = _seg(2.0, 4.0)                                   # no words: whole-segment rule
    c = _wseg(4.0, 6.0, [(4.0, 6.0, "Bye.")])
    out = assign_speakers([a, b, c], [SpeakerTurn(0.0, 2.0, "SPEAKER_00"),
                                      SpeakerTurn(2.0, 4.0, "SPEAKER_01"),
                                      SpeakerTurn(4.0, 6.0, "SPEAKER_00")])
    assert out == [a, b, c] and [s.speaker for s in out] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]


def test_a_tail_is_not_carried_across_a_long_pause():
    # Review finding: "Hello. So um" ... five minutes of silence ... "Okay, go." became one
    # 304 s segment with one speaker. A sentence only continues into a segment that follows
    # closely; after a real pause the tail stands alone.
    first = _wseg(7.0, 10.0, [(7.0, 8.0, "Hello."), (8.0, 9.0, "So"), (9.0, 10.0, "um")])
    second = _wseg(310.0, 314.0, [(310.0, 311.0, "Okay,"), (311.0, 312.0, "go."), (312.5, 314.0, "Next.")])
    out = assign_speakers([first, second], [SpeakerTurn(7.0, 10.0, "SPEAKER_00"),
                                            SpeakerTurn(310.0, 314.0, "SPEAKER_01")])
    assert [(s.speaker, s.text, s.start, s.end) for s in out] == [
        ("SPEAKER_00", "Hello. So um", 7.0, 10.0), ("SPEAKER_01", "Okay, go. Next.", 310.0, 314.0)]


def test_whole_segments_keep_their_asr_text_when_merged_into_one_run():
    # Review finding: OpenAI-style word lists carry no punctuation (and CJK words are single
    # characters), so rebuilding text from words destroyed it even though nothing was split.
    first = TranscriptSegment(0.0, 2.0, "Hello there. How are you?", words=[
        (0.0, 0.5, "Hello"), (0.5, 1.0, "there"), (1.0, 1.3, "How"), (1.3, 1.6, "are"), (1.6, 2.0, "you")])
    second = TranscriptSegment(2.0, 4.0, "今日は会議。", words=[(2.0, 3.0, "今日は"), (3.0, 4.0, "会議")])
    out = assign_speakers([first, second], [SpeakerTurn(0.0, 4.0, "SPEAKER_00")])
    assert [s.text for s in out] == ["Hello there. How are you? 今日は会議。"]
    assert (out[0].start, out[0].end, out[0].speaker) == (0.0, 4.0, "SPEAKER_00")


def test_rebuilt_text_rejoins_hyphenated_words():
    # Whisper times "time-consuming" as two words, "time" and "-consuming"; a cut segment is
    # rebuilt from its words and must not read "time -consuming".
    seg = _wseg(0.0, 5.0, [(0.0, 1.0, "Too"), (1.0, 2.0, "time"), (2.0, 3.0, "-consuming."),
                           (3.5, 5.0, "Right.")])
    seg.text = "Too time-consuming. Right."
    out = assign_speakers([seg], [SpeakerTurn(0.0, 3.2, "SPEAKER_00"), SpeakerTurn(3.2, 5.0, "SPEAKER_01")])
    assert [(s.speaker, s.text) for s in out] == [("SPEAKER_00", "Too time-consuming."), ("SPEAKER_01", "Right.")]
