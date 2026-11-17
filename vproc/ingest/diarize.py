import os
import re
import sys
import wave
from dataclasses import dataclass

from vproc.ingest.transcribe import TranscriptSegment, _collapse_repetition, _timestamp


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: str  # verbatim pyannote label, e.g. "SPEAKER_00"


def _load_pipeline(cfg):
    import torch  # lazy: heavy
    from pyannote.audio import Pipeline  # lazy: heavy, needs HF-gated model access

    pipe = Pipeline.from_pretrained(cfg.diarize_model, token=cfg.hf_token)
    if torch.backends.mps.is_available():
        # Unsupported MPS ops fall back to CPU instead of aborting diarization entirely.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        pipe.to(torch.device("mps"))
    return pipe


def read_wav(path: str):
    """(float32 samples in [-1, 1], sample rate) for the 16-bit mono wav ingest extracts, or
    None for anything else so the caller falls back to the file path."""
    import numpy as np

    try:
        with wave.open(path, "rb") as w:
            if w.getnchannels() != 1 or w.getsampwidth() != 2:
                return None
            rate, raw = w.getframerate(), w.readframes(w.getnframes())
    except (OSError, EOFError, wave.Error):
        return None
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0, rate


def progress_hook():
    """pyannote pipeline hook: diarization is the longest ingest stage, so report each new 10%
    of the steps that expose totals (embeddings) instead of minutes of silence."""
    reported: dict[str, int] = {}

    def hook(step_name, step_artifact, file=None, total=None, completed=None):
        if not total or completed is None:
            return
        tenth = min(10, int(10 * completed / total))  # batched steps can overshoot their total
        if tenth > reported.get(step_name, 0):
            reported[step_name] = tenth
            print(f"[ingest] speakers: {step_name} {tenth * 10}%", file=sys.stderr, flush=True)

    return hook


def _run_pipeline(pipe, wav_path: str):
    import torch  # lazy: heavy

    audio = read_wav(wav_path)
    if audio is None:
        return pipe(wav_path, hook=progress_hook())
    # Samples instead of a path: pyannote then never needs torchcodec, whose PyPI Linux wheel
    # cannot load without CUDA libraries. Same int16 -> float conversion the decoder applies.
    samples, rate = audio
    return pipe({"waveform": torch.from_numpy(samples).unsqueeze(0), "sample_rate": rate},
                hook=progress_hook())


def diarize(wav_path: str, cfg, diarizer=None) -> list[SpeakerTurn]:
    """Speaker turns for `wav_path`. `diarizer` is the injection seam: any callable
    wav_path -> annotation exposing .itertracks(yield_label=True)."""
    if diarizer is None:
        pipe = _load_pipeline(cfg)
        result = _run_pipeline(pipe, wav_path)
    else:
        result = diarizer(wav_path)
    # pyannote 4.x wraps its output (3.x doesn't) and also offers a non-overlapping variant made
    # for reconciling with a transcript: at most one speaker per instant, so a word's turn is
    # unambiguous.
    ann = getattr(result, "exclusive_speaker_diarization", None)
    if ann is None:
        ann = getattr(result, "speaker_diarization", result)
    return [SpeakerTurn(span.start, span.end, label)
            for span, _, label in ann.itertracks(yield_label=True)]


def _word_speaker(start: float, end: float, turns: list[SpeakerTurn]) -> str:
    mid = (start + end) / 2.0
    containing = [t for t in turns if t.start <= mid < t.end]
    if containing:  # overlapping speech: most overlap with the word, then the earliest turn
        return max(containing, key=lambda t: (min(end, t.end) - max(start, t.start), -t.start)).speaker
    # Outside every turn: distance to the turn's interval, not its midpoint, so a word just
    # after a long turn still belongs to it.
    return min(turns, key=lambda t: max(t.start - mid, mid - t.end, 0.0)).speaker


_SENTENCE_END = re.compile(r"[.?!][\"')\]]*$")
# A same-speaker run stops growing at about one ASR segment's length: the chunker groups whole
# segments into windows and cannot cut inside one. A run is also the unit that alignment
# places on a screen state by its midpoint, so a longer run would misplace more speech at a
# screen change. Lowering it changes every chunk's text: re-check the similarity floor.
MAX_RUN_S = 30.0
# A sentence only continues into a segment that follows closely; after a real pause the
# unfinished tail stands alone (otherwise one "sentence" could span minutes of silence).
MAX_TAIL_GAP_S = 2.0

Word = tuple[float, float, str]


def _join_words(words: list[Word]) -> str:
    """Space-joined, except that a hyphenated continuation ("time" + "-consuming") rejoins."""
    text = ""
    for _, _, token in words:
        text += token if text and re.match(r"-\w", token) else (" " if text else "") + token
    return text


def _sentences(segments: list[TranscriptSegment]) -> list[tuple[list[Word], list[TranscriptSegment]]]:
    """(words, source segments) per sentence over consecutive worded segments. ASR cuts segments
    mid-sentence, so a segment's unfinished tail joins the next segment's head, but only when
    that segment follows within MAX_TAIL_GAP_S and actually closes the sentence. A fully
    unpunctuated stream therefore never votes across segments: one unit per segment, which is
    the whole-segment rule."""
    units: list[tuple[list[Word], list[TranscriptSegment]]] = []
    tail: tuple[list[Word], list[TranscriptSegment]] = ([], [])
    for seg in segments:
        if tail[0] and (seg.words[0][0] - tail[0][-1][1] > MAX_TAIL_GAP_S
                        or not any(_SENTENCE_END.search(w[2]) for w in seg.words)):
            units.append(tail)
            tail = ([], [])
        words, sources = tail
        for word in seg.words:
            if not any(src is seg for src in sources):  # identity: equal-looking segments differ
                sources.append(seg)
            words.append(word)
            if _SENTENCE_END.search(word[2]):
                units.append((words, sources))
                words, sources = [], []
        tail = (words, sources)
    if tail[0]:
        units.append(tail)
    return units


def _attribute_sentences(segments: list[TranscriptSegment],
                         turns: list[SpeakerTurn]) -> list[TranscriptSegment]:
    """Attribute sentence by sentence: each sentence goes to the speaker of most of its word
    time, and consecutive same-speaker sentences form one output segment (up to MAX_RUN_S).
    Word-by-word labels are too jumpy on their own: diarization is often wrong for a moment (an
    overlapping "mm-hm", a last word landing in the next turn), while people rarely change
    mid-sentence. A back-channel ("Right.") is its own sentence, so it is still its own turn."""
    runs: list[tuple[str, list[Word], list[TranscriptSegment]]] = []
    for words, sources in _sentences(segments):
        spoken: dict[str, float] = {}
        for start, end, _ in words:  # insertion order breaks ties toward the earlier speaker
            speaker = _word_speaker(start, end, turns)
            spoken[speaker] = spoken.get(speaker, 0.0) + max(end - start, 0.01)
        speaker = max(spoken, key=spoken.get)
        if runs and runs[-1][0] == speaker and words[-1][1] - runs[-1][1][0][0] <= MAX_RUN_S:
            runs[-1][1].extend(words)
            runs[-1][2].extend(src for src in sources if not any(src is known for known in runs[-1][2]))
        else:
            runs.append((speaker, list(words), list(sources)))
    out: list[TranscriptSegment] = []
    for speaker, words, sources in runs:
        if len(sources) == 1 and len(words) == len(sources[0].words):
            sources[0].speaker = speaker  # an untouched ASR segment keeps its text and times
            out.append(sources[0])
        else:
            # Word tokens may lack punctuation (OpenAI-style lists) or spaces (CJK), so the ASR
            # text is kept whenever the run is made of whole segments; only a segment that was
            # actually cut is rebuilt from its words.
            whole = len(words) == sum(len(src.words) for src in sources)
            text = " ".join(src.text for src in sources) if whole else _join_words(words)
            out.append(TranscriptSegment(start=words[0][0], end=words[-1][1],
                                         text=_collapse_repetition(text), speaker=speaker, words=words))
    return out


def assign_speakers(transcript: list[TranscriptSegment],
                    turns: list[SpeakerTurn]) -> list[TranscriptSegment]:
    """Label the transcript with diarized speakers and return it. Segments with word
    timestamps are re-cut into sentences and attributed per sentence (see
    _attribute_sentences); one without is relabeled in place with the speaker overlapping it most (summed across turns; tie -> speaker whose
    overlapping turn starts earliest; zero overlap -> nearest turn by midpoint). Empty `turns`
    leaves labels untouched."""
    if not turns:
        return transcript
    # Validate the entire result before mutating any transcript labels. A corrupt
    # turn must enter the ingest's degradation/preservation path, not misattribute
    # words through NaN comparisons or leave a partly labelled transcript.
    validated: list[SpeakerTurn] = []
    for turn in turns:
        try:
            start, end = _timestamp(turn.start), _timestamp(turn.end)
            if (end < start or not isinstance(turn.speaker, str) or not turn.speaker.strip()
                    or any(ord(c) < 32 or ord(c) == 127 for c in turn.speaker)):
                raise ValueError("Invalid turn")
        except (AttributeError, TypeError, ValueError, OverflowError):
            raise ValueError("Invalid speaker turn returned by diarization") from None
        validated.append(SpeakerTurn(start, end, turn.speaker))
    turns = validated
    out: list[TranscriptSegment] = []
    worded: list[TranscriptSegment] = []  # consecutive segments that carry word timestamps
    for seg in transcript:
        if seg.words:
            worded.append(seg)
            continue
        out.extend(_attribute_sentences(worded, turns))
        worded = []
        out.append(seg)
        overlap: dict[str, float] = {}
        earliest: dict[str, float] = {}
        for t in turns:
            ov = min(seg.end, t.end) - max(seg.start, t.start)
            if ov > 0:
                overlap[t.speaker] = overlap.get(t.speaker, 0.0) + ov
                earliest[t.speaker] = min(earliest.get(t.speaker, float("inf")), t.start)
        if overlap:
            seg.speaker = max(overlap, key=lambda s: (overlap[s], -earliest[s]))
        else:
            mid = (seg.start + seg.end) / 2.0
            seg.speaker = min(turns, key=lambda t: abs((t.start + t.end) / 2.0 - mid)).speaker
    out.extend(_attribute_sentences(worded, turns))
    return out
