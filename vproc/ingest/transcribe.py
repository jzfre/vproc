import re
import sys
from dataclasses import dataclass

from vproc.llm import client

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: str = "SPEAKER_0"
    # (start, end, word) when the ASR backend returned word timestamps: lets diarization re-cut
    # segments into sentences and give each sentence to the speaker of most of it.
    words: list[tuple[float, float, str]] | None = None


def extract_audio_cmd(video: str, out_wav: str) -> list[str]:
    # -nostdin so a backgrounded ingest isn't stopped by SIGTTIN reading the TTY.
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", video,
            "-ac", "1", "-ar", "16000", "-y", out_wav]


def _collapse_repetition(text: str, threshold: int = 4) -> str:
    """Collapse a word repeated >= `threshold` times in a row down to a single
    occurrence. Whisper silence-hallucinations loop a token dozens of times
    ('struggle struggle ...'); natural speech ('no no no') stays under the threshold."""
    words = text.split()
    out: list[str] = []
    i = 0
    while i < len(words):
        j = i
        while j < len(words) and words[j].lower() == words[i].lower():
            j += 1
        out.append(words[i]) if j - i >= threshold else out.extend(words[i:j])
        i = j
    return " ".join(out)


def squash(text: str) -> str:
    """Letters and digits only, for comparing texts whose spacing/punctuation may differ."""
    return re.sub(r"[\W_]+", "", text.lower())


def _words(raw, text: str) -> list[tuple[float, float, str]] | None:
    """Word timestamps, only if they spell the segment: misaligned words would hand text to
    the wrong speaker, and whole-segment attribution is the safer fallback."""
    try:
        words = [(float(w["start"]), float(w["end"]), str(w["word"]).strip()) for w in raw or []]
    except (KeyError, TypeError, ValueError):
        return None
    words = [w for w in words if w[2]]
    if not words or squash("".join(w[2] for w in words)) != squash(text):
        return None
    return words


def segments_from_whisper(result: dict) -> list[TranscriptSegment]:
    out: list[TranscriptSegment] = []
    for s in result.get("segments", []):
        raw_text = (s.get("text") or "").strip()
        text = _collapse_repetition(raw_text)
        if text:
            out.append(TranscriptSegment(start=float(s["start"]), end=float(s["end"]), text=text,
                                         words=_words(s.get("words"), raw_text)))
    return out


def _transcribe_local(audio_path: str, model: str, transcriber=None) -> dict:
    if transcriber is None:
        import mlx_whisper  # lazy: heavy, Mac-only

        transcriber = mlx_whisper.transcribe
    # Anti-hallucination: condition_on_previous_text=False stops the repetition cascade
    # ("Ministry Ministry ...") on silence; the thresholds drop low-confidence / repetitive
    # decodes instead of emitting invented speech.
    def run(word_timestamps: bool) -> dict:
        return transcriber(
            audio_path,
            path_or_hf_repo=model,
            word_timestamps=word_timestamps,
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
        )

    try:
        return run(True)  # word times let diarization attribute speakers per sentence
    except Exception as e:
        # Word times improve speaker attribution; they are never worth losing the transcript.
        print(f"warning: transcription with word timestamps failed ({type(e).__name__}: {e}); "
              "retrying without them", file=sys.stderr)
        return run(False)


def transcribe(audio_path: str, endpoint=None, transcriber=None, audio_client=None) -> list[TranscriptSegment]:
    """Transcribe audio. `endpoint` (a config.Endpoint) selects the backend: an empty
    base_url runs local in-process whisper; a base_url uses an OpenAI-compatible ASR
    endpoint. Both paths funnel through segments_from_whisper (so the repetition guard
    applies regardless of backend)."""
    from vproc.config import Endpoint

    if endpoint is None:
        endpoint = Endpoint("", DEFAULT_MODEL)
    if endpoint.base_url:
        result = (audio_client or client.transcribe_audio)(endpoint.base_url, endpoint.model, audio_path)
    else:
        result = _transcribe_local(audio_path, endpoint.model, transcriber)
    return segments_from_whisper(result)
