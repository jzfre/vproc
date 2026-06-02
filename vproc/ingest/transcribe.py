from dataclasses import dataclass

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: str = "SPEAKER_0"


def extract_audio_cmd(video: str, out_wav: str) -> list[str]:
    return ["ffmpeg", "-hide_banner", "-i", video, "-ac", "1", "-ar", "16000", "-y", out_wav]


def segments_from_whisper(result: dict) -> list[TranscriptSegment]:
    out: list[TranscriptSegment] = []
    for s in result.get("segments", []):
        text = (s.get("text") or "").strip()
        if text:
            out.append(TranscriptSegment(start=float(s["start"]), end=float(s["end"]), text=text))
    return out


def transcribe(audio_path: str, model: str = DEFAULT_MODEL) -> list[TranscriptSegment]:
    import mlx_whisper  # lazy: heavy, Mac-only

    result = mlx_whisper.transcribe(audio_path, path_or_hf_repo=model, word_timestamps=False)
    return segments_from_whisper(result)
