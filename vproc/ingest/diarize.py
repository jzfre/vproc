import os
from dataclasses import dataclass

from vproc.ingest.transcribe import TranscriptSegment


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


def diarize(wav_path: str, cfg, diarizer=None) -> list[SpeakerTurn]:
    """Speaker turns for `wav_path`. `diarizer` is the injection seam: any callable
    wav_path -> annotation exposing .itertracks(yield_label=True)."""
    if diarizer is None:
        diarizer = _load_pipeline(cfg)
    result = diarizer(wav_path)
    ann = getattr(result, "speaker_diarization", result)  # pyannote 4.x wraps, 3.x doesn't
    return [SpeakerTurn(span.start, span.end, label)
            for span, _, label in ann.itertracks(yield_label=True)]


def assign_speakers(transcript: list[TranscriptSegment], turns: list[SpeakerTurn]) -> None:
    """Relabel each segment with the speaker overlapping it most (summed across turns;
    tie -> speaker whose overlapping turn starts earliest; zero overlap -> nearest turn
    by midpoint). Empty `turns` leaves labels untouched."""
    if not turns:
        return
    for seg in transcript:
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
