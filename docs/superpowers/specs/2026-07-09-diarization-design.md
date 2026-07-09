# vproc speaker diarization — design

**Date:** 2026-07-09 · **Status:** approved by owner (chat) · **Scope:** Phase 2, first slice

## Goal

Transcript segments carry distinct speaker labels (`SPEAKER_00`, `SPEAKER_01`, …) instead of
the hardcoded `SPEAKER_0`, so citations show who spoke and the existing `speaker` filter on
`/ask` and `/search` becomes meaningful.

**Out of scope** (stays deferred): named speakers from nameplate OCR (`SpeakerMap`),
word-level attribution (whisperX-style), remote/endpoint diarization.

## Approach

In-process pyannote on the already-extracted WAV, relabeling the existing
`TranscriptSegment`s. Chosen over (a) diarization-aware transcription — replaces the whole
ASR path and breaks local/remote endpoint symmetry — and (b) a custom diarization service
on voyage — new infra plus a nonstandard API for a step that runs fine locally.

## Components

### `vproc/ingest/diarize.py` (new)

```python
@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: str  # verbatim pyannote label, e.g. "SPEAKER_00"

def diarize(wav_path: str, cfg, diarizer=None) -> list[SpeakerTurn]: ...
def assign_speakers(transcript: list[TranscriptSegment], turns: list[SpeakerTurn]) -> None: ...
```

- `diarize` lazy-imports `pyannote.audio` (same pattern as HHEM / mlx-whisper), loads the
  pipeline named by `cfg.diarize_model` with `cfg.hf_token`, runs it on MPS when available
  else CPU, and returns turns. The `diarizer` parameter is the test-injection seam
  (a callable `wav_path -> list[SpeakerTurn]`-shaped output or the pyannote pipeline itself).
- `assign_speakers` is pure relabeling, mutating `seg.speaker` in place:
  - each segment gets the speaker with the greatest summed temporal overlap with
    `[seg.start, seg.end]`;
  - ties break to the speaker whose overlapping turn starts earliest;
  - a segment with zero overlap takes the turn whose midpoint is nearest the segment's
    midpoint;
  - empty `turns` leaves all labels untouched.

### Config

- `diarize_model: str = "pyannote/speaker-diarization-community-1"` — env
  `VPROC_DIARIZE_MODEL`; **empty string disables diarization** (pipeline skips the step,
  labels stay `SPEAKER_0`). Model is configurable per the project convention; execution
  location is in-process-local in v1, like HHEM.
- Auth reuses the existing `hf_token` field (`HF_TOKEN` env). The pipeline model is
  HF-gated: the owner must accept conditions on the model page and put a read token in
  `.env` (one-time).

### Pipeline integration (`vproc/ingest/pipeline.py`)

Immediately after `transcript = T.transcribe(...)`, when a wav exists, transcript is
non-empty, and `cfg.diarize_model` is set:

```python
turns = D.diarize(wav_path, cfg)
D.assign_speakers(transcript, turns)
```

Best-effort like OCR: any exception (missing/invalid token, model load failure, inference
error) prints a stderr warning and continues with unlabeled transcript. Speaker labels are
never worth losing an ingest over. A misconfigured token therefore degrades visibly-but-
softly; the warning includes the exception text so the cause is diagnosable.

Everything downstream already works unchanged: `align._subsplit` splits chunks on speaker
change, `Segment.speaker` flows into code-owned citations (`[title · mm:ss · SPEAKER_01]`),
the store schema already has `speaker`, and `build_where` already filters on it.

## Dependencies

`pyannote.audio` added as a project dependency (resolved cleanly against Python 3.14 /
torch 2.12 in a dry run; ~1 GB of extras, mostly torchaudio/torchcodec). Preferred pipeline
`pyannote/speaker-diarization-community-1`; if the installed `pyannote.audio` major version
can't load it, fall back to `pyannote/speaker-diarization-3.1` — decided empirically at
implementation, recorded in the plan.

## Testing

Fully offline except the last item:

1. `assign_speakers` unit tests: majority-overlap, tie-break, zero-overlap nearest-turn,
   empty-turns no-op.
2. `diarize` with injected fake diarizer: turn extraction/shape, no pyannote import needed.
3. Pipeline threading test: fake diarizer labels reach stored rows; disabled
   (`diarize_model=""`) path skips cleanly; diarizer exception degrades with warning and
   ingest still stores the transcript.
4. E2E on the real 22.5-min meeting (requires owner's `HF_TOKEN`): distinct `SPEAKER_xx`
   labels in stored rows, speaker filter round-trip through `/search`.

## Performance

Expected ~1–3 min added per 22-min video on the M5 Max (MPS); acceptable for v1. If it
proves slower on CPU-only hosts, that's a tuning concern for later, not this slice.

## Risks

- **Gated model friction:** no `HF_TOKEN` yet in the owner's `.env`; e2e verification
  blocks on the one-time token dance. Implementation and offline tests do not.
- **Version skew:** pyannote 4.x vs 3.x pipeline-name compatibility — handled by the
  empirical fallback above.
- **Segment-granularity attribution:** a single whisper segment spanning a speaker change
  gets one label (majority overlap). Word-level attribution is the known sharper tool,
  deliberately deferred.
