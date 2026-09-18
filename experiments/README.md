# experiments/ — on-screen transcript ingestion (prototype)

Exploratory work toward a more reliable ingestion path for **meeting recordings
that already contain a rendered, speaker-labeled transcript on screen** (e.g. a
Microsoft Teams "recap" page, Zoom/Meet transcript panels). These are screen
recordings where the platform's own transcript — speaker name + timestamp + exact
text — is visible in a side panel.

For such videos, that on-screen transcript is **ground truth** and beats the
default audio pipeline (Whisper ASR + pyannote diarization) on every axis:
exact words, correct speaker attribution, correct timing, and no speaker merging.

This directory contains a **standalone prototype** — it does not modify the vproc
pipeline yet.

## What's here

- `panel_reconstruct.py` — reconstructs a full `(speaker, start_ts, text)` transcript
  from a video by OCR'ing the on-screen transcript panel across sampled frames and
  deduplicating entries by `(speaker, time)`.

## Approach

1. **Sample frames** sparsely (default every 10s). The panel scrolls slowly with
   playback, and each frame shows several entries, so dense per-second sampling is
   unnecessary and wasteful — sparse sampling + dedup is the right model.
2. **Multi-region OCR.** For each frame, OCR several right-biased vertical strips
   (transcript panels dock on the right). Keep whichever regions return valid
   `name + M:SS + text` entries. This is robust to the app window being **moved or
   resized mid-recording** — a fixed crop is not.
3. **Dedup** across all frames/regions by `(speaker_lower, time_seconds)`, keeping
   the longest text seen for each entry (frames catch entries at different scroll
   positions / completeness).
4. **Vision model:** an OpenAI-compatible endpoint (tested with Ollama
   `qwen3.5:9b` and `qwen3-vl:4b`). Uses `enable_thinking: false` + a JSON schema +
   a generous token budget + one retry, to avoid a known qwen failure mode where the
   hidden reasoning channel consumes the whole budget and returns empty content.

## Findings (validated on a real 20-min Teams recap recording)

- **Accuracy:** reconstruction produced a clean, correctly-attributed 2-speaker
  transcript where the audio pipeline had **merged two people into one speaker**.
- **Robustness:** the recording's window was resized/moved partway through, which
  broke a fixed crop (0 entries in that stretch). Multi-region OCR **recovered**
  those entries.
- **Model tradeoff:** `qwen3-vl:4b` is ~2x faster than `qwen3.5:9b` but brittle on
  dense/awkward crops (empty/runaway output). `qwen3.5:9b` is slower but robust.
  On a capable GPU, prefer the robust model — ingestion is a background job.
- **Speed:** OCR dominates cost. On Apple Silicon (M-series) it's ~25-50s per
  region-crop, which is slow; on a discrete GPU it is far faster. Sampling density
  and model are configurable via env (`SAMPLE_EVERY`, `OCR_MODEL`).

## Usage

Requires an Ollama (or other OpenAI-compatible) vision endpoint on
`http://localhost:11434/v1` with a vision model pulled.

```sh
SAMPLE_EVERY=10 OCR_MODEL=qwen3.5:9b \
  python experiments/panel_reconstruct.py "/path/to/recording.mov" > transcript.json
```

Output: JSON array of `{speaker, start_ts, said_text}`, ordered by time.

## TODO — path to a generic, integrated feature

The goal is a **generic reliable ingestion** layer that detects the best available
signal per video and prefers on-screen transcripts when present, falling back to
the audio pipeline otherwise.

- [ ] **Layout detection.** Probe ~5 frames spread across the video; if >= 2 yield
      valid transcript entries, treat as a panel video (run panel reconstruction),
      else fall back to Whisper + pyannote. (Reuses the multi-region OCR — no extra
      detector needed.)
- [ ] **Close coverage gaps.** Add retry/denser sampling around stretches where no
      entries are found, to avoid dropping entries during fast exchanges or window
      changes.
- [ ] **Visual active-speaker path** for recordings WITHOUT an on-screen transcript
      (raw webcam/gallery view): detect the active speaker cheaply (highlighted tile
      / mic indicator / mouth movement via classical CV — NOT a per-frame vision LLM,
      which is far too slow) and use it to attribute Whisper segments.
- [ ] **Strategy selection in the pipeline.** Wire detection + the three strategies
      (on-screen transcript / visual active-speaker / audio-only) into
      `vproc/ingest/pipeline.py`, keeping audio as the timing/fallback backbone.
- [ ] **Generalize panel layouts** beyond Teams recap (Zoom, Meet) — the
      multi-region approach should extend with tuned/region presets per source.
- [ ] **Timestamp mapping.** The panel's `M:SS` is meeting time; confirm it maps to
      the recording's playback time (they matched 1:1 in the tested recording).
- [ ] **Text quality.** Consider light post-processing (dedupe repeated filler,
      merge consecutive same-speaker entries) before indexing.

## Related

- `vproc analyze <memory> "<question>"` (added to the CLI) answers freeform
  questions over a whole indexed transcript (summaries, action items, topics) —
  complementary to the grounded, retrieval-based `/ask`.
