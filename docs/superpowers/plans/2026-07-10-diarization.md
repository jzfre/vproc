# Speaker Diarization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transcript segments carry real speaker labels (`SPEAKER_00`, `SPEAKER_01`, …) from pyannote diarization instead of the hardcoded `SPEAKER_0`.

**Architecture:** A new `vproc/ingest/diarize.py` module runs pyannote in-process on the already-extracted WAV and relabels the existing `TranscriptSegment`s via a pure `assign_speakers` function. The pipeline calls it best-effort (like OCR) right after transcription; everything downstream (align sub-split on speaker change, citations, store schema, speaker filters) already handles speakers.

**Tech Stack:** Python 3.14 via `uv`, pyannote.audio (lazy-imported), torch 2.12 (already installed, MPS available), pytest (offline — every model/IO call injected with a fake).

**Spec:** `docs/superpowers/specs/2026-07-09-diarization-design.md`

## Global Constraints

- Run everything from `/Users/jzfre/Code/personal/vproc`; tests via `uv run pytest …`.
- Tests must be fully offline: no network, no model downloads, no pyannote import at module scope (lazy import inside the loader only, same pattern as HHEM / mlx_whisper).
- Match the existing compact code style; comments only for constraints code can't express.
- Default diarization model: `pyannote/speaker-diarization-community-1`; env var `VPROC_DIARIZE_MODEL`; empty string disables diarization.
- Speaker labels stored verbatim as pyannote emits them (`SPEAKER_00`, …). The undiarized default stays `SPEAKER_0`.
- Commit messages end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- Do NOT run `vproc ingest` against real endpoints in any task except Task 5.

---

### Task 1: Config field `diarize_model`

**Files:**
- Modify: `vproc/config.py` (Config dataclass ~line 41-60, `load_config()` ~line 68-86)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Config.diarize_model: str` (default `"pyannote/speaker-diarization-community-1"`), env `VPROC_DIARIZE_MODEL`, empty string = disabled. Tasks 2/4/5 rely on `cfg.diarize_model` and the existing `cfg.hf_token`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_config.py`:

```python
def test_diarize_model_default_and_disable(monkeypatch):
    for k in list(os.environ):
        if k.startswith("VPROC_"):
            monkeypatch.delenv(k, raising=False)
    assert load_config().diarize_model == "pyannote/speaker-diarization-community-1"
    monkeypatch.setenv("VPROC_DIARIZE_MODEL", "")   # empty string disables diarization
    assert load_config().diarize_model == ""
    monkeypatch.setenv("VPROC_DIARIZE_MODEL", "pyannote/speaker-diarization-3.1")
    assert load_config().diarize_model == "pyannote/speaker-diarization-3.1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::test_diarize_model_default_and_disable -q`
Expected: FAIL with `AttributeError: 'Config' object has no attribute 'diarize_model'`

- [ ] **Step 3: Write minimal implementation** — in `vproc/config.py`:

Add a module constant next to `DEFAULT_HHEM_MODEL` (top of file):

```python
DEFAULT_DIARIZE_MODEL = "pyannote/speaker-diarization-community-1"
```

Add the field at the END of the `Config` dataclass (after `grounding_thinking`; all trailing fields have defaults so positional construction in existing tests keeps working):

```python
    # Speaker diarization pipeline (HF-gated; uses hf_token). Empty string = disabled.
    diarize_model: str = DEFAULT_DIARIZE_MODEL
```

Add to `load_config()` (after the `grounding_thinking=` line):

```python
        diarize_model=os.environ.get("VPROC_DIARIZE_MODEL", DEFAULT_DIARIZE_MODEL),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add vproc/config.py tests/test_config.py
git commit -m "feat(config): VPROC_DIARIZE_MODEL (empty string disables diarization)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: `vproc/ingest/diarize.py` — SpeakerTurn, assign_speakers, diarize

**Files:**
- Create: `vproc/ingest/diarize.py`
- Test: `tests/test_diarize.py` (new)

**Interfaces:**
- Consumes: `TranscriptSegment` from `vproc/ingest/transcribe.py` (fields: `start: float, end: float, text: str, speaker: str = "SPEAKER_0"`); `cfg.diarize_model` / `cfg.hf_token` from Task 1.
- Produces (Tasks 4/5 rely on these exact names):
  - `SpeakerTurn` dataclass: `start: float, end: float, speaker: str`
  - `diarize(wav_path: str, cfg, diarizer=None) -> list[SpeakerTurn]` — `diarizer` is the injection seam: any callable `wav_path -> annotation` where annotation has `.itertracks(yield_label=True)` yielding `(segment_with_start_end, _, label)`.
  - `assign_speakers(transcript: list[TranscriptSegment], turns: list[SpeakerTurn]) -> None` — mutates `seg.speaker` in place.

- [ ] **Step 1: Write the failing tests** — create `tests/test_diarize.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_diarize.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'vproc.ingest.diarize'`

- [ ] **Step 3: Write the implementation** — create `vproc/ingest/diarize.py`:

```python
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
                earliest[t.speaker] = min(earliest.get(t.speaker, t.start), t.start)
        if overlap:
            seg.speaker = max(overlap, key=lambda s: (overlap[s], -earliest[s]))
        else:
            mid = (seg.start + seg.end) / 2.0
            seg.speaker = min(turns, key=lambda t: abs((t.start + t.end) / 2.0 - mid)).speaker
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_diarize.py -q`
Expected: 8 passed (pyannote is NOT installed yet — module must import without it)

- [ ] **Step 5: Run the whole suite** (guards against accidental heavy imports)

Run: `uv run pytest -q`
Expected: all pass, wall time similar to before (~2s)

- [ ] **Step 6: Commit**

```bash
git add vproc/ingest/diarize.py tests/test_diarize.py
git commit -m "feat(diarize): SpeakerTurn + assign_speakers + pyannote diarize with injection seam

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Add the pyannote.audio dependency

**Files:**
- Modify: `pyproject.toml` (+ `uv.lock`)

**Interfaces:**
- Produces: `from pyannote.audio import Pipeline` works in the venv, so Task 2's `_load_pipeline` is real. No code changes.

- [ ] **Step 1: Add the dependency**

Run: `uv add "pyannote.audio>=4.0"`
Expected: resolves and installs (~1 GB incl. torchaudio/torchcodec; resolution was verified with a dry run on 2026-07-09).

Contingency (only if install or the import below fails on Python 3.14): run `uv remove pyannote.audio && uv add "pyannote.audio>=3.3,<4"`, change `DEFAULT_DIARIZE_MODEL` to `"pyannote/speaker-diarization-3.1"` in `vproc/config.py` (and the expected default in `tests/test_config.py::test_diarize_model_default_and_disable`), and change `token=cfg.hf_token` to `use_auth_token=cfg.hf_token` in `_load_pipeline`. If BOTH majors fail to import, STOP and report — do not improvise further.

- [ ] **Step 2: Verify the lazy import works**

Run: `uv run python -c "from pyannote.audio import Pipeline; print('pyannote import OK')"`
Expected: `pyannote import OK` (warnings acceptable)

- [ ] **Step 3: Verify the suite still passes and stays fast**

Run: `uv run pytest -q`
Expected: all pass, wall time still ~2s (nothing imports pyannote at module scope)

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore(deps): add pyannote.audio for speaker diarization

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Pipeline integration (best-effort, like OCR)

**Files:**
- Modify: `vproc/ingest/pipeline.py` (imports ~line 1-13; the transcribe block inside `ingest_video`, currently `if _has_audio_stream(...)` / `else` around line 53-57)
- Test: `tests/test_pipeline.py` (`_cfg` helper ~line 32; new tests appended)

**Interfaces:**
- Consumes: `D.diarize(wav_path, cfg)` and `D.assign_speakers(transcript, turns)` from Task 2; `cfg.diarize_model` from Task 1.
- Produces: `ingest_video` relabels transcript speakers when `cfg.diarize_model` is non-empty; failures print `warning: diarization failed: …` to stderr and never abort ingest.

- [ ] **Step 1: Update the test cfg helper** — in `tests/test_pipeline.py`, the `_cfg` function returns a `types.SimpleNamespace`; add `diarize_model=""` so ALL existing tests run with diarization disabled:

```python
        frames_dir=str(tmp_path / "frames"),
        ocr_timeout=60.0,
        ocr_max_tokens=1024,
        diarize_model="",
    )
```

- [ ] **Step 2: Write the failing tests** — append to `tests/test_pipeline.py`:

```python
def test_diarization_labels_reach_store(tmp_path, monkeypatch):
    from vproc.ingest.diarize import SpeakerTurn
    _patch(monkeypatch, tmp_path,
           transcript=[TranscriptSegment(0.0, 4.0, "hello there"),
                       TranscriptSegment(4.0, 9.0, "hi back")])
    monkeypatch.setattr(P.D, "diarize",
                        lambda wav, cfg: [SpeakerTurn(0.0, 4.0, "SPEAKER_00"),
                                          SpeakerTurn(4.0, 9.0, "SPEAKER_01")])
    cfg = _cfg(tmp_path)
    cfg.diarize_model = "pyannote/fake"
    store = FakeStore()
    P.ingest_video("/videos/standup.mp4", cfg=cfg, store=store)
    speakers = {r["speaker"] for r in store.rows}
    assert {"SPEAKER_00", "SPEAKER_01"} <= speakers  # align sub-splits on speaker change


def test_diarization_disabled_when_model_empty(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    def _boom(wav, cfg):
        raise AssertionError("diarize must not be called when disabled")
    monkeypatch.setattr(P.D, "diarize", _boom)
    store = FakeStore()
    n = P.ingest_video("/videos/standup.mp4", cfg=_cfg(tmp_path), store=store)
    assert n >= 1  # ingest proceeded, diarize never invoked


def test_diarization_failure_degrades_to_unlabeled(tmp_path, monkeypatch, capsys):
    _patch(monkeypatch, tmp_path)
    def _boom(wav, cfg):
        raise RuntimeError("bad token")
    monkeypatch.setattr(P.D, "diarize", _boom)
    cfg = _cfg(tmp_path)
    cfg.diarize_model = "pyannote/fake"
    store = FakeStore()
    n = P.ingest_video("/videos/standup.mp4", cfg=cfg, store=store)
    assert n >= 1
    assert all(r["speaker"] == "SPEAKER_0" for r in store.rows)  # transcript survives unlabeled
    assert "diarization failed" in capsys.readouterr().err
```

(`TranscriptSegment` is already imported at the top of `tests/test_pipeline.py`; if not, add `from vproc.ingest.transcribe import TranscriptSegment` next to the existing imports.)

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_pipeline.py -q`
Expected: the three new tests FAIL with `AttributeError: module 'vproc.ingest.pipeline' has no attribute 'D'`; all pre-existing tests PASS (disabled by the `_cfg` default).

- [ ] **Step 4: Implement** — in `vproc/ingest/pipeline.py`:

Add to the imports block (alphabetical with the others):

```python
from vproc.ingest import diarize as D
```

Insert immediately after the transcribe `if/else` block (after `transcript = []  # no audio track: OCR-only ingest`), at the same indent level:

```python
        if transcript and cfg.diarize_model:
            try:
                D.assign_speakers(transcript, D.diarize(wav_path, cfg))
            except Exception as e:
                # Best-effort like OCR: speaker labels are never worth losing the transcript.
                print(f"warning: diarization failed: {e}", file=sys.stderr)
```

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest -q`
Expected: all pass (existing + 3 new)

- [ ] **Step 6: Commit**

```bash
git add vproc/ingest/pipeline.py tests/test_pipeline.py
git commit -m "feat(ingest): diarize transcript speakers (best-effort) when VPROC_DIARIZE_MODEL is set

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: End-to-end verification on the real meeting video (gated on HF_TOKEN)

**Files:** none (verification only; a follow-up commit only if a real bug is found and fixed)

**Interfaces:**
- Consumes: everything above; the owner's `HF_TOKEN` in `.env`; voyage endpoints from `.env`.

- [ ] **Step 1: Check the gate**

Run: `grep -c "^HF_TOKEN=hf_" /Users/jzfre/Code/personal/vproc/.env`
Expected: `1`. If `0`: STOP and report "blocked: owner must add HF_TOKEN to .env and accept conditions at huggingface.co/pyannote/speaker-diarization-community-1" — do not work around it.

- [ ] **Step 2: Real ingest with diarization**

```bash
cd /Users/jzfre/Code/personal/vproc
export VPROC_INDEX_PATH=./vproc_diar.lance VPROC_FRAMES_DIR=./vproc_frames_diar
rm -rf vproc_diar.lance vproc_frames_diar
uv run vproc ingest "2026-06-18 16-01-59.mp4"
```

Expected: `ingested N segments` with N ≥ 24 (diarization splits more chunks than the 24 seen undiarized), NO `warning: diarization failed` line. First run downloads the pyannote model (~1-2 min extra); total a few minutes.

- [ ] **Step 3: Verify distinct speakers in stored rows**

```bash
VPROC_INDEX_PATH=./vproc_diar.lance uv run python -c "
from vproc.store.lancedb_store import Store
import os
rows = Store(os.environ['VPROC_INDEX_PATH'])._table().to_arrow().to_pylist()
speakers = sorted({r['speaker'] for r in rows})
print('speakers:', speakers)
assert len([s for s in speakers if s.startswith('SPEAKER_') and s != 'SPEAKER_0']) >= 2, speakers
print('OK: distinct diarized speakers present')"
```

Expected: at least two distinct `SPEAKER_xx` labels (the meeting has multiple participants).

- [ ] **Step 4: Speaker filter round-trip through the service**

```bash
cd /Users/jzfre/Code/personal/vproc
VPROC_INDEX_PATH=./vproc_diar.lance VPROC_PORT=8797 VPROC_HOST=127.0.0.1 uv run vproc serve &
SRV=$!; sleep 5
LABEL=$(VPROC_INDEX_PATH=./vproc_diar.lance uv run python -c "
from vproc.store.lancedb_store import Store
rows = Store('./vproc_diar.lance')._table().to_arrow().to_pylist()
print(sorted({r['speaker'] for r in rows if r['speaker'] != 'SPEAKER_0'})[0])")
curl -sS -m 60 -X POST http://127.0.0.1:8797/search -H 'Content-Type: application/json' \
  -d "{\"query\": \"middleware\", \"speaker\": \"$LABEL\"}" \
  | python3 -c "
import json, sys
hits = json.load(sys.stdin)
assert hits, 'no hits'
assert all(h['speaker'] == '$LABEL' for h in hits), [h['speaker'] for h in hits]
print('OK:', len(hits), 'hits, all', '$LABEL')"
kill $SRV
```

Expected: `OK: <n> hits, all SPEAKER_xx`.

- [ ] **Step 5: Report** — wall-clock ingest time, speaker count, and a 2-3 row sample of `(start, speaker, said_text[:60])` so the owner can eyeball label plausibility. Clean up: `rm -rf vproc_diar.lance vproc_frames_diar` after the owner confirms.
