# vproc speaker naming (nameplate OCR) — design

**Date:** 2026-07-30 · **Status:** approved by owner (chat) · **Scope:** Phase 2, second slice

## Goal

Replace diarized `SPEAKER_xx` labels with real participant names read from the meeting app's
active-speaker indicators, so citations say `[test · 05:58 · Repan, Jozef]` and the speaker
filter takes names. Names only with evidence; wrong names must never be baked in silently.

**Feasibility (spike, 2026-07-26, tmp/test.mkv):** the VL model reads the participant
roster essentially perfectly across frames (spelling jitter like `Vanco/Yanco/Wanco`
aside) and identifies the active speaker correctly in a majority of probes (~3 s each).
3 samples/cluster produced correct majorities for 4 of 5 clusters; one needed more samples.

**Out of scope:** face recognition, per-app (Teams/Zoom/Meet) UI parsing, re-embedding
rows after a manual rename.

## Approach

Active-speaker-indicator voting: at diarized turn midpoints, extract a frame and ask the
VL model (the existing OCR endpoint — same model, calls are sequential) who the meeting UI
shows as speaking; canonicalize and majority-vote per cluster. Chosen over face/voice
re-identification (heavy new models for a problem the meeting app already solves on
screen) and manual-only renaming (no automation; survives as the fallback path here).

## Components

### `vproc/ingest/naming.py` (new)

```python
@dataclass
class NameVote:
    speaker: str          # diarized cluster label, e.g. "SPEAKER_04"
    t: float              # video timestamp probed
    name: str | None      # active speaker as read, None if not determinable
    visible_names: list[str]

def sample_name_votes(video_path: str, turns: list[SpeakerTurn], cfg,
                      probe=None) -> list[NameVote]: ...
def resolve_names(votes: list[NameVote]) -> tuple[dict[str, str], dict]: ...
def apply_names(transcript: list[TranscriptSegment], mapping: dict[str, str]) -> None: ...
```

- `sample_name_votes`: per cluster, up to 6 longest turns; frame at each turn midpoint via
  ffmpeg (`-nostdin -ss <t> -frames:v 1`, temp file); one VL call per frame with a strict-
  JSON prompt (`{"speaking": ..., "visible_names": [...]}`), parsed leniently (a bad reply
  = a lost vote, not an error). `probe` is the injection seam: a callable
  `(image_path) -> str` returning the model's raw reply; default wraps
  `client.ocr_image(cfg.ocr.base_url, cfg.ocr.model, path, PROMPT, timeout, max_tokens)`.
  Per-frame failures (ffmpeg or VL) skip that vote.
- `resolve_names`:
  1. **Canonicalization:** collect every read name (speaking + visible); group case-
     insensitively by edit-distance proximity (difflib `SequenceMatcher` ratio ≥ 0.75 —
     stdlib, no new deps); each group's canonical form is its most frequent spelling.
  2. **Voting:** per cluster over canonicalized non-null `speaking` votes: assign the top
     name iff it has ≥ 2 votes AND a strict majority (> half of that cluster's non-null
     votes). Two clusters sharing a winner both receive it (diarization over-splits; owner
     decision: merge under the name).
  3. Returns `(mapping, suggestions)`: `mapping` only for clusters that cleared the bar;
     `suggestions` carries per-cluster vote tallies + evidence timestamps for the rest.
- `apply_names`: in-place relabel of `TranscriptSegment.speaker` for labels in `mapping`
  (same pattern as `assign_speakers`).

### Persistence: `speakers.json`

Written at ingest end to `<frames_dir>/<project_id>/<title>/speakers.json` (the durable
per-memory dir that already holds frames):

```json
{"mapping": {"SPEAKER_04": "Repan, Jozef"},
 "suggestions": {"SPEAKER_02": {"votes": {"Selrico Lamont Martin": 1, "PATINO, DANIEL": 1},
                                 "evidence": [{"t": 992, "name": "Selrico Lamont Martin"}]}},
 "votes": [{"speaker": "SPEAKER_04", "t": 889, "name": "Repan, Jozef"}]}
```

This is the design doc's `SpeakerMap` ("name only with evidence") in v1 sidecar form.

### Pipeline integration (`vproc/ingest/pipeline.py`)

Inside the existing diarization guard, after `assign_speakers` — one combined best-effort
block (naming reuses the turns):

```python
if transcript and cfg.diarize_model:
    try:
        turns = D.diarize(wav_path, cfg)
        D.assign_speakers(transcript, turns)
        if cfg.speaker_naming and turns:
            votes = N.sample_name_votes(video_path, turns, cfg)
            mapping, suggestions = N.resolve_names(votes)
            N.apply_names(transcript, mapping)
    except Exception as e:
        print(f"warning: diarization failed ({cfg.diarize_model}): {e}", file=sys.stderr)
```

(Exact structure decided at planning; requirements: naming failure must not undo
diarization labels — its own inner try/except — and `speakers.json` + a one-line summary
per outcome are emitted at ingest end: `named: SPEAKER_04 -> Repan, Jozef (2/3)` /
`unresolved: SPEAKER_02 (votes split)`.)

Names flow into `build_segments` → `embed_text`, citations, store rows, and the existing
speaker filter — no downstream changes.

### CLI: `vproc speakers`

- `vproc speakers <memory>` — print `speakers.json` (mapping + suggestions with evidence).
- `vproc speakers <memory> --set 'SPEAKER_02=PATINO, DANIEL'` — manual assignment:
  updates matching store rows' `speaker` via LanceDB `table.update`, updates
  `speakers.json`. Documented limitation: `embed_text` keeps the old label string
  (retrieval unaffected in practice; no re-embed).

### Config

`speaker_naming: bool = True`, env `VPROC_SPEAKER_NAMING` (same on/off parsing as
`VPROC_GROUNDING_THINKING`). Naming runs only when diarization ran and produced turns.

## Cost

≤ 5 clusters × 6 probes × ~3 s ≈ 90–120 s per 22-min ingest (measured probe latency).

## Testing

Offline via the `probe` seam and fakes:
1. Canonicalization: the spike's real jitter (`Vanco, Pavol` / `Yanco, Pavol` /
   `Wanco, Pavol` / `Vanco, Pavlo` → one canonical; distinct names stay distinct).
2. Voting: strict-majority threshold, ≥2-vote floor, shared-winner merge, null votes
   excluded, split votes → suggestion not mapping.
3. `sample_name_votes`: probe seam used, longest-turns selection, per-frame failure skips.
4. Pipeline: names reach store rows; naming failure keeps diarized labels; disabled flag
   skips probes; `speakers.json` written with mapping + suggestions.
5. CLI: list + `--set` against a fake store; store update filter correctness.
6. E2E on `tmp/test.mkv`: expect `Vanco, Pavol`, `Selrico Lamont Martin` (two merged
   clusters), `Repan, Jozef` named; `SPEAKER_02` likely unresolved at 6 samples — verify
   the suggestion surfaces with evidence.

## Risks

- **Indicator lag / cross-talk mislabels frames** — mitigated by longest-turn midpoints,
  6 samples, strict majority; residual errors are correctable via `vproc speakers --set`.
- **Screen-share-only recordings** (no roster visible): votes come back null → clusters
  keep labels; feature degrades to today's behavior.
- **Same display name for two real people**: merge rule would conflate them; accepted for
  v1 (owner decision), manual `--set` can split later once the CLI exists.
