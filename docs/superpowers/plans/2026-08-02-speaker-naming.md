# Speaker Naming (Nameplate OCR) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace diarized `SPEAKER_xx` labels with real participant names read from the meeting app's active-speaker indicators, with evidence-backed suggestions (never silent wrong names) and a `vproc speakers` CLI for manual assignment.

**Architecture:** A new `vproc/ingest/naming.py` probes video frames at diarized-turn midpoints via the existing vision endpoint, canonicalizes OCR spelling jitter, and majority-votes a cluster→name mapping. The pipeline applies names right after diarization (before segments/embedding) as a nested best-effort step, persists `speakers.json` next to the memory's frames, and a CLI command lists/overrides the mapping (store rows updated via LanceDB `update`).

**Tech Stack:** Python 3.14 via `uv`, stdlib only for logic (`difflib`, `json`, `collections`), ffmpeg for frame grabs, existing `client.ocr_image` for probes, pytest offline via an injected `probe` seam.

**Spec:** `docs/superpowers/specs/2026-07-30-speaker-naming-design.md`

## Global Constraints

- Run everything from `/Users/jzfre/Code/personal/vproc`; tests via `uv run pytest …`.
- Tests fully offline: no network, no ffmpeg execution (patch the frame-grab helper), no model calls — every probe is injected.
- Match the existing compact code style; comments only for constraints code can't express.
- Voting rule (verbatim from spec): assign the top canonical name iff it has **≥ 2 votes AND a strict majority (> half of that cluster's non-null votes)**; clusters sharing a winner BOTH receive it; everything else becomes a suggestion with evidence and the cluster keeps `SPEAKER_xx`.
- Canonicalization (verbatim from spec): case-insensitive `difflib.SequenceMatcher` ratio **≥ 0.75** groups spellings; each group's canonical form is its **most frequent** spelling.
- Probe budget: up to **6 longest turns per cluster**, frame at each turn midpoint.
- Naming failure must never undo diarization labels or block ingest (nested best-effort, own stderr warning).
- Env/config: `VPROC_SPEAKER_NAMING`, field `speaker_naming: bool = True`, same on/off parsing as `VPROC_GROUNDING_THINKING`.
- Commit messages end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- Do NOT run real endpoints/models in any task except Task 6.

---

### Task 1: Config field `speaker_naming`

**Files:**
- Modify: `vproc/config.py` (Config dataclass, `load_config()`)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Config.speaker_naming: bool` (default True), env `VPROC_SPEAKER_NAMING` with off values `("0", "false", "off", "no")` case-insensitive. Task 4 reads `cfg.speaker_naming`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_config.py`:

```python
def test_speaker_naming_default_and_off(monkeypatch):
    for k in list(os.environ):
        if k.startswith("VPROC_"):
            monkeypatch.delenv(k, raising=False)
    assert load_config().speaker_naming is True
    for off in ("off", "false", "0", "no", "OFF"):
        monkeypatch.setenv("VPROC_SPEAKER_NAMING", off)
        assert load_config().speaker_naming is False, off
    monkeypatch.setenv("VPROC_SPEAKER_NAMING", "on")
    assert load_config().speaker_naming is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::test_speaker_naming_default_and_off -q`
Expected: FAIL with `AttributeError: 'Config' object has no attribute 'speaker_naming'`

- [ ] **Step 3: Implement** — in `vproc/config.py`, add at the END of the `Config` dataclass (after `diarize_model`):

```python
    # Nameplate speaker naming (needs diarization + the vision endpoint). Off = keep SPEAKER_xx.
    speaker_naming: bool = True
```

And in `load_config()` after the `diarize_model=` line:

```python
        speaker_naming=os.environ.get("VPROC_SPEAKER_NAMING", "on").strip().lower()
        not in ("0", "false", "off", "no"),
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_config.py -q` — Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add vproc/config.py tests/test_config.py
git commit -m "feat(config): VPROC_SPEAKER_NAMING toggle for nameplate speaker naming

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: `naming.py` resolution core — NameVote, canonicalization, voting, apply

**Files:**
- Create: `vproc/ingest/naming.py`
- Test: `tests/test_naming.py` (new)

**Interfaces:**
- Consumes: `TranscriptSegment` from `vproc/ingest/transcribe.py` (fields `start, end, text, speaker`).
- Produces (Tasks 3/4/5 rely on these exact names):
  - `NameVote` dataclass: `speaker: str, t: float, name: str | None, visible_names: list[str]`
  - `resolve_names(votes: list[NameVote]) -> tuple[dict[str, str], dict]` — `(mapping, suggestions)`
  - `apply_names(transcript: list[TranscriptSegment], mapping: dict[str, str]) -> None`
  - constants `PROMPT: str`, `MAX_PROBES_PER_SPEAKER = 6`, `SIMILARITY = 0.75`

- [ ] **Step 1: Write the failing tests** — create `tests/test_naming.py`:

```python
from vproc.ingest.naming import NameVote, apply_names, resolve_names
from vproc.ingest.transcribe import TranscriptSegment


def _v(spk, name, t=0.0, visible=()):
    return NameVote(speaker=spk, t=t, name=name, visible_names=list(visible))


def test_canonicalization_groups_ocr_jitter():
    # The spike's real jitter: 4 spellings of one person + 2 distinct other names.
    votes = [
        _v("SPEAKER_00", "Vanco, Pavol", visible=["Repan, Jozef", "PATINO, DANIEL"]),
        _v("SPEAKER_00", "Yanco, Pavol", visible=["Vanco, Pavol"]),
        _v("SPEAKER_00", "Wanco, Pavol", visible=["Vanco, Pavlo"]),
    ]
    mapping, suggestions = resolve_names(votes)
    # 3 jitter spellings converge on the most frequent form and clear the 2-vote majority.
    assert mapping == {"SPEAKER_00": "Vanco, Pavol"}
    assert suggestions == {}


def test_distinct_names_do_not_merge():
    votes = [_v("SPEAKER_00", "Repan, Jozef"), _v("SPEAKER_00", "PATINO, DANIEL"),
             _v("SPEAKER_00", "Repan, Jozef")]
    mapping, _ = resolve_names(votes)
    assert mapping == {"SPEAKER_00": "Repan, Jozef"}  # 2/3 strict majority


def test_split_votes_become_suggestion_not_mapping():
    votes = [_v("SPEAKER_02", "Selrico Lamont Martin", t=992),
             _v("SPEAKER_02", "PATINO, DANIEL", t=209),
             _v("SPEAKER_02", None, t=173)]
    mapping, suggestions = resolve_names(votes)
    assert mapping == {}
    s = suggestions["SPEAKER_02"]
    assert s["votes"] == {"Selrico Lamont Martin": 1, "PATINO, DANIEL": 1}
    assert {"t": 992, "name": "Selrico Lamont Martin"} in s["evidence"]


def test_single_vote_is_not_enough():
    mapping, suggestions = resolve_names([_v("SPEAKER_03", "Selrico Lamont Martin")])
    assert mapping == {} and "SPEAKER_03" in suggestions  # >=2 votes required


def test_two_clusters_sharing_winner_both_map():
    votes = ([_v("SPEAKER_01", "Selrico Lamont Martin")] * 2
             + [_v("SPEAKER_03", "Selrico Lamont Martin")] * 2)
    mapping, _ = resolve_names(votes)
    assert mapping == {"SPEAKER_01": "Selrico Lamont Martin",
                       "SPEAKER_03": "Selrico Lamont Martin"}


def test_null_votes_excluded_from_majority_denominator():
    # 2 name votes + 3 nulls: majority is over the 2 non-null votes.
    votes = [_v("SPEAKER_04", "Repan, Jozef")] * 2 + [_v("SPEAKER_04", None)] * 3
    mapping, _ = resolve_names(votes)
    assert mapping == {"SPEAKER_04": "Repan, Jozef"}


def test_apply_names_relabels_only_mapped():
    t = [TranscriptSegment(0.0, 1.0, "a", speaker="SPEAKER_00"),
         TranscriptSegment(1.0, 2.0, "b", speaker="SPEAKER_01")]
    apply_names(t, {"SPEAKER_00": "Repan, Jozef"})
    assert [s.speaker for s in t] == ["Repan, Jozef", "SPEAKER_01"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_naming.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'vproc.ingest.naming'`

- [ ] **Step 3: Implement** — create `vproc/ingest/naming.py`:

```python
import json
import os
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher

from vproc.ingest.diarize import SpeakerTurn
from vproc.ingest.transcribe import TranscriptSegment
from vproc.llm import client

PROMPT = (
    "This is a frame from a recorded video meeting. Look at the participant tiles / filmstrip "
    "and any visual speaking indicators (highlighted tile border, microphone icon, active-"
    "speaker name label). Answer in strict JSON: "
    '{"speaking": "<name of the participant currently speaking, or null>", '
    '"visible_names": ["<all participant names you can read>"]}'
)

MAX_PROBES_PER_SPEAKER = 6
SIMILARITY = 0.75  # difflib ratio: OCR jitter of one name groups; distinct names don't


@dataclass
class NameVote:
    speaker: str            # diarized cluster label, e.g. "SPEAKER_04"
    t: float                # video timestamp probed
    name: str | None        # active speaker as read; None = not determinable
    visible_names: list[str]


def _canonicalize(all_names: list[str]) -> dict[str, str]:
    """Map each OCR spelling variant to its group's most frequent form."""
    counts = Counter(n.strip() for n in all_names if n and n.strip())
    groups: list[list[str]] = []
    for name in sorted(counts, key=lambda n: -counts[n]):
        for g in groups:
            if SequenceMatcher(None, name.lower(), g[0].lower()).ratio() >= SIMILARITY:
                g.append(name)
                break
        else:
            groups.append([name])
    canon: dict[str, str] = {}
    for g in groups:
        best = max(g, key=lambda n: counts[n])
        for n in g:
            canon[n] = best
    return canon


def resolve_names(votes: list[NameVote]) -> tuple[dict[str, str], dict]:
    """(mapping, suggestions). A cluster maps to its top canonical name iff that name has
    >= 2 votes AND a strict majority of the cluster's non-null votes; clusters sharing a
    winner both map (diarization over-splits). The rest become evidence-backed suggestions."""
    all_names = [v.name for v in votes if v.name] + [n for v in votes for n in v.visible_names]
    canon = _canonicalize(all_names)
    by_speaker: dict[str, list[NameVote]] = defaultdict(list)
    for v in votes:
        by_speaker[v.speaker].append(v)
    mapping: dict[str, str] = {}
    suggestions: dict = {}
    for speaker, vs in sorted(by_speaker.items()):
        tally = Counter(canon.get(v.name.strip(), v.name.strip()) for v in vs if v.name and v.name.strip())
        total = sum(tally.values())
        if tally:
            top, n = tally.most_common(1)[0]
            if n >= 2 and n * 2 > total:
                mapping[speaker] = top
                continue
        suggestions[speaker] = {
            "votes": dict(tally),
            "evidence": [{"t": round(v.t), "name": canon.get(v.name.strip(), v.name.strip())}
                         for v in vs if v.name and v.name.strip()],
        }
    return mapping, suggestions


def apply_names(transcript: list[TranscriptSegment], mapping: dict[str, str]) -> None:
    for seg in transcript:
        if seg.speaker in mapping:
            seg.speaker = mapping[seg.speaker]
```

(`json`, `os`, `subprocess`, `tempfile`, `SpeakerTurn`, and `client` are unused until Task 3 — that is fine for one commit, or defer those imports to Task 3 if the linter objects.)

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_naming.py -q` — Expected: 7 passed
Then: `uv run pytest -q` — Expected: whole suite passes, still fast (~1s)

- [ ] **Step 5: Commit**

```bash
git add vproc/ingest/naming.py tests/test_naming.py
git commit -m "feat(naming): NameVote + canonicalization + majority-vote name resolution

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: `naming.py` sampling + persistence — probes and speakers.json

**Files:**
- Modify: `vproc/ingest/naming.py`
- Test: `tests/test_naming.py`

**Interfaces:**
- Consumes: `SpeakerTurn` from `vproc/ingest/diarize.py`; `client.ocr_image(base_url, model, path, prompt, timeout=..., max_tokens=...)`.
- Produces (Tasks 4/5 rely on these exact names):
  - `sample_name_votes(video_path: str, turns: list[SpeakerTurn], cfg, probe=None) -> list[NameVote]`
  - `save_speaker_map(mem_dir: str, data: dict) -> None` / `load_speaker_map(mem_dir: str) -> dict`
  - `_parse_vote(raw: str) -> tuple[str | None, list[str]]` (module-private, tested directly)

- [ ] **Step 1: Write the failing tests** — append to `tests/test_naming.py`:

```python
import json
import os

import vproc.ingest.naming as N
from vproc.ingest.diarize import SpeakerTurn


def test_parse_vote_lenient():
    assert N._parse_vote('{"speaking": "A B", "visible_names": ["A B", "C D"]}') == ("A B", ["A B", "C D"])
    assert N._parse_vote('noise before {"speaking": null, "visible_names": []} after') == (None, [])
    assert N._parse_vote('{"speaking": "  ", "visible_names": "not-a-list"}') == (None, [])
    assert N._parse_vote("total garbage") == (None, [])


def test_sample_name_votes_probes_longest_turns(monkeypatch):
    monkeypatch.setattr(N, "_frame_at", lambda video, t, out: None)  # no real ffmpeg
    probed = []
    def probe(path):
        probed.append(path)
        return '{"speaking": "Repan, Jozef", "visible_names": ["Repan, Jozef"]}'
    # 8 turns for one speaker: only the 6 longest get probed; 1 turn for another.
    turns = [SpeakerTurn(i * 10.0, i * 10.0 + 1.0 + i, "SPEAKER_00") for i in range(8)]
    turns.append(SpeakerTurn(500.0, 520.0, "SPEAKER_01"))
    votes = N.sample_name_votes("/v.mkv", turns, cfg=None, probe=probe)
    assert len([v for v in votes if v.speaker == "SPEAKER_00"]) == 6
    assert len([v for v in votes if v.speaker == "SPEAKER_01"]) == 1
    assert all(v.name == "Repan, Jozef" for v in votes)
    # longest turns won: the two shortest SPEAKER_00 turns (i=0,1) were skipped
    probed_ts = {v.t for v in votes if v.speaker == "SPEAKER_00"}
    assert (0.0 + 1.0) / 2 not in probed_ts and (10.0 + 12.0) / 2 not in probed_ts


def test_sample_name_votes_skips_failed_probes(monkeypatch):
    monkeypatch.setattr(N, "_frame_at", lambda video, t, out: None)
    calls = {"n": 0}
    def probe(path):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("slow")
        return '{"speaking": "X Y", "visible_names": []}'
    turns = [SpeakerTurn(0.0, 10.0, "SPEAKER_00"), SpeakerTurn(20.0, 30.0, "SPEAKER_00")]
    votes = N.sample_name_votes("/v.mkv", turns, cfg=None, probe=probe)
    assert len(votes) == 1  # first probe lost, ingest-level behavior unaffected


def test_speaker_map_round_trip(tmp_path):
    data = {"mapping": {"SPEAKER_04": "Repan, Jozef"},
            "suggestions": {"SPEAKER_02": {"votes": {"A": 1}, "evidence": []}},
            "votes": [{"speaker": "SPEAKER_04", "t": 889, "name": "Repan, Jozef"}]}
    mem_dir = str(tmp_path / "frames" / "default" / "test")
    N.save_speaker_map(mem_dir, data)
    assert N.load_speaker_map(mem_dir) == data
    assert os.path.exists(os.path.join(mem_dir, "speakers.json"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_naming.py -q`
Expected: the 4 new tests FAIL with `AttributeError: module 'vproc.ingest.naming' has no attribute '_parse_vote'` (etc.); Task 2's 7 still pass.

- [ ] **Step 3: Implement** — append to `vproc/ingest/naming.py`:

```python
def _frame_at(video_path: str, t: float, out_path: str) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
         "-ss", str(t), "-i", video_path, "-frames:v", "1", out_path, "-y"],
        check=True)


def _parse_vote(raw: str) -> tuple[str | None, list[str]]:
    """Lenient parse of a probe reply; a malformed reply is a lost vote, not an error."""
    try:
        i, j = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[i:j + 1])
        name = data.get("speaking")
        names = data.get("visible_names")
        return (name if isinstance(name, str) and name.strip() else None,
                [n for n in names if isinstance(n, str)] if isinstance(names, list) else [])
    except Exception:
        return None, []


def sample_name_votes(video_path: str, turns: list[SpeakerTurn], cfg, probe=None) -> list[NameVote]:
    """Probe frames at the midpoints of each cluster's longest turns and ask the vision
    model who is speaking. `probe` is the injection seam: (image_path) -> raw reply."""
    if probe is None:
        def probe(path):
            return client.ocr_image(cfg.ocr.base_url, cfg.ocr.model, path, PROMPT,
                                    timeout=cfg.ocr_timeout, max_tokens=cfg.ocr_max_tokens)
    by_speaker: dict[str, list[SpeakerTurn]] = defaultdict(list)
    for t in turns:
        by_speaker[t.speaker].append(t)
    votes: list[NameVote] = []
    with tempfile.TemporaryDirectory(prefix="vproc-name-") as work:
        for speaker, spk_turns in sorted(by_speaker.items()):
            longest = sorted(spk_turns, key=lambda t: t.end - t.start, reverse=True)
            for turn in longest[:MAX_PROBES_PER_SPEAKER]:
                mid = (turn.start + turn.end) / 2.0
                frame = os.path.join(work, f"{speaker}-{int(mid * 1000)}.png")
                try:
                    _frame_at(video_path, mid, frame)
                    name, visible = _parse_vote(probe(frame))
                except Exception:
                    continue  # a failed probe is a lost vote, never a failed ingest
                votes.append(NameVote(speaker, mid, name, visible))
    return votes


def save_speaker_map(mem_dir: str, data: dict) -> None:
    os.makedirs(mem_dir, exist_ok=True)
    with open(os.path.join(mem_dir, "speakers.json"), "w") as f:
        json.dump(data, f, indent=1)


def load_speaker_map(mem_dir: str) -> dict:
    with open(os.path.join(mem_dir, "speakers.json")) as f:
        return json.load(f)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_naming.py -q` — Expected: 11 passed
Then: `uv run pytest -q` — Expected: whole suite passes, ~1s (no ffmpeg/model calls leaked)

- [ ] **Step 5: Commit**

```bash
git add vproc/ingest/naming.py tests/test_naming.py
git commit -m "feat(naming): frame-probe sampling with injection seam + speakers.json persistence

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Pipeline integration

**Files:**
- Modify: `vproc/ingest/pipeline.py` (imports; the diarization guard inside `ingest_video`; the post-swap block)
- Test: `tests/test_pipeline.py` (`_cfg` helper; new tests)

**Interfaces:**
- Consumes: `N.sample_name_votes(video_path, turns, cfg)`, `N.resolve_names(votes)`, `N.apply_names(transcript, mapping)`, `N.save_speaker_map(mem_dir, data)` from Tasks 2–3; `cfg.speaker_naming` from Task 1.
- Produces: names flow into stored rows; `speakers.json` written to the durable `mem_dir` AFTER the frames swap (the swap `rmtree`s `mem_dir` — writing earlier would be destroyed); summary lines `named: SPEAKER_xx -> Name` / `unresolved speaker: SPEAKER_xx …` printed; naming failure warns `speaker naming failed` to stderr and keeps diarized labels.

- [ ] **Step 1: Update the test cfg helper** — in `tests/test_pipeline.py` `_cfg`, add after `diarize_model=""`:

```python
        speaker_naming=False,
```

(All pre-existing tests run with naming disabled; new tests enable it explicitly.)

- [ ] **Step 2: Write the failing tests** — append to `tests/test_pipeline.py`:

```python
def test_speaker_naming_applies_names_to_store(tmp_path, monkeypatch):
    from vproc.ingest.diarize import SpeakerTurn
    _patch(monkeypatch, tmp_path,
           transcript=[TranscriptSegment(0.0, 4.0, "hello there"),
                       TranscriptSegment(4.0, 9.0, "hi back")])
    monkeypatch.setattr(P.D, "diarize",
                        lambda wav, cfg: [SpeakerTurn(0.0, 4.0, "SPEAKER_00"),
                                          SpeakerTurn(4.0, 9.0, "SPEAKER_01")])
    monkeypatch.setattr(P.N, "sample_name_votes", lambda video, turns, cfg: [])
    monkeypatch.setattr(P.N, "resolve_names",
                        lambda votes: ({"SPEAKER_00": "Repan, Jozef"},
                                       {"SPEAKER_01": {"votes": {}, "evidence": []}}))
    cfg = _cfg(tmp_path)
    cfg.diarize_model = "pyannote/fake"
    cfg.speaker_naming = True
    store = FakeStore()
    P.ingest_video("/videos/standup.mp4", cfg=cfg, store=store)
    speakers = {r["speaker"] for r in store.rows}
    assert "Repan, Jozef" in speakers and "SPEAKER_01" in speakers
    import json as _json
    m = _json.load(open(os.path.join(cfg.frames_dir, "default", "standup", "speakers.json")))
    assert m["mapping"] == {"SPEAKER_00": "Repan, Jozef"}
    assert "SPEAKER_01" in m["suggestions"]


def test_naming_failure_keeps_diarized_labels(tmp_path, monkeypatch, capsys):
    from vproc.ingest.diarize import SpeakerTurn
    _patch(monkeypatch, tmp_path,
           transcript=[TranscriptSegment(0.0, 4.0, "hello"), TranscriptSegment(4.0, 9.0, "hi")])
    monkeypatch.setattr(P.D, "diarize",
                        lambda wav, cfg: [SpeakerTurn(0.0, 4.0, "SPEAKER_00"),
                                          SpeakerTurn(4.0, 9.0, "SPEAKER_01")])
    def boom(video, turns, cfg):
        raise RuntimeError("vision endpoint down")
    monkeypatch.setattr(P.N, "sample_name_votes", boom)
    cfg = _cfg(tmp_path)
    cfg.diarize_model = "pyannote/fake"
    cfg.speaker_naming = True
    store = FakeStore()
    n = P.ingest_video("/videos/standup.mp4", cfg=cfg, store=store)
    assert n >= 1
    assert {r["speaker"] for r in store.rows} >= {"SPEAKER_00", "SPEAKER_01"}  # labels survive
    assert "speaker naming failed" in capsys.readouterr().err


def test_naming_disabled_skips_probes(tmp_path, monkeypatch):
    from vproc.ingest.diarize import SpeakerTurn
    _patch(monkeypatch, tmp_path)
    monkeypatch.setattr(P.D, "diarize", lambda wav, cfg: [SpeakerTurn(0.0, 3.0, "SPEAKER_00")])
    def boom(video, turns, cfg):
        raise AssertionError("must not probe when speaker_naming is off")
    monkeypatch.setattr(P.N, "sample_name_votes", boom)
    cfg = _cfg(tmp_path)          # speaker_naming=False by default
    cfg.diarize_model = "pyannote/fake"
    store = FakeStore()
    assert P.ingest_video("/videos/standup.mp4", cfg=cfg, store=store) >= 1
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_pipeline.py -q`
Expected: the 3 new tests FAIL with `AttributeError: module 'vproc.ingest.pipeline' has no attribute 'N'`; all pre-existing tests PASS.

- [ ] **Step 4: Implement** — in `vproc/ingest/pipeline.py`:

Add to the imports block (alphabetical, after `from vproc.ingest import frames as F`... keep alphabetical: `naming as N` sorts after `frames as F` and before `ocr as O`):

```python
from vproc.ingest import naming as N
```

Replace the existing diarization block:

```python
        if transcript and cfg.diarize_model:
            try:
                D.assign_speakers(transcript, D.diarize(wav_path, cfg))
            except Exception as e:
                # Best-effort like OCR: speaker labels are never worth losing the transcript.
                print(f"warning: diarization failed ({cfg.diarize_model}): {e}", file=sys.stderr)
```

with:

```python
        speaker_map = None
        if transcript and cfg.diarize_model:
            try:
                turns = D.diarize(wav_path, cfg)
                D.assign_speakers(transcript, turns)
                if cfg.speaker_naming and turns:
                    try:
                        votes = N.sample_name_votes(video_path, turns, cfg)
                        mapping, suggestions = N.resolve_names(votes)
                        N.apply_names(transcript, mapping)
                        speaker_map = {
                            "mapping": mapping, "suggestions": suggestions,
                            "votes": [{"speaker": v.speaker, "t": round(v.t), "name": v.name}
                                      for v in votes],
                        }
                        for spk, name in sorted(mapping.items()):
                            print(f"named: {spk} -> {name}")
                        for spk in sorted(suggestions):
                            print(f"unresolved speaker: {spk} (review with 'vproc speakers {title}')")
                    except Exception as e:
                        # Naming must never undo diarization labels or block the ingest.
                        print(f"warning: speaker naming failed: {e}", file=sys.stderr)
            except Exception as e:
                # Best-effort like OCR: speaker labels are never worth losing the transcript.
                print(f"warning: diarization failed ({cfg.diarize_model}): {e}", file=sys.stderr)
```

Then AFTER the frames-swap block (the loop `for src, dest in moves.items(): shutil.move(src, dest)`) and BEFORE `store.add(rows)`, insert:

```python
        if speaker_map is not None:
            N.save_speaker_map(mem_dir, speaker_map)
```

(Placement constraint: the swap `rmtree`s `mem_dir`, so `speakers.json` must be written after it; `mem_dir` is guaranteed to exist there — the swap `os.makedirs(mem_dir, exist_ok=True)` runs unconditionally.)

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest -q` — Expected: all pass (existing + 3 new)

- [ ] **Step 6: Commit**

```bash
git add vproc/ingest/pipeline.py tests/test_pipeline.py
git commit -m "feat(ingest): nameplate speaker naming after diarization (best-effort, evidence persisted)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: `Store.update_speaker` + `vproc speakers` CLI

**Files:**
- Modify: `vproc/store/lancedb_store.py`, `vproc/cli.py`
- Test: `tests/test_store.py`, create `tests/test_cli.py`

**Interfaces:**
- Consumes: `N.load_speaker_map(mem_dir)` / `N.save_speaker_map(mem_dir, data)` from Task 3; `cfg.frames_dir`, `cfg.index_path`.
- Produces: `Store.update_speaker(memory_id: str, project_id: str | None, old_speaker: str, new_speaker: str) -> None`; CLI `vproc speakers <memory>` (list) and `vproc speakers <memory> --set 'SPEAKER_XX=Name'` (apply manual name).

- [ ] **Step 1: Write the failing store test** — append to `tests/test_store.py` (this file already builds real temp LanceDB stores; follow its existing fixture style for creating a store with rows):

```python
def test_update_speaker_renames_only_matching_rows(tmp_path):
    store = Store(str(tmp_path / "db"))
    store.add([
        {"id": "a", "memory_id": "m1", "project_id": "default", "speaker": "SPEAKER_02",
         "start_ts": 0.0, "end_ts": 1.0, "said_text": "x", "on_screen_text": "",
         "embed_text": "x", "source_video": "v", "frame_path": "", "vector": [1.0, 0.0]},
        {"id": "b", "memory_id": "m1", "project_id": "default", "speaker": "SPEAKER_01",
         "start_ts": 1.0, "end_ts": 2.0, "said_text": "y", "on_screen_text": "",
         "embed_text": "y", "source_video": "v", "frame_path": "", "vector": [0.0, 1.0]},
        {"id": "c", "memory_id": "m2", "project_id": "default", "speaker": "SPEAKER_02",
         "start_ts": 0.0, "end_ts": 1.0, "said_text": "z", "on_screen_text": "",
         "embed_text": "z", "source_video": "v", "frame_path": "", "vector": [1.0, 1.0]},
    ])
    store.update_speaker("m1", "default", "SPEAKER_02", "PATINO, DANIEL")
    rows = {r["id"]: r["speaker"] for r in store._table().to_arrow().to_pylist()}
    assert rows == {"a": "PATINO, DANIEL", "b": "SPEAKER_01", "c": "SPEAKER_02"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_store.py::test_update_speaker_renames_only_matching_rows -q`
Expected: FAIL with `AttributeError: 'Store' object has no attribute 'update_speaker'`

- [ ] **Step 3: Implement the store method** — append to `Store` in `vproc/store/lancedb_store.py` (check the installed lancedb 0.33 `table.update` signature in `.venv` before writing; it is `update(where=..., values=...)`-style):

```python
    def update_speaker(self, memory_id: str, project_id: str | None,
                       old_speaker: str, new_speaker: str) -> None:
        """Manual speaker rename (vproc speakers --set). embed_text keeps the old label
        string by design — retrieval is unaffected; no re-embed."""
        table = self._table()
        if table is None:
            return
        where = (f"memory_id = '{memory_id.replace(chr(39), chr(39) * 2)}' "
                 f"AND speaker = '{old_speaker.replace(chr(39), chr(39) * 2)}'")
        if project_id:
            where += f" AND project_id = '{project_id.replace(chr(39), chr(39) * 2)}'"
        table.update(where=where, values={"speaker": new_speaker})
```

Run: `uv run pytest tests/test_store.py -q` — Expected: all pass.

- [ ] **Step 4: Write the failing CLI tests** — create `tests/test_cli.py`:

```python
import json
import os

import vproc.cli as cli
from vproc.ingest import naming as N


class _FakeStore:
    def __init__(self, path):
        self.path = path
        self.renames = []

    def update_speaker(self, memory_id, project_id, old, new):
        self.renames.append((memory_id, project_id, old, new))


def _setup(tmp_path, monkeypatch):
    frames = tmp_path / "frames"
    mem_dir = str(frames / "default" / "standup")
    N.save_speaker_map(mem_dir, {
        "mapping": {"SPEAKER_00": "Repan, Jozef"},
        "suggestions": {"SPEAKER_02": {"votes": {"PATINO, DANIEL": 1}, "evidence": [{"t": 209, "name": "PATINO, DANIEL"}]}},
        "votes": []})
    monkeypatch.setenv("VPROC_FRAMES_DIR", str(frames))
    monkeypatch.setenv("VPROC_INDEX_PATH", str(tmp_path / "db"))
    fake = _FakeStore(str(tmp_path / "db"))
    monkeypatch.setattr(cli, "_store_for", lambda cfg: fake)
    return mem_dir, fake


def test_speakers_list(tmp_path, monkeypatch, capsys):
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr("sys.argv", ["vproc", "speakers", "standup"])
    cli.main()
    out = capsys.readouterr().out
    assert "SPEAKER_00" in out and "Repan, Jozef" in out
    assert "SPEAKER_02" in out and "PATINO, DANIEL" in out  # suggestion shown with evidence


def test_speakers_set_updates_store_and_map(tmp_path, monkeypatch, capsys):
    mem_dir, fake = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr("sys.argv", ["vproc", "speakers", "standup", "--set", "SPEAKER_02=PATINO, DANIEL"])
    cli.main()
    assert fake.renames == [("standup", "default", "SPEAKER_02", "PATINO, DANIEL")]
    m = N.load_speaker_map(mem_dir)
    assert m["mapping"]["SPEAKER_02"] == "PATINO, DANIEL"
    assert "SPEAKER_02" not in m["suggestions"]


def test_speakers_missing_map_is_friendly(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VPROC_FRAMES_DIR", str(tmp_path / "frames"))
    monkeypatch.setattr("sys.argv", ["vproc", "speakers", "nope"])
    cli.main()
    assert "no speaker map" in capsys.readouterr().out.lower()
```

- [ ] **Step 5: Run to verify they fail**

Run: `uv run pytest tests/test_cli.py -q`
Expected: FAIL with `AttributeError: module 'vproc.cli' has no attribute '_store_for'`

- [ ] **Step 6: Implement the CLI** — rewrite `vproc/cli.py` to add the command (current file dispatches on `args[0]` with plain `sys.argv`; keep that style):

```python
import os
import sys

from vproc.config import load_config, load_dotenv


def _store_for(cfg):
    from vproc.store.lancedb_store import Store  # lazy: heavy
    return Store(cfg.index_path)


def _speakers(args: list[str]) -> None:
    from vproc.ingest import naming as N

    cfg = load_config()
    memory = args[0]
    mem_dir = os.path.join(cfg.frames_dir, "default", memory)
    try:
        m = N.load_speaker_map(mem_dir)
    except FileNotFoundError:
        print(f"no speaker map for '{memory}' (expected {mem_dir}/speakers.json)")
        return
    if len(args) >= 3 and args[1] == "--set" and "=" in args[2]:
        old, _, new = args[2].partition("=")
        old, new = old.strip(), new.strip()
        _store_for(cfg).update_speaker(memory, "default", old, new)
        m["mapping"][old] = new
        m["suggestions"].pop(old, None)
        N.save_speaker_map(mem_dir, m)
        print(f"renamed {old} -> {new} in '{memory}'")
        return
    for spk, name in sorted(m.get("mapping", {}).items()):
        print(f"{spk} = {name}")
    for spk, s in sorted(m.get("suggestions", {}).items()):
        votes = ", ".join(f"{n} x{c}" for n, c in sorted(s.get("votes", {}).items()))
        print(f"{spk} = ? (votes: {votes or 'none'})")
        for ev in s.get("evidence", []):
            print(f"    seen at {ev['t']}s: {ev['name']}")


def main() -> None:
    load_dotenv()  # auto-load ./.env so `vproc` works without manually sourcing it
    args = sys.argv[1:]
    if len(args) >= 2 and args[0] == "ingest":
        from vproc.ingest.pipeline import ingest_video

        n = ingest_video(args[1])
        print(f"ingested {n} segments from {args[1]}")
        if n == 0:
            print("warning: 0 segments ingested (no transcript and no readable frames?)", file=sys.stderr)
    elif args[:1] == ["serve"]:
        import uvicorn

        from vproc.service import create_app

        cfg = load_config()
        uvicorn.run(create_app(), host=cfg.host, port=cfg.port)
    elif len(args) >= 2 and args[0] == "speakers":
        _speakers(args[1:])
    else:
        print("usage: vproc [ingest <video.mp4> | serve | speakers <memory> [--set 'SPEAKER_XX=Name']]")
        sys.exit(1)
```

(IMPORTANT: read the current `vproc/cli.py` first and preserve its existing `ingest`/`serve` branches exactly as they are today — including the existing 0-segment warning wording if it differs from the above — only ADD `_store_for`, `_speakers`, the `speakers` dispatch branch, and the widened usage line.)

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/test_cli.py tests/test_store.py -q` — Expected: all pass
Then: `uv run pytest -q` — Expected: whole suite passes

- [ ] **Step 8: Commit**

```bash
git add vproc/cli.py vproc/store/lancedb_store.py tests/test_cli.py tests/test_store.py
git commit -m "feat(cli): vproc speakers — list evidence-backed name map, manual --set rename

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: End-to-end on the real test video (local stack)

**Files:** none (verification; a follow-up commit only if a real bug is found and fixed)

**Interfaces:**
- Consumes: everything above; LM Studio on `localhost:1234` (start with `~/.lmstudio/bin/lms server start`; if the vision model was TTL-evicted, `~/.lmstudio/bin/lms load qwen/qwen3-vl-8b --context-length 16384 --ttl 3600`); `HF_TOKEN` in `.env` (present).

- [ ] **Step 1: Preconditions**

```bash
curl -sf -m 5 http://localhost:1234/v1/models | grep -q qwen3-vl-8b && echo OK || echo "START LM STUDIO FIRST"
grep -c "^HF_TOKEN=hf_" /Users/jzfre/Code/personal/vproc/.env   # expect 1
```

If LM Studio can't be brought up, STOP and report — do not point at other endpoints.

- [ ] **Step 2: Full ingest with naming**

```bash
cd /Users/jzfre/Code/personal/vproc
export VPROC_INDEX_PATH=./vproc_name.lance VPROC_FRAMES_DIR=./vproc_frames_name
rm -rf vproc_name.lance vproc_frames_name
uv run vproc ingest tmp/test.mkv
```

Expected: `named: SPEAKER_xx -> ...` lines appear; based on the 2026-07-26 spike, expect
`Vanco, Pavol`, `Selrico Lamont Martin` (possibly on two merged clusters), and
`Repan, Jozef` named; zero or one `unresolved speaker:` lines (likely `SPEAKER_02`);
NO `speaker naming failed` warning; `ingested N segments` with N in the 55–75 range —
naming runs BEFORE segment building, so merging two clusters under one name REDUCES
speaker-change splits vs the unnamed run's 73; a lower N here is correct, not a
regression. Wall time ≈ 8 min (5.5 min pipeline + ~2 min probes).

- [ ] **Step 3: Verify names in stored rows**

```bash
VPROC_INDEX_PATH=./vproc_name.lance uv run python -c "
from vproc.store.lancedb_store import Store
from collections import Counter
rows = Store('./vproc_name.lance')._table().to_arrow().to_pylist()
c = Counter(r['speaker'] for r in rows)
print(dict(c))
named = [s for s in c if not s.startswith('SPEAKER_')]
assert len(named) >= 2, c
print('OK: real names in rows:', named)"
```

- [ ] **Step 4: CLI round-trip**

```bash
uv run vproc speakers test                      # shows mapping + any suggestion with evidence
# if a suggestion exists (e.g. SPEAKER_02), resolve it manually using the evidence:
uv run vproc speakers test --set 'SPEAKER_02=PATINO, DANIEL'
uv run vproc speakers test                      # now shows the manual mapping
```

Expected: list output matches Step 2's summary; after `--set`, re-dump rows (Step 3 command) and confirm the renamed label appears.

- [ ] **Step 5: Name-filtered search through the service**

```bash
VPROC_INDEX_PATH=./vproc_name.lance VPROC_PORT=8795 VPROC_HOST=127.0.0.1 uv run vproc serve &
SRV=$!; sleep 6
curl -sS -m 60 -X POST http://127.0.0.1:8795/search -H 'Content-Type: application/json' \
  -d '{"query": "middleware", "speaker": "Repan, Jozef"}' \
  | python3 -c "
import json, sys
hits = json.load(sys.stdin)
assert hits and all(h['speaker'] == 'Repan, Jozef' for h in hits), [h.get('speaker') for h in hits]
print('OK:', len(hits), 'hits, all Repan, Jozef')"
kill $SRV
```

(Note: the name contains a comma and space — the `_safe` filter accepts both since the injection-guard rewrite.)

- [ ] **Step 6: Report** — named clusters + vote counts, any unresolved cluster and its evidence, wall time, and a 3-row sample of `(start, speaker, said_text[:60])`. Clean up: `rm -rf vproc_name.lance vproc_frames_name` after the owner confirms.
