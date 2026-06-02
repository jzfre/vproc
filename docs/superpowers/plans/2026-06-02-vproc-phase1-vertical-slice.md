# vproc Phase 1 — Vertical Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ingest one meeting `.mp4` into a grounded, searchable index and answer questions about it from VS Code over HTTP — with a hallucination-guard test proving off-topic questions return "Not discussed in these meetings."

**Architecture:** One Python HTTP service holds all business logic (ffmpeg, transcription, retrieval, faithfulness, grounding orchestration) and calls three external OpenAI-compatible AI endpoints (OCR, embeddings, grounding) configured by env var. Grounding is *structural*: code owns citations, code decides abstention, an independent NLI model (HHEM) verifies each claim. Every AI/IO call is injected so the logic is unit-testable offline with fakes.

**Tech Stack:** Python 3.12 (uv) · openai SDK (any `/v1`) · LanceDB (hybrid vector+FTS) · mlx-whisper (ASR) · ffmpeg + imagehash/Pillow (frames) · transformers (Vectara HHEM-2.1-Open) · FastAPI + the `mcp` SDK (REST + MCP over HTTP) · pytest.

**Spec:** `docs/superpowers/specs/2026-06-02-vproc-design.md`

---

## File structure (created in this plan)

```
vproc/
  __init__.py
  config.py                 # Config/Endpoint dataclasses from env (Task 2)
  models.py                 # Project, Memory, Segment, Evidence, Citation, Claim, Answer, mmss (Task 3)
  llm/
    __init__.py
    client.py               # OpenAI /v1: embed_texts, chat_json, ocr_image (Task 4)
  ingest/
    __init__.py
    frames.py               # ffmpeg cmd, parse pts_time, pHash dedup (Task 5)
    transcribe.py           # mlx-whisper wrapper + raw→TranscriptSegment (Task 6)
    ocr.py                  # OCR endpoint call (Task 7)
    align.py                # screen-state intervals + segment building (Task 8)
    embed_index.py          # embed segments + write to store (Task 10)
    pipeline.py             # orchestrate one-memory ingest (Task 11)
  store/
    __init__.py
    lancedb_store.py        # Store: add / vector_search / fts_search (Task 9)
  retrieve/
    __init__.py
    retriever.py            # embed query + hybrid RRF + top-sim (Task 12)
  answer/
    __init__.py
    evidence.py             # build Evidence (E1..En) + evidence_block (Task 13)
    generate.py             # grounding call + JSON parse + drop empty-evidence claims (Task 14)
    faithfulness.py         # HHEM wrapper + filter_claims (Task 15)
    ask.py                  # ask_memory / search_memory orchestration (Task 16)
  service.py                # FastAPI REST (/ask,/search,/healthz) + MCP mount (Task 17)
  cli.py                    # `vproc ingest <video>` / `vproc serve` (Task 17)
tests/                      # one test module per source module
.env.example
.vscode/mcp.json
pyproject.toml
```

**Conventions:** every function that calls an AI endpoint, the filesystem, or a subprocess takes the side-effecting callable/handle as a parameter with a real default, so unit tests inject a fake. Tests never touch the network or load real models.

---

## Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `vproc/__init__.py`, `vproc/llm/__init__.py`, `vproc/ingest/__init__.py`, `vproc/store/__init__.py`, `vproc/retrieve/__init__.py`, `vproc/answer/__init__.py`, `tests/__init__.py`, `tests/test_smoke.py`, `.env.example`

- [ ] **Step 1: Ensure `uv` is installed**

Run: `command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh`
Expected: a path to `uv` is printed (restart shell if just installed).

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "vproc"
version = "0.1.0"
description = "Grounded meeting-memory from video"
requires-python = ">=3.12"
dependencies = [
    "openai>=1.40",
    "lancedb>=0.13",
    "pyarrow>=17",
    "pydantic>=2.7",
    "pillow>=10",
    "imagehash>=4.3",
    "fastapi>=0.115",
    "uvicorn>=0.30",
    "mcp>=1.27",
    "numpy>=1.26",
]

[project.optional-dependencies]
# Heavy real-model deps — needed only for real ingest/answering (Task 18), NOT for the
# offline unit tests (which inject fakes; transformers/mlx-whisper are imported lazily).
runtime = [
    "transformers>=4.51",
    "torch>=2.4",
    "mlx-whisper>=0.4; sys_platform == 'darwin'",
]

[project.scripts]
vproc = "vproc.cli:main"

[dependency-groups]
dev = ["pytest>=8", "httpx>=0.27", "ruff>=0.6"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
pythonpath = ["."]
```

- [ ] **Step 3: Create empty package markers**

Create these files, each empty:
`vproc/__init__.py`, `vproc/llm/__init__.py`, `vproc/ingest/__init__.py`, `vproc/store/__init__.py`, `vproc/retrieve/__init__.py`, `vproc/answer/__init__.py`, `tests/__init__.py`.

- [ ] **Step 4: Create `.env.example`**

```bash
# AI endpoints (OpenAI-compatible). Copy to .env and adjust.
VPROC_OCR_BASE_URL=http://voyage:8000/v1
VPROC_OCR_MODEL=QuantTrio/Qwen3.5-9B-AWQ
VPROC_EMBED_BASE_URL=http://localhost:1234/v1
VPROC_EMBED_MODEL=Qwen3-Embedding-0.6B
VPROC_GROUNDING_BASE_URL=http://localhost:1234/v1
VPROC_GROUNDING_MODEL=Qwen3-32B-AWQ
# Service + storage
VPROC_HOST=0.0.0.0
VPROC_PORT=8765
VPROC_INDEX_PATH=./vproc.lance
# Tuning
VPROC_SIM_FLOOR=0.25
VPROC_HHEM_THRESHOLD=0.5
```

- [ ] **Step 5: Write the smoke test** — `tests/test_smoke.py`

```python
def test_package_imports():
    import vproc
    assert vproc is not None
```

- [ ] **Step 6: Sync deps and run the smoke test**

Run: `uv sync && uv run pytest tests/test_smoke.py -v`
Expected: 1 passed. (First `uv sync` downloads torch/transformers — may take a few minutes.)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock vproc tests .env.example
git commit -m "chore: scaffold vproc package (uv, deps, smoke test)"
```

---

## Task 2: Config from environment

**Files:**
- Create: `vproc/config.py`, `tests/test_config.py`

- [ ] **Step 1: Write the failing test** — `tests/test_config.py`

```python
from vproc.config import load_config

def test_defaults(monkeypatch):
    for k in list(__import__("os").environ):
        if k.startswith("VPROC_"):
            monkeypatch.delenv(k, raising=False)
    cfg = load_config()
    assert cfg.ocr.model == "QuantTrio/Qwen3.5-9B-AWQ"
    assert cfg.grounding.base_url.endswith("/v1")
    assert cfg.port == 8765
    assert cfg.sim_floor == 0.25
    assert cfg.hhem_threshold == 0.5

def test_env_overrides(monkeypatch):
    monkeypatch.setenv("VPROC_GROUNDING_BASE_URL", "http://x:9/v1")
    monkeypatch.setenv("VPROC_GROUNDING_MODEL", "my-model")
    monkeypatch.setenv("VPROC_PORT", "9000")
    cfg = load_config()
    assert cfg.grounding.base_url == "http://x:9/v1"
    assert cfg.grounding.model == "my-model"
    assert cfg.port == 9000
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vproc.config'`.

- [ ] **Step 3: Implement** — `vproc/config.py`

```python
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Endpoint:
    base_url: str
    model: str


@dataclass(frozen=True)
class Config:
    ocr: Endpoint
    embed: Endpoint
    grounding: Endpoint
    index_path: str
    host: str
    port: int
    sim_floor: float
    hhem_threshold: float
    hf_token: str | None


def _ep(prefix: str, default_url: str, default_model: str) -> Endpoint:
    return Endpoint(
        base_url=os.environ.get(f"{prefix}_BASE_URL", default_url),
        model=os.environ.get(f"{prefix}_MODEL", default_model),
    )


def load_config() -> Config:
    return Config(
        ocr=_ep("VPROC_OCR", "http://voyage:8000/v1", "QuantTrio/Qwen3.5-9B-AWQ"),
        embed=_ep("VPROC_EMBED", "http://localhost:1234/v1", "Qwen3-Embedding-0.6B"),
        grounding=_ep("VPROC_GROUNDING", "http://localhost:1234/v1", "Qwen3-32B-AWQ"),
        index_path=os.environ.get("VPROC_INDEX_PATH", "./vproc.lance"),
        host=os.environ.get("VPROC_HOST", "0.0.0.0"),
        port=int(os.environ.get("VPROC_PORT", "8765")),
        sim_floor=float(os.environ.get("VPROC_SIM_FLOOR", "0.25")),
        hhem_threshold=float(os.environ.get("VPROC_HHEM_THRESHOLD", "0.5")),
        hf_token=os.environ.get("HF_TOKEN"),
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add vproc/config.py tests/test_config.py
git commit -m "feat: env-driven Config with 3 AI endpoints"
```

---

## Task 3: Domain models

**Files:**
- Create: `vproc/models.py`, `tests/test_models.py`

- [ ] **Step 1: Write the failing test** — `tests/test_models.py`

```python
from vproc.models import Segment, Evidence, Citation, Claim, Answer, mmss

def test_mmss():
    assert mmss(0) == "00:00"
    assert mmss(75.4) == "01:15"
    assert mmss(3661) == "61:01"

def test_segment_roundtrips():
    s = Segment(id="s1", project_id="p", memory_id="m", screen_state_id="ss0",
                start_ts=1.0, end_ts=2.0, speaker="SPEAKER_0", said_text="hi",
                on_screen_text="slide", on_screen_confidence=None,
                frame_path="/f.png", source_video="/v.mp4", embed_text="[SCREEN]\nslide")
    assert Segment(**s.model_dump()) == s

def test_answer_defaults():
    a = Answer(answered=False, abstained=True, text="Not discussed in these meetings.",
               claims=[], evidence=[])
    assert a.answered is False and a.claims == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vproc.models'`.

- [ ] **Step 3: Implement** — `vproc/models.py`

```python
from pydantic import BaseModel, Field


def mmss(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


class Project(BaseModel):
    id: str
    name: str
    created_at: str


class Memory(BaseModel):
    id: str
    project_id: str
    title: str
    source_video: str
    duration_s: float | None = None
    status: str = "ready"
    created_at: str | None = None


class Segment(BaseModel):
    id: str
    project_id: str
    memory_id: str
    screen_state_id: str | None = None
    start_ts: float
    end_ts: float
    speaker: str
    said_text: str
    on_screen_text: str = ""
    on_screen_confidence: float | None = None
    frame_path: str | None = None
    source_video: str
    embed_text: str


class Evidence(BaseModel):
    key: str
    segment_id: str
    memory_title: str
    start_ts: float
    end_ts: float
    speaker: str
    text: str


class Citation(BaseModel):
    memory_title: str
    start_ts: float
    end_ts: float
    speaker: str


class Claim(BaseModel):
    text: str
    evidence_ids: list[str]
    citations: list[Citation] = Field(default_factory=list)


class Answer(BaseModel):
    answered: bool
    abstained: bool
    text: str
    claims: list[Claim]
    evidence: list[Evidence]
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_models.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add vproc/models.py tests/test_models.py
git commit -m "feat: pydantic domain models + mmss helper"
```

---

## Task 4: LLM client (OpenAI `/v1` wrappers)

**Files:**
- Create: `vproc/llm/client.py`, `tests/test_llm_client.py`

- [ ] **Step 1: Write the failing test** — `tests/test_llm_client.py`

```python
import vproc.llm.client as c

class _Resp:
    def __init__(self, payload): self._p = payload

class _FakeEmbeddings:
    def create(self, model, input):
        class D:  # noqa: N801
            def __init__(self, e): self.embedding = e
        return type("R", (), {"data": [D([float(len(t))]) for t in input]})()

class _FakeChat:
    def __init__(self, capture): self.capture = capture
    @property
    def completions(self):
        outer = self
        class C:
            def create(self, **kw):
                outer.capture.update(kw)
                msg = type("M", (), {"content": "OK"})()
                return type("R", (), {"choices": [type("Ch", (), {"message": msg})()]})()
        return C()

class _FakeClient:
    def __init__(self, capture):
        self.embeddings = _FakeEmbeddings()
        self.chat = _FakeChat(capture)

def test_embed_texts(monkeypatch):
    monkeypatch.setattr(c, "_client", lambda base_url: _FakeClient({}))
    out = c.embed_texts("u", "m", ["a", "bb"])
    assert out == [[1.0], [2.0]]

def test_ocr_image_builds_data_url(tmp_path, monkeypatch):
    cap = {}
    monkeypatch.setattr(c, "_client", lambda base_url: _FakeClient(cap))
    img = tmp_path / "f.png"; img.write_bytes(b"\x89PNG\r\n\x1a\n")
    out = c.ocr_image("u", "m", str(img), "PROMPT")
    assert out == "OK"
    content = cap["messages"][0]["content"]
    assert content[0]["text"] == "PROMPT"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

def test_chat_json_sets_json_format(monkeypatch):
    cap = {}
    monkeypatch.setattr(c, "_client", lambda base_url: _FakeClient(cap))
    out = c.chat_json("u", "m", "sys", "usr")
    assert out == "OK"
    assert cap["response_format"] == {"type": "json_object"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_llm_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vproc.llm.client'`.

- [ ] **Step 3: Implement** — `vproc/llm/client.py`

```python
import base64
import mimetypes

from openai import OpenAI


def _client(base_url: str) -> OpenAI:
    # api_key is required by the SDK but unused by local servers
    return OpenAI(base_url=base_url, api_key="not-needed")


def embed_texts(base_url: str, model: str, texts: list[str]) -> list[list[float]]:
    resp = _client(base_url).embeddings.create(model=model, input=texts)
    return [list(d.embedding) for d in resp.data]


def chat_json(base_url: str, model: str, system: str, user: str, temperature: float = 0.1) -> str:
    resp = _client(base_url).chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


def ocr_image(base_url: str, model: str, image_path: str, prompt: str) -> str:
    data = base64.b64encode(open(image_path, "rb").read()).decode()
    mime = mimetypes.guess_type(image_path)[0] or "image/png"
    resp = _client(base_url).chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
        ]}],
        temperature=0.0,
    )
    return resp.choices[0].message.content
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_llm_client.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add vproc/llm/client.py tests/test_llm_client.py
git commit -m "feat: OpenAI /v1 client (embed, chat_json, ocr_image)"
```

---

## Task 5: Frame sampling (ffmpeg + pHash dedup)

**Files:**
- Create: `vproc/ingest/frames.py`, `tests/test_frames.py`

- [ ] **Step 1: Write the failing test** — `tests/test_frames.py`

```python
from PIL import Image
from vproc.ingest.frames import ffmpeg_sample_cmd, parse_frames_log, phash_dedup, RawFrame

def _img(tmp_path, name, color):
    p = tmp_path / name
    Image.new("RGB", (64, 64), color).save(p)
    return str(p)

def test_cmd_has_filters():
    cmd = ffmpeg_sample_cmd("in.mp4", "/out", "/out/frames.log", scene=0.08)
    joined = " ".join(cmd)
    assert "mpdecimate" in joined and "gt(scene,0.08)" in joined
    assert "metadata=print:file=/out/frames.log" in joined

def test_parse_frames_log_pairs_paths_to_times():
    log = "frame:0 pts_time:0.000000\nframe:1 pts_time:12.500000\n"
    frames = parse_frames_log(log, ["/o/00000002.png", "/o/00000001.png"])
    assert [f.t for f in frames] == [0.0, 12.5]
    # paths are sorted before pairing
    assert frames[0].path == "/o/00000001.png"

def test_phash_dedup_drops_near_duplicates(tmp_path):
    a = _img(tmp_path, "a.png", (0, 0, 0))
    a2 = _img(tmp_path, "a2.png", (0, 0, 0))      # identical → dropped
    b = _img(tmp_path, "b.png", (255, 255, 255))  # different → kept
    frames = [RawFrame(a, 0.0), RawFrame(a2, 1.0), RawFrame(b, 2.0)]
    kept = phash_dedup(frames, threshold=6, floor_s=999)
    assert [f.path for f in kept] == [a, b]

def test_phash_dedup_floor_forces_anchor(tmp_path):
    a = _img(tmp_path, "a.png", (0, 0, 0))
    a2 = _img(tmp_path, "a2.png", (0, 0, 0))
    frames = [RawFrame(a, 0.0), RawFrame(a2, 40.0)]  # identical but 40s apart
    kept = phash_dedup(frames, threshold=6, floor_s=30)
    assert [f.path for f in kept] == [a, a2]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_frames.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vproc.ingest.frames'`.

- [ ] **Step 3: Implement** — `vproc/ingest/frames.py`

```python
import re
from dataclasses import dataclass

import imagehash
from PIL import Image


@dataclass
class RawFrame:
    path: str
    t: float


def ffmpeg_sample_cmd(video: str, out_dir: str, log_path: str, scene: float = 0.08) -> list[str]:
    vf = f"mpdecimate,select='gt(scene,{scene})',metadata=print:file={log_path}"
    return ["ffmpeg", "-hide_banner", "-i", video, "-vf", vf,
            "-fps_mode", "vfr", "-frame_pts", "1", f"{out_dir}/%08d.png"]


def parse_frames_log(log_text: str, frame_paths: list[str]) -> list[RawFrame]:
    times = [float(x) for x in re.findall(r"pts_time:([0-9.]+)", log_text)]
    paths = sorted(frame_paths)
    return [RawFrame(path=p, t=t) for p, t in zip(paths, times)]


def phash_dedup(frames: list[RawFrame], threshold: int = 6, floor_s: float = 30.0) -> list[RawFrame]:
    kept: list[RawFrame] = []
    last_hash = None
    last_t = None
    for f in frames:
        h = imagehash.phash(Image.open(f.path))
        changed = last_hash is None or (h - last_hash) > threshold
        anchor = last_t is None or (f.t - last_t) >= floor_s
        if changed or anchor:
            kept.append(f)
            last_hash, last_t = h, f.t
    return kept
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_frames.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add vproc/ingest/frames.py tests/test_frames.py
git commit -m "feat: frame sampling cmd + pts_time parse + pHash dedup"
```

---

## Task 6: Transcription (mlx-whisper wrapper)

**Files:**
- Create: `vproc/ingest/transcribe.py`, `tests/test_transcribe.py`

- [ ] **Step 1: Write the failing test** — `tests/test_transcribe.py`

```python
from vproc.ingest.transcribe import segments_from_whisper, extract_audio_cmd, TranscriptSegment

def test_extract_audio_cmd():
    cmd = extract_audio_cmd("in.mp4", "/o/a.wav")
    assert cmd[:1] == ["ffmpeg"]
    assert "-ar" in cmd and "16000" in cmd and cmd[-1] == "/o/a.wav"

def test_segments_from_whisper_filters_empty_and_strips():
    result = {"segments": [
        {"start": 0.0, "end": 1.0, "text": " hello "},
        {"start": 1.0, "end": 2.0, "text": "   "},
        {"start": 2.0, "end": 3.0, "text": "world"},
    ]}
    segs = segments_from_whisper(result)
    assert segs == [
        TranscriptSegment(0.0, 1.0, "hello"),
        TranscriptSegment(2.0, 3.0, "world"),
    ]
    assert segs[0].speaker == "SPEAKER_0"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_transcribe.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vproc.ingest.transcribe'`.

- [ ] **Step 3: Implement** — `vproc/ingest/transcribe.py`

```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_transcribe.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add vproc/ingest/transcribe.py tests/test_transcribe.py
git commit -m "feat: mlx-whisper transcription wrapper + segment parsing"
```

---

## Task 7: Screen OCR (endpoint call)

**Files:**
- Create: `vproc/ingest/ocr.py`, `tests/test_ocr.py`

- [ ] **Step 1: Write the failing test** — `tests/test_ocr.py`

```python
from vproc.config import Endpoint
from vproc.ingest.ocr import ocr_frame, OCR_PROMPT

def test_ocr_frame_calls_endpoint_with_prompt():
    calls = {}
    def fake_chat(base_url, model, image_path, prompt):
        calls.update(base_url=base_url, model=model, image_path=image_path, prompt=prompt)
        return "  Quarterly Plan  "
    ep = Endpoint("http://voyage:8000/v1", "QuantTrio/Qwen3.5-9B-AWQ")
    text = ocr_frame(ep, "/frame.png", chat=fake_chat)
    assert text == "Quarterly Plan"
    assert calls["base_url"] == "http://voyage:8000/v1"
    assert calls["image_path"] == "/frame.png"
    assert "verbatim" in OCR_PROMPT.lower() or "only" in OCR_PROMPT.lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_ocr.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vproc.ingest.ocr'`.

- [ ] **Step 3: Implement** — `vproc/ingest/ocr.py`

```python
from vproc.config import Endpoint
from vproc.llm import client

OCR_PROMPT = (
    "You are an OCR transcription engine. Transcribe ONLY the text that is visibly "
    "rendered as characters in the image, in natural reading order (top-to-bottom, "
    "left-to-right). Do NOT infer, complete, translate, or correct spelling. If a region "
    "is unreadable, output the token [illegible] for it. Output only the literal on-screen "
    "text, with no commentary."
)


def ocr_frame(ocr: Endpoint, image_path: str, chat=client.ocr_image) -> str:
    return chat(ocr.base_url, ocr.model, image_path, OCR_PROMPT).strip()
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_ocr.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add vproc/ingest/ocr.py tests/test_ocr.py
git commit -m "feat: OCR via configured endpoint with strict verbatim prompt"
```

---

## Task 8: Align & build segments (the core logic)

**Files:**
- Create: `vproc/ingest/align.py`, `tests/test_align.py`

- [ ] **Step 1: Write the failing test** — `tests/test_align.py`

```python
from vproc.ingest.frames import RawFrame
from vproc.ingest.transcribe import TranscriptSegment as TS
from vproc.ingest.align import build_screen_states, assign_segments, build_segments, make_embed_text

def test_build_screen_states_intervals():
    frames = [RawFrame("a.png", 0.0), RawFrame("b.png", 10.0)]
    states = build_screen_states(frames, end_time=25.0)
    assert (states[0].t_start, states[0].t_end) == (0.0, 10.0)
    assert (states[1].t_start, states[1].t_end) == (10.0, 25.0)

def test_assign_by_midpoint_when_straddling():
    frames = [RawFrame("a.png", 0.0), RawFrame("b.png", 10.0)]
    states = build_screen_states(frames, end_time=20.0)
    # segment 8.0-12.0 straddles the 10.0 boundary; midpoint 10.0 → state b
    seg = TS(8.0, 12.0, "boundary talk")
    buckets = assign_segments(states, [seg])
    assert buckets["ss0"] == []
    assert buckets["ss1"] == [seg]

def test_build_segments_attaches_screen_and_spoken():
    frames = [RawFrame("a.png", 0.0)]
    states = build_screen_states(frames, end_time=10.0)
    states[0].on_screen_text = "Roadmap Q3"
    transcript = [TS(1.0, 3.0, "we ship in July", speaker="SPEAKER_0")]
    segs = build_segments("p", "meeting-1", "/v.mp4", states, transcript)
    assert len(segs) == 1
    s = segs[0]
    assert s.said_text == "we ship in July"
    assert s.on_screen_text == "Roadmap Q3"
    assert "[SCREEN]\nRoadmap Q3" in s.embed_text
    assert "[SPOKEN]\nSPEAKER_0 (00:01): we ship in July" in s.embed_text
    assert s.start_ts == 1.0 and s.end_ts == 3.0

def test_build_segments_skips_empty_states():
    frames = [RawFrame("a.png", 0.0)]
    states = build_screen_states(frames, end_time=10.0)  # no OCR text, no transcript
    assert build_segments("p", "m", "/v.mp4", states, []) == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_align.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vproc.ingest.align'`.

- [ ] **Step 3: Implement** — `vproc/ingest/align.py`

```python
import uuid
from dataclasses import dataclass, field

from vproc.ingest.frames import RawFrame
from vproc.ingest.transcribe import TranscriptSegment
from vproc.models import Segment, mmss


@dataclass
class ScreenState:
    screen_state_id: str
    t_start: float
    t_end: float
    frame_path: str
    on_screen_text: str = ""


def build_screen_states(frames: list[RawFrame], end_time: float) -> list[ScreenState]:
    states: list[ScreenState] = []
    for i, f in enumerate(frames):
        t_end = frames[i + 1].t if i + 1 < len(frames) else end_time
        states.append(ScreenState(f"ss{i}", f.t, t_end, f.path))
    return states


def _midpoint(seg: TranscriptSegment) -> float:
    return (seg.start + seg.end) / 2.0


def assign_segments(states: list[ScreenState],
                    transcript: list[TranscriptSegment]) -> dict[str, list[TranscriptSegment]]:
    buckets: dict[str, list[TranscriptSegment]] = {s.screen_state_id: [] for s in states}
    for seg in transcript:
        mid = _midpoint(seg)
        chosen = next((s for s in states if s.t_start <= mid < s.t_end), None)
        if chosen is None and states:
            chosen = states[-1] if mid >= states[-1].t_start else states[0]
        if chosen is not None:
            buckets[chosen.screen_state_id].append(seg)
    return buckets


def make_embed_text(on_screen: str, segs: list[TranscriptSegment]) -> str:
    spoken = "\n".join(f"{s.speaker} ({mmss(s.start)}): {s.text}" for s in segs)
    return f"[SCREEN]\n{on_screen}\n[SPOKEN]\n{spoken}"


def build_segments(project_id: str, memory_id: str, source_video: str,
                   states: list[ScreenState],
                   transcript: list[TranscriptSegment]) -> list[Segment]:
    buckets = assign_segments(states, transcript)
    out: list[Segment] = []
    for s in states:
        segs = buckets[s.screen_state_id]
        if not segs and not s.on_screen_text.strip():
            continue
        said = " ".join(x.text for x in segs)
        start = segs[0].start if segs else s.t_start
        end = segs[-1].end if segs else s.t_end
        speaker = segs[0].speaker if segs else "SPEAKER_0"
        out.append(Segment(
            id=str(uuid.uuid4()),
            project_id=project_id, memory_id=memory_id, screen_state_id=s.screen_state_id,
            start_ts=start, end_ts=end, speaker=speaker,
            said_text=said, on_screen_text=s.on_screen_text, on_screen_confidence=None,
            frame_path=s.frame_path, source_video=source_video,
            embed_text=make_embed_text(s.on_screen_text, segs),
        ))
    return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_align.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add vproc/ingest/align.py tests/test_align.py
git commit -m "feat: screen-state intervals + midpoint assignment + segment building"
```

---

## Task 9: LanceDB store (hybrid add/search)

**Files:**
- Create: `vproc/store/lancedb_store.py`, `tests/test_store.py`

- [ ] **Step 1: Write the failing test** — `tests/test_store.py`

```python
from vproc.store.lancedb_store import Store

def _row(id, project, vec, text="x"):
    return {"id": id, "project_id": project, "memory_id": "m", "speaker": "SPEAKER_0",
            "start_ts": 0.0, "end_ts": 1.0, "said_text": text, "on_screen_text": "",
            "embed_text": text, "source_video": "/v.mp4", "frame_path": "", "vector": vec}

def test_vector_search_orders_by_similarity(tmp_path):
    store = Store(str(tmp_path / "db.lance"))
    store.add([_row("near", "p", [1.0, 0.0]), _row("far", "p", [0.0, 1.0])])
    hits = store.vector_search([1.0, 0.0], k=2)
    assert hits[0]["id"] == "near"
    assert "_distance" in hits[0]

def test_metadata_prefilter(tmp_path):
    store = Store(str(tmp_path / "db.lance"))
    store.add([_row("a", "p1", [1.0, 0.0]), _row("b", "p2", [1.0, 0.0])])
    hits = store.vector_search([1.0, 0.0], k=5, where="project_id = 'p2'")
    assert [h["id"] for h in hits] == ["b"]

def test_search_on_empty_store_returns_empty(tmp_path):
    store = Store(str(tmp_path / "db.lance"))
    assert store.vector_search([1.0, 0.0], k=3) == []
    assert store.fts_search("anything", k=3) == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vproc.store.lancedb_store'`.

- [ ] **Step 3: Implement** — `vproc/store/lancedb_store.py`

```python
import lancedb


class Store:
    """Embedded LanceDB hybrid store. Table is created lazily on first add
    (so the vector dimension is taken from the data)."""

    TABLE = "segments"

    def __init__(self, path: str):
        self.db = lancedb.connect(path)

    def _table(self):
        if self.TABLE in self.db.table_names():
            return self.db.open_table(self.TABLE)
        return None

    def add(self, rows: list[dict]) -> None:
        if not rows:
            return
        table = self._table()
        if table is None:
            table = self.db.create_table(self.TABLE, data=rows)
        else:
            table.add(rows)
        try:
            table.create_fts_index("embed_text", use_tantivy=False, replace=True)
        except Exception:
            pass  # FTS is best-effort; vector search still works

    def vector_search(self, vector, k: int, where: str | None = None) -> list[dict]:
        table = self._table()
        if table is None:
            return []
        q = table.search(vector).metric("cosine").limit(k)
        if where:
            q = q.where(where, prefilter=True)
        return q.to_list()

    def fts_search(self, text: str, k: int, where: str | None = None) -> list[dict]:
        table = self._table()
        if table is None:
            return []
        try:
            q = table.search(text, query_type="fts").limit(k)
            if where:
                q = q.where(where, prefilter=True)
            return q.to_list()
        except Exception:
            return []
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_store.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add vproc/store/lancedb_store.py tests/test_store.py
git commit -m "feat: LanceDB store with lazy schema, vector + fts search, metadata prefilter"
```

---

## Task 10: Embed & index

**Files:**
- Create: `vproc/ingest/embed_index.py`, `tests/test_embed_index.py`

- [ ] **Step 1: Write the failing test** — `tests/test_embed_index.py`

```python
from vproc.config import Config, Endpoint
from vproc.models import Segment
from vproc.store.lancedb_store import Store
from vproc.ingest.embed_index import embed_and_store

def _cfg():
    ep = Endpoint("u", "m")
    return Config(ep, ep, ep, "x", "0.0.0.0", 8765, 0.25, 0.5, None)

def _seg(i):
    return Segment(id=f"s{i}", project_id="p", memory_id="m", start_ts=0.0, end_ts=1.0,
                   speaker="SPEAKER_0", said_text="t", source_video="/v.mp4", embed_text=f"text {i}")

def test_embed_and_store_writes_rows(tmp_path):
    store = Store(str(tmp_path / "db.lance"))
    def fake_embed(base_url, model, texts):
        return [[float(len(t)), 1.0] for t in texts]
    embed_and_store(_cfg(), store, [_seg(1), _seg(2)], embed=fake_embed)
    hits = store.vector_search([6.0, 1.0], k=2)
    assert {h["id"] for h in hits} == {"s1", "s2"}

def test_embed_and_store_noop_when_empty(tmp_path):
    store = Store(str(tmp_path / "db.lance"))
    embed_and_store(_cfg(), store, [], embed=lambda *a: [])  # must not raise
    assert store.vector_search([0.0, 0.0], k=1) == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_embed_index.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vproc.ingest.embed_index'`.

- [ ] **Step 3: Implement** — `vproc/ingest/embed_index.py`

```python
from vproc.llm import client
from vproc.models import Segment


def _row(seg: Segment, vector: list[float]) -> dict:
    return {
        "id": seg.id, "project_id": seg.project_id, "memory_id": seg.memory_id,
        "speaker": seg.speaker, "start_ts": seg.start_ts, "end_ts": seg.end_ts,
        "said_text": seg.said_text, "on_screen_text": seg.on_screen_text,
        "embed_text": seg.embed_text, "source_video": seg.source_video,
        "frame_path": seg.frame_path or "", "vector": vector,
    }


def embed_and_store(cfg, store, segments: list[Segment], embed=client.embed_texts) -> None:
    if not segments:
        return
    vectors = embed(cfg.embed.base_url, cfg.embed.model, [s.embed_text for s in segments])
    store.add([_row(seg, vec) for seg, vec in zip(segments, vectors)])
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_embed_index.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add vproc/ingest/embed_index.py tests/test_embed_index.py
git commit -m "feat: embed segments via endpoint and write to store"
```

---

## Task 11: Ingest pipeline + CLI ingest

**Files:**
- Create: `vproc/ingest/pipeline.py`, `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test** — `tests/test_pipeline.py`

This test exercises the orchestration with the ffmpeg/whisper/ocr/embed boundaries faked, so it runs with no media or models. It monkeypatches the subprocess calls and the module-level helpers `pipeline` imports.

```python
import vproc.ingest.pipeline as P
from vproc.config import Config, Endpoint
from vproc.store.lancedb_store import Store
from vproc.ingest.frames import RawFrame
from vproc.ingest.transcribe import TranscriptSegment

def _cfg():
    ep = Endpoint("u", "m")
    return Config(ep, ep, ep, "x", "0.0.0.0", 8765, 0.25, 0.5, None)

def test_ingest_video_builds_segments(tmp_path, monkeypatch):
    store = Store(str(tmp_path / "db.lance"))

    monkeypatch.setattr(P.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(P, "_list_frames", lambda d: ["/f/0.png", "/f/1.png"])
    monkeypatch.setattr(P.F, "parse_frames_log",
                        lambda text, paths: [RawFrame("/f/0.png", 0.0), RawFrame("/f/1.png", 10.0)])
    monkeypatch.setattr(P.F, "phash_dedup", lambda frames, **k: frames)
    monkeypatch.setattr(P, "_read_log", lambda p: "")
    monkeypatch.setattr(P.T, "transcribe",
                        lambda wav: [TranscriptSegment(1.0, 3.0, "we ship in July")])
    monkeypatch.setattr(P.O, "ocr_frame", lambda ocr, path: "Roadmap Q3")
    monkeypatch.setattr(P.EI, "embed_and_store",
                        lambda cfg, store, segs, **k: store.add(
                            [{"id": s.id, "project_id": s.project_id, "memory_id": s.memory_id,
                              "speaker": s.speaker, "start_ts": s.start_ts, "end_ts": s.end_ts,
                              "said_text": s.said_text, "on_screen_text": s.on_screen_text,
                              "embed_text": s.embed_text, "source_video": s.source_video,
                              "frame_path": s.frame_path or "", "vector": [1.0, 0.0]} for s in segs]))

    n = P.ingest_video("/videos/standup.mp4", cfg=_cfg(), store=store)
    assert n == 2  # two screen states, both have content
    hits = store.vector_search([1.0, 0.0], k=5)
    assert any("we ship in July" in h["said_text"] for h in hits)
    assert all(h["memory_id"] == "standup" for h in hits)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vproc.ingest.pipeline'`.

- [ ] **Step 3: Implement** — `vproc/ingest/pipeline.py`

```python
import os
import pathlib
import subprocess
import tempfile

from vproc.config import load_config
from vproc.ingest import align as A
from vproc.ingest import embed_index as EI
from vproc.ingest import frames as F
from vproc.ingest import ocr as O
from vproc.ingest import transcribe as T
from vproc.store.lancedb_store import Store


def _list_frames(frames_dir: str) -> list[str]:
    return sorted(str(p) for p in pathlib.Path(frames_dir).glob("*.png"))


def _read_log(path: str) -> str:
    return pathlib.Path(path).read_text() if os.path.exists(path) else ""


def ingest_video(video_path: str, cfg=None, store=None, project_id: str = "default") -> int:
    cfg = cfg or load_config()
    store = store or Store(cfg.index_path)
    title = pathlib.Path(video_path).stem  # doubles as memory_id / citation title in v1

    workdir = tempfile.mkdtemp(prefix="vproc-")
    frames_dir = os.path.join(workdir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    log_path = os.path.join(workdir, "frames.log")
    wav_path = os.path.join(workdir, "audio.wav")

    subprocess.run(F.ffmpeg_sample_cmd(video_path, frames_dir, log_path), check=True)
    raw = F.parse_frames_log(_read_log(log_path), _list_frames(frames_dir))
    kept = F.phash_dedup(raw)

    subprocess.run(T.extract_audio_cmd(video_path, wav_path), check=True)
    transcript = T.transcribe(wav_path)

    end_time = transcript[-1].end if transcript else (kept[-1].t if kept else 0.0)
    states = A.build_screen_states(kept, end_time)
    for state in states:
        state.on_screen_text = O.ocr_frame(cfg.ocr, state.frame_path)

    segments = A.build_segments(project_id, title, video_path, states, transcript)
    EI.embed_and_store(cfg, store, segments)
    return len(segments)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add vproc/ingest/pipeline.py tests/test_pipeline.py
git commit -m "feat: ingest pipeline orchestrating frames→whisper→ocr→align→index"
```

---

## Task 12: Retriever (hybrid RRF + top-similarity)

**Files:**
- Create: `vproc/retrieve/retriever.py`, `tests/test_retriever.py`

- [ ] **Step 1: Write the failing test** — `tests/test_retriever.py`

```python
from vproc.config import Config, Endpoint
from vproc.store.lancedb_store import Store
from vproc.retrieve.retriever import retrieve, rrf

def _cfg():
    ep = Endpoint("u", "m")
    return Config(ep, ep, ep, "x", "0.0.0.0", 8765, 0.25, 0.5, None)

def _row(id, vec, text):
    return {"id": id, "project_id": "p", "memory_id": "m", "speaker": "SPEAKER_0",
            "start_ts": 0.0, "end_ts": 1.0, "said_text": text, "on_screen_text": "",
            "embed_text": text, "source_video": "/v.mp4", "frame_path": "", "vector": vec}

def test_rrf_merges_rankings():
    scores = rrf([["a", "b"], ["b", "c"]])
    assert scores["b"] > scores["a"]  # b appears in both lists

def test_retrieve_returns_hits_and_top_sim(tmp_path):
    store = Store(str(tmp_path / "db.lance"))
    store.add([_row("near", [1.0, 0.0], "petrol engine cycle"),
               _row("far", [0.0, 1.0], "hiring budget")])
    def fake_embed(base_url, model, texts):
        return [[1.0, 0.0]]  # query embeds near "near"
    hits, top_sim = retrieve(store, _cfg(), "engine", k=2, embed=fake_embed)
    assert hits[0]["id"] == "near"
    assert top_sim > 0.9  # cosine sim of identical direction ≈ 1

def test_retrieve_empty_store(tmp_path):
    store = Store(str(tmp_path / "db.lance"))
    hits, top_sim = retrieve(store, _cfg(), "anything", embed=lambda *a: [[1.0, 0.0]])
    assert hits == [] and top_sim == 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_retriever.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vproc.retrieve.retriever'`.

- [ ] **Step 3: Implement** — `vproc/retrieve/retriever.py`

```python
from vproc.llm import client


def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, sid in enumerate(ranking):
            scores[sid] = scores.get(sid, 0.0) + 1.0 / (k + rank + 1)
    return scores


def retrieve(store, cfg, query: str, k: int = 8, where: str | None = None,
             embed=client.embed_texts):
    qvec = embed(cfg.embed.base_url, cfg.embed.model, [query])[0]
    vhits = store.vector_search(qvec, k, where)
    fhits = store.fts_search(query, k, where)
    if not vhits and not fhits:
        return [], 0.0
    top_sim = (1.0 - float(vhits[0]["_distance"])) if vhits else 0.0
    by_id = {h["id"]: h for h in (vhits + fhits)}
    ranked = sorted(
        rrf([[h["id"] for h in vhits], [h["id"] for h in fhits]]).items(),
        key=lambda kv: kv[1], reverse=True,
    )
    merged = [by_id[sid] for sid, _ in ranked if sid in by_id][:k]
    return merged, top_sim
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_retriever.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add vproc/retrieve/retriever.py tests/test_retriever.py
git commit -m "feat: hybrid retrieve (vector + fts RRF) with top-similarity"
```

---

## Task 13: Evidence builder

**Files:**
- Create: `vproc/answer/evidence.py`, `tests/test_evidence.py`

- [ ] **Step 1: Write the failing test** — `tests/test_evidence.py`

```python
from vproc.answer.evidence import build_evidence, evidence_block

def _hit(id, said, screen=""):
    return {"id": id, "memory_id": "standup", "start_ts": 12.0, "end_ts": 14.0,
            "speaker": "Tim V", "said_text": said, "on_screen_text": screen}

def test_build_evidence_assigns_sequential_keys():
    ev = build_evidence([_hit("s1", "cars are great"), _hit("s2", "budget is tight", "Q3 Budget")])
    assert [e.key for e in ev] == ["E1", "E2"]
    assert ev[0].segment_id == "s1"
    assert ev[0].memory_title == "standup" and ev[0].speaker == "Tim V"
    assert "budget is tight" in ev[1].text and "[screen] Q3 Budget" in ev[1].text

def test_evidence_block_format():
    ev = build_evidence([_hit("s1", "hello world")])
    block = evidence_block(ev)
    assert block == '[E1] "hello world"'
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_evidence.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vproc.answer.evidence'`.

- [ ] **Step 3: Implement** — `vproc/answer/evidence.py`

```python
from vproc.models import Evidence


def _text(hit: dict) -> str:
    said = hit.get("said_text", "") or ""
    screen = (hit.get("on_screen_text") or "").strip()
    return said + (f"\n[screen] {screen}" if screen else "")


def build_evidence(hits: list[dict]) -> list[Evidence]:
    out: list[Evidence] = []
    for i, h in enumerate(hits, start=1):
        out.append(Evidence(
            key=f"E{i}", segment_id=h["id"], memory_title=h.get("memory_id", ""),
            start_ts=float(h["start_ts"]), end_ts=float(h["end_ts"]),
            speaker=h.get("speaker", "SPEAKER_0"), text=_text(h),
        ))
    return out


def evidence_block(evidence: list[Evidence]) -> str:
    return "\n".join(f'[{e.key}] "{e.text}"' for e in evidence)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_evidence.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add vproc/answer/evidence.py tests/test_evidence.py
git commit -m "feat: opaque-key evidence list + prompt block"
```

---

## Task 14: Grounded generation

**Files:**
- Create: `vproc/answer/generate.py`, `tests/test_generate.py`

- [ ] **Step 1: Write the failing test** — `tests/test_generate.py`

```python
from vproc.config import Config, Endpoint
from vproc.answer.generate import generate, SYSTEM

def _cfg():
    ep = Endpoint("u", "m")
    return Config(ep, ep, ep, "x", "0.0.0.0", 8765, 0.25, 0.5, None)

def test_generate_parses_and_keeps_cited_claims():
    def fake_chat(base_url, model, system, user):
        assert "EVIDENCE" in user
        return '{"answered": true, "claims": [{"text": "ships in July", "evidence_ids": ["E1"]}]}'
    out = generate(_cfg(), "when does it ship?", "[E1] \"ships in July\"", chat=fake_chat)
    assert out["answered"] is True
    assert out["claims"] == [{"text": "ships in July", "evidence_ids": ["E1"]}]

def test_generate_drops_claims_without_evidence():
    def fake_chat(*a, **k):
        return '{"answered": true, "claims": [{"text": "made up", "evidence_ids": []}]}'
    out = generate(_cfg(), "q", "[E1] \"x\"", chat=fake_chat)
    assert out["answered"] is False and out["claims"] == []

def test_generate_recovers_from_noisy_json():
    def fake_chat(*a, **k):
        return 'Sure!\n{"answered": false, "claims": []}\nThanks'
    out = generate(_cfg(), "q", "[E1] \"x\"", chat=fake_chat)
    assert out["answered"] is False

def test_system_prompt_forbids_outside_knowledge():
    assert "only" in SYSTEM.lower() and "outside" in SYSTEM.lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_generate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vproc.answer.generate'`.

- [ ] **Step 3: Implement** — `vproc/answer/generate.py`

```python
import json

from vproc.llm import client

SYSTEM = (
    "You answer ONLY from the numbered EVIDENCE provided below. For every claim you make, "
    "cite the evidence id(s) it comes from. If the evidence does not contain the answer, set "
    'answered to false and return an empty claims list. Never use outside knowledge or your '
    "own assumptions. "
    'Respond as strict JSON: {"answered": bool, "claims": [{"text": str, "evidence_ids": [str]}]}'
)


def _parse(raw: str) -> dict:
    try:
        return json.loads(raw)
    except Exception:
        i, j = raw.find("{"), raw.rfind("}")
        if i >= 0 and j > i:
            try:
                return json.loads(raw[i:j + 1])
            except Exception:
                pass
    return {"answered": False, "claims": []}


def generate(cfg, question: str, evidence_block: str, chat=client.chat_json) -> dict:
    user = f"QUESTION:\n{question}\n\nEVIDENCE:\n{evidence_block}\n\nJSON:"
    data = _parse(chat(cfg.grounding.base_url, cfg.grounding.model, SYSTEM, user))
    claims = [c for c in data.get("claims", []) if c.get("evidence_ids")]
    return {"answered": bool(data.get("answered")) and len(claims) > 0, "claims": claims}
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_generate.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add vproc/answer/generate.py tests/test_generate.py
git commit -m "feat: grounded generation with JSON parse + empty-evidence rejection"
```

---

## Task 15: Faithfulness gate (HHEM)

**Files:**
- Create: `vproc/answer/faithfulness.py`, `tests/test_faithfulness.py`

- [ ] **Step 1: Write the failing test** — `tests/test_faithfulness.py`

```python
from vproc.models import Evidence
from vproc.answer.faithfulness import filter_claims

def _ev(key, text):
    return Evidence(key=key, segment_id=key, memory_title="m", start_ts=0, end_ts=1,
                    speaker="SPEAKER_0", text=text)

def test_filter_keeps_supported_drops_unsupported():
    by_key = {"E1": _ev("E1", "the product ships in July")}
    claims = [
        {"text": "ships in July", "evidence_ids": ["E1"]},   # supported
        {"text": "ships in March", "evidence_ids": ["E1"]},  # contradicted
    ]
    # fake scorer: high if the claim's key words appear in premise
    def scorer(premise, hypothesis):
        return 0.9 if "July" in premise and "July" in hypothesis else 0.1
    kept = filter_claims(claims, by_key, scorer, threshold=0.5)
    assert kept == [{"text": "ships in July", "evidence_ids": ["E1"]}]

def test_filter_drops_when_no_premise():
    claims = [{"text": "x", "evidence_ids": ["E9"]}]  # E9 not in evidence
    kept = filter_claims(claims, {}, lambda p, h: 1.0, threshold=0.5)
    assert kept == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_faithfulness.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vproc.answer.faithfulness'`.

- [ ] **Step 3: Implement** — `vproc/answer/faithfulness.py`

```python
from vproc.models import Evidence

HHEM_MODEL = "vectara/hallucination_evaluation_model"


class HHEM:
    """Lazy wrapper around Vectara HHEM-2.1-Open (loaded on first use)."""

    def __init__(self, model_id: str = HHEM_MODEL):
        from transformers import AutoModelForSequenceClassification  # lazy: heavy import

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, trust_remote_code=True
        )

    def score(self, premise: str, hypothesis: str) -> float:
        return float(self.model.predict([(premise, hypothesis)])[0])


def filter_claims(claims: list[dict], evidence_by_key: dict[str, Evidence],
                  scorer, threshold: float) -> list[dict]:
    kept: list[dict] = []
    for claim in claims:
        premise = "\n".join(
            evidence_by_key[k].text for k in claim["evidence_ids"] if k in evidence_by_key
        )
        if premise and scorer(premise, claim["text"]) >= threshold:
            kept.append(claim)
    return kept
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_faithfulness.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add vproc/answer/faithfulness.py tests/test_faithfulness.py
git commit -m "feat: HHEM faithfulness gate (injectable scorer)"
```

---

## Task 16: ask_memory / search_memory orchestration (+ hallucination guard)

**Files:**
- Create: `vproc/answer/ask.py`, `tests/test_ask.py`

- [ ] **Step 1: Write the failing test** — `tests/test_ask.py`

This is the load-bearing test: it proves grounded answering AND the "Not discussed" guard, with all model boundaries faked.

```python
from vproc.config import Config, Endpoint
from vproc.answer.ask import ask_memory, search_memory, ABSTAIN

def _cfg(sim_floor=0.25):
    ep = Endpoint("u", "m")
    return Config(ep, ep, ep, "x", "0.0.0.0", 8765, sim_floor, 0.5, None)

def _hit(id, said):
    return {"id": id, "memory_id": "standup", "start_ts": 12.0, "end_ts": 14.0,
            "speaker": "Tim V", "said_text": said, "on_screen_text": ""}

def test_answer_is_grounded_and_cited():
    def fake_retrieve(store, cfg, q, k=8, where=None):
        return [_hit("s1", "Tim said cars are electric now")], 0.9
    def fake_generate(cfg, q, block):
        return {"answered": True, "claims": [{"text": "cars are electric now", "evidence_ids": ["E1"]}]}
    ans = ask_memory(None, _cfg(), "did Tim talk about cars?", scorer=lambda p, h: 0.9,
                     _retrieve=fake_retrieve, _generate=fake_generate)
    assert ans.answered is True
    assert "cars are electric now" in ans.text
    assert "[standup · 00:12 · Tim V]" in ans.text     # code-owned citation
    assert ans.claims[0].citations[0].speaker == "Tim V"

def test_hallucination_guard_abstains_on_empty_retrieval():
    def fake_retrieve(*a, **k):
        return [], 0.0
    ans = ask_memory(None, _cfg(), "how does a petrol engine work?", scorer=lambda p, h: 0.9,
                     _retrieve=fake_retrieve, _generate=lambda *a: 1 / 0)  # must not be called
    assert ans.answered is False and ans.abstained is True
    assert ans.text == ABSTAIN

def test_abstains_below_similarity_floor():
    def fake_retrieve(*a, **k):
        return [_hit("s1", "unrelated")], 0.10  # below floor 0.25
    ans = ask_memory(None, _cfg(), "q", scorer=lambda p, h: 0.9,
                     _retrieve=fake_retrieve, _generate=lambda *a: 1 / 0)
    assert ans.text == ABSTAIN

def test_abstains_when_hhem_rejects_all():
    def fake_retrieve(*a, **k):
        return [_hit("s1", "Tim said cars are electric")], 0.9
    def fake_generate(*a):
        return {"answered": True, "claims": [{"text": "cars are petrol", "evidence_ids": ["E1"]}]}
    ans = ask_memory(None, _cfg(), "q", scorer=lambda p, h: 0.1,  # HHEM rejects
                     _retrieve=fake_retrieve, _generate=fake_generate)
    assert ans.text == ABSTAIN

def test_search_memory_returns_evidence():
    def fake_retrieve(*a, **k):
        return [_hit("s1", "cars")], 0.9
    ev = search_memory(None, _cfg(), "cars", _retrieve=fake_retrieve)
    assert ev[0].key == "E1" and ev[0].speaker == "Tim V"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_ask.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vproc.answer.ask'`.

- [ ] **Step 3: Implement** — `vproc/answer/ask.py`

```python
from vproc.answer import generate as gen
from vproc.answer.evidence import build_evidence, evidence_block
from vproc.answer.faithfulness import filter_claims
from vproc.models import Answer, Citation, Claim, Evidence, mmss
from vproc.retrieve.retriever import retrieve

ABSTAIN = "Not discussed in these meetings."


def _abstain() -> Answer:
    return Answer(answered=False, abstained=True, text=ABSTAIN, claims=[], evidence=[])


def _cite_tag(c: Citation) -> str:
    return f"[{c.memory_title} · {mmss(c.start_ts)} · {c.speaker}]"


def ask_memory(store, cfg, question: str, scorer, where: str | None = None, k: int = 8,
               _retrieve=retrieve, _generate=gen.generate) -> Answer:
    hits, top_sim = _retrieve(store, cfg, question, k=k, where=where)
    if not hits or top_sim < cfg.sim_floor:
        return _abstain()

    evidence = build_evidence(hits)
    by_key: dict[str, Evidence] = {e.key: e for e in evidence}
    result = _generate(cfg, question, evidence_block(evidence))
    if not result["answered"]:
        return _abstain()

    kept = filter_claims(result["claims"], by_key, scorer, cfg.hhem_threshold)
    if not kept:
        return _abstain()

    out_claims: list[Claim] = []
    lines: list[str] = []
    for c in kept:
        cites = [
            Citation(memory_title=by_key[k].memory_title, start_ts=by_key[k].start_ts,
                     end_ts=by_key[k].end_ts, speaker=by_key[k].speaker)
            for k in c["evidence_ids"] if k in by_key
        ]
        out_claims.append(Claim(text=c["text"], evidence_ids=c["evidence_ids"], citations=cites))
        tags = " ".join(_cite_tag(cit) for cit in cites)
        lines.append(f"{c['text']} {tags}".strip())

    return Answer(answered=True, abstained=False, text="\n".join(lines),
                  claims=out_claims, evidence=evidence)


def search_memory(store, cfg, query: str, where: str | None = None, k: int = 8,
                  _retrieve=retrieve) -> list[Evidence]:
    hits, _ = _retrieve(store, cfg, query, k=k, where=where)
    return build_evidence(hits)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_ask.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add vproc/answer/ask.py tests/test_ask.py
git commit -m "feat: ask_memory/search_memory with code-owned citations + abstention + HHEM"
```

---

## Task 17: HTTP service (REST + MCP) and CLI

**Files:**
- Create: `vproc/service.py`, `vproc/cli.py`, `.vscode/mcp.json`, `tests/test_service.py`

- [ ] **Step 1: Write the failing test** — `tests/test_service.py`

```python
from fastapi.testclient import TestClient
from vproc.config import Config, Endpoint
from vproc.models import Answer
from vproc.service import create_app

def _cfg():
    ep = Endpoint("u", "m")
    return Config(ep, ep, ep, "x", "0.0.0.0", 8765, 0.25, 0.5, None)

def test_healthz():
    app = create_app(store=object(), scorer=lambda p, h: 1.0, cfg=_cfg(),
                     ask=lambda *a, **k: Answer(answered=False, abstained=True,
                                                text="Not discussed in these meetings.",
                                                claims=[], evidence=[]),
                     search=lambda *a, **k: [])
    client = TestClient(app)
    assert client.get("/healthz").json() == {"ok": True}

def test_ask_route_returns_answer_json():
    captured = {}
    def fake_ask(store, cfg, question, scorer, where=None):
        captured["question"] = question
        return Answer(answered=True, abstained=False, text="ships in July [standup · 00:12 · Tim V]",
                      claims=[], evidence=[])
    app = create_app(store=object(), scorer=lambda p, h: 1.0, cfg=_cfg(),
                     ask=fake_ask, search=lambda *a, **k: [])
    client = TestClient(app)
    r = client.post("/ask", json={"question": "when does it ship?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answered"] is True and "July" in body["text"]
    assert captured["question"] == "when does it ship?"

def test_where_filter_built_from_request():
    seen = {}
    def fake_ask(store, cfg, question, scorer, where=None):
        seen["where"] = where
        return Answer(answered=False, abstained=True, text="x", claims=[], evidence=[])
    app = create_app(store=object(), scorer=lambda p, h: 1.0, cfg=_cfg(),
                     ask=fake_ask, search=lambda *a, **k: [])
    TestClient(app).post("/ask", json={"question": "q", "speaker": "Tim V", "memory": "standup"})
    assert "speaker = 'Tim V'" in seen["where"] and "memory_id = 'standup'" in seen["where"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_service.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vproc.service'`.

- [ ] **Step 3: Implement** — `vproc/service.py`

```python
from fastapi import FastAPI
from pydantic import BaseModel

from vproc.answer.ask import ask_memory, search_memory
from vproc.config import load_config


class Query(BaseModel):
    question: str | None = None
    query: str | None = None
    project: str | None = None
    memory: str | None = None
    speaker: str | None = None


def build_where(q: Query) -> str | None:
    clauses = []
    if q.project:
        clauses.append(f"project_id = '{q.project}'")
    if q.memory:
        clauses.append(f"memory_id = '{q.memory}'")
    if q.speaker:
        clauses.append(f"speaker = '{q.speaker}'")
    return " AND ".join(clauses) if clauses else None


def create_app(store=None, scorer=None, cfg=None, ask=ask_memory, search=search_memory) -> FastAPI:
    cfg = cfg or load_config()
    if store is None:
        from vproc.store.lancedb_store import Store
        store = Store(cfg.index_path)
    if scorer is None:
        from vproc.answer.faithfulness import HHEM
        scorer = HHEM().score

    app = FastAPI(title="vproc")

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.post("/ask")
    def ask_route(q: Query):
        answer = ask(store, cfg, q.question or "", scorer, where=build_where(q))
        return answer.model_dump()

    @app.post("/search")
    def search_route(q: Query):
        evidence = search(store, cfg, q.query or q.question or "", where=build_where(q))
        return [e.model_dump() for e in evidence]

    _mount_mcp(app, store, cfg, scorer, ask, search)
    return app


def _mount_mcp(app, store, cfg, scorer, ask, search) -> None:
    """Expose the same logic as MCP tools at /mcp. Best-effort: if the installed
    mcp SDK's mounting API differs, REST still works and this is a no-op."""
    try:
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("vproc")

        @mcp.tool()
        def ask_memory_tool(question: str) -> dict:
            return ask(store, cfg, question, scorer).model_dump()

        @mcp.tool()
        def search_memory_tool(query: str) -> list:
            return [e.model_dump() for e in search(store, cfg, query)]

        app.mount("/mcp", mcp.streamable_http_app())
    except Exception:
        pass
```

- [ ] **Step 4: Run service tests to verify they pass**

Run: `uv run pytest tests/test_service.py -v`
Expected: 3 passed.

- [ ] **Step 5: Implement the CLI** — `vproc/cli.py`

```python
import sys

from vproc.config import load_config


def main() -> None:
    args = sys.argv[1:]
    if len(args) >= 2 and args[0] == "ingest":
        from vproc.ingest.pipeline import ingest_video

        n = ingest_video(args[1])
        print(f"ingested {n} segments from {args[1]}")
    elif args[:1] == ["serve"]:
        import uvicorn

        from vproc.service import create_app

        cfg = load_config()
        uvicorn.run(create_app(), host=cfg.host, port=cfg.port)
    else:
        print("usage: vproc [ingest <video.mp4> | serve]")
        sys.exit(1)
```

- [ ] **Step 6: Create `.vscode/mcp.json`**

```json
{
  "servers": {
    "vproc": { "type": "http", "url": "http://127.0.0.1:8765/mcp" }
  }
}
```

- [ ] **Step 7: Run the full test suite**

Run: `uv run pytest -v`
Expected: all tests pass (Tasks 1–17).

- [ ] **Step 8: Commit**

```bash
git add vproc/service.py vproc/cli.py .vscode/mcp.json tests/test_service.py
git commit -m "feat: HTTP service (REST /ask,/search,/healthz + MCP /mcp) and CLI"
```

---

## Task 18: Manual end-to-end verification (real models, one meeting)

This task is **manual** (no automated test) — it exercises the real endpoints once. Do it after Task 17.

- [ ] **Step 1: Prepare endpoints**
  - Confirm voyage OCR is reachable: `curl http://voyage:8000/v1/models` lists `QuantTrio/Qwen3.5-9B-AWQ`.
  - In **LM Studio** (Mac), start a server (`http://localhost:1234/v1`) with an **embedding** model (`Qwen3-Embedding-0.6B`) and the **grounding** model (`Qwen3-32B-AWQ`) loaded.
  - Install real-model deps: `uv sync --extra runtime` (downloads torch/transformers/mlx-whisper).
  - `cp .env.example .env` and adjust hosts if needed. Export them: `set -a && source .env && set +a`.

- [ ] **Step 2: Verify the OCR model accepts an image** (spec risk #6)

Run a one-off: create a screenshot `sample.png` with some text, then
`uv run python -c "from vproc.config import load_config; from vproc.ingest.ocr import ocr_frame; print(ocr_frame(load_config().ocr, 'sample.png'))"`
Expected: the visible text is printed. If it errors on image input, set `VPROC_OCR_*` to a dedicated `Qwen3-VL-8B-Instruct` endpoint (spec §10).

- [ ] **Step 3: Ingest one real meeting**

Run: `uv run vproc ingest /path/to/meeting1.mp4`
Expected: prints `ingested N segments …`; a `./vproc.lance` directory appears.

- [ ] **Step 4: Serve and ask via REST**

Run (in one terminal): `uv run vproc serve`
Run (in another): 
`curl -s localhost:8765/ask -H 'content-type: application/json' -d '{"question":"give an overall description of the meeting"}' | python -m json.tool`
Expected: `answered: true` with text containing `[<title> · mm:ss · …]` citations.

- [ ] **Step 5: Confirm the hallucination guard on real models**

Run: `curl -s localhost:8765/ask -H 'content-type: application/json' -d '{"question":"how does a petrol engine work?"}'`
Expected: `"text": "Not discussed in these meetings."` (assuming engines weren't discussed).

- [ ] **Step 6: Wire VS Code**
  - Ensure `.vscode/mcp.json` points at the running service (`http://127.0.0.1:8765/mcp`, or the LAN IP if VS Code runs on another machine).
  - In Copilot Chat **Agent mode**, confirm the `vproc` tools appear and answer from the meeting with citations.

- [ ] **Step 7: Commit a quickstart note** — append a "Quickstart" section to `README.md` documenting steps 1–6, then:

```bash
git add README.md
git commit -m "docs: phase-1 quickstart (endpoints, ingest, serve, VS Code)"
```

---

## Self-review (completed during planning)

**Spec coverage:** ingest pipeline (frames/transcribe/ocr/align/embed) ✓ Tasks 5–11 · LanceDB hybrid store ✓ Task 9 · retrieval + abstention ✓ Tasks 12, 16 · code-owned citations ✓ Task 16 · grounded generation ✓ Task 14 · HHEM faithfulness ✓ Task 15 · REST+MCP HTTP service ✓ Task 17 · env-configured endpoints ✓ Task 2 · hallucination-guard test ✓ Task 16. **Deferred to Phase 2 (out of scope here, per spec §14):** diarization + speaker naming, constrained decoding (XGrammar/GBNF — v1 uses prompt + `json_object` + validation), reranker (retrieval is vector+FTS RRF only), `get_meeting_overview` map-reduce summary, Project/Memory tables (v1 uses the video stem as `memory_id`/title).

**Placeholder scan:** none — every step has runnable code and an exact command.

**Type consistency:** `Endpoint`/`Config` (Task 2) used identically everywhere; `Segment`/`Evidence`/`Citation`/`Claim`/`Answer` (Task 3) flow unchanged through Tasks 10/13/16/17; retrieval hit dicts use the same keys (`id, memory_id, start_ts, end_ts, speaker, said_text, on_screen_text, _distance`) in Tasks 9/12/13/16; injection seams (`embed=`, `chat=`, `scorer=`, `_retrieve=`, `_generate=`, `ask=`, `search=`) are consistent between each module and its tests.

---

## Notes for the implementer

- **Tests run fully offline** — every model/IO call is injected with a fake in tests. Only Task 18 touches real endpoints.
- **DRY/YAGNI:** v1 deliberately omits the reranker, constrained decoding, and diarization (Phase 2). Don't add them here.
- **Grounding is structural:** never let the grounding model emit timestamps/speakers — Task 16 resolves citations in code from the evidence map. Keep it that way.
- **If `lancedb` FTS (`use_tantivy=False`) errors** on the installed version, `Store.add` swallows it and vector search still works; revisit hybrid in Phase 2.
