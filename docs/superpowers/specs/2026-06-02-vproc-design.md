# vproc — Grounded Meeting-Memory from Video

**Status:** Draft for review · **Date:** 2026-06-02 · **Owner:** jr@aqui.technology

---

## 1. Problem & goal

Important meetings were recorded as screen-share + audio (`.mp4`) while the owner was out
sick. Watching them back is too slow. We want to turn each recording into searchable,
**grounded** "memory" — a time-aligned record of *what was said*, *by whom*, and *what was
on screen* — and let the owner query it from **GitHub Copilot in VS Code**.

The defining requirement: **answers must come only from the videos.** No fact may be
supplied by a model's general knowledge or invented ("hallucinated"). When something was
not discussed, the system must say so rather than guess.

### In scope (v1)
- Ingest local `.mp4` recordings (screen-share + audio).
- Local transcription with word-level timestamps, speaker diarization, and **named**
  speakers (names read from on-screen nameplates, confirmed once).
- Local OCR of screen content (slides, code, docs, dashboards).
- Time-aligned **segments** stored in a local hybrid search index.
- A local **RAG** answerer that produces grounded, cited answers (or "not discussed").
- A local **MCP server** exposing this to VS Code / Copilot.

### Out of scope (v1, may follow later)
- Standalone CLI / web chat front-ends (the RAG core is reused when we add them).
- Vision-model *description* of non-text visuals (screens are mostly text).
- Cloud/company-API answerer (kept as a pluggable swap; not built in v1).
- Real-time / live-meeting processing (batch only).

### Current scale
**2 meetings, ~50 minutes each.** One **Project**, two **Memories**. Ingest is minutes on
the Blackwell box. The architecture is unchanged by scale — grounding is the point, not size.

---

## 2. Hard constraints

| Constraint | Consequence |
|---|---|
| **100% open-source + local** in the pipeline | No proprietary frameworks (⇒ **no** Apple Vision OCR). Whisper, pyannote, Qwen, LanceDB, etc. — all Apache/MIT/BSD. The **only** gated piece is pyannote's model (free HF token + one-time accept; CC-BY-4.0). |
| **No cloud generation** | Copilot/VS Code is a **front-end only** — it performs **zero** generation. All OCR/ASR/embedding/answering runs on local open models. Confidential content never leaves the owner's machines. |
| **Cross-platform** | **Dev/test:** Apple Silicon M5 Max (64 GB, Metal/MLX/MPS). **Prod:** Ubuntu Linux + NVIDIA RTX PRO 4000 **Blackwell** (24 GB VRAM, CUDA sm_120). One codebase, backend chosen by config. |
| **24 GB VRAM is the binding budget** | Size every model against the prod box, not the 64 GB Mac. Exploit that **ingest-time** and **query-time** models are *never co-resident*. |
| **Grounding is sacred** | Anti-hallucination is an *architecture* (Section 6), not a model setting. |

---

## 3. Core principle — why it can't hallucinate

The system **never authors facts. It hands back verbatim quotes with citations, and a local
model phrases an answer using only those quotes.** Three structural guarantees:

1. **Code owns the citation, not the model.** The answering LLM sees evidence as opaque keys
   (`[E3] "<verbatim text>"`). It may only *reference* keys. Code resolves
   `E3 → (memory, timestamp, speaker)` **after** generation. ⇒ a timestamp or speaker name is
   *structurally impossible to hallucinate*.
2. **Abstention is a code decision.** If retrieval is empty or below a score floor, the system
   returns the fixed string **"Not discussed in these meetings."** *before the LLM ever runs.*
   We never rely on the model to volunteer "I don't know."
3. **An independent verifier gates every claim.** A small NLI model (Vectara HHEM-2.1-Open)
   checks each generated sentence against *its* cited evidence; unsupported sentences are
   dropped/flagged. The checker is a *different* model than the generator.

---

## 4. Architecture overview

```
INGEST (batch, on the Blackwell box; owns full 24 GB)
  meeting.mp4
    ├─ ffmpeg ──► audio.wav (mono 16 kHz)
    │              └─ WhisperX ──► word-level transcript
    │                   └─ pyannote ──► speaker turns ──► speaker-attributed words
    └─ ffmpeg (mpdecimate+scene) + pHash dedup ──► one frame per distinct screen state
                   └─ Qwen3-VL OCR (verbatim) + verifier ──► on-screen text [+confidence]
                          └─ nameplate OCR ──► cluster→name suggestion ──► owner confirms once
                                   │
                     ALIGN by timestamp ──► Segment{ t, speaker, said_text, on_screen_text }
                                   │
                     embed (Qwen3-Embedding) ──► LanceDB (vector + BM25 + metadata)

QUERY (interactive; co-resident in 24 GB)
  VS Code / Copilot ──MCP──► search_memory(q)         ──► raw cited verbatim segments
                       └────► ask_memory(q)
                                  ├─ hybrid retrieve + rerank
                                  ├─ score floor? ── empty ──► "Not discussed in these meetings."
                                  ├─ evidence list E1..En (opaque keys)
                                  ├─ Qwen3 answerer (non-thinking, JSON schema, cite-or-refuse)
                                  ├─ HHEM faithfulness gate (per claim)
                                  └─ resolve En → (memory, mm:ss, speaker) ──► cited answer
```

---

## 5. Data model

```
Project   { id, name, created_at }
  └─ Memory { id, project_id, title, source_video, recorded_on, duration_s,
              speakers: [SpeakerMap], status, created_at }
       └─ Segment { id, project_id, memory_id, screen_state_id,
                    start_ts, end_ts, speaker,            # resolved name or "SPEAKER_xx"
                    said_text,                            # verbatim transcript for the interval
                    on_screen_text, on_screen_confidence, # verbatim OCR; low-conf spans tagged [unverified]
                    frame_path, source_video,
                    embed_text,                           # "[SCREEN]\n…\n[SPOKEN]\n<spk> (mm:ss): …"
                    vector }                              # fixed-dim; embedder/dim locked at ingest

SpeakerMap { cluster_id, name, confidence, evidence: {frame_ts, ocr_box} }  # name only with evidence
```

- **Project** groups related meetings; v1 has one project, two memories.
- **Memory** = one processed meeting.
- **Segment** = the retrievable atom: a screen-state interval `[t_start, t_end)` bundling the
  speaker-attributed transcript that overlaps it with that interval's OCR text. This single unit
  answers all three example queries (overview = summarize a memory's segments; "did Tim V say X" =
  filter `speaker` + search; "how does a petrol engine work" = search → empty ⇒ "not discussed").
- **Chunking rule:** one segment per screen-state interval. Long monologues over one slide are
  sub-split by speaker turn / ~60–90 s **but keep the same `screen_state_id` + OCR attached**, so
  visual context never detaches. A transcript span straddling two screen states is assigned to the
  interval holding its midpoint (one consistent rule — never duplicated across slides).

---

## 6. Ingestion pipeline (per memory)

Module boundaries under `vproc/ingest/` — each does one thing, testable in isolation.

| # | Module | Does | Key detail / gotcha |
|---|---|---|---|
| 1 | `frames.py` | Sample screen frames | `ffmpeg … -vf "mpdecimate,select='gt(scene,0.08)',metadata=print:file=frames.log" -fps_mode vfr`. **Must** read `pts_time` from the log — never infer time from filenames. Then pHash dedup (Hamming > ~6 on 64-bit hash) ⇒ one frame per distinct screen. Add a 30 s safety-floor anchor. |
| 2 | `transcribe.py` | Word-level transcript | WhisperX (faster-whisper `large-v3-turbo` + wav2vec2 forced alignment). VAD on; `condition_on_previous_text=False` (kills repetition loops on silence). **Blackwell: `compute_type="float16"`** (INT8 crashes on sm_120). Mac: CPU transcribe + MPS align. |
| 3 | `diarize.py` | Who spoke when | pyannote.audio 4.x, `speaker-diarization-community-1` (needs `HF_TOKEN` + one-time accept). Fuse to words by **max temporal overlap** ⇒ speaker-attributed segments. If sm_120 kernel error ⇒ fall back to CPU diarization. |
| 4 | `ocr.py` | Read screen text | Qwen3-VL-8B-Instruct via OpenAI `/v1`. **Verbatim prompt** ("transcribe only visibly-rendered text; `[illegible]` for unreadable; no inference/translation/spelling-fix"). Request per-line `{bbox_2d, text_content}` JSON. **Verifier:** deterministic OCR (PaddleOCR/docTR) on each bbox crop for char-agreement + token-logprob/2-resolution self-consistency ⇒ tag low-confidence spans `[unverified]`. |
| 5 | `names.py` | cluster → real name | Match an OCR'd **nameplate** to the active-speaker time window; require temporal overlap. **Never infer a name from voice.** Suggest → owner confirms once (CLI prompt in v1). Persist `cluster_id → name` + mean speaker-embedding for reuse. No evidence ⇒ keep `SPEAKER_xx`. |
| 6 | `align.py` | Build segments | Turn deduped frames into screen-state intervals; attach overlapping speaker-attributed transcript; emit `Segment`s + `embed_text`. |
| 7 | `embed_index.py` | Embed + store | Qwen3-Embedding-0.6B (**instruction prefix on queries only**, `padding_side="left"`). Write vectors + metadata to LanceDB. |
| `pipeline.py` orchestrates 1→7 for one memory and writes the `Memory` record. |

**Speakers are fully visible on the nameplates in both target meetings**, so step 5 auto-naming is
expected to succeed; the manual confirm is a quick safety check.

---

## 7. Query / answering pipeline

Under `vproc/retrieve/` and `vproc/answer/`.

1. **Retrieve** (`store/`): LanceDB **hybrid** (dense + BM25, RRF) with `prefilter=True` metadata
   filter (`project`, `memory`, `speaker`, `start_ts` range). Native FTS (`use_tantivy=False`).
2. **Rerank**: Qwen3-Reranker-0.6B cross-encoder over candidates.
3. **Abstention gate** (code): if no candidates or top rerank score `< floor` ⇒ return
   **"Not discussed in these meetings."** (`answered=false`). *No LLM call.*
4. **Evidence list**: assign opaque keys `E1..En` to the top-k; build a numbered list of verbatim
   `said_text` / `on_screen_text` (low-conf OCR spans excluded or marked).
5. **Constrained generation**: Qwen3 answerer, **non-thinking mode**, low temperature, JSON schema
   `{answered: bool, claims: [{text, evidence_ids: [str]}]}` enforced by XGrammar (prod) / GBNF
   (Mac). Reject any claim with empty `evidence_ids`; all-empty ⇒ abstain.
6. **Faithfulness gate**: HHEM-2.1-Open NLI per claim (premise = cited evidence text, hypothesis =
   claim). Drop/flag below threshold (~0.5, calibrated).
7. **Citation resolution** (code): `evidence_ids → (memory_title, mm:ss–mm:ss, speaker)`. Assemble
   prose with inline `[Memory · mm:ss · Speaker]` citations. Return `Answer` + the evidence array
   so every claim is auditable in VS Code.

**Overview / summary** ("give an overall description") = grounded-abstractive map-reduce over a
memory's segments under the *same* schema, each sentence HHEM-gated, with an **extractive fallback**
(return top cited verbatim segments) if the abstractive summary fails the gate.

---

## 8. MCP server / VS Code integration

`vproc/mcp_server.py` — official `mcp` Python SDK (FastMCP), stdio transport.

Tools:
- `list_projects()` / `list_memories(project)` — navigation.
- `search_memory(query, project?, memory?, speaker?, k=8)` → **raw cited verbatim segments**, no
  synthesis (impossible to hallucinate; lets Copilot/owner audit).
- `ask_memory(question, project?, memory?, speaker?)` → **local grounded answer** with inline
  citations + the evidence array, or "Not discussed in these meetings."
- `get_meeting_overview(memory)` → grounded summary (Section 7).

VS Code registration — `.vscode/mcp.json`, **top-level key `servers`** (not `mcpServers`):
```json
{ "servers": { "vproc": {
    "command": "uv", "args": ["run", "python", "-m", "vproc.mcp_server"],
    "env": { "HF_TOKEN": "${env:HF_TOKEN}" } } } }
```
Tools surface in **Copilot Chat Agent mode**. `ask_memory` **must** generate locally — never let
Copilot's cloud model synthesize (that would leak confidential segments).

---

## 9. Cross-platform serving seam

One **OpenAI-compatible `/v1`** endpoint per model role + a config registry
`{role → {backend, base_url, model_id, dtype}}`. No backend-specific SDK in business logic — only a
thin client (`vproc/llm/`) knows the backend.

| Role | Mac (dev) | Blackwell (prod) |
|---|---|---|
| VLM-OCR, answerer | vllm-mlx / Ollama 0.19+ (MLX) | vLLM (cu130 image, pinned tag) |
| ASR | faster-whisper (CPU) / mlx-whisper draft | faster-whisper (CUDA) |
| Embed / rerank | MLX or sentence-transformers (MPS) | vLLM / sentence-transformers (cu128) |
| Store | LanceDB (arm64 wheel) | LanceDB (x86_64 wheel) — directory is byte-portable |

---

## 10. The validated stack

| Component | Choice | Version / model | License |
|---|---|---|---|
| Demux + frames | ffmpeg + pHash | ffmpeg 7.x (LGPL), imagehash, Pillow | LGPL/BSD/HPND |
| ASR + word ts | WhisperX | whisperx 3.7.x · faster-whisper 1.2.1 · ct2 4.7.2 · `large-v3-turbo` | BSD/MIT |
| Diarization | pyannote.audio | 4.x · `speaker-diarization-community-1` (MIT `3.1` fallback) | MIT code / CC-BY-4.0 weights (gated) |
| Screen OCR | Qwen3-VL | `Qwen3-VL-8B-Instruct` (+`-FP8` prod) | Apache-2.0 |
| Embeddings | Qwen3-Embedding | `Qwen3-Embedding-0.6B` | Apache-2.0 |
| Hybrid store | LanceDB | native FTS (`use_tantivy=False`) | Apache-2.0 |
| Reranker | Qwen3-Reranker | `Qwen3-Reranker-0.6B` | Apache-2.0 |
| Answerer | Qwen3 | `Qwen3-32B-AWQ` (official quant, 32K ctx) — non-thinking | Apache-2.0 |
| Faithfulness | Vectara HHEM | `HHEM-2.1-Open` (~440 MB) | Apache-2.0 |
| Constrained decode | XGrammar / GBNF | vLLM default / llama.cpp | Apache-2.0 / MIT |
| Front-end | MCP Python SDK | `mcp` 1.27.2 (FastMCP) | MIT |

**Answerer note:** default to the **official `Qwen3-32B-AWQ`** (audit-grade reproducible quant) for a
grounding-sacred pipeline; its 32K context is ample for a 2-meeting corpus. (Third-party
`Qwen3.6-27B` quants offer 262K context but are unofficial — kept as an alternative, not the
default.) **OCR model is config-swappable**: if the prod box already runs a specific Qwen-VL build,
point the config at it; otherwise pull `Qwen3-VL-8B-Instruct`.

---

## 11. VRAM plan (24 GB Blackwell — the binding box)

Ingest and query models are **never co-resident** — run ingest as a separate batch process, release
VRAM, then start the query server.

- **Ingest phase** (sequential sub-steps to be safe): WhisperX ASR FP16 ~1.6–3 GB + wav2vec2 ~0.5 GB;
  pyannote ~2–3 GB; Qwen3-VL-8B-FP8 OCR ~9–10 GB (+ViT/KV, ~14 GB headroom for high-res tokens). Cap
  `max_pixels` / `--limit-mm-per-prompt`; pHash-dedup first (don't OCR every frame).
- **Query phase** (co-resident): Qwen3-Embedding-0.6B ~1.2 GB + Qwen3-Reranker-0.6B ~1.3 GB +
  HHEM <0.6 GB (CPU-capable) + answerer Qwen3-32B-AWQ ~17–19 GB. Cap `--max-model-len` to 32K and set
  `--gpu-memory-utilization 0.85`.
- **Mac (64 GB)**: everything co-fits with room; use it to validate long-context behavior the 24 GB
  card can't hold.

**Rough timing:** Blackwell ingests both meetings in **a few minutes each**; answers in seconds.
Mac ingest is slower (CPU-bound ASR, ~20–40 min/meeting) but fine for dev + interactive answering.

---

## 12. Error handling & edge cases

- **Missing frame timestamps** ⇒ hard error (never infer from filename/fps).
- **Whisper silence hallucination** ⇒ VAD on, `condition_on_previous_text=False`.
- **Blackwell prereqs** ⇒ assert `torch.cuda.get_device_capability() == (12, 0)`; require CUDA 12.8+/
  cuDNN 9 / torch cu128; **pin** ctranslate2/vLLM versions (sm_120 is a moving target — see Risks).
- **Missing `HF_TOKEN`** ⇒ clear setup message (pyannote is gated); MIT `3.1` fallback documented.
- **OCR low-confidence spans** ⇒ tagged `[unverified]`, excluded from evidence; answerer told to ignore.
- **No nameplate for a cluster** ⇒ keep `SPEAKER_xx` (never guess from voice).
- **Off-topic question** ⇒ deterministic abstention; **the LLM is never asked**.
- **Abstractive summary fails HHEM** ⇒ extractive verbatim fallback.

---

## 13. Testing strategy

- **Unit:** pHash dedup, `pts_time` parsing, interval/midpoint chunking, abstention-gate logic,
  citation resolution (`En → tuple`), instruction-prefix/padding in the embed wrapper.
- **Grounding integration (the load-bearing tests):**
  - Golden Q&A over a small fixture meeting: assert each citation points to a real verbatim span.
  - **Hallucination guard:** ask a question whose answer is *not* in the fixture (e.g. "how does a
    petrol engine work") ⇒ **must abstain** ("Not discussed…"), zero fabricated content.
  - Speaker filter: "did <name> mention <X>" returns only that speaker's segments (or abstains).
  - Faithfulness: assert HHEM drops a deliberately-unsupported claim.
- **OCR faithfulness:** fixture frame ⇒ verbatim text + correct `[unverified]` tagging.
- **Cross-platform smoke:** the same `ask_memory` runs against `/v1` on Mac (MLX/Ollama) and would
  run on Blackwell (vLLM) — backend swapped by config only.
- **Threshold calibration:** rerank floor + HHEM threshold tuned on a small labeled subset of the two
  real meetings.

---

## 14. Build phases

- **Phase 0 — scaffold:** `uv` project, config registry, `/v1` client abstraction, LanceDB schema,
  data models, MCP server skeleton.
- **Phase 1 — MVP (vertical slice):** ingest **one** meeting → frames+pHash → WhisperX transcript
  (*anonymous speakers*) → Qwen3-VL OCR + verifier → align → segments → embed → LanceDB; retrieve +
  rerank + abstention; `ask_memory` with code-owned citations + HHEM gate; `search_memory`; **working
  in VS Code Agent mode.** This already answers questions about one meeting.
- **Phase 2 — speakers + summaries:** diarization → nameplate name resolution → owner confirm;
  Project/Memory model; `get_meeting_overview` grounded summaries; constrained decoding.
- **Phase 3 — both meetings + harden:** ingest the second meeting; calibrate thresholds; Blackwell
  deployment + version pinning; (optional) company-key answerer swap; (optional) CLI/web front-end.

---

## 15. Repo layout

```
vproc/
  config.py            # config registry + platform profiles + thresholds
  models.py            # Project, Memory, Segment, Evidence, Citation, Answer
  llm/                 # OpenAI /v1 client: chat, embed, rerank (backend-agnostic)
  ingest/              # frames, transcribe, diarize, ocr, names, align, embed_index, pipeline
  store/               # LanceDB schema + hybrid search + metadata filters
  retrieve/            # hybrid retrieve + rerank + score-gate
  answer/              # evidence builder, constrained gen, citation resolver, HHEM gate, summarizer
  mcp_server.py        # FastMCP: list_*, search_memory, ask_memory, get_meeting_overview
  cli.py               # `vproc ingest <video>` (and the speaker-confirm prompt)
tests/                 # unit + grounding integration fixtures
docs/superpowers/specs/2026-06-02-vproc-design.md
.vscode/mcp.json
pyproject.toml
vproc.toml             # model roles → {backend, base_url, model_id, dtype}, paths, thresholds
```

---

## 16. Open questions / risks (carry into the plan)

1. **vLLM sm_120 tag is a moving target** — one source flags v0.18.1 as a ~3× regression vs v0.17.1;
   another recommends ≥0.19.x / cu130 nightly. **Benchmark + pin** on the actual RTX PRO 4000 at
   deploy time. Have the AWQ-INT4 path + the `--dtype fp16 --kv-cache-dtype fp16 --quantization fp8`
   workaround ready.
2. **OCR faithfulness ceiling** — VLM OCR can still invent text; the verifier mitigates but doesn't
   eliminate. Mitigation: transcript is *primary* evidence; OCR is *secondary*; raw frame kept for
   human verification.
3. **pyannote on Blackwell** has had intermittent sm_120 kernel errors — verify; CPU fallback ready.
4. **CC-BY-4.0 attribution** for pyannote community-1 differs from the Apache/MIT rest — surface
   attribution in an MCP "about" resource, or use MIT `3.1`.
5. **Calibration** — rerank floor + HHEM threshold must be tuned on the real meetings, or the system
   over-/under-refuses.

---

## 17. Success criteria

- Ask in Copilot: *"Give an overall description of meeting 1"* → a cited summary, every sentence
  traceable to a timestamp+speaker.
- *"Did <name> talk about <topic>?"* → answer drawn only from that speaker's segments, or "not
  discussed."
- *"How does a petrol engine work?"* (not in the videos) → **"Not discussed in these meetings."**
- Zero facts in any answer that aren't backed by a resolvable citation into the source video.
