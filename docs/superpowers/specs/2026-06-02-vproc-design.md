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
- A local **HTTP service** (REST `+` MCP) exposing this to VS Code / Copilot and any LAN client.

### Out of scope (v1, may follow later)
- Standalone CLI / web chat front-ends (the RAG core is reused when we add them).
- Vision-model *description* of non-text visuals (screens are mostly text).
- Cloud/company-API answerer (kept as a pluggable swap; not built in v1).
- Real-time / live-meeting processing (batch only).

### Current scale
**2 meetings, ~50 minutes each.** One **Project**, two **Memories**. Ingest is quick (minutes per
meeting). The architecture is unchanged by scale — grounding is the point, not size.

---

## 2. Hard constraints

| Constraint | Consequence |
|---|---|
| **100% open-source + local** in the pipeline | No proprietary frameworks (⇒ **no** Apple Vision OCR). Whisper, pyannote, Qwen, LanceDB, etc. — all Apache/MIT/BSD. The **only** gated piece is pyannote's model (free HF token + one-time accept; CC-BY-4.0). |
| **No cloud generation** | Copilot/VS Code is a **front-end only** — it performs **zero** generation. All OCR/ASR/embedding/answering runs on local open models. Confidential content never leaves the owner's machines. |
| **Endpoint-configurable** | vproc is a base app (ffmpeg, ASR, retrieval, HHEM, MCP) + **3 env-configured AI endpoints** (OCR, embeddings, grounding). Not tied to a machine; runs on macOS in dev. OCR → voyage's 9B vLLM; embeddings/grounding → Mac LM Studio (or voyage's 9B). See §9. |
| **One app + external AI endpoints** | The app holds **all** business logic and exposes an HTTP API; it calls 3 AI endpoints (OCR/embed/grounding). They're external for one reason: we want **local + private**, but **no single box has the RAM to hold every model at once** — so each model runs where there's room, as a plain OpenAI-compatible endpoint. See §9. |
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
INGEST (batch; base app runs the pipeline, calls the OCR endpoint)
  meeting.mp4
    ├─ ffmpeg ──► audio.wav (mono 16 kHz)
    │              └─ WhisperX ──► word-level transcript
    │                   └─ pyannote ──► speaker turns ──► speaker-attributed words
    └─ ffmpeg (mpdecimate+scene) + pHash dedup ──► one frame per distinct screen state
                   └─ OCR endpoint (Qwen3.5-9B, verbatim) ──► on-screen text (secondary)
                          └─ nameplate OCR ──► cluster→name suggestion ──► owner confirms once
                                   │
                     ALIGN by timestamp ──► Segment{ t, speaker, said_text, on_screen_text }
                                   │
                     embed (embed endpoint) ──► LanceDB (vector + BM25 + metadata)

QUERY (interactive; base app + grounding/embed endpoints)
  VS Code / Copilot ──MCP(HTTP)──► search_memory(q)   ──► raw cited verbatim segments
                       └────► ask_memory(q)
                                  ├─ hybrid retrieve (+ optional rerank)
                                  ├─ score floor? ── empty ──► "Not discussed in these meetings."
                                  ├─ evidence list E1..En (opaque keys)
                                  ├─ grounding endpoint (non-thinking, JSON schema, cite-or-refuse)
                                  ├─ HHEM faithfulness gate (per claim, in-process)
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
                    on_screen_text, on_screen_confidence, # verbatim OCR (secondary evidence; confidence optional)
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
| 4 | `ocr.py` | Read screen text | **One call** to the **OCR endpoint** (`VPROC_OCR_*`, default voyage's `Qwen3.5-9B-AWQ`). **Verbatim prompt** ("transcribe only visibly-rendered text; `[illegible]` for unreadable; no inference/translation/spelling-fix"). OCR is *just a model* — no extra verifier in v1. The **spoken transcript stays the primary evidence**; on-screen text is secondary. *(Optional later: logprob/confidence tagging.)* |
| 5 | `names.py` | cluster → real name | Match an OCR'd **nameplate** to the active-speaker time window; require temporal overlap. **Never infer a name from voice.** Suggest → owner confirms once (CLI prompt in v1). Persist `cluster_id → name` + mean speaker-embedding for reuse. No evidence ⇒ keep `SPEAKER_xx`. |
| 6 | `align.py` | Build segments | Turn deduped frames into screen-state intervals; attach overlapping speaker-attributed transcript; emit `Segment`s + `embed_text`. |
| 7 | `embed_index.py` | Embed + store | Qwen3-Embedding-0.6B (**instruction prefix on queries only**, `padding_side="left"`). Write vectors + metadata to LanceDB. |
| `pipeline.py` orchestrates 1→7 for one memory and writes the `Memory` record. |

**Placement (§9):** ffmpeg, ASR + diarization (2–3, in-process via MLX/MPS), alignment, and LanceDB all
run **inside the base app** (the Mac in dev). Only **OCR (4)** and **embedding (7)** are external
endpoints (`VPROC_OCR_*`, `VPROC_EMBED_*`).

**Speakers are fully visible on the nameplates in both target meetings**, so step 5 auto-naming is
expected to succeed; the manual confirm is a quick safety check.

---

## 7. Query / answering pipeline

Under `vproc/retrieve/` and `vproc/answer/`.

1. **Retrieve** (`store/`): LanceDB **hybrid** (dense + BM25, RRF) with `prefilter=True` metadata
   filter (`project`, `memory`, `speaker`, `start_ts` range). Native FTS (`use_tantivy=False`).
2. **Rerank** *(optional, in-process)*: Qwen3-Reranker-0.6B cross-encoder over candidates — most useful as the corpus grows; for 2 meetings, hybrid search alone may suffice (add it in Phase 2 if precision needs it).
3. **Abstention gate** (code): if no candidates or top rerank score `< floor` ⇒ return
   **"Not discussed in these meetings."** (`answered=false`). *No LLM call.*
4. **Evidence list**: assign opaque keys `E1..En` to the top-k; build a numbered list of verbatim
   `said_text` / `on_screen_text` (transcript is primary evidence, OCR secondary).
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

vproc runs as **one HTTP service** on the LAN (`VPROC_HOST` / `VPROC_PORT`, default `:8765`) that
encapsulates *all* the business logic — ffmpeg, ASR, retrieval, HHEM, grounding. It exposes that logic
two ways over the same functions: a **plain REST API** (`POST /ask`, `POST /search`) for any client,
and an **MCP interface** (`/mcp`, official `mcp` SDK / FastMCP, HTTP transport) so **VS Code / Copilot —
anywhere on the local network** — can call it as a tool in Agent mode. Nothing runs on the editor's
machine; it just does `POST http://<vproc-host>:8765/…`.

Operations (exposed as both REST routes and MCP tools):
- `list_projects()` / `list_memories(project)` — navigation.
- `search_memory(query, project?, memory?, speaker?, k=8)` → **raw cited verbatim segments**, no
  synthesis (impossible to hallucinate; lets Copilot/owner audit).
- `ask_memory(question, project?, memory?, speaker?)` → **grounded answer** (generated *inside* vproc
  via the grounding endpoint) with inline citations + the evidence array, or "Not discussed in these
  meetings."
- `get_meeting_overview(memory)` → grounded summary (Section 7).

VS Code registration — `.vscode/mcp.json`, top-level key `servers`, HTTP transport:
```json
{ "servers": { "vproc": { "type": "http", "url": "http://<vproc-host>:8765/mcp" } } }
```
Non-Copilot clients hit REST directly, e.g. `POST http://192.168.1.103:8765/ask {"question": "…"}`
→ grounded, cited JSON. Tools surface in **Copilot Chat Agent mode**. Generation happens **inside
vproc** (via the grounding endpoint) — Copilot's cloud model never synthesizes, so confidential
segments never leave the tailnet.

---

## 9. Deployment — a base app + three configurable AI endpoints

vproc is **not tied to a machine.** It's a **base app** that you run wherever (macOS in dev), plus
**three AI services configured by environment variable** as OpenAI-compatible URLs. ffmpeg and the
whole pipeline run inside the app; only OCR / embeddings / grounding are external calls.

**In-process (the base app — runs wherever vproc runs):** ffmpeg, ASR + diarization (WhisperX +
pyannote via MLX/MPS), alignment/chunking, retrieval over **LanceDB**, the optional reranker, the
**HHEM** faithfulness check, grounding orchestration, and the **MCP HTTP server**.

**External AI endpoints (env-configured OpenAI-compatible URLs):**

| Env var | Default | What it is |
|---|---|---|
| `VPROC_OCR_BASE_URL` / `VPROC_OCR_MODEL` | `http://voyage:8000/v1` · `QuantTrio/Qwen3.5-9B-AWQ` | voyage's already-running vision vLLM. **OCR is *just* this model call** — nothing else. |
| `VPROC_EMBED_BASE_URL` / `VPROC_EMBED_MODEL` | Mac LM Studio `http://localhost:1234/v1` · `Qwen3-Embedding-0.6B` | query/segment embeddings |
| `VPROC_GROUNDING_BASE_URL` / `VPROC_GROUNDING_MODEL` | **Mac LM Studio** `http://localhost:1234/v1` · `Qwen3-32B-AWQ` *(default)* | the answerer — Mac by default (more context than voyage's 65 K); voyage 9B is the fallback |

- **Why the models are external endpoints (and split across machines):** we want everything **local +
  private**, but **no single box has the RAM to hold every model at once**. So each model runs where
  there's room, exposed as a plain OpenAI-compatible endpoint, and the app just calls it. (Bonus: each
  stays resident, so there's no reload thrash mid-run.)
- **Defaults:** `VPROC_OCR` → voyage's running 9B (`voyage:8000`, vision); `VPROC_EMBED` and
  `VPROC_GROUNDING` → the **Mac** (LM Studio, `localhost:1234`) — grounding on the Mac 32B by default.
  All hosts on the `aqui.technology` tailnet (near wire-speed on the LAN).
- **Context window:** voyage's 9B is served at **65 K** (`--max-model-len 65536`). That's actually
  ample for *retrieval* grounding — we feed only the handful of retrieved segments, never a whole
  meeting — but grounding defaults to the **Mac 32B** for extra headroom and smoother summaries.
- **Verify first:** confirm the running `Qwen3.5-9B-AWQ` actually accepts image input
  (`/v1/chat/completions` with an `image_url`) before trusting it for OCR. `/v1/models` is confirmed
  live (logprobs enabled).
- **No backend-specific SDK in business logic** — a thin client (`vproc/llm/`) speaks only OpenAI
  `/v1`, so any endpoint (vLLM, LM Studio, Ollama, …) is interchangeable.

---

## 10. The validated stack

| Component | Choice | Version / model | License |
|---|---|---|---|
| Demux + frames | ffmpeg + pHash | ffmpeg 7.x (LGPL), imagehash, Pillow | LGPL/BSD/HPND |
| ASR + word ts | WhisperX | whisperx 3.7.x · faster-whisper 1.2.1 · ct2 4.7.2 · `large-v3-turbo` | BSD/MIT |
| Diarization | pyannote.audio | 4.x · `speaker-diarization-community-1` (MIT `3.1` fallback) | MIT code / CC-BY-4.0 weights (gated) |
| Screen OCR | **reuse voyage's running** Qwen3.5-9B (vision) | `QuantTrio/Qwen3.5-9B-AWQ` @ `voyage:8000` | Apache-2.0 (AWQ) |
| Embeddings | Qwen3-Embedding | `Qwen3-Embedding-0.6B` | Apache-2.0 |
| Hybrid store | LanceDB | native FTS (`use_tantivy=False`) | Apache-2.0 |
| Reranker | Qwen3-Reranker | `Qwen3-Reranker-0.6B` | Apache-2.0 |
| Answerer (grounding) | **Qwen3-32B-AWQ on the Mac** *(default)* / voyage 9B *(fallback)* | `Qwen3-32B-AWQ` · `QuantTrio/Qwen3.5-9B-AWQ` | Apache-2.0 |
| Faithfulness | Vectara HHEM | `HHEM-2.1-Open` (~440 MB) | Apache-2.0 |
| Constrained decode | XGrammar / GBNF | vLLM default / llama.cpp | Apache-2.0 / MIT |
| Front-end | MCP Python SDK | `mcp` 1.27.2 (FastMCP) | MIT |

**Grounding/answerer** (endpoint `VPROC_GROUNDING_*`): the answerer only *reads provided segments and
writes a cited answer or abstains* — comprehension + formatting, not world-knowledge reasoning — and
grounding is enforced **structurally** (code-owned citations, retrieval-gated abstention, HHEM), not by
model size. **Default: the Mac's `Qwen3-32B-AWQ`** (LM Studio) — more context than voyage's 65 K and smoother
summaries. The 9B is a perfectly good **fallback** (grounding is structural, so even 9B holds the
contract) when the Mac is busy — one env var. **HHEM** (Vectara HHEM-2.1-Open, ~440 MB NLI) runs
**in-process in the base app** to double-check each answer sentence against its cited evidence — it is
not an endpoint. **OCR:** reuse voyage's `Qwen3.5-9B-AWQ`; swap to a dedicated `Qwen3-VL-8B-Instruct`
only if its OCR proves weak.

---

## 11. Resource sizing

Because the AI models sit on **always-resident endpoints** (no in-flight reload), sizing is simple:

- **voyage (24 GB Blackwell):** ~90 % committed to the running `Qwen3.5-9B-AWQ`
  (`--gpu-memory-utilization 0.90`). It serves **OCR** (and, if you choose, **grounding**) and is left
  alone. Cap OCR image size and pHash-dedup first so a high-res slide's vision tokens don't strain the
  already-busy card.
- **Mac (64 GB unified):** runs the **base app** (ffmpeg + WhisperX ≈ 3 GB + pyannote ≈ 2 GB + HHEM
  ≈ 0.5 GB + LanceDB) **plus** LM Studio hosting **embeddings** ≈ 1 GB and, optionally, the **32B
  answerer** ≈ 18–20 GB. Everything co-fits in 64 GB with wide headroom.
- **Default grounding = Mac 32B** → queries are self-contained on the Mac (only ingest-OCR needs
  voyage) with more context than voyage's 65 K. Fallback to voyage's 9B if the Mac is busy — one env
  var; grounding is structural, so 9B still holds the contract.

**Rough timing (2 × 50-min meetings):** OCR on voyage's warm GPU + ASR on the Mac (MLX) overlap; each
meeting ingests in minutes. Interactive answers return in a few seconds either way.

---

## 12. Error handling & edge cases

- **Missing frame timestamps** ⇒ hard error (never infer from filename/fps).
- **Whisper silence hallucination** ⇒ VAD on, `condition_on_previous_text=False`.
- **If ASR/diarization run on a Blackwell GPU** (not the default Mac/MLX path) ⇒ `compute_type="float16"`
  (INT8 crashes on sm_120), CUDA 12.8+/cuDNN 9/torch cu128, pinned ctranslate2. (The voyage vLLM
  endpoints are managed outside vproc via `qwen3.service`.)
- **Missing `HF_TOKEN`** ⇒ clear setup message (pyannote is gated); MIT `3.1` fallback documented.
- **OCR `[illegible]` / weak spans** ⇒ kept as-is but treated as *secondary* evidence; the spoken transcript is primary, so a bad OCR span can't by itself fabricate an answer.
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
- **OCR sanity:** fixture frame ⇒ the OCR endpoint returns the visible text (spot-check; OCR is secondary evidence).
- **Cross-platform smoke:** the same `ask_memory` runs against `/v1` on Mac (MLX/Ollama) and would
  run on Blackwell (vLLM) — backend swapped by config only.
- **Threshold calibration:** rerank floor + HHEM threshold tuned on a small labeled subset of the two
  real meetings.

---

## 14. Build phases

- **Phase 0 — scaffold:** `uv` project, config registry, `/v1` client abstraction, LanceDB schema,
  data models, MCP server skeleton.
- **Phase 1 — MVP (vertical slice, base app on the Mac):** ingest **one** meeting → frames+pHash →
  Whisper in-process (MLX, *anonymous speakers*) → OCR via `VPROC_OCR` (voyage 9B) → align → segments →
  embed via `VPROC_EMBED` → LanceDB; retrieve + abstention; `ask_memory` via `VPROC_GROUNDING`
  (default Mac 32B) with code-owned citations + in-process HHEM; `search_memory`; **HTTP service
  (REST + MCP) on the LAN, queried from VS Code Agent mode.**
- **Phase 2 — speakers + summaries:** add diarization (pyannote, in-process) + nameplate name
  resolution → owner confirm; Project/Memory model; optional reranker; `get_meeting_overview` grounded
  summaries; constrained decoding; (optional) wire voyage's 9B as a grounding fallback.
- **Phase 3 — both meetings + harden:** ingest the second meeting; calibrate thresholds; verify the 9B
  accepts images / pin voyage vLLM; (optional) company-key grounding endpoint; (optional) CLI front-end.

---

## 15. Repo layout

```
vproc/
  config.py            # loads .env endpoints + thresholds + paths
  models.py            # Project, Memory, Segment, Evidence, Citation, Answer
  llm/                 # OpenAI /v1 client: chat, embed, rerank (backend-agnostic)
  ingest/              # frames, transcribe, diarize, ocr, names, align, embed_index, pipeline
  store/               # LanceDB schema + hybrid search + metadata filters
  retrieve/            # hybrid retrieve + rerank + score-gate
  answer/              # evidence builder, constrained gen, citation resolver, HHEM gate, summarizer
  service.py           # the HTTP service: REST (/ask, /search) + MCP (/mcp) over the same logic
  cli.py               # `vproc ingest <video>` (and the speaker-confirm prompt)
tests/                 # unit + grounding integration fixtures
docs/superpowers/specs/2026-06-02-vproc-design.md
.vscode/mcp.json
pyproject.toml
.env                   # VPROC_OCR_* / VPROC_EMBED_* / VPROC_GROUNDING_* / VPROC_HOST / VPROC_PORT / VPROC_INDEX_PATH / HF_TOKEN
```

---

## 16. Open questions / risks (carry into the plan)

1. **vLLM sm_120 tag is a moving target** — one source flags v0.18.1 as a ~3× regression vs v0.17.1;
   another recommends ≥0.19.x / cu130 nightly. **Benchmark + pin** on the actual RTX PRO 4000 at
   deploy time. Have the AWQ-INT4 path + the `--dtype fp16 --kv-cache-dtype fp16 --quantization fp8`
   workaround ready.
2. **OCR faithfulness ceiling** — VLM OCR can invent text. v1 keeps it simple (no extra verifier);
   mitigation is structural — the spoken **transcript is primary** evidence, OCR is **secondary**, and
   the raw frame is kept for human verification. Add logprob/confidence tagging later only if OCR
   errors actually bite.
3. **pyannote on Blackwell** has had intermittent sm_120 kernel errors — verify; CPU fallback ready.
4. **CC-BY-4.0 attribution** for pyannote community-1 differs from the Apache/MIT rest — surface
   attribution in an MCP "about" resource, or use MIT `3.1`.
5. **Calibration** — rerank floor + HHEM threshold must be tuned on the real meetings, or the system
   over-/under-refuses.
6. **Reused OCR model** — verify `Qwen3.5-9B-AWQ` accepts image input before relying on it; the
   VLM-OCR faithfulness caveat (risk 2) applies. Spot-check OCR quality on a real slide early; if weak,
   swap in a dedicated `Qwen3-VL-8B-Instruct` (config-only change).
7. **Host availability** — the base app + embeddings + **grounding (default)** are the Mac; **OCR** is
   voyage. A *query* needs the Mac up (embed + grounding); *ingest* additionally needs voyage (OCR).
   Fallback: repoint grounding to voyage's 9B if the Mac is busy.
8. **voyage GPU is ~90 % committed** to the running model — never schedule GPU ingest there; ASR runs
   on the Mac (or voyage CPU in the MVP).

---

## 17. Success criteria

- Ask in Copilot: *"Give an overall description of meeting 1"* → a cited summary, every sentence
  traceable to a timestamp+speaker.
- *"Did <name> talk about <topic>?"* → answer drawn only from that speaker's segments, or "not
  discussed."
- *"How does a petrol engine work?"* (not in the videos) → **"Not discussed in these meetings."**
- Zero facts in any answer that aren't backed by a resolvable citation into the source video.
