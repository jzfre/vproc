# Vproc architecture and operating boundaries

## Current implementation

- `vproc/cli.py` exposes ingest, serve, analyze, speaker listing/corrections,
  help/version, and prerequisite checks (`doctor`, offline by default; `--online`
  probes every configured server and Hugging Face access with injectable probes).
  Shell query clients live in `scripts/`. Configuration comes from
  `vproc/config.py`, with the CLI loading `.env` from its working directory;
  `KNOWN_ENV` lists every setting (plus `VPROC_ASK_URL` for the helper scripts)
  so doctor can flag misspelled names.
- `vproc/llm/client.py` is the only model-server client. Endpoints may carry a
  bearer key (`VPROC_<NAME>_API_KEY[_FILE]`); `client.configure(cfg)` registers
  keys per base URL at every entry point (ingest, `create_app`, analyze,
  `doctor --online`),
  and the client cache is keyed by (base URL, key). Generation calls are never
  retried by the SDK and use a short connect timeout; a generation cut off at
  `max_tokens` is logged and treated as no answer, never parsed from the
  reasoning channel.
- `vproc/ingest/pipeline.py` orchestrates an embedding preflight, ffmpeg
  frames/audio, ASR, optional diarization/nameplate naming, OCR, alignment,
  embedding, and index replacement, printing one stderr line per stage.
  Local ASR uses mlx-whisper on Apple Silicon; remote ASR is configurable. Both
  request word timestamps: `diarize.assign_speakers` re-cuts worded segments
  into sentences (a sentence may continue across an ASR segment boundary when
  the next segment closes it), gives each sentence to the speaker of most of its
  word time (pyannote's exclusive, non-overlapping turns), and emits
  same-speaker runs of at most 30 s, so chunks end on sentence boundaries.
  Word-by-word labels were measured and rejected: diarization noise caused 27
  mid-sentence speaker flips on the reference meeting, sentence voting 0. It
  falls back to whole-segment max-overlap when words are absent or do not spell
  the segment. A tail joins the next segment only when that segment follows
  within 2 s and closes the sentence, so a fully unpunctuated stream is never
  voted across segments; runs made of whole segments keep the ASR text (word
  tokens may lack punctuation or spaces). Diarization receives the wav as an in-memory waveform, so
  it does not depend on torchcodec. An empty `VPROC_OCR_BASE_URL` disables OCR
  and nameplate naming.
- `scripts/local_models.py` (+ `.lock`) is an optional CPU server for ASR
  (faster-whisper, hotwords for consistent casing/punctuation) and embeddings;
  `deploy/systemd/` has user-unit templates. Neither is part of the package.
- `vproc/store/lancedb_store.py` owns the embedded `segments` table. Its schema
  and vector dimension are inferred on first insert. Memory identity is the pair
  `(project_id, filename stem)`; there is no independent persisted Memory table.
- Retrieval combines cosine-vector search and full-text search with reciprocal
  rank fusion. Only a full token match can lift the similarity floor for a
  literal keyword query. Keep the embedding model stable within an index.
- `vproc/answer/` builds evidence, requests structured claims, rejects malformed
  output, checks each claim against cited evidence with HHEM, then resolves
  citations in code. Abstention is not proof a subject never occurred: only the
  top eight retrieved segments are considered by default. Answers and evidence
  carry diagnostics (`abstain_reason`, `top_sim`, `hhem_scores`, per-hit
  `similarity`/`keyword_match`) as numbers and enums only: the text of a
  rejected claim is never returned. A query-only embedding prefix
  (`VPROC_EMBED_QUERY_PREFIX`) supports instruction-tuned embedders without
  touching stored vectors; `sim_floor` is embedding-model specific. Scoring each
  cited evidence item separately was evaluated and rejected: it gives every
  claim several attempts at the gate. The answer schema bounds the claims array
  (`MAX_CLAIMS`): an unbounded one let a non-thinking model run to `max_tokens`.
- `vproc/service.py` serves REST, streamable HTTP MCP, media, and a static UI.
  Metadata queries project out vectors and filter meetings inside LanceDB.
  HHEM loads on the first ask, before generation, and scoring is serialized per
  app; a load failure is a fixed-text 503, is retried on the next ask, and shows
  in `/api/status` as `verifier`.
  REST/MCP share query validation, filters, and sanitized backend errors.
  `/api/status` reads only local index compatibility and MCP setup availability.
- `vproc/ui/app.js` is dependency-free browser JavaScript. The player lists the
  default project; ask/search are global. Tests exercise asynchronous state with
  Node's built-in runner.

## Persistence and failure boundaries

- New ingest rows use absolute source/frame paths. New screenshots have unique
  generation prefixes; old screenshots are cleaned only after index replacement.
- Empty extraction preserves an existing memory. OCR, diarization, and naming
  may degrade with warnings and one final `DEGRADED` summary; embedding/index
  failures stop ingest, and the embedding backend is checked before any media
  work. A failed diarization never replaces a memory that already has speaker
  labels. ffmpeg/ffprobe failures surface as `IngestError` messages.
- `vproc/locking.py` uses process-backed, thread-reentrant FileLock objects.
  Store mutations share one index lock object; ingest/CLI edits also lock frame
  storage. Contenders fail promptly. Locks release on process exit; never unlink
  their files. These guarantees assume cooperating vproc writers on local storage.
- Memory replacement is one scoped LanceDB merge commit. Frames are staged first
  and cleanup removes only previous indexed paths within the current memory's
  directory, and only frames that an index row references are staged at all.
  A failed/ambiguous commit keeps screenshots; orphan files are not deleted
  speculatively. The filesystem and database are not one transaction.
- `vproc/paths.py` validates child identifiers and rejects symlinked project/memory
  directories before writes. Speaker-map saves use atomic file replacement.
- New index schema metadata records `vproc:embedding_model` and
  `vproc:embedding_dimension`. Production reads/writes validate both; unknown or
  malformed identity is rejected. Legacy metadata/media remain readable, but
  model-dependent operations require a new index and re-ingest. Same-name server
  aliases changing weights are not detectable. See README for recovery commands.
- Separate libraries should use separate frame directories because speaker
  sidecars are keyed by project/memory, not by index identity.
- The service is intentionally unauthenticated and defaults to all interfaces.
  Preserve this documented LAN behavior unless explicitly changing deployment.
- Historical `docs/superpowers/` specs describe intent and earlier alternatives;
  current source and these notes describe what actually runs.

## Verification

Use the commands in `README.md`. Offline checks validate pipeline contracts,
temporary real LanceDB persistence/atomic replacement, process-lock termination,
synthetic ffmpeg ingest, HTTP media ranges, MCP calls/filtering, and UI state.
Real ASR/OCR/diarization/answer quality needs a
representative recording and configured model endpoints.

## Release installation

- The source archive includes the lockfile, configuration example, docs, scripts,
  tests, and bundled UI; explicit Hatch file lists exclude private workspace data.
  Use `uv sync --locked --no-dev --extra runtime --python 3.12` from its directory.
- On Linux the lock resolves `torch`, `torchaudio` and `torchcodec` to PyTorch's
  CPU index through `[tool.uv.sources]` (no CUDA code path exists; PyPI's Linux
  torchcodec needs CUDA libraries to load). `tests/test_packaging.py` guards the
  lock. `uv sync` without `--extra runtime` strips the model libraries from a
  shared `.venv`.
- Base dependencies support the service and remote model calls. Optional extras
  `answer`, `diarization`, and `local-asr` split local model libraries; `runtime`
  combines them. MLX is packaged only for macOS arm64. Other platforms need ASR
  endpoints. Current binary dependencies exclude Intel Macs; see README limits.
- MCP is capped below SDK 2 because the service uses `mcp.server.fastmcp`.
  The source lock fixes dependency versions; installing a wheel alone re-resolves.
- `scripts/smoke_release.py` exercises an installed distribution in temporary
  storage with loopback fake models, real FFmpeg/LanceDB, and HTTP startup/restart.
  It rejects editable installs and source-tree imports. It does not assess models.
- Configuration files use UTF-8 with optional BOM. FFmpeg metadata paths require
  both option-value and filtergraph escaping, including Windows drive separators.
