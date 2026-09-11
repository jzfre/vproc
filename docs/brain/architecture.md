# Vproc architecture and operating boundaries

## Current implementation

- `vproc/cli.py` exposes ingest, serve, speaker corrections, help/version, and
  offline prerequisite checks (`doctor`); shell query
  clients live in `scripts/`. Configuration comes from `vproc/config.py`, with
  the CLI loading `.env` from its working directory.
- `vproc/ingest/pipeline.py` orchestrates ffmpeg frames/audio, ASR, optional
  diarization/nameplate naming, OCR, alignment, embedding, and index replacement.
  Local ASR uses mlx-whisper on Apple Silicon; remote ASR is configurable.
- `vproc/store/lancedb_store.py` owns the embedded `segments` table. Its schema
  and vector dimension are inferred on first insert. Memory identity is the pair
  `(project_id, filename stem)`; there is no independent persisted Memory table.
- Retrieval combines cosine-vector search and full-text search with reciprocal
  rank fusion. Only a full token match can lift the similarity floor for a
  literal keyword query. Keep the embedding model stable within an index.
- `vproc/answer/` builds evidence, requests structured claims, rejects malformed
  output, checks each claim against cited evidence with HHEM, then resolves
  citations in code. Abstention is not proof a subject never occurred: only the
  top eight retrieved segments are considered by default.
- `vproc/service.py` serves REST, streamable HTTP MCP, media, and a static UI.
  Metadata queries project out vectors and filter meetings inside LanceDB.
  HHEM loads on the first scoring request and scoring is serialized per app.
  REST/MCP share query validation, filters, and sanitized backend errors.
  `/api/status` reads only local index compatibility and MCP setup availability.
- `vproc/ui/app.js` is dependency-free browser JavaScript. The player lists the
  default project; ask/search are global. Tests exercise asynchronous state with
  Node's built-in runner.

## Persistence and failure boundaries

- New ingest rows use absolute source/frame paths. New screenshots have unique
  generation prefixes; old screenshots are cleaned only after index replacement.
- Empty extraction preserves an existing memory. OCR, diarization, and naming
  may degrade with warnings; embedding/index failures stop ingest.
- `vproc/locking.py` uses process-backed, thread-reentrant FileLock objects.
  Store mutations share one index lock object; ingest/CLI edits also lock frame
  storage. Contenders fail promptly. Locks release on process exit; never unlink
  their files. These guarantees assume cooperating vproc writers on local storage.
- Memory replacement is one scoped LanceDB merge commit. Frames are staged first
  and cleanup removes only previous indexed paths within the current memory's
  directory. A failed/ambiguous commit keeps screenshots; orphan files are not
  deleted speculatively. The filesystem and database are not one transaction.
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
