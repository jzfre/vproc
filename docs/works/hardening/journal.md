# Index and service hardening — 2026-09-10

## Asked

Continue the reliability pass by hardening the remaining persistence and failure boundaries.

## Design and plan

- Keep the local LanceDB architecture, filename-based memory identity, and LAN
  deployment. Preserve the previous pass's uncommitted changes and real data.
- [x] Add reentrant process-backed writer locks that fail promptly on contention
  and release when a process exits. Lock the index and frame writes together.
- [x] Replace a memory's rows with one LanceDB merge commit, scoped by project and
  memory. Stage unique frames before commit; clean only previously indexed files
  after success. Leave ambiguous write outcomes recoverable.
- [x] Store model identity with new indexes and reject incompatible vector reads
  or writes. Keep legacy metadata/media readable; require a new index and re-ingest
  when model identity is absent. Do not invent historical provenance.
- [x] Validate configuration and REST/MCP requests, expose matching MCP filters,
  and return actionable backend/configuration failures without leaking credentials.
- [x] Verify real temporary-store replacement, failure recovery, lock contention
  and process exit, model mismatch, HTTP/MCP boundaries, and existing UI tests.

## Decisions

- Database atomicity covers rows; files are staged first with unique names. The
  filesystem and database are not a single distributed transaction.
- A configured model identifier and vector dimension detect configuration drift;
  they cannot prove weights behind a server alias have remained unchanged.
- Use the same writer-lock object inside a Store for nested operations. Distinct
  Store instances/processes contend on a stable lock file; never unlink lock files.
- Index and frame locks use distinct filenames so both can share a directory.
- Legacy model adoption is intentionally absent: dimensions cannot establish
  provenance. Explicit new index/frame directories preserve the old library.

## Changed

- Store/ingest: process locks, atomic scoped merge replacement, collision and
  vector validation, persisted model/dimension contract, safe FTS failure warnings.
- Paths/CLI/naming: reject unsafe identifiers and symlinked child directories,
  serialize edits with ingest, preserve zero-match mappings, atomic sidecar saves,
  and print actionable configuration/index errors with unsuccessful exit status.
- Service: shared bounded REST/MCP validation and metadata filters, sanitized
  timeout/unavailable/index errors, MCP setup warnings, local `/api/status`.
- Shell ask/search helpers preserve server error guidance and fail cleanly for
  HTTP, connection, and malformed-response errors.
- Config: reject invalid numeric domains, endpoint syntax, empty model/path
  settings and boolean typos; ignore malformed dotenv keys/NUL lines.
- Declared filelock directly; raised LanceDB floor to the tested 0.33 API.

## Remaining limits

- Actual ASR/OCR/diarization/grounding quality still needs a representative video.
- Locks protect cooperating local writers, not unsupported shared/network storage
  or third-party code bypassing vproc's locking protocol.
- Filesystem and database commit together are not atomic; crashes may leave
  unreferenced screenshots. Separate libraries should have separate frame roots.
- Model names/dimensions do not identify the actual remote weight revision.
- Existing unauthenticated LAN deployment and top-k retrieval limitations remain.

## Validated

- `uv run pytest -q`: 476 passed; three pre-existing dependency warnings.
- `node --test tests/test_ui.cjs`: 11 passed; Node/Bash syntax checks pass.
- Production Ruff, `uv lock --check`, `uv build`, and `git diff --check` pass.
  Dependency versions are unchanged; the lockfile records the explicit filelock
  dependency, tested LanceDB floor, and resolver-normalized platform markers.
- Real temporary LanceDB tests cover single row commits, failed-commit recovery,
  model/dimension mismatches and REST/MCP SQL filtering. Process tests verify
  lock release after normal exit and forced termination. Shell tests use a local
  fixture HTTP server; synthetic video tests run ffmpeg and FastAPI end to end.
- No existing meeting data was migrated or modified; no real model calls made.
