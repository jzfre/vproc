# Reliability and correctness pass — 2026-09-10

## Asked

Understand the project, identify gaps and bugs, and improve it before manual testing.

## Approach and decisions

- Baseline: 158 Python tests passed. Independent investigations covered ingest,
  answer/retrieval boundaries, UI state, and service/storage behavior.
- Preserve the local-first architecture, existing UI, and documented LAN bind.
  Reproduce consequential failures and add focused regressions; no model calls
  against real recordings or changes to existing meeting data.
- Unique screenshot generations preserve files referenced by old rows across
  failed re-ingest. Keep the existing API filename shape and speaker sidecar path.
- Keep browsing/search independent of optional HHEM loading. Remove stale media
  caching and filter/project metadata inside LanceDB to avoid full-library reads.

## Changed

- Ingest preservation across failed writes/empty extraction; stable absolute
  paths; embedding count checks; real video duration for silent-screen intervals.
- Embedding response validation, strict answer JSON, valid faithfulness scores,
  full-token lexical matches, and speaker/time metadata available as evidence.
- UI meeting-switch and media-error races, failed citation jumps, and global
  query wording; dependency-free behavioral UI regressions.
- Lazy serialized HHEM loading, refreshed media paths, directory/missing-UI
  errors, and frame symlink containment. SQL metadata filtering escapes values.
- README now documents actual CLI commands, runtime prerequisites, recovery,
  model limitations, and testing. `.env.example` covers supported tuning values.

## Remaining gaps and next manual test

The later [hardening pass](../hardening/journal.md) addresses the persistence,
embedding-identity, and service gaps recorded below; see it for current boundaries.

- Real-model quality is unmeasured in this pass. Ingest one representative short
  recording, verify transcript/speakers/screens, then ask a known-answer question
  and an unrelated question; inspect citations and abstention against playback.
- UI: rapidly switch meetings, seek via transcript/timeline, follow citations
  across meetings, then re-ingest a recording and confirm new media is served.
- Ingest is not transactional or concurrency-safe for the same memory. Use one
  ingest at a time; interrupted writes may need a successful re-ingest.
- Same filename stems intentionally replace each other. There is no persisted
  embedding-model identity, schema migration system, or distinct memory catalog.
- Retrieval considers a small top-k window; long-meeting summaries and negative
  answers can miss material. OCR/ASR mistakes and scorer errors remain possible.
- Browser playback depends on the source codec/container; no transcoding layer.
- MCP tool calls have no project filters, and MCP/FTS setup errors are still
  best effort with limited visibility. These remain separate follow-up work.
- Existing test-style lint debt and dependency deprecation warnings remain;
  production lint is the check documented for this pass.

## Brain

Current architecture and recovery boundaries are in `docs/brain/architecture.md`.

## Validated

- `uv run pytest -q`: 213 passed; three existing dependency deprecation warnings.
- `node --test tests/test_ui.cjs`: 11 passed. `uv run ruff check vproc` and
  `git diff --check` pass. `uv build` produces a wheel containing all UI assets.
- Synthetic integration: real ffmpeg/ffprobe → LanceDB → FastAPI segments,
  screenshots, and HTTP video ranges; same-stem replacement from another source
  directory updates the running app and removes stale screenshots.
- Fixture Chromium: search citation lands at 24s, answer citation at 10s;
  failed citations preserve playback, rapid switches preserve selected state,
  missing media permits transcript navigation, and empty catalog behaves cleanly.
  No JavaScript exceptions; only injected 404s. Temporary fixture server stopped.
- No live-model inference or existing meeting-data writes performed.
