# vproc UI app — design

**Date:** 2026-08-04 · **Status:** approved by owner (chat) · **Scope:** Phase 2, third slice

## Goal

A local web UI over ingested meetings: watch the video with a who-is-speaking timeline and
seek controls, read the speaker-attributed transcript in sync, and use ask/search with
citations that deep-link into the video. Served by `vproc serve` itself — open
`http://localhost:8765`.

**Out of scope (v1):** remux-on-ingest for Safari mkv playback, a full library page with
thumbnails, editing/renaming speakers from the UI, multi-project support (UI assumes
project `default`, like the CLI), auth (LAN-trusted, like the REST API).

## Approach

Zero-dependency static frontend (vanilla `index.html` + `app.js` + `style.css`, no build
step, no CDN/network assets — must work offline) served by the existing FastAPI app, plus
four read-only `/api` endpoints. Chosen over a React/Vite SPA (second toolchain + build
step in a Python repo for a one-page tool) and a desktop wrapper (packaging overhead, no
benefit for a localhost tool).

## Components

### Static frontend: `vproc/ui/`

- `index.html` — single page: header (memory picker, meeting title), player pane
  (`<video>`, "now speaking" name, speaker-timeline strip), transcript pane, ask/search
  panel.
- `app.js` — all behavior, vanilla ES modules, no external deps:
  - boot: `GET /api/memories` → populate picker; select `?memory=` URL param if present,
    else first entry; update the URL param on switch.
  - load memory: `GET /api/memories/{id}/segments` → render timeline + transcript;
    set `<video src="/api/media/{id}">`.
  - timeline strip: one horizontal lane per speaker (consistent color per speaker,
    same palette as transcript), a rect per segment spanning `start_ts..end_ts`
    proportionally; playhead line driven by `timeupdate`; click → `video.currentTime`.
  - transcript: one block per segment — speaker name (colored), `mm:ss` start, said_text;
    when `on_screen_text` is non-empty, show it in a styled sub-block with the frame
    thumbnail (`/api/frames/{id}/{frame_name}`) when `frame_name` is present. Active
    segment (currentTime within `[start_ts, end_ts)`) highlighted and auto-scrolled into
    view (suppressed while the user is hovering the pane); click → seek.
  - "now speaking": the active segment's speaker name, empty between segments.
  - ask/search panel: one input + mode toggle. ask → `POST /ask {question}` → answered:
    claim list with citation chips (`title · mm:ss · speaker`); abstained: show the
    abstention text as-is. search → `POST /search {query}` → evidence rows
    (`mm:ss–mm:ss · speaker · text excerpt`). Clicking a chip/row seeks; if its
    `memory_title` differs from the current memory, switch memory first, seek after the
    segments load. A visible spinner while the request runs (ask can take a minute on a
    thinking grounding model).
- `style.css` — dark-first, deliberate design (frontend-design pass at implementation);
  speaker palette must be identical between timeline lanes and transcript names;
  responsive enough for a laptop window, no mobile requirement.

### Service endpoints (`vproc/service.py`)

All read-only, registered BEFORE the static mount so API routes win; the static mount
(`StaticFiles(html=True)` on `vproc/ui`) is added last at `/`. The existing
`/ask`, `/search`, `/healthz`, `/mcp` are unchanged.

- `GET /api/memories` → `[{memory_id, segment_count, duration_s, speakers: [str]}]`.
  Aggregated by scanning the table (small data; no new store indexes). `duration_s` =
  max `end_ts`. `speakers` sorted, excluding none (include `SPEAKER_0`/`SPEAKER_xx`
  labels as-is — the UI renders whatever the store has).
- `GET /api/memories/{id}/segments` → ordered by `start_ts`:
  `[{start_ts, end_ts, speaker, said_text, on_screen_text, frame_name}]` where
  `frame_name` is `basename(frame_path)` when the row has one, else `null`.
  404 when the memory has no rows.
- `GET /api/media/{id}` → streams the memory's `source_video` file with HTTP Range
  support (implemented by hand — Starlette's `FileResponse` does not honor Range, and
  `<video>` seeking requires 206 responses). Path resolved as stored (CWD-relative paths
  resolve against the service CWD). 404 with a JSON body naming the missing path when
  the file doesn't exist. Content-Type from the file extension (`video/mp4`,
  `video/x-matroska`, fallback `application/octet-stream`). Range semantics: single
  range only; `bytes=start-[end]` → 206 with `Content-Range`; unsatisfiable → 416;
  no Range header → 200 full body.
- `GET /api/frames/{id}/{name}` → the PNG from
  `<frames_dir>/default/<id>/<name>`. Path safety: `name` must match
  `^[A-Za-z0-9._-]+$` and must not contain `..`; `id` goes through the same validation
  as filter values (`_safe`-equivalent: length + control-char checks). 404 when absent.
- Store gain: `Store.memory_rows(memory_id: str, project_id: str | None = None) ->
  list[dict]` — all rows for a memory (used by both new GET endpoints; keeps LanceDB
  querying inside the store class). Aggregation happens in the service.

### Data flow

`/api/memories` (once at boot) → `/api/memories/{id}/segments` (per selection) → static
rendering; `timeupdate` drives all sync client-side. Ask/search POSTs go to the existing
endpoints; citations map to seeks via `(memory_title, start_ts)`.

## Error handling

- Missing video file: player pane replaced by a message naming the expected path
  ("ingested from a different directory?"); transcript, timeline, ask/search stay fully
  functional. The `<video>` error event triggers the fallback (also covers codec
  failures, e.g. mkv in Safari — message mentions Chrome for mkv).
- Empty store / no memories: friendly empty state pointing at `vproc ingest`.
- Ask/search failures (endpoint down, 4xx): inline error text in the panel, no crash.
- `/api` endpoints never 500 on missing data: 404s with JSON `{detail}` bodies.

## Testing

- Endpoint tests (fake/real temp store, following existing `test_service.py` seams):
  memories aggregation (count/duration/speakers), segment ordering + `frame_name`
  mapping, media Range semantics (200 no-header, 206 with correct `Content-Range`
  and byte slice, 416 out-of-range, 404 missing file), frames path-traversal rejection
  (`..`, slashes, absolute paths → 404/400, never a file outside the frames dir).
- Static smoke: `GET /` serves `index.html` (200, `text/html`); API routes still win
  over the mount (`/healthz`, `/ask` unaffected).
- Frontend logic is deliberately thin; its correctness is verified end-to-end: drive the
  real UI in Chrome against the ingested `tmp/test.mkv` (picker, seek from timeline,
  seek from transcript, auto-scroll sync, ask with citation deep-link, search) as the
  final task, with screenshots.

## Risks

- **mkv in Safari** won't play (documented; Chrome works for h264/aac mkv; mp4 plays
  everywhere). Remux option deferred.
- **Moved/renamed source videos** dangle (stored path is ingest-time). The 404 + UI
  fallback makes it visible, not silent; re-ingest fixes.
- **Large tables**: `/api/memories` scans all rows; fine at personal scale (hundreds of
  segments per meeting), revisit if libraries grow to thousands of memories.
