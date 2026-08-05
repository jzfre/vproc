# vproc

Grounded meeting-memory from video. `vproc` ingests a recorded meeting (audio
transcript + diarized speakers + on-screen text OCR'd from frames), indexes it
in an embedded LanceDB store, and answers questions about it with citations
back to the exact speaker/timestamp/frame — abstaining rather than guessing
when the transcript doesn't support an answer. A small local web UI plays the
video alongside a synced, speaker-colored transcript and timeline.

## Quickstart

```sh
uv run vproc ingest <video.mp4>   # transcribe, diarize, OCR, embed, index
uv run vproc serve                # -> http://localhost:8765
```

Then open the UI, or use `vproc ask` / `vproc search` (see `scripts/ask.sh`,
`scripts/search.sh`) to query from the CLI. Configure model endpoints via
`VPROC_*` env vars (see `.env.example`); `.env` is auto-loaded.

## Network posture

`vproc serve` is an **unauthenticated, LAN-trusted service**. Anyone who can
reach the port can list meetings and stream their video, frames, and
transcripts — there is no login and no access control. It defaults to
`VPROC_HOST=0.0.0.0` (bind all interfaces) so it's reachable from other
devices on your network by design. If that's not what you want, set:

```sh
VPROC_HOST=127.0.0.1
```

in `.env` for a private-by-default, localhost-only service.
