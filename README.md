# vproc

Grounded meeting-memory from video. `vproc` ingests a recorded meeting (audio
transcript + diarized speakers + on-screen text OCR'd from frames), indexes it
in an embedded LanceDB store, and answers questions about it with citations
back to the exact speaker/timestamp/frame — abstaining rather than guessing
when the transcript doesn't support an answer. A small local web UI plays the
video alongside a synced, speaker-colored transcript and timeline.

## Before installing on another computer

The release contains the Python application and web UI. **It does not include
FFmpeg, a Python environment, model servers, model weights, or your meeting
library.** There is no separate database server or Node.js runtime to install
for normal use; LanceDB is embedded and the UI is bundled.

Use Python **3.12** for the locked installation below. `uv` can download this
Python version as part of installation. Plan disk space for the environment,
downloaded model weights, original videos, screenshots, and the index. The
runtime includes large PyTorch dependencies; local-model memory requirements
depend on the models you select.

| Computer | Transcription setup | Installation boundary |
| --- | --- | --- |
| Apple Silicon Mac | Local MLX Whisper, or an ASR endpoint | Use a native arm64 Python and macOS 14 or newer. |
| Windows x64 / Linux x64 | Configure an OpenAI-compatible ASR endpoint | vproc does not package a local ASR engine for these platforms. |
| Intel Mac | A remote ASR endpoint would be needed | Unsupported by the current locked distribution: LanceDB and the local-model dependencies lack compatible Intel Mac wheels. |

vproc's packaged `local-asr` extra is restricted to Apple Silicon macOS. The
[MLX installation requirements](https://ml-explore.github.io/mlx/build/html/install.html)
explain the native Python/macOS requirements. Locked Python 3.12 binary-wheel
resolution has been checked for Windows x64 and Linux x64. Native end-to-end
testing on those systems has not been performed for this release.

### Install uv and FFmpeg

Install `uv` using the [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/).
For macOS/Linux:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For Windows PowerShell:

```powershell
winget install --id=astral-sh.uv -e
```

Open a new terminal after installation, then confirm `uv --version` works.

Install **both `ffmpeg` and `ffprobe`**, with their executable directory on
`PATH`. On macOS with Homebrew, use
[`brew install ffmpeg`](https://formulae.brew.sh/formula/ffmpeg). On Debian/Ubuntu:

```sh
sudo apt update
sudo apt install ffmpeg
```

For Windows and other Linux distributions, use the executable/package links
on the [official FFmpeg download page](https://ffmpeg.org/download.html).
On Windows, extract a **shared** build and add its `bin` directory to `PATH`.
Diarization uses TorchCodec, which also needs FFmpeg's shared libraries; a
standalone static `ffmpeg.exe` can pass the command check while diarization
still fails. See [TorchCodec installation](https://github.com/meta-pytorch/torchcodec#installing-torchcodec).

Verify in the terminal that will run vproc:

```sh
ffmpeg -version
ffprobe -version
```

### Install from the source release (recommended)

Copy the release's `vproc-0.1.0.tar.gz` to the new computer and extract it, or
copy a source checkout. Open a terminal **inside the extracted project
directory**, where `pyproject.toml`, `uv.lock`, and `.env.example` are present.
Keep this directory in a stable location. Do not copy the old `.venv`; install
the dependencies for the destination computer:

```sh
uv sync --locked --no-dev --extra runtime --python 3.12
```

This creates `.venv` and uses the versions in the supplied `uv.lock`. An
internet connection is needed for uncached Python/packages. Copy the example
configuration using the command for your shell:

```sh
# macOS/Linux
cp .env.example .env
```

```powershell
# Windows PowerShell
Copy-Item .env.example .env
```

Edit `.env` using the configuration instructions below before ingestion. Save
it as UTF-8; a Windows UTF-8 byte-order mark is supported.
The subsequent `uv run --no-sync` commands use the environment just installed,
without dropping its optional dependencies.

For a smaller installation, choose the features you need:

| Extra | Adds |
| --- | --- |
| No extra | Web UI, HTTP/MCP service, embedded index, remote ASR, OCR, embeddings, and search clients |
| `answer` | PyTorch and Transformers 4.x for local HHEM answer verification |
| `diarization` | pyannote speaker diarization |
| `local-asr` | MLX Whisper on Apple Silicon macOS |
| `runtime` | All the above extras, with MLX installed only on Apple Silicon macOS |

For example, remote ASR plus verified answers without diarization:

```sh
uv sync --locked --no-dev --extra answer --python 3.12
```

Set `VPROC_DIARIZE_MODEL=` in `.env` when omitting diarization. For a base-only
install, omit `--extra runtime` entirely. Browsing does not need model servers;
search needs the configured embedding server, and answering also needs the
grounding server and `answer` extra.

### Configure model backends

**The localhost addresses and `replace-with-your-...` model names in
`.env.example` are placeholders.** Replace them with running endpoints and
the exact model IDs those endpoints serve. A backend can run on this computer
or another host; `localhost` always refers to the computer running vproc.

| Settings | Backend capability | Used for |
| --- | --- | --- |
| `VPROC_OCR_BASE_URL`, `VPROC_OCR_MODEL` | OpenAI-compatible chat completions with image input | On-screen text and speaker nameplates |
| `VPROC_EMBED_BASE_URL`, `VPROC_EMBED_MODEL` | OpenAI-compatible embeddings | Ingest, search, and retrieval for answers |
| `VPROC_GROUNDING_BASE_URL`, `VPROC_GROUNDING_MODEL` | OpenAI-compatible chat completions with structured JSON output | Answers grounded in retrieved evidence |
| `VPROC_TRANSCRIBE_BASE_URL`, `VPROC_TRANSCRIBE_MODEL` | OpenAI-compatible audio transcriptions | Required ASR mode on Windows and Linux; optional on Apple Silicon |

Use base URLs such as `http://localhost:8000/v1`, not the full
`/chat/completions` or `/audio/transcriptions` route. The client currently
sends the fixed API key `not-needed`; it expects self-hosted endpoints or a
proxy that accepts this value. There is no configurable provider API-key
setting. Meeting audio, images, or text are sent to the endpoints configured
for each operation.

For endpoint transcription, set these values to your ASR server and model:

```dotenv
VPROC_TRANSCRIBE_BASE_URL=http://localhost:8001/v1
VPROC_TRANSCRIBE_MODEL=replace-with-your-asr-model
```

The ASR server should support `response_format=verbose_json` with timestamped
segments. Text-only responses become one segment spanning the recording,
which loses precise transcript timing. On Apple Silicon, leave the ASR base
URL empty to use local MLX and retain the MLX model ID from `.env.example`.
First use downloads that model from Hugging Face, as described by
[MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper).

Diarization is enabled in the example configuration. To use the default
`pyannote/speaker-diarization-community-1` model:

1. Sign in to Hugging Face and accept the conditions on the
   [model page](https://huggingface.co/pyannote/speaker-diarization-community-1).
2. Create a [Hugging Face access token](https://huggingface.co/settings/tokens)
   with permission to read the model, then add `HF_TOKEN=your-token` to your
   private `.env` or process environment.
3. Keep the `diarization` or `runtime` extra installed. First use downloads
   the model; access approval and a valid token are separate from installing
   the Python package.

To disable diarization and automatic speaker naming, set
`VPROC_DIARIZE_MODEL=`. OCR, diarization, and speaker naming are best effort;
failures produce warnings and can reduce the resulting evidence.

Answer verification runs locally using
`vectara/hallucination_evaluation_model`. HHEM loads only when an answer first
needs checking and may download weights and model code from Hugging Face.
The implementation enables `trust_remote_code=True`; use a trusted model
repository if changing `VPROC_HHEM_MODEL`. Installing `runtime` does not
pre-download any model weights or create model servers.

### Check setup and run

From the same directory containing `.env`:

```sh
uv run --no-sync vproc --version
uv run --no-sync vproc --help
uv run --no-sync vproc doctor
uv run --no-sync vproc ingest "meeting.mp4"
uv run --no-sync vproc serve
```

Replace `meeting.mp4` with a real recording path; quote paths containing
spaces. `doctor` checks local prerequisites without contacting model servers,
loading models, or downloading weights. Missing required local prerequisites
return a nonzero exit status. A successful check does not establish endpoint
reachability, model access, or real-model quality. A base-only install can
still browse/search even when `doctor` reports missing answer dependencies.

Open [the UI](http://localhost:8765), play the ingested recording, inspect its
transcript, and ask a question whose answer you know. This is the final check
that your actual media and model backends work together. Stop the service
with Ctrl+C. It runs in the foreground; no background service is installed.

With the service running, visit
[`/healthz`](http://localhost:8765/healthz) and
[`/api/status`](http://localhost:8765/api/status) in a browser. On Windows,
the same checks work in PowerShell:

```powershell
Invoke-RestMethod http://localhost:8765/healthz
Invoke-RestMethod http://localhost:8765/api/status
```

The CLI commands are `ingest`, `serve`, `speakers`, and `doctor`; ask/search
are HTTP endpoints exposed through the UI. The source release also includes
Bash helpers for macOS/Linux:

```sh
scripts/ask.sh "What did we decide about the release?"
scripts/search.sh "release date"
```

### Wheel installation alternative

The built `vproc-0.1.0-py3-none-any.whl` includes the application and UI. Copy
it and `.env.example` to a stable working directory on the destination PC,
then run:

```sh
uv venv --python 3.12
uv pip install "./vproc-0.1.0-py3-none-any.whl[runtime]"
```

Copy and configure `.env` as above. On macOS/Linux run `.venv/bin/vproc doctor`
and `.venv/bin/vproc serve`; in Windows PowerShell use
`.\.venv\Scripts\vproc.exe doctor` and `.\.venv\Scripts\vproc.exe serve`.
Use the same executable for `ingest`. This wheel installation resolves
dependencies from package metadata; it does **not** use `uv.lock`. Use the
source release when you want the locked dependency set. The wheel does not
install the Bash helper scripts.

## Configuration, storage, and moving a library

`.env` is read from the **current working directory**, not from the installed
package directory. Existing environment variables take precedence over `.env`.
Keep launching from the same directory, or use absolute storage paths and
ensure the intended configuration is present in the new working directory.
Restart the service after changing configuration.

The default persistent directories are `./vproc.lance` (index and transcripts)
and `./vproc_frames` (screenshots and speaker-map sidecars). Source videos stay
where you ingested them; vproc does not copy them into the library. The Python
environment and downloaded model caches are separate from these directories.

To back up an existing library, stop the service and all ingest/speaker writers,
then copy the **entire** configured index and frame directories, original
videos, and your private configuration. Keep secrets and meeting data out of
release archives. Do not copy files while an ingest is replacing the index.

New records store **absolute video and screenshot paths**. Copying the index
to another PC or changing `VPROC_FRAMES_DIR` does not rewrite those records.
Restoring the exact paths may work on a matching filesystem layout; for a new
layout or operating system, keep the backup intact, set fresh index/frame
paths as shown below, and re-ingest the original videos from their new stable
location. Retain the same embedding model and weights when continuing an
existing index. A different embedding model requires a fresh index. Preserve
speaker-map backups and reapply any manual corrections after re-ingestion.

## Ingest and playback behavior

- A memory is identified by its video filename stem within a project. Ingesting
  another `standup.mp4` in the same project replaces that memory, even if the
  source directory differs. Use distinct filenames for distinct meetings.
- Re-ingest writes new screenshot filenames before replacing index rows. Failed
  embedding or index writes preserve the previous screenshots; a zero-segment
  extraction leaves the previous memory intact and emits a CLI warning.
- Ingest and speaker edits acquire process locks on the index and frame storage.
  A competing writer exits promptly with a retry message; reads continue. Locks
  release when a process exits, including after a crash. Leave lock files in place.
- Each meeting's rows are replaced in one LanceDB commit. Screenshots are staged
  first, then only previously indexed screenshots are removed after success.
  Files and rows are not one transaction: interruptions can leave unused frames.
  Unreferenced files are preserved because another index may still need them.
- New ingests store absolute video and frame paths. Keep the original video in
  place for playback. Older records with relative paths may need re-ingesting
  if the service is started from another directory.
- The UI lists the default project; ask/search query across the library. REST
  and MCP requests may narrow results with `project`, `memory`, and `speaker`
  filters. Queries are limited to 8,192 characters; filters to 128 characters
  without control characters.
- Citation times and speaker labels come from indexed evidence. Transcription,
  OCR, speaker attribution, and faithfulness scoring can still be wrong; verify
  consequential answers against the recording.

## Embedding compatibility and recovery

New indexes record the configured embedding model name and vector dimension.
Changing either is rejected before incompatible data is searched or appended.
This checks configured identity; it cannot detect changed weights behind the
same server model alias.

Older indexes without that metadata remain browseable, but **ask/search and
further ingest require a fresh index**. The old model cannot safely be inferred
from vector dimensions. Set new paths in `.env`, keep the intended embedding
model configured, and re-ingest the source videos:

```sh
VPROC_INDEX_PATH=./vproc-v2.lance
VPROC_FRAMES_DIR=./vproc_frames-v2
```

Restart the service after changing configuration. This leaves the previous
index and frames intact. Use separate frame directories for separate libraries
so their speaker-map sidecars remain independent.

`GET /api/status` reports MCP availability and local index compatibility without
calling model endpoints; `/healthz` reports process liveness. Neither proves a
model backend is ready. Ask/search return HTTP 409 for incompatible indexes,
503 for unavailable backends or a busy writer, and 504 for model timeouts. MCP
returns corresponding tool errors. FTS and MCP setup failures also emit warnings.

Invalid ports, thresholds, timeouts, token limits, endpoint URLs, and misspelled
boolean settings fail at startup. Supported booleans are `on/off`, `true/false`,
`yes/no`, and `1/0`. The CLI prints configuration and index errors and exits
nonzero. A speaker rename matching no rows leaves its saved mapping unchanged.
The ask/search shell helpers print server error guidance and return nonzero on
HTTP or connection failures.

## Development, testing, and building a release

Use the source checkout or source release. Node.js is needed only for the UI
tests below. Install the development dependencies first:

```sh
uv sync --locked --python 3.12
uv run --no-sync pytest -q
node --test tests/test_ui.cjs
uv run --no-sync ruff check vproc
uv build
```

`uv build` writes the source archive and wheel to `dist/`. The source archive
includes `uv.lock`, `.env.example`, README, helper scripts, and tests. Transfer
the source archive for the recommended installation above; it excludes your
private `.env`, environment, meeting media, frames, and index. Build artifacts
are local files; this command does not publish a release.

Verify the installed wheel independently of the checkout:

```sh
uv run --no-project --python 3.12 --refresh-package vproc --with ./dist/vproc-0.1.0-py3-none-any.whl python scripts/smoke_release.py
```

This creates an isolated install, rejects source-code imports, starts the real
service, and checks bundled UI assets, MCP setup, synthetic FFmpeg ingestion,
index persistence after restart, media playback endpoints, and search. Model
responses come from a temporary loopback server; no model weights or meeting
data are used. The default test requires FFmpeg; append `--http-only` to check
startup/UI without media ingestion.

Python tests use temporary stores and fake model responses. The synthetic video
integration test exercises real ffmpeg/ffprobe and LanceDB when ffmpeg is
installed. UI state regressions use Node's built-in test runner; no npm install
is required. These checks do not establish real-model transcription or answer
quality. See [the project notes](docs/brain/architecture.md),
[the improvement journal](docs/works/reliability-pass/journal.md), and
[the hardening journal](docs/works/hardening/journal.md) for boundaries and the
next manual test.

## Network posture

`vproc serve` is an **unauthenticated, LAN-trusted service**. Anyone who can
reach the port can list meetings and stream their video, frames, and
transcripts — there is no login and no access control. It defaults to
`VPROC_HOST=0.0.0.0` (bind all interfaces) so it's reachable from other
devices on your network by design. If that's not what you want, set:

```sh
VPROC_HOST=127.0.0.1
```

in `.env` for a localhost-only service. For LAN access, keep `0.0.0.0` and
restrict incoming access to the configured port with the host firewall.
Do not expose this unauthenticated service directly to the internet.
