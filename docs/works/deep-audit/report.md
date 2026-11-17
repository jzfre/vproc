# Comprehensive vProc audit — 2026-11-17

## Outcome and evidence standard

This audit covers the application's ingestion, storage, retrieval, answers, clients,
service, UI, local model server, CLI, dependencies and release workflow. It follows
an earlier targeted pass that was insufficient to establish production readiness.
Findings below were reproduced before fixing, or established by a specific runtime
experiment. Existing tests were treated as a baseline, not as evidence of new work.

The result is a materially hardened application and repeatable verification tools.
It is **not a certification of real-model accuracy or public-internet safety**.
This host has no configured `.env`, no `VPROC_*` environment settings, no running
configured model endpoints and no Hugging Face model cache. A real embedding
assessment was attempted and failed at corpus embedding: `APITimeoutError`, zero
completed cases. No simulated answer scores are presented as model-quality results.

## Coverage and fixes

| Area | Confirmed defect or risk | Fix and evidence |
| --- | --- | --- |
| Ingest preservation | Empty diarization output was treated as success and could replace existing named speakers with placeholders. | Empty turns for a nonempty transcript enter the existing preservation/degradation path. Invalid turns are validated before labels mutate. Regression tests cover named-memory preservation and new-memory degradation. |
| Timestamp/citation correctness | Overlapping transcript segments could produce an end time earlier than some included speech; midpoint bucketing could emit chunks out of chronological order. | Use the full interval and sort final chunks by start; overlapping-screen/utterance regressions. |
| Frame alignment | Scientific notation was misread, and mismatched frame-file/metadata counts were silently truncated by `zip`. | Validate numeric times/order and exact counts; fail before replacement on incomplete extraction. Real FFmpeg tests plus a full available recording were processed. |
| Ingest failure privacy | Model exceptions could print response content or credentials in OCR, naming, diarization and fatal transcription paths. | Class-only diagnostics and sanitized `IngestError`; malformed ASR shapes fail cleanly. Regression assertions exclude injected private exception strings. |
| Probe filesystem safety | Diarization speaker labels entered temporary frame filenames. | Use internal sequence numbers; malicious path-like labels cannot choose output paths. Image handles close deterministically. |
| Numerical retrieval | Finite Python embeddings overflowed/underflowed in float32, or overflowed/underflowed the cosine norm; examples included `1e300`, `1e-300`, `1e20`, `1e-30`. | Stable scale-first unit normalization on write and query, preserving cosine direction and ordinary existing indexes. No mutation of caller inputs. Invalid returned distances yield explicit compatibility errors. |
| Concurrent writes/recovery | Correct behavior around actual writer death was insufficiently exercised. | Spawned-process tests kill a writer before commit and after commit before FTS refresh; readers see the appropriate old/new committed state, contenders cannot write, OS locks release, and subsequent writes succeed. Lock-file inode preservation is explicit with the upgraded FileLock. |
| Catalogue scalability | Listing meetings loaded every transcript and embedded-text string. Speaker renaming also loaded rows to count matches. | Project and stream only memory/speaker/end-time batches; database count for rename. A 10,000-row equivalence benchmark measures memory and latency. |
| Answer parsing | A valid JSON schema example or reasoning draft could mask the final answer. | Ignore tagged reasoning and scan complete answer objects without reparsing nested claim JSON; regressions cover schema/draft/final response layouts. Malformed output still abstains. |
| Verification/citations | Duplicate evidence IDs repeated verifier input and emitted duplicate citations. | Filter and deduplicate IDs before verification; premise and emitted citation tests. Whole-transcript analysis now returns a textual fallback for malformed nonstring answers. |
| Network clients | ASR's one-hour read window also allowed a one-hour connect timeout; concurrent first use built duplicate pools. Flat ASR word assignment was quadratic. | Five-second ASR connect window, locked pool initialization, canonical trailing-slash cache keys, deque word consumption. Online embedding doctor probes have a 15-second read window and no retries. |
| Query availability | Slow model work could fill shared server workers; stopping a browser wait does not stop backend work. | Shared REST/MCP admission limit, default four; excess work returns 503 with Retry-After. Async health endpoint; slots remain held until abandoned backend work finishes. Tested both controlled load and real socket disconnect. |
| Request resources | Question length validation occurred only after parsing an arbitrarily large JSON body. | 256 KiB body cap before query/MCP parsing, including chunked requests and mounted apps. Media streaming is unaffected. |
| MCP lifecycle | Stateful initialization retained unnecessary server sessions for tools with no client-local state. | Stateless HTTP plus patched SDK 1.30; repeated initialization retains no sessions, normal tools and filters still work. |
| Frame reads | A symlinked project directory could redefine the permitted root; frame route also exposed sidecars and rejected legitimate names with internal `..`. | Reuse ingest's path policy, reject child/file symlinks, serve PNGs only. Unicode control characters are rejected consistently across ingest/API; ordinary Unicode names remain valid. |
| HTTP/CLI diagnostics | Metadata endpoints lacked the sanitized backend error wrapper; CLI analysis could expose SDK error bodies. Config repr exposed HF tokens/URL credentials. | Sanitized handlers and repr fields; regression tests inject secret strings. Failed speaker-sidecar writes now explain that the authoritative index rename already committed. |
| UI project correctness | Global queries could return another project's identically named meeting, but the player opens only the default project. Search could leave controls locked while waiting indefinitely. | UI requests explicitly select default project; search gains Stop waiting. API/MCP retain explicit cross-project support. Chromium tests cover playback/citations and escaped hostile text. |
| Local CPU model resource use | ASR requests queued in workers, embedding inference ran concurrently, failures leaked text, and cleanup/recovery lacked coverage. | One active request per model with fast 503, 32-item embedding batch cap, sanitized model errors, eager/lazy failure cleanup tests. |
| Dependencies | Base and optional locks contained reported security advisories. | Compatible updates and install minimums; base scan becomes clean. Optional engine constraints retain eight unique advisory IDs, with explicit reachability/mitigation analysis in [dependencies.md](dependencies.md). |

## Measured experiments

Numbers are local observations, not throughput guarantees. Temporary data was
removed after each run; original recordings and operational indexes were not modified.

### Available recording: real media mechanics

A 1,347.67-second, 1920×1080 H.264/AAC recording (84.6 MB) was decoded using the
actual extraction commands. Frame extraction took 26.54 seconds, retained 11
scene-change frames, and produced 3.83 MB of PNGs. Perceptual hashing took 0.27
seconds. Mono 16 kHz audio extraction took 3.12 seconds and produced 43.12 MB.
First/last extracted frame times were 0 and 1344.57 seconds; peak FFmpeg child RSS
was about 260 MiB. This establishes decoding/timing mechanics for this file,
not speech recognition, scene-selection recall, OCR or speaker accuracy.

### Library size: synthetic 10,000 chunks

The committed `scripts/benchmark_store.py` creates 100 memories, 128-dimensional
vectors, and 2,000 characters each of spoken/embedded text in temporary storage.
On Python 3.12.14, Linux x86-64, LanceDB 0.33 and PyArrow 24:

| Operation | Observed result |
| --- | --- |
| Previous catalogue aggregation | 0.187 s, 47.09 MiB peak Python allocations |
| Streaming metadata catalogue | 0.069 s, 0.37 MiB peak Python allocations |
| Filtered cosine query | 0.010 s |
| Initial add plus full-text index | 0.729 s |
| Replace 100 chunks plus full-text refresh | 0.287 s |

The old and new catalogues were asserted equal. Allocation figures use
`tracemalloc`, not total RSS. The remaining full-text refresh rebuilds the index;
very large libraries may need a separate indexing strategy.

### Request overload and disconnects

Forty simultaneous ASGI clients with a controlled 250 ms backend and default
capacity four produced four successful requests and 36 immediate 503 responses.
Maximum measured rejection time was 66 ms; health responded in 20 ms. These are
controlled application-overload measurements, not real-model throughput.

A separate real Uvicorn/raw-TCP MCP experiment disconnected the caller after a
blocking model tool started. Health remained 200, REST search returned 503 while
the abandoned backend retained its slot, and returned 200 after work finished.
No falsely early release or permanent slot leak was observed.

### Browser execution

`scripts/smoke_ui.py` ran in real headless Chromium against bundled assets and a
generated six-second VP8 video. It checked playback, seeking, cross-meeting citation
navigation, default-project ask/search bodies, literal rendering of hostile HTML,
and no JavaScript errors. The screenshot was inspected locally. API responses are
synthetic; this check does not claim generation quality or Safari/Firefox coverage.

## Reproduce and extend

```sh
# Use a separate environment if a service depends on an existing runtime venv.
UV_PROJECT_ENVIRONMENT=/tmp/vproc-audit uv sync --locked --python 3.12
/tmp/vproc-audit/bin/python -m pytest -q
node --test tests/test_ui.cjs
/tmp/vproc-audit/bin/ruff check vproc
/tmp/vproc-audit/bin/python scripts/benchmark_store.py --rows 10000

uv run --no-project --with playwright python -m playwright install --with-deps chromium
uv run --no-project --with playwright python scripts/smoke_ui.py

uv build
uv run --no-project --python 3.12 --refresh-package vproc --with ./dist/vproc-0.1.0-py3-none-any.whl python scripts/smoke_release.py
```

The GitHub workflow runs Python/UI checks, lint, builds, installed-wheel smoke and
real Chromium smoke. Model evaluation is deliberately opt-in: CI has no model
credentials and a mocked model cannot establish quality.

The new [live evaluation guide](../../../evals/README.md) describes the gold
corpus and commands. It runs actual configured embeddings, generation and HHEM,
uses a temporary index, and reports retrieval recall, abstention correctness,
citation targets, explicitly lexical phrase checks and latency. Reports include
configuration/dataset fingerprints but omit meeting content and model responses.
Backend errors and failed expectations return nonzero status. Use a labelled,
consented representative dataset in addition to the supplied synthetic corpus;
exact phrase matching is not a semantic quality judge.

## Remaining operating boundaries and acceptance gaps

- **Live-model acceptance is unavailable on this host.** No successful real ASR,
  OCR, diarization, HHEM or generation evaluation was performed. The attempted
  retrieval-only evaluation completed zero of eight cases because the configured
  embedding server was unavailable. Deployment acceptance still needs those
  real measurements; threshold tuning based on test fixtures would be unsound.
- **Trusted-LAN service.** HTTP/MCP and media have no authentication and bind all
  interfaces by default. Use localhost, firewall/VPN restrictions, or an
  authenticated reverse proxy. The query cap and sanitization are not tenant
  isolation or a reason to expose the service publicly. DNS rebinding protection
  remains disabled for arbitrary LAN hostnames; don't treat browser origin checks
  as access control.
- **Trusted model code.** HHEM enables `trust_remote_code`; arbitrary configured
  repositories are executable code. Optional runtime advisories and migration
  constraints are listed in the dependency audit. Their absence from the base
  install does not make every optional model safe.
- **One configuration per process.** Endpoint credentials remain global to the
  client module. Separate service processes are required for independent credential
  sets; this API is not a multi-tenant application factory.
- **Local cooperating writers.** Process tests do not prove network filesystem,
  arbitrary external-writer, disk-power-loss, or cross-filesystem transaction
  guarantees. Frames and database are separate commits; interrupted staging may
  leave orphans intentionally. Previously corrupted zero/overflow index rows are
  not repaired automatically; re-ingest into a fresh index with working models.
- **Timing/quality limits.** A long ASR segment without word timestamps cannot be
  split accurately without inventing times. OCR samples selected scene changes,
  and perceptual deduplication may discard relevant visual changes. Speaker names
  are probabilistic. `analyze` remains unverified whole-transcript generation and
  may exceed model context on long meetings.
- **Resource limits.** Query admission is per process, not across multiple worker
  processes. Local model upload parsing precedes route admission; it is a trusted
  loopback helper, not a hostile-upload endpoint. Large-library/CPU memory figures
  beyond the measured fixtures, optional model loading, and multi-hour recordings
  require target-host profiling.
- **Platform limits.** Linux/Python 3.12/Chromium were executed here. macOS, Windows,
  Apple Silicon acceleration and other browsers were not executed; their behavior
  must be checked before making platform-specific production guarantees.

## Final local verification

- **762 Python tests passed**, no skips, versus the prior 648-test baseline.
- **18 Node UI tests passed**, versus the prior 15-test baseline.
- Application and new utility Ruff checks and whitespace checks passed.
- Source archive and wheel built. The source archive was inspected for the gold
  corpus, evaluation guide, audit report and runnable benchmark/browser scripts.
- The isolated installed wheel passed startup, bundled UI, MCP setup, synthetic
  FFmpeg/OCR/embedding ingestion, restart persistence, media and search checks.
- Real Chromium smoke passed; its synthetic-backend scope is described above.
- Both committed dependency locks validate; CPU packaging guards pass.
- Independent reviews checked ingestion/answers/storage changes and root service
  changes. The latter caught a mounted-app body-cap bypass and a Unicode path/API
  mismatch during review; both were reproduced, fixed and regression-tested.
- Real model evaluation remains **not passed**: the attempted live retrieval run
  returned a backend timeout and zero completed cases. This is recorded as missing
  deployment evidence, never as a skipped test that makes the quality gate green.

The original uncommitted README, architecture notes and host-migration files were
left outside this change. The GitHub workflow provides the final remote check of
these committed artifacts; its result is reported separately when the push runs.
