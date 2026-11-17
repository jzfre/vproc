# Reliability review — 2026-11-17

## Scope and findings

Reviewed ingestion and alignment, store replacement and locking, hybrid retrieval,
answer parsing and verification, HTTP/MCP integration, packaging, and the existing
regression coverage. The baseline had 610 passing Python tests and 15 passing UI
tests; two Python checks initially skipped because FFmpeg was absent on this host.

Two correctness gaps were reproduced with failing tests and fixed:

- ASR segment and word timestamps accepted NaN, infinity, negative values, and
  reversed intervals. Invalid word times could corrupt sentence-level speaker
  attribution; invalid segment times could reach chunking and timestamp rendering.
  Local and endpoint ASR now share timestamp validation. Invalid nonempty segment
  times raise `IngestError` before replacement, preserving previous rows and frames.
  Invalid optional word timing falls back to segment-level attribution. Word times
  must lie within their segment and progress chronologically; overlapping words
  remain supported if their starts and ends progress. Zero-duration intervals and
  numeric timestamp strings remain accepted. Segments are stably sorted by start.
- The literal-match relevance gate was computed before reciprocal-rank fusion
  truncated the evidence. A keyword match dropped by fusion could allow generation
  on unrelated retained evidence. The gate now uses only retained hits. Retained
  literal matches can also qualify when optional FTS is unavailable. Search ranking,
  configured thresholds, and citation construction are unchanged.

## Repeatable checks

The new `.github/workflows/checks.yml` runs locked base dependencies on Python 3.12,
Python tests (including evaluations), Node UI tests, application lint, release
builds, and the installed-wheel smoke test on Linux. FFmpeg and ffprobe are installed
for synthetic-media coverage. Optional local models and private recordings are not
required. No secrets or model service credentials are used by these checks.

Run the focused synthetic evaluations with:

```sh
uv run --no-sync pytest -v tests/test_answer_evals.py
```

These ten cases exercise real LanceDB retrieval, answer parsing, the verification
gate, and citations together. They cover spoken facts, OCR facts, speaker identity,
contradictions, mixed supported/unsupported claims, invented/missing citations,
off-topic questions, model abstention, and project isolation. All data is synthetic.
Embeddings, generated responses, and verifier scores are controlled fixtures: this
suite measures application behavior, **not real-model accuracy**.

For the complete release check, use the commands in the README's development
section. If an existing virtual environment belongs to another machine, create a
separate test environment without removing an operational environment's extras:

```sh
UV_PROJECT_ENVIRONMENT=/tmp/vproc-checks uv sync --locked --python 3.12
/tmp/vproc-checks/bin/python -m pytest -q
```

## Deployment limits and real-model acceptance

This review does not certify production readiness for every deployment. The service
remains unauthenticated and intended for a trusted LAN, as documented in the README.
Windows/macOS behavior and real ASR, diarization, OCR, generation, and HHEM quality
require validation on the target host and configured models.

Before changing model IDs or thresholds, use a consented representative recording
with manually checked questions and citations. Include spoken and screen-only facts,
speaker identity, a contradiction, an absent topic, and an exact identifier. Record
retrieval recall, supported-answer accuracy, false answers on absent topics,
citation correctness, abstention rate, latency, and degraded-ingest warnings.
Compare against the current configuration on the same recording. Do not commit
private recordings, transcripts, answers, or credentials as evaluation fixtures.

## Verification results

- Full Python suite after the fixes and initial evaluations: **642 passed**, no
  skips once FFmpeg was installed. A subsequent FTS-unavailable regression also
  passed with the focused retrieval suite (**17 passed**).
- UI state regressions: **15 passed**.
- Ruff application checks and whitespace validation passed.
- Source archive and wheel built successfully. The isolated installed wheel passed
  startup/UI/MCP setup and synthetic FFmpeg ingestion, restart persistence,
  media playback endpoints, and search against temporary loopback model responses.
- Hosted CI is added but its first GitHub run has not been observed locally.

## Follow-up: zero embeddings

A real LanceDB reproduction showed that an all-zero embedding was accepted on
write but omitted from cosine search; a zero query also returned no results.
Cosine similarity is undefined for zero vectors, so this must be an error rather
than a successful ingest or an apparent lack of meeting evidence. The embedding
client now rejects zero backend output, and the store rejects zero writes and
queries. Five regression cases failed before the fix and pass afterward,
including checks that rejected additions and replacements preserve both rows
and the existing table version. No existing index is rewritten by this fix;
recordings previously ingested with zero embeddings need re-ingestion with a
working backend.

Final follow-up verification: **648 Python tests passed**, no skips; application
lint and whitespace checks passed. Source archive and wheel rebuilt successfully;
the rebuilt installed wheel again passed the full synthetic-media smoke test.
A separate code review of the preceding timestamp/retrieval commit found no
regressions and passed its 177 focused tests. Publishing remains incomplete:
the remote is readable anonymously, but this host has no authenticated GitHub
CLI session or SSH key, and HTTPS push fails for missing credentials.
