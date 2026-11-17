# Live answer and retrieval evaluations

`answers.json` is a small, synthetic gold corpus: 15 source segments (including ten
unrelated distractors) and eight questions. Cases exercise release facts, speaker
identity, exact error identifiers, screen-only facts, cross-project collisions,
absent topics, and project/speaker filters. No private recording is included.

Run from an installed source checkout with its real model configuration:

```sh
uv run --no-sync python scripts/evaluate_answers.py --output /tmp/vproc-answer-eval.json
```

The command loads `.env` from the working directory, contacts the configured
embedding and grounding endpoints, and loads the configured HHEM verifier. The
`answer` or `runtime` extra must be installed for the default full evaluation.
Model weights may be downloaded on first use. The CLI has no fake-model mode.
The corpus is embedded into a temporary LanceDB directory that is removed after
use. **The configured meeting index and frames are not read or modified.**

To measure retrieval without a grounding backend or local HHEM:

```sh
uv run --no-sync python scripts/evaluate_answers.py --retrieval-only --k 8 \
  --output /tmp/vproc-retrieval-eval.json
```

This still calls the real embedding endpoint. Unavailable models are failures;
the script never silently substitutes mocks. A backend failure ends the run
promptly and records only its stage and exception type.

## Metrics and exit status

Reports contain case IDs and numerical/boolean results, not questions, source
content, model responses, endpoint URLs, keys, or exception messages. Use opaque
IDs for private datasets. Without `--output`, the JSON report goes to stdout.
Existing application/backend logs still follow their ordinary logging settings.

- `recall_at_k`: fraction of the gold source IDs present in the actual retrieved
  top-k. Cases with no expected source IDs report `null`, not a perfect score.
- `scope_correct`: all returned hits obey the requested project/memory/speaker
  filters.
- `abstention_correct`: the verified answer's abstention matches the gold label.
  Full answer mode only.
- `citation_source_accuracy`: fraction of emitted citations pointing to a gold
  source **and** its correct memory, speaker, start, and end timestamps. No emitted
  citations report `null`; answerable cases require actual correct citations.
- `lexical_phrase_recall`: fraction of case-insensitive expected phrases present
  in the kept claims. This is a lexical check, **not a semantic accuracy score**:
  a correct paraphrase can fail, and matching words alone do not prove correctness.
- Per-case latency includes retrieval and, in full mode, generation/verification.
  Total latency also includes verifier loading, corpus embedding, and index setup.

Summary means omit `null` values and use completed cases only; inspect `completed`
and `error` before comparing runs. Retrieval-only mode leaves all answer metrics
`null`; absent-topic cases without gold sources test filter behavior only and do
not establish that generation would abstain.

Exit status is 0 only when all evaluated cases pass: their recall meets
`min_recall` (default 1.0), scope is correct, and in full mode abstention and
citation expectations are satisfied and all requested phrases match. Exit 1
means a failed expectation or model/index error. Exit 2 means invalid dataset,
configuration, command-line arguments, or report output location.

Record configuration/model versions and compare repeated runs on the same
corpus when changing models, query prefixes, thresholds, or retrieval code.
This small synthetic set is a smoke evaluation, not a representative benchmark
or a claim about private meeting accuracy. Real ASR/OCR/diarization quality needs
separate labeled media; these evaluations begin at text segments.

## Custom gold data

Copy `answers.json` and pass `--dataset /path/to/private-gold.json`. Each source has
an ID, project/memory/speaker, ordered nonnegative finite times, and spoken and/or
screen text. Each case has a unique ID, question, strict boolean `should_abstain`,
expected source IDs, optional expected phrases, and optional project/memory/speaker
filters. Answerable cases require at least one known source; sources outside
filters, duplicate IDs, invalid times, and empty content are rejected before any
model calls. The format limits runs to 200 sources and 100 cases to bound accidental
backend work. The selected dataset's text is sent to the configured endpoints.
Keep private gold datasets and reports outside the repository.

`tests/test_evaluate_answers.py` validates the harness using explicitly injected
model functions, real LanceDB, and the real answer pipeline. Those tests are
application-contract tests and are never reported as live-model quality results.

Reports also record model identifiers, thresholds, a hash of the query prefix and
a hash of the validated dataset. Keep those fields with results when comparing
configurations; they do not identify underlying weights if a server silently
changes the weights behind the same model alias.
