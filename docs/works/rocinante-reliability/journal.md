# Rocinante run: smarter and more reliable processing — 2026-09-20

## Asked

vproc ran for the first time on a headless Linux server (CPU models, a shared text-only vLLM
behind a bearer key). Analyze the codebase for what that run showed could be smarter or more
reliable, and apply the fixes.

## Approach

Six audit dimensions (ingest, client/config, retrieval/answer, service/UI, Linux deployment,
and an empirical audit of the live index against the recording's on-screen Teams captions),
each followed by a skeptic who tried to refute every finding against source, tests and the
live system. Only confirmed items were implemented, test-first. A second reviewed pass
(four reviewers, each finding again attacked by a skeptic) confirmed 10 defects in the new code
itself; all were fixed test-first. Pytest went 502 -> 612, the UI tests 11 -> 15.

## Decisions

- **API keys per endpoint**, registered per base URL by `client.configure(cfg)` at the three
  entry points; call sites keep passing `(base_url, model)`. A key belongs to a server, so
  endpoints sharing a base URL share it and conflicting keys are rejected. This removed the
  key-adding proxy the first run needed.
- **Speakers are attributed per sentence, using word timestamps.** The audit found the
  diarization clusters were perfect (4 clusters = 4 people) but 25 of 49 chunks held another
  person's words, because a whole ASR segment went to one speaker; one gate-passed "who said
  X" answer was wrong. Pure word-level splitting surfaced diarization noise instead. On
  captured ASR words and turns of the reference meeting, speaker changes in mid-sentence
  were: whole-segment 8, word-level 27, per sentence within a segment 3, per sentence across
  segment boundaries 0 (and 0 of 67 chunks end mid-sentence; the first count is zero by
  construction, the evidence is the caption-checked spots below). An unfinished tail only
  joins a segment that follows within 2 s and closes the sentence, so a fully unpunctuated
  stream keeps the old rule. Accepted cost: a real interruption inside one sentence goes to
  whoever spoke most of it.
- **Rejected: scoring each cited evidence item separately in the HHEM gate.** It gives every
  claim n+1 attempts and would show citations that did not take part in verification. HHEM-2.1
  does not truncate at 512 tokens, so the motivating worry was unfounded.
- **Rejected: reading vLLM's `reasoning` field as an answer fallback.** A generation cut off at
  `max_tokens` leaves only a draft there; it is now logged and treated as no answer.
- **Rejected for now:** a stage cache for ASR/diarization (stale-result risks; the embedding
  preflight removes the main late failure), admission control and in-flight dedup for `/ask`
  (no duplicate asks observed; vLLM already serializes), an HTTP speaker-rename route (first
  mutating route on an unauthenticated service), MCP progress heartbeats (the clients in use
  do not extend timeouts on progress), multi-address `VPROC_HOST`, caption OCR and tile-border
  speaker naming (Teams-specific; no OCR engine on the host), `VPROC_TRANSCRIBE_PROMPT`.
- **Claims are bounded** (`MAX_CLAIMS = 6`, as `maxItems` in the answer schema and a slice in
  `generate()`). Found on the live system by the new truncation warning: with thinking off the
  27B model answered correctly, then kept appending claims about everything in the evidence
  until `max_tokens` (8192 tokens, 259 s), and the truncated JSON became an abstention. With
  the bound the same question answers in about 10 s. Not validated on LM Studio.
- Generation requests are never retried by the SDK: a timed-out generation is usually still
  running, and a retry queues it again behind itself on a single-sequence server.
- OCR is disabled by an explicit empty `VPROC_OCR_BASE_URL`, like the existing transcribe and
  diarize conventions, instead of a stub server.
- The similarity floor stays a code decision but is embedding-model specific. With
  Qwen3-Embedding it never fired at 0.25; a query-only instruction prefix plus 0.35 rejects
  8 of 10 off-topic probes and 0 of 12 answerable ones on the final index. In-domain-sounding
  questions (0.44) overlap broad answerable ones (0.42), so the rest is left to the gate.
- Linux takes PyTorch CPU wheels via `[tool.uv.sources]`. Re-locked with temporary pins, then
  without: the only lock change is 19 CUDA packages out, three `+cpu` forks in; macOS wheels
  are unchanged.
- The CPU model server is a tracked, locked, tested script outside the package; systemd
  templates live in `deploy/systemd/` and are not shipped in the sdist.

## Validated

- `vproc doctor --online` on the live host: all probes pass; with a wrong key and an invalid
  `HF_TOKEN` it fails in seconds with the setting to fix (both failures of the first run).
- Real pyannote run through the in-memory waveform path; real HHEM load with an invalid
  ambient token; real vLLM structured reply through the keyed client (no proxy).
- Known spots checked against the on-screen captions: a sentence during which the diarizer
  flips speakers for 2.8 s, and a turn that word-level labels gave to the wrong person, are
  now attributed correctly, and one-word back-channels stay their own turns. Word
  times also corrected early citation times that were about 30 s off.
- Re-ingest of the 22.5 min recording: live stage and percentage lines, 7m31s. Words in
  unpunctuated chunks 59% -> 0% (hotwords reach every Whisper window; `initial_prompt` only
  the first). A participant name that was misheard is now right 8 of 8 times. Hotwords drop
  fillers, about 9% fewer words.
- The question that returned a wrong, gate-passed attribution in 2m35s now answers correctly
  in about 5 s; off-topic questions abstain at the floor in 0.1 s without an LLM call.
- Not validated here: the Mac. `word_timestamps=True` changes mlx-whisper decoding (a failure
  falls back to segment-level) and diarization now receives a waveform. Do one Mac ingest
  before relying on it.

- Review fixes worth remembering: a sentence must not be carried across a long pause (it made
  a 304 s "sentence"); text rebuilt from word tokens loses punctuation with OpenAI-style word
  lists and adds spaces in CJK, so whole segments keep their ASR text; literal-match detection
  must ignore the `[SCREEN]`/`[SPOKEN]`/`(mm:ss):` scaffolding of indexed text; the UI may only
  claim what it knows (top results, not the library; "still finishing" only after a real wait);
  existence checks must expand `~` like the store does.

## Changed

- Client/config/doctor: endpoint keys, no-retry generation with a 5 s connect timeout,
  truncation warning, OCR-off switch, query prefix, `KNOWN_ENV`, `doctor --online`, config
  source and endpoint lines.
- Ingest: embedding preflight, stage/percentage progress, quiet ffmpeg, word timestamps,
  in-memory diarization, downgrade guard, `DEGRADED` summary, referenced-frames-only,
  `IngestError`.
- Answering/service/UI: abstain reason, `top_sim`, gate scores, per-hit similarity and keyword
  match, verifier preflight and status, backend status in logs, elapsed/stop-waiting, badges.
- Packaging/deploy: CPU torch sources and lock guard test, `scripts/local_models.py` with lock
  and tests, `deploy/systemd/`, `vproc speakers` summary.

## Brain

`docs/brain/architecture.md` updated: client keys and retry policy, ingest order and speaker
attribution, diagnostics, verifier state, Linux packaging, optional model server.
