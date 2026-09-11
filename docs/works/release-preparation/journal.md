# Release preparation

## Asked

Prepare vproc for installation on another computer, including prerequisites and
README instructions. Keep delivery local for subsequent real-model testing.

## Decisions and changes

- Recommend the source archive and Python 3.12 with `uv sync --locked`; a wheel
  alternative is documented, with its fresh dependency resolution made explicit.
- Split local model dependencies into answer, diarization, and Apple Silicon
  local-ASR extras; keep runtime as the complete convenience extra. Core server
  installs no longer require PyTorch/pyannote. Preserve the existing LAN default.
- Cap MCP below SDK 2: a clean wheel install selected v2 and lost FastMCP setup.
  The installed-release smoke reproduced this and passed after the constraint.
- Explicitly include release source, lock, example config, docs, tests, scripts,
  and bundled UI; exclude private config, environments, recordings, and indexes.
- Add offline doctor and help/version commands, neutral endpoint examples, model
  service/download requirements, OS limits, and backup/re-ingest instructions.
- Fix FFmpeg filter-path escaping (real failure with colon/backslash/apostrophe
  and filter delimiters) and UTF-8 BOM configuration loading (first key ignored).
  Both regressions failed before their fixes and passed afterward.

## Validated

- Fresh Python 3.12 base + dev environment: 502 Python tests passed, three
  existing dependency deprecation warnings. Node UI suite: 11 passed.
- Production code and smoke script Ruff checks, lock consistency, and whitespace
  checks passed. Source and wheel builds passed; UI/config/lock inclusions and
  absence of private/cache/media files were inspected in the actual artifacts.
- Fresh Apple Silicon source install with all runtime extras succeeded; torch,
  torchaudio, torchcodec, pyannote.audio, transformers, and mlx_whisper imported.
- Installed wheel smoke passed using real local HTTP, FFmpeg, LanceDB, bundled
  assets, ingest, restart persistence, frame/video serving, and search. It used
  synthetic media and loopback model stubs; editable/source imports were rejected.
- Locked Python 3.12 binary dependencies resolved for Windows x64 and Linux x64.
  Native execution there is untested. Intel Mac lacks compatible LanceDB wheels.
- Real ASR/OCR/diarization/answer quality and destination hardware remain manual
  acceptance checks. No models were downloaded and no existing library was used.

Durable installation boundaries are in `docs/brain/architecture.md` and README.
