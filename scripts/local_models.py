# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fastapi>=0.115",
#   "uvicorn>=0.30",
#   "python-multipart>=0.0.9",
#   "faster-whisper>=1.2,<2",
#   "sentence-transformers>=5,<6",
#   "transformers>=4.51,<5",
#   "torch==2.12.0",
#   "numpy>=1.26",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Optional CPU model server for hosts without a local ASR/embedding backend.

vproc packages no speech or embedding engine for Linux/Windows, and a GPU that is fully used by
an LLM server cannot host them. This one process serves, on CPU, the two OpenAI-compatible
routes vproc needs besides a chat model:

  POST /v1/audio/transcriptions   faster-whisper (verbose_json: segments, optional word times)
  POST /v1/embeddings             sentence-transformers (float or base64)

Run (dependencies are locked next to this file in local_models.py.lock):

  uv run --locked --script scripts/local_models.py

then point vproc at it:

  VPROC_TRANSCRIBE_BASE_URL=http://127.0.0.1:8011/v1   VPROC_TRANSCRIBE_MODEL=large-v3-turbo
  VPROC_EMBED_BASE_URL=http://127.0.0.1:8011/v1        VPROC_EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B

It deliberately has no chat route: vproc reaches the LLM directly (VPROC_GROUNDING_API_KEY_FILE for
a keyed server) and OCR is switched off with an empty VPROC_OCR_BASE_URL. It binds 127.0.0.1 and
has no authentication. Settings: MODELS_HOST, MODELS_PORT, WHISPER_MODEL, WHISPER_THREADS,
WHISPER_HOTWORDS, EMBED_MODEL, EMBED_THREADS.
"""

import base64
import os
import pathlib
import sys
import tempfile
import threading
import time
from typing import Literal

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

HOST = os.environ.get("MODELS_HOST", "127.0.0.1")
PORT = int(os.environ.get("MODELS_PORT", "8011"))
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
WHISPER_THREADS = int(os.environ.get("WHISPER_THREADS", "16"))
# Whisper decodes each 30 s window separately here (condition_on_previous_text=False guards
# against repetition loops), so without a style hint some windows come out lowercase and
# unpunctuated. initial_prompt only reaches the first window; hotwords are applied to all.
WHISPER_HOTWORDS = os.environ.get(
    "WHISPER_HOTWORDS", "Hello, everyone. Let's get started: this is a meeting, with punctuation.")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBED_THREADS = int(os.environ.get("EMBED_THREADS", "8"))
EMBED_MAX_TOKENS = 2048  # vproc chunks are far shorter; bounds memory for any other caller

app = FastAPI(title="vproc local models (CPU)")
_models: dict[str, object] = {}
_load_locks = {"whisper": threading.Lock(), "embed": threading.Lock()}
_whisper_busy = threading.Lock()  # one transcription at a time; each uses most CPU threads


def log(msg: str) -> None:
    print(f"[models] {msg}", flush=True)


def _whisper():
    with _load_locks["whisper"]:
        if "whisper" not in _models:
            from faster_whisper import WhisperModel

            t0 = time.monotonic()
            _models["whisper"] = WhisperModel(
                WHISPER_MODEL, device="cpu", compute_type="int8", cpu_threads=WHISPER_THREADS)
            log(f"loaded whisper {WHISPER_MODEL} in {time.monotonic() - t0:.1f}s")
        return _models["whisper"]


def _embedder():
    with _load_locks["embed"]:
        if "embed" not in _models:
            import torch
            from sentence_transformers import SentenceTransformer

            torch.set_num_threads(EMBED_THREADS)
            t0 = time.monotonic()
            model = SentenceTransformer(EMBED_MODEL, device="cpu")
            model.max_seq_length = min(model.max_seq_length or EMBED_MAX_TOKENS, EMBED_MAX_TOKENS)
            _models["embed"] = model
            log(f"loaded embedder {EMBED_MODEL} in {time.monotonic() - t0:.1f}s")
        return _models["embed"]


def _require_model(requested: str, served: str) -> None:
    # Never substitute silently: an index records its embedding model by name.
    if requested != served:
        raise HTTPException(404, f"model {requested!r} is not served here; use {served!r}")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "loaded": sorted(_models)}


@app.get("/v1/models")
def models() -> dict:
    return {"object": "list", "data": [{"id": i, "object": "model", "owned_by": "local"}
                                       for i in (WHISPER_MODEL, EMBED_MODEL)]}


@app.post("/v1/audio/transcriptions")
def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(...),
    response_format: Literal["json", "text", "verbose_json"] = Form("json"),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    timestamp_granularities: list[str] = Form(default=[], alias="timestamp_granularities[]"),
) -> Response:
    _require_model(model, WHISPER_MODEL)
    whisper = _whisper()
    want_words = "word" in timestamp_granularities
    hotwords = " ".join(part for part in (WHISPER_HOTWORDS, (prompt or "").strip()) if part)
    suffix = pathlib.Path(file.filename or "audio.wav").suffix or ".wav"
    # delete_on_close=False: the decoder reopens the file by name, which Windows only allows
    # then; the file is still removed when the block exits.
    with tempfile.NamedTemporaryFile(suffix=suffix, delete_on_close=False) as tmp:
        while chunk := file.file.read(1 << 20):
            tmp.write(chunk)
        tmp.flush()
        with _whisper_busy:
            t0 = time.monotonic()
            # Same anti-hallucination settings as vproc's local mlx path, plus Silero VAD so
            # silence is skipped instead of decoded (faster on CPU, fewer invented words).
            segments, info = whisper.transcribe(
                tmp.name, language=language or None, hotwords=hotwords or None,
                beam_size=5, vad_filter=True, word_timestamps=want_words,
                condition_on_previous_text=False, compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0, no_speech_threshold=0.6)
            log(f"transcribing {info.duration:.0f}s of audio (language={info.language}, "
                f"p={info.language_probability:.2f}, words={'on' if want_words else 'off'})")
            out = []
            for s in segments:
                seg = {"id": s.id, "seek": s.seek, "start": s.start, "end": s.end, "text": s.text,
                       "tokens": s.tokens, "temperature": s.temperature,
                       "avg_logprob": s.avg_logprob, "compression_ratio": s.compression_ratio,
                       "no_speech_prob": s.no_speech_prob}
                if want_words and s.words:
                    seg["words"] = [{"start": w.start, "end": w.end, "word": w.word} for w in s.words]
                out.append(seg)
                if len(out) % 50 == 0:
                    log(f"  {len(out)} segments, at {s.end:.0f}s / {info.duration:.0f}s")
            took = time.monotonic() - t0
            log(f"transcribed {len(out)} segments in {took:.0f}s "
                f"({info.duration / max(took, 1e-6):.1f}x realtime)")
    text = "".join(s["text"] for s in out).strip()
    if response_format == "text":
        return Response(text, media_type="text/plain")
    if response_format == "verbose_json":
        body = {"task": "transcribe", "language": info.language, "duration": info.duration,
                "text": text, "segments": out}
        if want_words:  # nested per segment (faster-whisper servers) and flat (OpenAI)
            body["words"] = [w for s in out for w in s.get("words", [])]
        return JSONResponse(body)
    return JSONResponse({"text": text})


class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]
    encoding_format: Literal["float", "base64"] | None = "float"


@app.post("/v1/embeddings")
def embeddings(req: EmbeddingRequest) -> dict:
    _require_model(req.model, EMBED_MODEL)
    texts = [req.input] if isinstance(req.input, str) else req.input
    if not texts:
        raise HTTPException(400, "input must not be empty")
    vectors = np.asarray(_embedder().encode(texts, batch_size=32, normalize_embeddings=True,
                                            convert_to_numpy=True), dtype=np.float32)
    data = []
    for i, v in enumerate(vectors):
        # The OpenAI SDK asks for base64 by default and decodes float32 itself.
        emb = (base64.b64encode(v.astype("<f4").tobytes()).decode()
               if req.encoding_format == "base64" else v.tolist())
        data.append({"object": "embedding", "index": i, "embedding": emb})
    return {"object": "list", "data": data, "model": req.model,
            "usage": {"prompt_tokens": 0, "total_tokens": 0}}


def _preload() -> None:
    for load in (_embedder, _whisper):
        try:
            load()
        except Exception as e:  # surfaced again on the first request; keep the server up
            print(f"[models] preload failed: {e}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    threading.Thread(target=_preload, daemon=True).start()
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
