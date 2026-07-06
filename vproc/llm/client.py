import base64
import mimetypes
import wave

from openai import OpenAI

_clients: dict[str, OpenAI] = {}


def _client(base_url: str) -> OpenAI:
    # api_key is required by the SDK but unused by local servers. Cache one client per
    # base_url so the httpx connection pool is reused instead of leaking a socket per call.
    inst = _clients.get(base_url)
    if inst is None:
        inst = _clients[base_url] = OpenAI(base_url=base_url, api_key="not-needed")
    return inst


def embed_texts(base_url: str, model: str, texts: list[str]) -> list[list[float]]:
    resp = _client(base_url).embeddings.create(model=model, input=texts)
    # The API pairs each vector to its input via `index`; list order is not contractual.
    data = sorted(resp.data, key=lambda d: d.index)
    return [list(d.embedding) for d in data]


def chat_json(
    base_url: str,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.1,
    schema: dict | None = None,
) -> str:
    # Prefer constrained decoding (json_schema) when a schema is given; fall back to the
    # looser json_object mode otherwise. Newer OpenAI-compatible servers (e.g. current
    # LM Studio) reject json_object and require json_schema or text.
    if schema is not None:
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": schema},
        }
    else:
        response_format = {"type": "json_object"}
    resp = _client(base_url).chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        response_format=response_format,
    )
    msg = resp.choices[0].message
    # Some thinking models (e.g. Qwen3 via LM Studio) emit the constrained JSON into the
    # reasoning channel and leave `content` empty; fall back to reasoning_content.
    return msg.content or getattr(msg, "reasoning_content", None) or ""


def _wav_duration(path: str) -> float:
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def transcribe_audio(base_url: str, model: str, audio_path: str) -> dict:
    """Transcribe via an OpenAI-compatible ASR endpoint (/v1/audio/transcriptions).
    Returns the same {"segments": [{start, end, text}]} shape as the local whisper path."""
    with open(audio_path, "rb") as f:
        # Hour-long meetings exceed the SDK's 600s default; max_retries=0 so a slow response
        # doesn't silently re-upload the ~100MB wav two more times.
        resp = _client(base_url).with_options(timeout=3600, max_retries=0).audio.transcriptions.create(
            model=model, file=f, response_format="verbose_json"
        )
    segs = getattr(resp, "segments", None) or []
    if segs:
        return {"segments": [{"start": float(s.start), "end": float(s.end), "text": s.text} for s in segs]}
    # Backends that downgrade verbose_json return only `text`; synthesize one segment
    # spanning the whole recording rather than dropping the transcript. No text = true silence.
    text = (getattr(resp, "text", None) or "").strip()
    if not text:
        return {"segments": []}
    return {"segments": [{"start": 0.0, "end": _wav_duration(audio_path), "text": text}]}


def ocr_image(base_url: str, model: str, image_path: str, prompt: str, timeout: float = 60.0) -> str:
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    mime = mimetypes.guess_type(image_path)[0] or "image/png"
    # Bounded timeout + no retries: a slow vision backend degrades one frame's OCR (caller
    # falls back to empty on_screen_text) instead of hanging the whole ingest for 600s x N.
    resp = _client(base_url).with_options(timeout=timeout, max_retries=0).chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
        ]}],
        temperature=0.0,
    )
    return resp.choices[0].message.content or ""
