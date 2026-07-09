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


# Embedding servers cap the per-request batch (TEI defaults to 32); send chunks that fit
# any backend rather than one request holding every segment of a long video.
EMBED_BATCH = 32


def embed_texts(base_url: str, model: str, texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        resp = _client(base_url).embeddings.create(model=model, input=texts[i:i + EMBED_BATCH])
        # The API pairs each vector to its input via `index`; list order is not contractual.
        data = sorted(resp.data, key=lambda d: d.index)
        out.extend(list(d.embedding) for d in data)
    return out


def chat_json(
    base_url: str,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.1,
    schema: dict | None = None,
    max_tokens: int = 8192,
    think: bool = True,
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
    # max_tokens bounds a reasoning model's thinking channel: unbounded, some questions
    # send it into a runaway loop that fills the context over many minutes. A truncated
    # runaway yields unparseable output, which the caller treats as an abstention.
    # think=False additionally disables the thinking channel entirely (hybrid models like
    # Qwen3.5); non-thinking backends ignore the template kwarg.
    extra = {} if think else {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    resp = _client(base_url).chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        response_format=response_format,
        max_tokens=max_tokens,
        **extra,
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


def ocr_image(base_url: str, model: str, image_path: str, prompt: str,
              timeout: float = 60.0, max_tokens: int = 1024) -> str:
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    mime = mimetypes.guess_type(image_path)[0] or "image/png"
    # OCR is transcription, not reasoning. Cap the output AND disable the model's thinking
    # channel: a Qwen-style reasoning model otherwise "thinks" about a screenshot until it
    # burns the whole context (65k tokens over many minutes) and returns no text. Bounded
    # timeout + no retries so a slow frame degrades (caller falls back to empty on_screen_text)
    # instead of stalling the ingest. chat_template_kwargs is ignored by non-thinking backends.
    resp = _client(base_url).with_options(timeout=timeout, max_retries=0).chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
        ]}],
        temperature=0.0,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return resp.choices[0].message.content or ""
