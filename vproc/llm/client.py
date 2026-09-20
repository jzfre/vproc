import base64
import logging
import math
import mimetypes
import wave

from openai import BadRequestError, OpenAI, Timeout, UnprocessableEntityError

logger = logging.getLogger(__name__)
_clients: dict[tuple[str, str], OpenAI] = {}
_api_keys: dict[str, str] = {}


def configure(cfg) -> None:
    """Register the configured bearer keys by base URL. Entry points call this once with
    their Config; call sites keep passing plain (base_url, model) pairs."""
    key_for = getattr(cfg, "api_key_for", None)
    if key_for is None:
        return  # direct stand-in configs (tests, embedders) inject their own clients
    for ep in (cfg.ocr, cfg.embed, cfg.grounding, cfg.transcribe):
        key = key_for(ep.base_url)
        if key is None:
            _api_keys.pop(ep.base_url.rstrip("/"), None)
        else:
            _api_keys[ep.base_url.rstrip("/")] = key


def _client(base_url: str) -> OpenAI:
    # The SDK requires an api_key; keyless local servers ignore the placeholder. Cache one
    # client per (base_url, key) so the httpx connection pool is reused instead of leaking
    # a socket per call, and a client built before configure() is never reused with a stale key.
    key = _api_keys.get(base_url.rstrip("/"), "not-needed")
    inst = _clients.get((base_url, key))
    if inst is None:
        inst = _clients[(base_url, key)] = OpenAI(base_url=base_url, api_key=key)
    return inst


# Embedding servers cap the per-request batch (TEI defaults to 32); send chunks that fit
# any backend rather than one request holding every segment of a long video.
EMBED_BATCH = 32


def embed_texts(base_url: str, model: str, texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    dimensions = None
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        resp = _client(base_url).embeddings.create(model=model, input=batch)
        # The API pairs each vector to its input via `index`; list order is not contractual.
        # Reject incomplete/duplicate batches before callers can pair them with segments.
        if (not isinstance(resp.data, list) or len(resp.data) != len(batch)
                or any(type(d.index) is not int for d in resp.data)):
            raise ValueError("invalid embedding response indices")
        data = sorted(resp.data, key=lambda d: d.index)
        if [d.index for d in data] != list(range(len(batch))):
            raise ValueError("invalid embedding response indices")
        for d in data:
            vector = d.embedding
            if (not isinstance(vector, list) or not vector
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in vector)):
                raise ValueError("invalid embedding response vector")
            if dimensions is None:
                dimensions = len(vector)
            if len(vector) != dimensions:
                raise ValueError("inconsistent embedding response vector dimensions")
            out.append(list(vector))
    return out


CHAT_TIMEOUT = 600.0  # read window for slow reasoning models (the SDK default, made explicit)
CONNECT_TIMEOUT = 5.0


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
    # No SDK retries: a generation that timed out or 5xx'd is usually still running, and a
    # resubmission queues the same minutes-long request behind itself on a shared or
    # single-sequence server. A bare float would also stretch the connect timeout, so an
    # unreachable host keeps failing in seconds.
    resp = _client(base_url).with_options(
        timeout=Timeout(CHAT_TIMEOUT, connect=CONNECT_TIMEOUT), max_retries=0,
    ).chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        response_format=response_format,
        max_tokens=max_tokens,
        **extra,
    )
    choice = resp.choices[0]
    if getattr(choice, "finish_reason", None) == "length":
        # Cut off at max_tokens (usually a runaway thinking channel). Whatever sits in the
        # reasoning channel is a draft, never the answer; callers treat "" as an abstention.
        # Counts only: generated content must not reach logs.
        logger.warning("Generation by %s hit max_tokens=%s (completion tokens: %s); "
                       "treated as no answer", model, max_tokens,
                       getattr(getattr(resp, "usage", None), "completion_tokens", "unknown"))
        return ""
    msg = choice.message
    # Some thinking models (e.g. Qwen3 via LM Studio) emit the constrained JSON into the
    # reasoning channel and leave `content` empty; fall back to reasoning_content.
    return msg.content or getattr(msg, "reasoning_content", None) or ""


def _wav_duration(path: str) -> float:
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def _word(w) -> dict:
    get = w.get if isinstance(w, dict) else lambda key: getattr(w, key, None)
    return {"start": float(get("start")), "end": float(get("end")), "word": str(get("word") or "")}


def _attach_words(out: list[dict], segs, flat) -> None:
    """Add a "words" list to each segment dict that has them. faster-whisper style servers nest
    words in their segment; OpenAI-style servers return one flat list, filed here under the
    segment whose text it spells, in order. Anything unexpected just leaves words off."""
    from vproc.ingest.transcribe import squash

    try:
        nested = [getattr(s, "words", None) for s in segs]
        if any(nested):
            for seg, words in zip(out, nested):
                if words:
                    seg["words"] = [_word(w) for w in words]
            return
        pending = [_word(w) for w in flat or []]
        for seg in out:
            target, spelled, taken = squash(seg["text"]), "", []
            while pending and len(spelled) < len(target):
                taken.append(pending.pop(0))
                spelled += squash(taken[-1]["word"])
            if spelled != target:
                return  # misaligned from here on; earlier segments keep their words
            if taken:
                seg["words"] = taken
    except (TypeError, ValueError):
        return


def transcribe_audio(base_url: str, model: str, audio_path: str) -> dict:
    """Transcribe via an OpenAI-compatible ASR endpoint (/v1/audio/transcriptions).
    Returns the same {"segments": [{start, end, text}]} shape as the local whisper path."""
    def create(**extra):
        with open(audio_path, "rb") as f:
            # Hour-long meetings exceed the SDK's 600s default; max_retries=0 so a slow response
            # doesn't silently re-upload the ~100MB wav two more times.
            return _client(base_url).with_options(timeout=3600, max_retries=0).audio.transcriptions.create(
                model=model, file=f, response_format="verbose_json", **extra)

    try:
        # Word times let diarization attribute speakers per sentence instead of per segment.
        resp = create(timestamp_granularities=["word", "segment"])
    except (BadRequestError, UnprocessableEntityError):
        resp = create()  # strict server without word timestamps: whole-segment attribution
    segs = getattr(resp, "segments", None) or []
    if segs:
        out = [{"start": float(s.start), "end": float(s.end), "text": s.text} for s in segs]
        _attach_words(out, segs, getattr(resp, "words", None))
        return {"segments": out}
    # Backends that downgrade verbose_json return only `text`; synthesize one segment
    # spanning the whole recording rather than dropping the transcript. No text = true silence.
    text = (getattr(resp, "text", None) or "").strip()
    if not text:
        return {"segments": []}
    logger.warning("ASR backend returned no segment timestamps: the whole transcript becomes ONE "
                   "chunk with one speaker. Use a server that supports response_format=verbose_json.")
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
    resp = _client(base_url).with_options(
        timeout=Timeout(timeout, connect=CONNECT_TIMEOUT), max_retries=0,
    ).chat.completions.create(
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
