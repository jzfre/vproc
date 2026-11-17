import wave
from types import SimpleNamespace

import pytest

import vproc.llm.client as c

class _Resp:
    def __init__(self, payload): self._p = payload

class _FakeEmbeddings:
    def __init__(self, order=None): self.order = order; self.batch_sizes = []
    def create(self, model, input):
        self.batch_sizes.append(len(input))
        class D:  # noqa: N801
            def __init__(self, e, i): self.embedding = e; self.index = i
        order = self.order if self.order is not None else list(range(len(input)))
        return type("R", (), {"data": [D([float(len(input[i]))], i) for i in order]})()

class _FakeChat:
    def __init__(self, capture): self.capture = capture
    @property
    def completions(self):
        outer = self
        class C:
            def create(self, **kw):
                outer.capture.update(kw)
                msg = type("M", (), {"content": "OK"})()
                return type("R", (), {"choices": [type("Ch", (), {"message": msg})()]})()
        return C()

class _FakeClient:
    def __init__(self, capture, embed_order=None):
        self.embeddings = _FakeEmbeddings(embed_order)
        self.chat = _FakeChat(capture)
    def with_options(self, **kw): return self

def test_embed_texts(monkeypatch):
    monkeypatch.setattr(c, "_client", lambda base_url: _FakeClient({}))
    out = c.embed_texts("u", "m", ["a", "bb"])
    assert out == [[1.0], [2.0]]

def test_embed_texts_sorts_by_index(monkeypatch):
    # Server returns data ordered by completion (scrambled); pairing must follow `index`.
    monkeypatch.setattr(c, "_client", lambda base_url: _FakeClient({}, embed_order=[1, 0]))
    out = c.embed_texts("u", "m", ["a", "bb"])
    assert out == [[1.0], [2.0]]

def test_embed_texts_chunks_large_batches(monkeypatch):
    # Embedding servers cap the per-request batch (TEI: 32); a long video's segments must
    # be sent in chunks, with results concatenated in input order.
    fake = _FakeClient({})
    monkeypatch.setattr(c, "_client", lambda base_url: fake)
    texts = ["x" * (i + 1) for i in range(70)]
    out = c.embed_texts("u", "m", texts)
    assert len(out) == 70
    assert out == [[float(i + 1)] for i in range(70)]  # order preserved across chunks
    assert fake.embeddings.batch_sizes == [32, 32, 6]  # no request exceeds the cap


@pytest.mark.parametrize("indices", [[], [0], [0, 0], [1, 2], [-1, 0]])
def test_embed_texts_rejects_missing_or_mispaired_vectors(monkeypatch, indices):
    response = SimpleNamespace(data=[
        SimpleNamespace(index=i, embedding=[1.0, 2.0]) for i in indices
    ])
    fake = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kw: response))
    monkeypatch.setattr(c, "_client", lambda base_url: fake)
    with pytest.raises(ValueError, match="embedding.*indices"):
        c.embed_texts("u", "m", ["first segment", "second segment"])


@pytest.mark.parametrize("vectors", [
    [[], []], [[1.0], [1.0, 2.0]], [[float("nan")], [1.0]],
    [[float("inf")], [1.0]], [["1.0"], [1.0]],
])
def test_embed_texts_rejects_invalid_vectors(monkeypatch, vectors):
    response = SimpleNamespace(data=[
        SimpleNamespace(index=i, embedding=vector) for i, vector in enumerate(vectors)
    ])
    fake = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kw: response))
    monkeypatch.setattr(c, "_client", lambda base_url: fake)
    with pytest.raises(ValueError, match="embedding.*vector"):
        c.embed_texts("u", "m", ["first segment", "second segment"])


def test_embed_texts_rejects_dimension_changes_between_batches(monkeypatch):
    def create(**kwargs):
        texts = kwargs["input"]
        return SimpleNamespace(data=[
            SimpleNamespace(index=i, embedding=[1.0] * (2 if text == "last" else 1))
            for i, text in enumerate(texts)
        ])
    fake = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    monkeypatch.setattr(c, "_client", lambda base_url: fake)
    with pytest.raises(ValueError, match="embedding.*vector"):
        c.embed_texts("u", "m", ["first"] * 32 + ["last"])

def test_ocr_image_builds_data_url(tmp_path, monkeypatch):
    cap = {}
    monkeypatch.setattr(c, "_client", lambda base_url: _FakeClient(cap))
    img = tmp_path / "f.png"; img.write_bytes(b"\x89PNG\r\n\x1a\n")
    out = c.ocr_image("u", "m", str(img), "PROMPT")
    assert out == "OK"
    content = cap["messages"][0]["content"]
    assert content[0]["text"] == "PROMPT"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    # OCR must cap output and disable the reasoning channel (else a thinking model runs away)
    assert cap["max_tokens"] == 1024
    assert cap["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False

def test_ocr_image_fails_fast_when_the_vision_host_is_unreachable(tmp_path, monkeypatch):
    # A bare float would stretch the connect timeout too: a dead host must cost seconds per
    # frame, not the whole per-frame read budget.
    options = {}

    class Client(_FakeClient):
        def with_options(self, **kw):
            options.update(kw)
            return self

    monkeypatch.setattr(c, "_client", lambda base_url: Client({}))
    img = tmp_path / "f.png"; img.write_bytes(b"\x89PNG\r\n\x1a\n")
    c.ocr_image("u", "m", str(img), "P", timeout=45.0)
    assert options["timeout"].read == 45.0 and options["timeout"].connect == 5.0
    assert options["max_retries"] == 0


def test_ocr_image_none_content_returns_empty(tmp_path, monkeypatch):
    class _NoneChat:
        @property
        def completions(self):
            class C:
                def create(self, **kw):
                    msg = type("M", (), {"content": None})()
                    return type("R", (), {"choices": [type("Ch", (), {"message": msg})()]})()
            return C()
    monkeypatch.setattr(c, "_client", lambda base_url: type(
        "X", (), {"chat": _NoneChat(), "with_options": lambda self, **kw: self})())
    img = tmp_path / "f.png"; img.write_bytes(b"\x89PNG\r\n\x1a\n")
    assert c.ocr_image("u", "m", str(img), "P") == ""

def test_chat_json_sets_json_format(monkeypatch):
    cap = {}
    monkeypatch.setattr(c, "_client", lambda base_url: _FakeClient(cap))
    out = c.chat_json("u", "m", "sys", "usr")
    assert out == "OK"
    assert cap["response_format"] == {"type": "json_object"}
    assert "extra_body" not in cap  # thinking stays on by default: no template override sent

def test_chat_json_think_off_disables_reasoning(monkeypatch):
    cap = {}
    monkeypatch.setattr(c, "_client", lambda base_url: _FakeClient(cap))
    c.chat_json("u", "m", "sys", "usr", think=False)
    assert cap["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}

def test_chat_json_never_resubmits_a_generation(monkeypatch):
    # A generation that times out or 5xxs was probably still running: an SDK retry would
    # queue the same long request again behind itself on a single-sequence server.
    options = {}

    class Client(_FakeClient):
        def with_options(self, **kw):
            options.update(kw)
            return self

    monkeypatch.setattr(c, "_client", lambda base_url: Client({}))
    c.chat_json("u", "m", "sys", "usr")
    assert options["max_retries"] == 0
    # Generous read window for slow reasoning models, but an unreachable host fails fast.
    assert options["timeout"].read == 600.0 and options["timeout"].connect == 5.0


def test_client_key_lookup_ignores_a_trailing_slash(monkeypatch):
    calls = []
    monkeypatch.setattr(c, "OpenAI", lambda **kw: calls.append(kw) or object())
    monkeypatch.setattr(c, "_api_keys", {"http://llm/v1": "s3cret"})
    c._clients.clear()
    c._client("http://llm/v1/")
    assert calls[0]["api_key"] == "s3cret"


def _chat_client(message, finish_reason):
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    create = lambda **kw: SimpleNamespace(  # noqa: E731
        choices=[choice], usage=SimpleNamespace(completion_tokens=24, prompt_tokens=300))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client.with_options = lambda **kw: client
    return client


def test_chat_json_truncated_generation_is_empty_and_logged(monkeypatch, caplog):
    # A reasoning model cut off at max_tokens leaves draft JSON in its thinking channel; that
    # draft must never be parsed as the answer, and the operator must be able to see why an
    # answerable question abstained. Counts only: never log content.
    message = SimpleNamespace(content=None, reasoning_content='draft {"answered": true, "claims": []}')
    monkeypatch.setattr(c, "_client", lambda base_url: _chat_client(message, "length"))
    with caplog.at_level("WARNING"):
        assert c.chat_json("u", "m", "sys", "usr", max_tokens=24) == ""
    assert "max_tokens" in caplog.text and "24" in caplog.text
    assert "draft" not in caplog.text


def test_chat_json_keeps_reasoning_channel_fallback_for_complete_generations(monkeypatch):
    message = SimpleNamespace(content="", reasoning_content='{"answered": false, "claims": []}')
    monkeypatch.setattr(c, "_client", lambda base_url: _chat_client(message, "stop"))
    assert c.chat_json("u", "m", "sys", "usr") == '{"answered": false, "claims": []}'


def test_client_cached_per_base_url(monkeypatch):
    calls = []
    monkeypatch.setattr(c, "OpenAI", lambda **kw: calls.append(kw) or object())
    c._clients.clear()
    a = c._client("http://x/v1"); b = c._client("http://x/v1"); d = c._client("http://y/v1")
    assert a is b and a is not d
    assert len(calls) == 2

def test_client_sends_placeholder_key_by_default(monkeypatch):
    calls = []
    monkeypatch.setattr(c, "OpenAI", lambda **kw: calls.append(kw) or object())
    monkeypatch.setattr(c, "_api_keys", {})
    c._clients.clear()
    c._client("http://x/v1")
    assert calls[0]["api_key"] == "not-needed"  # the SDK requires a value; local servers ignore it


def test_configure_registers_endpoint_api_keys(monkeypatch):
    from vproc.config import Config, Endpoint

    calls = []
    monkeypatch.setattr(c, "OpenAI", lambda **kw: calls.append(kw) or object())
    monkeypatch.setattr(c, "_api_keys", {})
    c._clients.clear()
    plain = Endpoint("http://x/v1", "m")
    keyed = Endpoint("http://llm/v1", "m", api_key="s3cret")
    c.configure(Config(plain, plain, keyed, "./index", "localhost", 8765, 0.25, 0.5, None))
    c._client("http://llm/v1"); c._client("http://x/v1")
    assert [(k["base_url"], k["api_key"]) for k in calls] == [
        ("http://llm/v1", "s3cret"), ("http://x/v1", "not-needed")]


def test_configure_replaces_a_client_cached_with_a_stale_key(monkeypatch):
    from vproc.config import Config, Endpoint

    calls = []
    monkeypatch.setattr(c, "OpenAI", lambda **kw: calls.append(kw) or object())
    monkeypatch.setattr(c, "_api_keys", {})
    c._clients.clear()
    plain = Endpoint("http://x/v1", "m")
    stale = c._client("http://llm/v1")
    keyed = Endpoint("http://llm/v1", "m", api_key="s3cret")
    c.configure(Config(plain, plain, keyed, "./index", "localhost", 8765, 0.25, 0.5, None))
    assert c._client("http://llm/v1") is not stale
    assert [k["api_key"] for k in calls] == ["not-needed", "s3cret"]


def test_configure_ignores_direct_configs_without_keys(monkeypatch):
    # Tests and embedders pass partial stand-in configs with injected fake clients.
    monkeypatch.setattr(c, "_api_keys", {})
    c.configure(SimpleNamespace(ocr=SimpleNamespace(base_url="u", model="m")))
    assert c._api_keys == {}


class _Seg:
    def __init__(self, s, e, t): self.start, self.end, self.text = s, e, t

class _FakeAudioClient:
    def __init__(self, cap, resp): self.cap, self.resp = cap, resp
    def with_options(self, **kw): self.cap.update(kw); return self
    @property
    def audio(self):
        outer = self
        class T:
            def create(self, **kw):
                outer.cap.update(kw)
                return outer.resp
        return type("A", (), {"transcriptions": T()})()

def _write_wav(path, seconds, rate=16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))

def test_transcribe_audio_calls_asr_endpoint(tmp_path, monkeypatch):
    cap = {}
    resp = type("R", (), {"segments": [_Seg(0.0, 1.0, "hello"), _Seg(1.0, 2.0, "world")]})()
    monkeypatch.setattr(c, "_client", lambda base_url: _FakeAudioClient(cap, resp))
    wav = tmp_path / "a.wav"; wav.write_bytes(b"RIFFxxxx")
    out = c.transcribe_audio("http://x/v1", "whisper-1", str(wav))
    assert cap["model"] == "whisper-1"
    assert cap["response_format"] == "verbose_json"
    assert cap["timeout"].read == 3600 and cap["max_retries"] == 0
    assert out == {"segments": [
        {"start": 0.0, "end": 1.0, "text": "hello"},
        {"start": 1.0, "end": 2.0, "text": "world"},
    ]}

def test_transcribe_audio_synthesizes_segment_from_text(tmp_path, monkeypatch, caplog):
    resp = type("R", (), {"segments": None, "text": "  full transcript  "})()
    monkeypatch.setattr(c, "_client", lambda base_url: _FakeAudioClient({}, resp))
    wav = tmp_path / "a.wav"; _write_wav(wav, 2.0)
    out = c.transcribe_audio("http://x/v1", "whisper-1", str(wav))
    assert out == {"segments": [{"start": 0.0, "end": 2.0, "text": "full transcript"}]}
    # One recording-long chunk with one speaker is a big quality loss: say so.
    assert "no segment timestamps" in caplog.text

def test_transcribe_audio_empty_when_no_segments_no_text(tmp_path, monkeypatch):
    resp = type("R", (), {"segments": None, "text": ""})()
    monkeypatch.setattr(c, "_client", lambda base_url: _FakeAudioClient({}, resp))
    wav = tmp_path / "a.wav"; _write_wav(wav, 1.0)
    assert c.transcribe_audio("http://x/v1", "whisper-1", str(wav)) == {"segments": []}


class _Word:
    def __init__(self, s, e, w): self.start, self.end, self.word = s, e, w


def test_transcribe_audio_requests_words_and_files_them_under_their_segment(tmp_path, monkeypatch):
    # OpenAI-style servers return one flat word list next to the segments.
    cap = {}
    resp = type("R", (), {
        "segments": [_Seg(0.0, 1.0, " hello big"), _Seg(1.0, 2.0, " world")],
        "words": [_Word(0.0, 0.4, "hello"), _Word(0.5, 1.0, "big"), _Word(1.1, 2.0, "world")]})()
    monkeypatch.setattr(c, "_client", lambda base_url: _FakeAudioClient(cap, resp))
    wav = tmp_path / "a.wav"; wav.write_bytes(b"RIFFxxxx")
    out = c.transcribe_audio("http://x/v1", "whisper-1", str(wav))
    assert cap["timestamp_granularities"] == ["word", "segment"]
    assert out["segments"][0]["words"] == [{"start": 0.0, "end": 0.4, "word": "hello"},
                                           {"start": 0.5, "end": 1.0, "word": "big"}]
    assert out["segments"][1]["words"] == [{"start": 1.1, "end": 2.0, "word": "world"}]


def test_transcribe_audio_accepts_words_nested_in_segments(tmp_path, monkeypatch):
    # faster-whisper style servers nest each segment's words inside it.
    seg = _Seg(0.0, 1.0, "hi there")
    seg.words = [{"start": 0.0, "end": 0.5, "word": " hi"}, {"start": 0.5, "end": 1.0, "word": " there"}]
    resp = type("R", (), {"segments": [seg], "words": None})()
    monkeypatch.setattr(c, "_client", lambda base_url: _FakeAudioClient({}, resp))
    wav = tmp_path / "a.wav"; wav.write_bytes(b"RIFFxxxx")
    out = c.transcribe_audio("http://x/v1", "whisper-1", str(wav))
    assert [w["word"] for w in out["segments"][0]["words"]] == [" hi", " there"]


@pytest.mark.parametrize("status", [400, 422])
def test_transcribe_audio_retries_without_word_timestamps_on_a_strict_server(tmp_path, monkeypatch, status):
    import httpx
    from openai import BadRequestError, UnprocessableEntityError

    error = BadRequestError if status == 400 else UnprocessableEntityError

    attempts = []
    resp = type("R", (), {"segments": [_Seg(0.0, 1.0, "hello")]})()

    class Strict(_FakeAudioClient):
        @property
        def audio(self):
            def create(**kw):
                attempts.append(sorted(kw))
                if "timestamp_granularities" in kw:
                    raise error("unsupported", body=None, response=httpx.Response(
                        status, request=httpx.Request("POST", "http://x/v1/audio/transcriptions")))
                return resp
            return SimpleNamespace(transcriptions=SimpleNamespace(create=create))

    monkeypatch.setattr(c, "_client", lambda base_url: Strict({}, resp))
    wav = tmp_path / "a.wav"; wav.write_bytes(b"RIFFxxxx")
    out = c.transcribe_audio("http://x/v1", "whisper-1", str(wav))
    assert out == {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]}
    assert len(attempts) == 2 and "timestamp_granularities" not in attempts[1]


@pytest.mark.parametrize("start", [True, None, float("nan"), -1])
def test_remote_asr_invalid_times_reach_shared_validation(tmp_path, monkeypatch, start):
    from vproc.config import Endpoint
    from vproc.errors import IngestError
    from vproc.ingest.transcribe import transcribe

    response = SimpleNamespace(segments=[_Seg(start, 1, "hello")])
    monkeypatch.setattr(c, "_client", lambda _: _FakeAudioClient({}, response))
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"placeholder")
    with pytest.raises(IngestError, match="timestamps"):
        transcribe(str(wav), endpoint=Endpoint("http://asr/v1", "asr"))


def test_embed_texts_rejects_zero_vector_from_backend(monkeypatch):
    response = SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[0.0, -0.0])])
    fake = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kw: response))
    monkeypatch.setattr(c, "_client", lambda _: fake)
    with pytest.raises(ValueError, match="embedding.*vector"):
        c.embed_texts("u", "m", ["a real transcript chunk"])


def test_asr_timeout_keeps_connection_failure_short(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(c, "_client", lambda _: _FakeAudioClient(captured, SimpleNamespace(segments=[])))
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"placeholder")
    c.transcribe_audio("http://asr/v1", "asr", str(wav))
    assert captured["timeout"].read == 3600.0
    assert captured["timeout"].connect == 5.0
    assert captured["max_retries"] == 0


def test_concurrent_first_requests_share_one_client_pool(monkeypatch):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    start = threading.Barrier(8)
    created = []

    def constructor(**kwargs):
        # Let all contenders reach the uninitialized cache before construction finishes.
        time.sleep(0.02)
        instance = object()
        created.append(instance)
        return instance

    monkeypatch.setattr(c, "OpenAI", constructor)
    monkeypatch.setattr(c, "_clients", {})
    monkeypatch.setattr(c, "_api_keys", {})

    def request(_):
        start.wait(timeout=5)
        return c._client("http://pool/v1")

    with ThreadPoolExecutor(max_workers=8) as pool:
        clients = list(pool.map(request, range(8)))
    assert len(created) == 1
    assert all(client is clients[0] for client in clients)


def test_client_pool_normalizes_trailing_slash(monkeypatch):
    monkeypatch.setattr(c, "OpenAI", lambda **kw: object())
    monkeypatch.setattr(c, "_clients", {})
    assert c._client("http://pool/v1") is c._client("http://pool/v1/")


def test_embedding_probe_can_bound_timeout_without_retries(monkeypatch):
    captured = {}

    class Quick(_FakeClient):
        def with_options(self, **kwargs):
            captured.update(kwargs)
            return self

    monkeypatch.setattr(c, "_client", lambda _: Quick({}))
    assert c.embed_texts("u", "m", ["probe"], timeout=15.0) == [[5.0]]
    assert captured["timeout"].read == 15.0
    assert captured["timeout"].connect == 5.0
    assert captured["max_retries"] == 0


def test_large_flat_word_list_preserves_assignment_order():
    # Long ASR responses must consume the word queue once, without dropping or reusing words.
    count = 20000
    segments = [{"text": f"word{i}"} for i in range(count)]
    words = [{"word": f"word{i}", "start": i, "end": i + 1} for i in range(count)]
    c._attach_words(segments, [SimpleNamespace(words=None) for _ in segments], words)
    assert [seg["words"][0]["start"] for seg in segments] == list(range(count))
    assert all(seg["words"][0]["word"] == seg["text"] for seg in segments)
