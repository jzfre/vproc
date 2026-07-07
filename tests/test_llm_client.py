import wave

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

def test_client_cached_per_base_url(monkeypatch):
    calls = []
    monkeypatch.setattr(c, "OpenAI", lambda **kw: calls.append(kw) or object())
    c._clients.clear()
    a = c._client("http://x/v1"); b = c._client("http://x/v1"); d = c._client("http://y/v1")
    assert a is b and a is not d
    assert len(calls) == 2

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
    assert cap["timeout"] == 3600 and cap["max_retries"] == 0
    assert out == {"segments": [
        {"start": 0.0, "end": 1.0, "text": "hello"},
        {"start": 1.0, "end": 2.0, "text": "world"},
    ]}

def test_transcribe_audio_synthesizes_segment_from_text(tmp_path, monkeypatch):
    resp = type("R", (), {"segments": None, "text": "  full transcript  "})()
    monkeypatch.setattr(c, "_client", lambda base_url: _FakeAudioClient({}, resp))
    wav = tmp_path / "a.wav"; _write_wav(wav, 2.0)
    out = c.transcribe_audio("http://x/v1", "whisper-1", str(wav))
    assert out == {"segments": [{"start": 0.0, "end": 2.0, "text": "full transcript"}]}

def test_transcribe_audio_empty_when_no_segments_no_text(tmp_path, monkeypatch):
    resp = type("R", (), {"segments": None, "text": ""})()
    monkeypatch.setattr(c, "_client", lambda base_url: _FakeAudioClient({}, resp))
    wav = tmp_path / "a.wav"; _write_wav(wav, 1.0)
    assert c.transcribe_audio("http://x/v1", "whisper-1", str(wav)) == {"segments": []}
