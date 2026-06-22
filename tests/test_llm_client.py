import vproc.llm.client as c

class _Resp:
    def __init__(self, payload): self._p = payload

class _FakeEmbeddings:
    def create(self, model, input):
        class D:  # noqa: N801
            def __init__(self, e): self.embedding = e
        return type("R", (), {"data": [D([float(len(t))]) for t in input]})()

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
    def __init__(self, capture):
        self.embeddings = _FakeEmbeddings()
        self.chat = _FakeChat(capture)

def test_embed_texts(monkeypatch):
    monkeypatch.setattr(c, "_client", lambda base_url: _FakeClient({}))
    out = c.embed_texts("u", "m", ["a", "bb"])
    assert out == [[1.0], [2.0]]

def test_ocr_image_builds_data_url(tmp_path, monkeypatch):
    cap = {}
    monkeypatch.setattr(c, "_client", lambda base_url: _FakeClient(cap))
    img = tmp_path / "f.png"; img.write_bytes(b"\x89PNG\r\n\x1a\n")
    out = c.ocr_image("u", "m", str(img), "PROMPT")
    assert out == "OK"
    content = cap["messages"][0]["content"]
    assert content[0]["text"] == "PROMPT"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

def test_chat_json_sets_json_format(monkeypatch):
    cap = {}
    monkeypatch.setattr(c, "_client", lambda base_url: _FakeClient(cap))
    out = c.chat_json("u", "m", "sys", "usr")
    assert out == "OK"
    assert cap["response_format"] == {"type": "json_object"}

class _Seg:
    def __init__(self, s, e, t): self.start, self.end, self.text = s, e, t

class _FakeAudioClient:
    def __init__(self, cap): self.cap = cap
    @property
    def audio(self):
        outer = self
        class T:
            def create(self, **kw):
                outer.cap.update(kw)
                return type("R", (), {"segments": [_Seg(0.0, 1.0, "hello"), _Seg(1.0, 2.0, "world")]})()
        return type("A", (), {"transcriptions": T()})()

def test_transcribe_audio_calls_asr_endpoint(tmp_path, monkeypatch):
    cap = {}
    monkeypatch.setattr(c, "_client", lambda base_url: _FakeAudioClient(cap))
    wav = tmp_path / "a.wav"; wav.write_bytes(b"RIFFxxxx")
    out = c.transcribe_audio("http://x/v1", "whisper-1", str(wav))
    assert cap["model"] == "whisper-1"
    assert cap["response_format"] == "verbose_json"
    assert out == {"segments": [
        {"start": 0.0, "end": 1.0, "text": "hello"},
        {"start": 1.0, "end": 2.0, "text": "world"},
    ]}
