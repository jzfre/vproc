"""scripts/local_models.py with fake models: the HTTP contract vproc's client relies on."""
import base64
import importlib.util
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "local_models.py"


@pytest.fixture
def server(monkeypatch):
    spec = importlib.util.spec_from_file_location("local_models", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Whisper:
    def __init__(self):
        self.kwargs = None

    def transcribe(self, path, **kwargs):
        self.kwargs = kwargs
        word = lambda s, e, w: SimpleNamespace(start=s, end=e, word=w)  # noqa: E731
        segment = SimpleNamespace(
            id=0, seek=0, start=0.0, end=1.5, text=" Hello there.", tokens=[1], temperature=0.0,
            avg_logprob=-0.1, compression_ratio=1.0, no_speech_prob=0.0,
            words=[word(0.0, 0.6, " Hello"), word(0.7, 1.5, " there.")] if kwargs["word_timestamps"] else None)
        info = SimpleNamespace(duration=1.5, language="en", language_probability=0.99)
        return iter([segment]), info


def _transcribe(client, server, **data):
    return client.post("/v1/audio/transcriptions", files={"file": ("a.wav", b"RIFF", "audio/wav")},
                       data={"model": server.WHISPER_MODEL, "response_format": "verbose_json", **data})


def test_transcription_returns_segments_and_words_when_asked(server, monkeypatch):
    whisper = _Whisper()
    monkeypatch.setattr(server, "_whisper", lambda: whisper)
    body = _transcribe(TestClient(server.app), server,
                       **{"timestamp_granularities[]": ["word", "segment"]}).json()
    assert whisper.kwargs["word_timestamps"] is True
    assert body["segments"][0]["text"] == " Hello there."
    assert body["segments"][0]["words"] == [{"start": 0.0, "end": 0.6, "word": " Hello"},
                                            {"start": 0.7, "end": 1.5, "word": " there."}]
    assert body["words"] == body["segments"][0]["words"]  # OpenAI-style flat list as well


def test_transcription_without_word_granularity_skips_alignment(server, monkeypatch):
    whisper = _Whisper()
    monkeypatch.setattr(server, "_whisper", lambda: whisper)
    body = _transcribe(TestClient(server.app), server).json()
    assert whisper.kwargs["word_timestamps"] is False and "words" not in body["segments"][0]


def test_style_hint_is_sent_as_hotwords_so_it_reaches_every_window(server, monkeypatch):
    # With condition_on_previous_text=False, initial_prompt only reaches the first 30 s window;
    # hotwords are re-applied per window, which keeps casing and punctuation consistent.
    whisper = _Whisper()
    monkeypatch.setattr(server, "_whisper", lambda: whisper)
    _transcribe(TestClient(server.app), server)
    assert whisper.kwargs["hotwords"] == server.WHISPER_HOTWORDS and server.WHISPER_HOTWORDS
    assert whisper.kwargs["condition_on_previous_text"] is False
    _transcribe(TestClient(server.app), server, prompt="Ada Lovelace, Alan Turing.")
    assert whisper.kwargs["hotwords"].endswith("Ada Lovelace, Alan Turing.")


def test_unknown_model_or_format_is_rejected_not_silently_substituted(server, monkeypatch):
    monkeypatch.setattr(server, "_whisper", lambda: _Whisper())
    client = TestClient(server.app)
    wrong_model = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", b"RIFF")},
                              data={"model": "some-other-whisper"})
    assert wrong_model.status_code == 404
    assert _transcribe(client, server, response_format="srt").status_code == 422
    assert client.post("/v1/embeddings", json={"model": "other", "input": "x"}).status_code == 404


class _Embedder:
    def encode(self, texts, **kwargs):
        import numpy as np

        return np.array([[float(len(t)), 0.5] for t in texts], dtype=np.float32)


def test_embeddings_in_float_and_base64(server, monkeypatch):
    monkeypatch.setattr(server, "_embedder", lambda: _Embedder())
    client = TestClient(server.app)
    floats = client.post("/v1/embeddings", json={"model": server.EMBED_MODEL, "input": ["ab", "abcd"]}).json()
    assert [d["index"] for d in floats["data"]] == [0, 1]
    assert floats["data"][1]["embedding"] == [4.0, 0.5]
    encoded = client.post("/v1/embeddings", json={
        "model": server.EMBED_MODEL, "input": "ab", "encoding_format": "base64"}).json()
    assert struct.unpack("<2f", base64.b64decode(encoded["data"][0]["embedding"])) == (2.0, 0.5)


def test_health_reports_loaded_models_and_there_is_no_chat_route(server):
    client = TestClient(server.app)
    assert client.get("/health").json()["ok"] is True
    # vproc talks to the LLM directly (VPROC_GROUNDING_API_KEY_FILE) and disables OCR with an
    # empty VPROC_OCR_BASE_URL: this server must not grow a key-carrying proxy or an OCR stub.
    assert client.post("/v1/chat/completions", json={}).status_code == 404


@pytest.mark.parametrize("kind", ["asr", "embed"])
def test_busy_model_rejects_work_without_queueing_or_blocking_health(server, monkeypatch, kind):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    entered, release = Event(), Event()
    class BlockingWhisper(_Whisper):
        def transcribe(self, *args, **kwargs):
            entered.set()
            assert release.wait(3), "test release timed out"
            return super().transcribe(*args, **kwargs)
    class BlockingEmbedder(_Embedder):
        def encode(self, *args, **kwargs):
            entered.set()
            assert release.wait(3), "test release timed out"
            return super().encode(*args, **kwargs)
    monkeypatch.setattr(server, "_whisper", lambda: BlockingWhisper())
    monkeypatch.setattr(server, "_embedder", lambda: BlockingEmbedder())
    client = TestClient(server.app)
    request = (lambda: _transcribe(client, server)) if kind == "asr" else (
        lambda: client.post("/v1/embeddings", json={"model": server.EMBED_MODEL, "input": "hello"}))
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(request)
        try:
            assert entered.wait(2)
            second = pool.submit(request)
            busy = second.result(timeout=1)
            assert busy.status_code == 503 and busy.headers["retry-after"] == "1"
            assert client.get("/health").status_code == 200
        finally:
            release.set()
        assert first.result(timeout=2).status_code == 200
    assert request().status_code == 200  # admission released after the first request


@pytest.mark.parametrize("failure", ["decode", "iteration"])
def test_transcription_failure_cleans_tempfile_and_does_not_leak_details(server, monkeypatch, failure, capsys):
    files = []
    class BrokenWhisper(_Whisper):
        def transcribe(self, path, **kwargs):
            files.append(Path(path))
            assert Path(path).read_bytes() == b"RIFF"
            if failure == "decode":
                raise RuntimeError("private token and transcript")
            def broken_segments():
                raise RuntimeError("private token and transcript")
                yield
            _, info = super().transcribe(path, **kwargs)
            return broken_segments(), info
    monkeypatch.setattr(server, "_whisper", lambda: BrokenWhisper())
    client = TestClient(server.app, raise_server_exceptions=False)
    response = _transcribe(client, server)
    assert response.status_code == 503
    assert files and all(not p.exists() for p in files)
    assert "private token" not in response.text + capsys.readouterr().err
    monkeypatch.setattr(server, "_whisper", lambda: _Whisper())
    assert _transcribe(client, server).status_code == 200


def test_oversized_embedding_batch_rejected_before_loading_model(server, monkeypatch):
    monkeypatch.setattr(server, "_embedder", lambda: pytest.fail("oversized request loaded a model"))
    response = TestClient(server.app).post("/v1/embeddings", json={
        "model": server.EMBED_MODEL, "input": ["hello"] * 33})
    assert response.status_code == 400
