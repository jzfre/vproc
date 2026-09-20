import json
from types import SimpleNamespace

import vproc.answer.analyze as A
from vproc.config import Endpoint


class _Store:
    def __init__(self, rows): self.rows = rows
    def memory_rows(self, memory_id, project_id): return list(self.rows)


def _cfg():
    return SimpleNamespace(grounding=Endpoint("u", "m"), grounding_max_tokens=512)


def test_analyze_sends_whole_transcript_in_time_order(monkeypatch):
    seen = {}

    def chat(base_url, model, system, user, **kw):
        seen.update(kw, user=user)
        return json.dumps({"answer": "Ann ships Friday."})

    monkeypatch.setattr(A.client, "chat_json", chat)
    rows = [{"start_ts": 9.0, "speaker": "Bob", "said_text": "ok"},
            {"start_ts": 2.0, "speaker": "Ann", "said_text": "we ship Friday"}]
    assert A.analyze_memory(_Store(rows), _cfg(), "standup", "When?") == "Ann ships Friday."
    assert seen["user"].index("[2s Ann] we ship Friday") < seen["user"].index("[9s Bob] ok")
    assert seen["think"] is False and seen["max_tokens"] == 512


def test_analyze_without_transcript_never_calls_the_model(monkeypatch):
    monkeypatch.setattr(A.client, "chat_json", lambda *a, **k: 1 / 0)
    assert "No transcript" in A.analyze_memory(_Store([]), _cfg(), "standup", "When?")


def test_analyze_registers_endpoint_api_keys(monkeypatch):
    seen = []
    monkeypatch.setattr(A.client, "configure", seen.append)
    monkeypatch.setattr(A.client, "chat_json", lambda *a, **k: "{}")
    cfg = _cfg()
    A.analyze_memory(_Store([{"start_ts": 0.0, "speaker": "Ann", "said_text": "hi"}]), cfg, "m", "q")
    assert seen == [cfg]
