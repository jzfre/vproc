from fastapi.testclient import TestClient
from vproc.config import Config, Endpoint
from vproc.models import Answer
from vproc.service import create_app

def _cfg():
    ep = Endpoint("u", "m")
    return Config(ep, ep, ep, "x", "0.0.0.0", 8765, 0.25, 0.5, None)

def test_healthz():
    app = create_app(store=object(), scorer=lambda p, h: 1.0, cfg=_cfg(),
                     ask=lambda *a, **k: Answer(answered=False, abstained=True,
                                                text="Not discussed in these meetings.",
                                                claims=[], evidence=[]),
                     search=lambda *a, **k: [])
    client = TestClient(app)
    assert client.get("/healthz").json() == {"ok": True}

def test_ask_route_returns_answer_json():
    captured = {}
    def fake_ask(store, cfg, question, scorer, where=None):
        captured["question"] = question
        return Answer(answered=True, abstained=False, text="ships in July [standup · 00:12 · Tim V]",
                      claims=[], evidence=[])
    app = create_app(store=object(), scorer=lambda p, h: 1.0, cfg=_cfg(),
                     ask=fake_ask, search=lambda *a, **k: [])
    client = TestClient(app)
    r = client.post("/ask", json={"question": "when does it ship?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answered"] is True and "July" in body["text"]
    assert captured["question"] == "when does it ship?"

def test_where_filter_built_from_request():
    seen = {}
    def fake_ask(store, cfg, question, scorer, where=None):
        seen["where"] = where
        return Answer(answered=False, abstained=True, text="x", claims=[], evidence=[])
    app = create_app(store=object(), scorer=lambda p, h: 1.0, cfg=_cfg(),
                     ask=fake_ask, search=lambda *a, **k: [])
    TestClient(app).post("/ask", json={"question": "q", "speaker": "Tim V", "memory": "standup"})
    assert "speaker = 'Tim V'" in seen["where"] and "memory_id = 'standup'" in seen["where"]
