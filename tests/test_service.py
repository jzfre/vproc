import dataclasses

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import APIConnectionError, APIStatusError, APITimeoutError
from vproc.config import Config, Endpoint
from vproc.errors import IndexBusyError, IndexCompatibilityError
from vproc.models import Answer
from vproc.service import create_app

def _cfg():
    ep = Endpoint("u", "m")
    return Config(ep, ep, ep, "x", "0.0.0.0", 8765, 0.25, 0.5, None)

def _cfg_with(cfg, **overrides):
    return dataclasses.replace(cfg, **overrides)

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

def test_filter_injection_is_neutralized_by_escaping():
    seen = {}
    def fake_ask(store, cfg, question, scorer, where=None):
        seen["where"] = where
        return Answer(answered=False, abstained=True, text="x", claims=[], evidence=[])
    app = create_app(store=object(), scorer=lambda p, h: 1.0, cfg=_cfg(),
                     ask=fake_ask, search=lambda *a, **k: [])
    r = TestClient(app).post("/ask", json={"question": "q", "project": "x' OR '1'='1"})
    assert r.status_code == 200                        # not rejected, escaped instead
    # every single quote doubled, so the value stays inside the SQL string literal
    assert "project_id = 'x'' OR ''1''=''1'" in seen["where"]

def test_legit_filter_ids_with_apostrophes_and_unicode_accepted():
    seen = {}
    def fake_ask(store, cfg, question, scorer, where=None):
        seen["where"] = where
        return Answer(answered=False, abstained=True, text="x", claims=[], evidence=[])
    app = create_app(store=object(), scorer=lambda p, h: 1.0, cfg=_cfg(),
                     ask=fake_ask, search=lambda *a, **k: [])
    r = TestClient(app).post("/ask", json={"question": "q", "memory": "Tim's réunion (2)",
                                           "speaker": "O'Brien"})
    assert r.status_code == 200
    assert "memory_id = 'Tim''s réunion (2)'" in seen["where"]
    assert "speaker = 'O''Brien'" in seen["where"]

def test_control_char_and_overlength_filters_rejected():
    app = create_app(store=object(), scorer=lambda p, h: 1.0, cfg=_cfg(),
                     ask=lambda *a, **k: Answer(answered=False, abstained=True, text="x",
                                                claims=[], evidence=[]),
                     search=lambda *a, **k: [])
    client = TestClient(app)
    assert client.post("/ask", json={"question": "q", "memory": "a\nb"}).status_code == 400
    assert client.post("/ask", json={"question": "q", "memory": "x" * 129}).status_code == 400

def test_empty_question_and_query_rejected():
    called = {"n": 0}
    def fake_ask(store, cfg, question, scorer, where=None):
        called["n"] += 1
        return Answer(answered=False, abstained=True, text="x", claims=[], evidence=[])
    app = create_app(store=object(), scorer=lambda p, h: 1.0, cfg=_cfg(),
                     ask=fake_ask, search=lambda *a, **k: [])
    client = TestClient(app)
    assert client.post("/ask", json={}).status_code == 400            # no question
    assert client.post("/ask", json={"question": "   "}).status_code == 400  # whitespace
    assert client.post("/search", json={}).status_code == 400         # no query/question
    assert called["n"] == 0            # embedding backend never reached with ""

def test_mcp_endpoint_initializes():
    app = create_app(store=object(), scorer=lambda p, h: 1.0, cfg=_cfg(),
                     ask=lambda *a, **k: Answer(answered=False, abstained=True, text="x",
                                                claims=[], evidence=[]),
                     search=lambda *a, **k: [])
    headers = {"Accept": "application/json, text/event-stream",
               "Content-Type": "application/json"}
    init = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "t", "version": "1"}}}
    with TestClient(app) as client:   # context manager so the lifespan actually runs
        r = client.post("/mcp", json=init, headers=headers)
    assert r.status_code == 200        # no 404, no 500 'Task group is not initialized'
    assert '"serverInfo"' in r.text and '"protocolVersion"' in r.text


def _ui_rows():
    return [
        {"id": "1", "memory_id": "standup", "project_id": "default", "speaker": "Repan, Jozef",
         "start_ts": 10.0, "end_ts": 20.0, "said_text": "hello", "on_screen_text": "",
         "embed_text": "", "source_video": "tmp/standup.mkv", "frame_path": ""},
        {"id": "2", "memory_id": "standup", "project_id": "default", "speaker": "Vanco, Pavol",
         "start_ts": 0.0, "end_ts": 10.0, "said_text": "hi", "on_screen_text": "agenda",
         "embed_text": "", "source_video": "tmp/standup.mkv",
         "frame_path": "./vproc_frames/default/standup/00000001.png"},
    ]


class _UIStore:
    def __init__(self, rows):
        self.rows = rows

    def all_rows(self):
        return list(self.rows)

    def memory_rows(self, memory_id, project_id=None):
        return [r for r in self.rows if r["memory_id"] == memory_id
                and (project_id is None or r["project_id"] == project_id)]


def test_api_memories_aggregates():
    app = create_app(store=_UIStore(_ui_rows()), scorer=lambda p, h: 1.0, cfg=_cfg())
    c = TestClient(app)
    r = c.get("/api/memories")
    assert r.status_code == 200
    (m,) = r.json()
    assert m["memory_id"] == "standup" and m["segment_count"] == 2
    assert m["duration_s"] == 20.0
    assert m["speakers"] == ["Repan, Jozef", "Vanco, Pavol"]


def test_api_memories_excludes_non_default_project():
    rows = _ui_rows() + [
        {"id": "3", "memory_id": "other-proj-meeting", "project_id": "acme",
         "speaker": "X", "start_ts": 0.0, "end_ts": 5.0, "said_text": "", "on_screen_text": "",
         "embed_text": "", "source_video": "tmp/other.mkv", "frame_path": ""},
    ]
    app = create_app(store=_UIStore(rows), scorer=lambda p, h: 1.0, cfg=_cfg())
    memory_ids = [m["memory_id"] for m in TestClient(app).get("/api/memories").json()]
    assert "other-proj-meeting" not in memory_ids
    assert memory_ids == ["standup"]


def test_api_memories_sorted_by_memory_id():
    rows = _ui_rows() + [
        {"id": "3", "memory_id": "all-hands", "project_id": "default", "speaker": "X",
         "start_ts": 0.0, "end_ts": 5.0, "said_text": "", "on_screen_text": "",
         "embed_text": "", "source_video": "tmp/all-hands.mkv", "frame_path": ""},
    ]
    app = create_app(store=_UIStore(rows), scorer=lambda p, h: 1.0, cfg=_cfg())
    memory_ids = [m["memory_id"] for m in TestClient(app).get("/api/memories").json()]
    assert memory_ids == ["all-hands", "standup"]  # sorted, not insertion order


def test_api_segments_ordered_with_frame_name():
    app = create_app(store=_UIStore(_ui_rows()), scorer=lambda p, h: 1.0, cfg=_cfg())
    c = TestClient(app)
    segs = c.get("/api/memories/standup/segments").json()
    assert [s["start_ts"] for s in segs] == [0.0, 10.0]  # ordered by start_ts
    assert segs[0]["frame_name"] == "00000001.png"
    assert segs[1]["frame_name"] is None
    assert set(segs[0]) == {"start_ts", "end_ts", "speaker", "said_text",
                            "on_screen_text", "frame_name"}


def test_api_segments_unknown_memory_404():
    app = create_app(store=_UIStore([]), scorer=lambda p, h: 1.0, cfg=_cfg())
    r = TestClient(app).get("/api/memories/nope/segments")
    assert r.status_code == 404 and "detail" in r.json()


def _media_app(tmp_path, body=b"0123456789abcdef"):
    video = tmp_path / "clip.mp4"
    video.write_bytes(body)
    rows = [{"id": "1", "memory_id": "clip", "project_id": "default", "speaker": "S",
             "start_ts": 0.0, "end_ts": 1.0, "said_text": "", "on_screen_text": "",
             "embed_text": "", "source_video": str(video), "frame_path": ""}]
    return create_app(store=_UIStore(rows), scorer=lambda p, h: 1.0, cfg=_cfg())


def test_media_full_and_ranges(tmp_path):
    c = TestClient(_media_app(tmp_path))
    full = c.get("/api/media/clip")
    assert full.status_code == 200 and full.content == b"0123456789abcdef"
    assert full.headers["content-type"] == "video/mp4"

    part = c.get("/api/media/clip", headers={"Range": "bytes=4-7"})
    assert part.status_code == 206 and part.content == b"4567"
    assert part.headers["content-range"] == "bytes 4-7/16"

    tail = c.get("/api/media/clip", headers={"Range": "bytes=12-"})
    assert tail.status_code == 206 and tail.content == b"cdef"

    suffix = c.get("/api/media/clip", headers={"Range": "bytes=-4"})
    assert suffix.status_code == 206 and suffix.content == b"cdef"
    assert suffix.headers["content-range"] == "bytes 12-15/16"

    bad = c.get("/api/media/clip", headers={"Range": "bytes=99-"})
    assert bad.status_code == 416


def test_media_missing_file_404(tmp_path):
    rows = [{"id": "1", "memory_id": "gone", "project_id": "default", "speaker": "S",
             "start_ts": 0.0, "end_ts": 1.0, "said_text": "", "on_screen_text": "",
             "embed_text": "", "source_video": str(tmp_path / "missing.mkv"),
             "frame_path": ""}]
    app = create_app(store=_UIStore(rows), scorer=lambda p, h: 1.0, cfg=_cfg())
    r = TestClient(app).get("/api/media/gone")
    assert r.status_code == 404 and "missing.mkv" in r.json()["detail"]
    assert TestClient(app).get("/api/media/nope").status_code == 404


def test_api_frames_serves_and_blocks_traversal(tmp_path, monkeypatch):
    frames = tmp_path / "frames" / "default" / "standup"
    frames.mkdir(parents=True)
    (frames / "00000001.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    secret = tmp_path / "secret.txt"
    secret.write_text("nope")
    cfg = _cfg()
    # inject frames_dir into whatever cfg object _cfg() returns (follow its type)
    cfg = _cfg_with(cfg, frames_dir=str(tmp_path / "frames"))
    app = create_app(store=_UIStore([]), scorer=lambda p, h: 1.0, cfg=cfg)
    c = TestClient(app)
    ok = c.get("/api/frames/standup/00000001.png")
    assert ok.status_code == 200 and ok.content.startswith(b"\x89PNG")
    assert c.get("/api/frames/standup/../../secret.txt").status_code in (404, 400)
    assert c.get("/api/frames/standup/%2e%2e%2fsecret.txt").status_code in (404, 400)
    assert c.get("/api/frames/standup/absent.png").status_code == 404


def test_api_frames_traversal_payloads_that_reach_the_guard(tmp_path):
    # The two payloads above never actually exercise api_frame's ".." checks: httpx
    # normalizes "../../" client-side before the request is sent, and "%2e%2e%2f"
    # decodes to a literal "/" that makes the {name} path segment fail to match the
    # route at all. Neither reaches the handler, so deleting its ".." checks wouldn't
    # fail those assertions. These payloads DO reach the handler (confirmed: FastAPI
    # matches memory_id="..", name="..png" respectively) and are only blocked by the
    # explicit ".." checks in api_frame, since the character-class regex alone allows
    # dots and would otherwise accept them.
    frames = tmp_path / "frames" / "default" / "standup"
    frames.mkdir(parents=True)
    (frames / "00000001.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    secret = tmp_path / "secret.txt"
    secret.write_text("nope")
    cfg = _cfg_with(_cfg(), frames_dir=str(tmp_path / "frames"))
    app = create_app(store=_UIStore([]), scorer=lambda p, h: 1.0, cfg=cfg)
    c = TestClient(app)

    r1 = c.get("/api/frames/%2e%2e/secret.txt")  # memory_id decodes to ".."
    assert r1.status_code == 404
    assert "nope" not in r1.text
    # Pin the exact rejection reason, not just the status code: os.path.isfile() would
    # also 404 (as "no such frame") for these paths even with the ".." checks deleted,
    # since the fixed "default" path segment means a single ".." can only cancel it out
    # and land back inside frames_dir itself, not reach tmp_path/secret.txt (one level
    # further up) — so a bare status-code assertion can't distinguish "guard fired" from
    # "guard absent, file coincidentally missing". The detail message can.
    assert r1.json()["detail"] == "bad frame name"

    r2 = c.get("/api/frames/standup/..png")  # name passes the char-class regex
    assert r2.status_code == 404
    assert r2.json()["detail"] == "bad frame name"


def test_missing_ui_dir_does_not_crash_app_creation(tmp_path, monkeypatch):
    # An API-only install must keep working and return a normal 404 for the UI.
    import vproc.service as svc
    monkeypatch.setattr(svc, "__file__", str(tmp_path / "service.py"))
    app = create_app(store=_UIStore([]), scorer=lambda p, h: 1.0, cfg=_cfg())
    assert TestClient(app).get("/healthz").json() == {"ok": True}
    assert TestClient(app).get("/").status_code == 404


def test_root_serves_ui_and_api_wins():
    app = create_app(store=_UIStore([]), scorer=lambda p, h: 1.0, cfg=_cfg())
    c = TestClient(app)
    r = c.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert c.get("/healthz").json() == {"ok": True}  # routes still beat the mount


def test_browsing_and_search_do_not_load_answer_verifier(monkeypatch):
    import vproc.answer.faithfulness as faithfulness

    def unavailable_model(*args):
        raise AssertionError("Browsing must not load or download the answer model")

    monkeypatch.setattr(faithfulness, "HHEM", unavailable_model)
    app = create_app(store=_UIStore([]), cfg=_cfg(), search=lambda *a, **k: [])
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/healthz").json() == {"ok": True}
        assert client.get("/api/memories").json() == []
        assert client.post("/search", json={"query": "release"}).json() == []


def test_answer_verifier_loads_on_first_score_and_is_reused(monkeypatch):
    import vproc.answer.faithfulness as faithfulness

    loaded = []

    class Verifier:
        def __init__(self, model):
            loaded.append(model)

        def score(self, premise, hypothesis):
            return 0.8

    def answer_with_score(store, cfg, question, scorer, where=None):
        return Answer(answered=True, abstained=False,
                      text=str(scorer("evidence", question)), claims=[], evidence=[])

    monkeypatch.setattr(faithfulness, "HHEM", Verifier)
    app = create_app(store=_UIStore([]), cfg=_cfg(), ask=answer_with_score)
    assert loaded == []
    with TestClient(app) as client:
        for question in ("first", "second"):
            assert client.post("/ask", json={"question": question}).json()["text"] == "0.8"
    assert loaded == [_cfg().hhem_model]


def test_media_uses_updated_index_after_reingest(tmp_path):
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"old recording")
    second.write_bytes(b"new recording")
    store = _UIStore([{**_ui_rows()[0], "source_video": str(first)}])
    client = TestClient(create_app(store=store, cfg=_cfg(), scorer=lambda p, h: 1.0))
    assert client.get("/api/media/standup").content == b"old recording"
    store.rows[0]["source_video"] = str(second)
    assert client.get("/api/media/standup").content == b"new recording"
    store.rows.clear()
    assert client.get("/api/media/standup").status_code == 404


def test_media_directory_is_reported_as_missing(tmp_path):
    store = _UIStore([{**_ui_rows()[0], "source_video": str(tmp_path)}])
    client = TestClient(create_app(store=store, cfg=_cfg(), scorer=lambda p, h: 1.0))
    assert client.get("/api/media/standup").status_code == 404


def test_frame_symlink_cannot_escape_frames_directory(tmp_path):
    frames = tmp_path / "frames" / "default" / "standup"
    frames.mkdir(parents=True)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside frame storage")
    (frames / "00000001.png").symlink_to(outside)
    cfg = _cfg_with(_cfg(), frames_dir=str(tmp_path / "frames"))
    client = TestClient(create_app(store=_UIStore([]), cfg=cfg, scorer=lambda p, h: 1.0))
    assert client.get("/api/frames/standup/00000001.png").status_code == 404


@pytest.mark.parametrize("route, field", [("/ask", "question"), ("/search", "query")])
def test_query_length_limit_rejects_before_backend_and_allows_boundary(route, field):
    def answer(store, cfg, question, scorer, where=None):
        assert len(question) <= 8192, "overlong input reached backend"
        return Answer(answered=False, abstained=True, text="offline", claims=[], evidence=[])

    def search(store, cfg, query, where=None):
        assert len(query) <= 8192, "overlong input reached backend"
        return []

    client = TestClient(create_app(store=object(), cfg=_cfg(), scorer=lambda *_: 1,
                                   ask=answer, search=search))
    assert client.post(route, json={field: "x" * 8193}).status_code == 400
    assert client.post(route, json={field: "x" * 8192}).status_code == 200


@pytest.mark.parametrize("field", ["project", "memory", "speaker"])
@pytest.mark.parametrize("value", ["x\x00y", "x\x85y", "x" * 129])
def test_search_filter_validation_blocks_invalid_values(field, value):
    def search(*args, **kwargs):
        pytest.fail("invalid filter reached the retrieval backend")

    client = TestClient(create_app(store=object(), cfg=_cfg(), scorer=lambda *_: 1,
                                   search=search))
    assert client.post("/search", json={"query": "q", field: value}).status_code == 400


def _backend_errors():
    request = httpx.Request("POST", "http://backend.invalid/v1", headers={"Authorization": "secret-token"})
    return [
        (APITimeoutError(request=request), 504, "timed out"),
        (APIConnectionError(message="secret-token connection body", request=request), 503, "unavailable"),
        (APIStatusError("secret-token response body", response=httpx.Response(503, request=request),
                        body={"secret": "secret-token"}), 503, "unavailable"),
        (RuntimeError("secret-token unexpected backend response"), 500, "failed"),
        (IndexCompatibilityError("secret-token index metadata"), 409, "re-ingest"),
        (IndexBusyError("secret-token index path"), 503, "retry"),
    ]


@pytest.mark.parametrize("route, field", [("/ask", "question"), ("/search", "query")])
@pytest.mark.parametrize("error, status, detail", _backend_errors())
def test_backend_failures_are_safe_and_next_request_recovers(route, field, error, status, detail, caplog):
    pending_error = error

    def backend(*args, **kwargs):
        nonlocal pending_error
        if pending_error is not None:
            exc, pending_error = pending_error, None
            raise exc
        return (Answer(answered=False, abstained=True, text="recovered", claims=[], evidence=[])
                if route == "/ask" else [])

    app = create_app(store=object(), cfg=_cfg(), scorer=lambda *_: 1, ask=backend, search=backend)
    with TestClient(app) as client:
        response = client.post(route, json={field: "release?"})
        assert response.status_code == status
        assert detail in response.json()["detail"].lower()
        assert "secret-token" not in response.text
        assert client.get("/healthz").json() == {"ok": True}
        assert client.post(route, json={field: "release?"}).status_code == 200
    assert "secret-token" not in caplog.text


def test_mcp_setup_failure_is_visible_without_breaking_rest(monkeypatch, caplog):
    from mcp.server.fastmcp import FastMCP

    def broken_setup(self):
        raise RuntimeError("secret-token setup details")

    monkeypatch.setattr(FastMCP, "streamable_http_app", broken_setup)
    client = TestClient(create_app(store=object(), cfg=_cfg(), scorer=lambda *_: 1,
                                   search=lambda *a, **k: []))
    assert client.get("/healthz").json() == {"ok": True}
    assert client.post("/search", json={"query": "release"}).status_code == 200
    status = client.get("/api/status")
    assert status.status_code == 200
    assert status.json()["mcp_available"] is False
    assert "MCP" in caplog.text and "unavailable" in caplog.text.lower()
    assert "secret-token" not in status.text + caplog.text


def test_status_checks_empty_and_legacy_index_without_model_calls(tmp_path, monkeypatch):
    from vproc.llm import client as llm
    from vproc.store.lancedb_store import Store

    def no_model_calls(*args, **kwargs):
        pytest.fail("index status must not probe model endpoints")

    monkeypatch.setattr(llm, "_client", no_model_calls)
    store = Store(str(tmp_path / "index"))
    client = TestClient(create_app(store=store, cfg=_cfg(), scorer=lambda *_: 1))
    status = client.get("/api/status")
    assert status.status_code == 200
    assert status.json()["mcp_available"] is True
    assert status.json()["index_compatible"] is True

    store.db.create_table(store.TABLE, data=[{**_ui_rows()[0], "vector": [1.0, 0.0]}])
    status = client.get("/api/status")
    assert status.status_code == 200
    assert status.json()["index_compatible"] is False
    assert "re-ingest" in status.json()["detail"].lower()
    assert client.get("/healthz").json() == {"ok": True}
    assert client.get("/api/memories").status_code == 200


def test_create_app_registers_endpoint_api_keys(monkeypatch):
    import vproc.service as service

    seen = []
    monkeypatch.setattr(service.client, "configure", seen.append)
    cfg = _cfg()
    create_app(store=object(), scorer=lambda p, h: 1.0, cfg=cfg)
    assert seen == [cfg]


def test_unloadable_verifier_fails_ask_before_any_generation_and_recovers(monkeypatch):
    # The gate model loads before the (minutes-long) generation: a broken verifier must cost
    # seconds and say so, not a full generation per ask ending in a generic 500.
    import vproc.answer.faithfulness as faithfulness

    asked = []
    healthy = []

    class Verifier:
        def __init__(self, model):
            if not healthy:
                raise OSError("401 for https://huggingface.co/... token=secret-token")

        def score(self, premise, hypothesis):
            return 0.9

    def answer(store, cfg, question, scorer, where=None):
        asked.append(question)
        return Answer(answered=True, abstained=False, text=str(scorer("e", question)),
                      claims=[], evidence=[])

    monkeypatch.setattr(faithfulness, "HHEM", Verifier)
    app = create_app(store=_UIStore([]), cfg=_cfg(), ask=answer)
    with TestClient(app) as client:
        assert client.get("/api/status").json()["verifier"] == "not_loaded"
        failed = client.post("/ask", json={"question": "first"})
        assert failed.status_code == 503
        assert "verifier" in failed.json()["detail"].lower()
        assert "secret-token" not in failed.text
        assert asked == []
        assert client.get("/api/status").json()["verifier"] == "failed"
        healthy.append(True)  # e.g. the token was fixed: no restart needed
        assert client.post("/ask", json={"question": "second"}).json()["text"] == "0.9"
        assert client.get("/api/status").json()["verifier"] == "ready"


def test_injected_scorer_reports_no_verifier_state():
    client = TestClient(create_app(store=_UIStore([]), cfg=_cfg(), scorer=lambda p, h: 1.0))
    assert "verifier" not in client.get("/api/status").json()


def test_backend_failure_log_carries_the_http_status_but_never_the_body(caplog):
    # "InternalServerError" alone could not tell a proxy 503 from a vLLM 500.
    request = httpx.Request("POST", "http://backend/v1/chat/completions")

    def failing(store, cfg, question, scorer, where=None):
        raise APIStatusError("secret-token response body",
                             response=httpx.Response(503, request=request), body=None)

    client = TestClient(create_app(store=object(), scorer=lambda p, h: 1.0, cfg=_cfg(), ask=failing))
    with caplog.at_level("WARNING"):
        assert client.post("/ask", json={"question": "q"}).status_code == 503
    assert "APIStatusError" in caplog.text and "503" in caplog.text
    assert "secret-token" not in caplog.text


def test_frame_read_uses_same_valid_memory_names_as_ingest(tmp_path):
    from vproc.paths import memory_directory

    cfg = _cfg_with(_cfg(), frames_dir=str(tmp_path / "frames"))
    folder = memory_directory(cfg.frames_dir, "default", "planning..draft")
    folder.mkdir(parents=True)
    (folder / "frame.png").write_bytes(b"png")
    client = TestClient(create_app(store=_UIStore([]), scorer=lambda *_: 1, cfg=cfg))
    assert client.get("/api/frames/planning..draft/frame.png").status_code == 200


def test_frame_route_rejects_sidecars_and_symlinked_project(tmp_path):
    cfg = _cfg_with(_cfg(), frames_dir=str(tmp_path / "frames"))
    root = tmp_path / "frames"
    root.mkdir()
    outside = tmp_path / "outside" / "meeting"
    outside.mkdir(parents=True)
    (outside / "frame.png").write_bytes(b"outside")
    (root / "default").symlink_to(outside.parent, target_is_directory=True)
    client = TestClient(create_app(store=_UIStore([]), scorer=lambda *_: 1, cfg=cfg))
    assert client.get("/api/frames/meeting/frame.png").status_code == 404
    (root / "default").unlink()
    folder = root / "default" / "meeting"
    folder.mkdir(parents=True)
    (folder / "speakers.json").write_text('{"private":"sidecar"}')
    assert client.get("/api/frames/meeting/speakers.json").status_code == 404


def test_model_requests_fail_fast_at_capacity_and_release_slots():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    entered, release = Event(), Event()

    def answer(*args, **kwargs):
        entered.set()
        assert release.wait(5), "test never released model call"
        raise RuntimeError("backend failed")

    cfg = _cfg_with(_cfg(), max_concurrent_queries=1)
    app = create_app(store=_UIStore([]), scorer=lambda *_: 1, cfg=cfg,
                     ask=answer, search=lambda *a, **kw: [])
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(client.post, "/ask", json={"question": "slow"})
        try:
            assert entered.wait(5)
            for path, body in [("/ask", {"question": "next"}), ("/search", {"query": "next"})]:
                rejected = client.post(path, json=body)
                assert rejected.status_code == 503
                assert rejected.headers["retry-after"] == "1"
            assert client.get("/healthz").json() == {"ok": True}
            assert client.get("/api/memories").json() == []
        finally:
            release.set()
        assert pending.result(timeout=5).status_code == 500
        assert client.post("/search", json={"query": "now free"}).status_code == 200


@pytest.mark.parametrize("route", ["/api/memories", "/api/memories/m/segments", "/api/media/m"])
def test_metadata_backend_errors_are_sanitized(route, caplog):
    class BrokenStore:
        def all_rows(self):
            raise RuntimeError("private transcript or credentials")

        def memory_rows(self, *args):
            return self.all_rows()

    client = TestClient(create_app(store=BrokenStore(), scorer=lambda *_: 1, cfg=_cfg()),
                        raise_server_exceptions=False)
    response = client.get(route)
    assert response.status_code == 500
    assert response.json()["detail"] == "Request failed. Check the service logs and retry."
    assert "private transcript" not in caplog.text
