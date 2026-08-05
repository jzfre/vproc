import dataclasses

from fastapi.testclient import TestClient
from vproc.config import Config, Endpoint
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


def test_root_serves_ui_and_api_wins():
    app = create_app(store=_UIStore([]), scorer=lambda p, h: 1.0, cfg=_cfg())
    c = TestClient(app)
    r = c.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert c.get("/healthz").json() == {"ok": True}  # routes still beat the mount
