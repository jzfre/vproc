import json
from contextlib import contextmanager

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import APIConnectionError, APITimeoutError

from vproc.config import Config, Endpoint
from vproc.errors import IndexBusyError, IndexCompatibilityError
from vproc.models import Answer, Evidence
from vproc.service import create_app


def _app(ask=None, search=None, store=None):
    ep = Endpoint("http://backend.invalid/v1", "offline-test")
    cfg = Config(ep, ep, ep, "unused", "0.0.0.0", 8765, 0.25, 0.5, None)
    return create_app(store=store if store is not None else object(), cfg=cfg, scorer=lambda *_: 1,
                      ask=ask or (lambda *a, **k: Answer(answered=False, abstained=True,
                                                       text="offline", claims=[], evidence=[])),
                      search=search or (lambda *a, **k: []))


@contextmanager
def _mcp_session(app):
    headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
    with TestClient(app) as client:
        response = client.post("/mcp", headers=headers, json={
            "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": "2025-06-18", "capabilities": {},
                "clientInfo": {"name": "offline-test", "version": "1"},
            },
        })
        assert response.status_code == 200
        headers["mcp-session-id"] = response.headers["mcp-session-id"]
        headers["mcp-protocol-version"] = "2025-06-18"
        response = client.post("/mcp", headers=headers,
                               json={"jsonrpc": "2.0", "method": "notifications/initialized"})
        assert response.status_code == 202
        request_id = 1

        def request(method, params=None):
            nonlocal request_id
            request_id += 1
            response = client.post("/mcp", headers=headers, json={
                "jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {},
            })
            assert response.status_code == 200
            messages = [json.loads(line.removeprefix("data: "))
                        for line in response.text.splitlines() if line.startswith("data: ")]
            return next(message["result"] for message in messages if message.get("id") == request_id)

        yield request


def test_mcp_tools_expose_and_apply_all_rest_filters():
    def answer(store, cfg, question, scorer, where=None):
        return Answer(answered=True, abstained=False, text=where or "unfiltered", claims=[], evidence=[])

    def search(store, cfg, query, where=None):
        return [Evidence(key="E1", segment_id="1", memory_title="offline", start_ts=0,
                         end_ts=1, speaker="S", text=where or "unfiltered")]

    with _mcp_session(_app(ask=answer, search=search)) as request:
        listed = request("tools/list")["tools"]
        assert {tool["name"] for tool in listed} == {"ask_memory_tool", "search_memory_tool"}
        for tool in listed:
            assert {"project", "memory", "speaker"} <= set(tool["inputSchema"]["properties"])
        expected = "project_id = 'demo' AND memory_id = 'Tim''s réunion' AND speaker = 'O''Brien'"
        for name, field in [("ask_memory_tool", "question"), ("search_memory_tool", "query")]:
            result = request("tools/call", {"name": name, "arguments": {
                field: "release?", "project": "demo", "memory": "Tim's réunion", "speaker": "O'Brien",
            }})
            assert result["isError"] is False
            assert expected in result["content"][0]["text"]


@pytest.mark.parametrize("transport", ["rest", "mcp"])
def test_filters_restrict_real_index_and_sql_payload_cannot_broaden_results(tmp_path, transport):
    from vproc.store.lancedb_store import Store

    store = Store(str(tmp_path / "index"))
    row = {"id": "wanted", "project_id": "demo", "memory_id": "Tim's réunion", "speaker": "O'Brien",
           "start_ts": 0.0, "end_ts": 1.0, "said_text": "release", "on_screen_text": "",
           "embed_text": "release", "source_video": "offline.mp4", "frame_path": "", "vector": [1.0, 0.0]}
    store.add([row, {**row, "id": "other", "project_id": "other"}], embedding_model="offline-test")

    def search(store, cfg, query, where=None):
        rows = store.vector_search([1.0, 0.0], 8, where=where, embedding_model=cfg.embed.model)
        return [Evidence(key=r["id"], segment_id=r["id"], memory_title=r["memory_id"], start_ts=r["start_ts"],
                         end_ts=r["end_ts"], speaker=r["speaker"], text=r["said_text"]) for r in rows]

    app = _app(search=search, store=store)
    filters = {"query": "release", "project": "demo", "memory": "Tim's réunion", "speaker": "O'Brien"}
    if transport == "rest":
        with TestClient(app) as client:
            hits = client.post("/search", json=filters).json()
            assert [hit["segment_id"] for hit in hits] == ["wanted"]
            assert client.post("/search", json={**filters, "project": "demo' OR '1'='1"}).json() == []
    else:
        with _mcp_session(app) as request:
            args = {"name": "search_memory_tool", "arguments": filters}
            result = request("tools/call", args)
            assert result["isError"] is False
            hits = json.loads(result["content"][0]["text"])
            assert hits["segment_id"] == "wanted"
            args["arguments"] = {**filters, "project": "demo' OR '1'='1"}
            result = request("tools/call", args)
            assert result["isError"] is False
            assert result["content"] == []


@pytest.mark.parametrize("name, field", [("ask_memory_tool", "question"), ("search_memory_tool", "query")])
def test_mcp_query_and_filter_size_boundaries_are_accepted(name, field):
    with _mcp_session(_app()) as request:
        result = request("tools/call", {"name": name, "arguments": {
            field: "x" * 8192, "project": "p" * 128, "memory": "m" * 128, "speaker": "s" * 128,
        }})
        assert result["isError"] is False


@pytest.mark.parametrize("name, field", [("ask_memory_tool", "question"), ("search_memory_tool", "query")])
def test_mcp_invalid_inputs_never_reach_backend_and_session_recovers(name, field):
    backend_calls = []

    def backend(*args, **kwargs):
        backend_calls.append(args)
        return (Answer(answered=False, abstained=True, text="unexpected call", claims=[], evidence=[])
                if name == "ask_memory_tool" else [])

    with _mcp_session(_app(ask=backend, search=backend)) as request:
        invalid = [{}, {field: ""}, {field: " \t\n"}, {field: "x" * 8193}, {field: ["q"]}]
        for filter_name in ("project", "memory", "speaker"):
            invalid.extend({field: "q", filter_name: value}
                           for value in ("x" * 129, "a\nb", "a\x85b"))
        for arguments in invalid:
            result = request("tools/call", {"name": name, "arguments": arguments})
            assert result["isError"] is True
            assert backend_calls == []
        assert len(request("tools/list")["tools"]) == 2


@pytest.mark.parametrize("name, field", [("ask_memory_tool", "question"), ("search_memory_tool", "query")])
@pytest.mark.parametrize("kind, detail", [("timeout", "timed out"), ("connection", "unavailable"),
                                         ("unexpected", "failed"), ("incompatible", "re-ingest"),
                                         ("busy", "retry")])
def test_mcp_backend_failures_are_safe_and_session_recovers(name, field, kind, detail, caplog):
    backend_request = httpx.Request("POST", "http://backend.invalid/v1", headers={"Authorization": "secret-token"})
    errors = {
        "timeout": APITimeoutError(request=backend_request),
        "connection": APIConnectionError(message="secret-token response", request=backend_request),
        "unexpected": RuntimeError("secret-token response"),
        "incompatible": IndexCompatibilityError("secret-token index metadata"),
        "busy": IndexBusyError("secret-token index path"),
    }
    pending_error = errors[kind]

    def backend(*args, **kwargs):
        nonlocal pending_error
        if pending_error is not None:
            error, pending_error = pending_error, None
            raise error
        return (Answer(answered=False, abstained=True, text="recovered", claims=[], evidence=[])
                if name == "ask_memory_tool" else [])

    with _mcp_session(_app(ask=backend, search=backend)) as request:
        args = {"name": name, "arguments": {field: "release?"}}
        result = request("tools/call", args)
        assert result["isError"] is True
        assert detail in result["content"][0]["text"].lower()
        assert "secret-token" not in json.dumps(result)
        assert request("tools/call", args)["isError"] is False
    assert "secret-token" not in caplog.text
