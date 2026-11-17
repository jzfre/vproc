import asyncio

import pytest
from fastapi.testclient import TestClient

from vproc.config import Config, Endpoint
from vproc.http_limits import MAX_QUERY_BODY_BYTES, QueryBodyLimit
from vproc.service import create_app


@pytest.mark.parametrize("path", ["/ask", "/search", "/mcp", "/mcp/"])
def test_oversized_query_body_is_rejected_before_parsing_or_model_work(path):
    ep = Endpoint("u", "m")
    cfg = Config(ep, ep, ep, "unused", "127.0.0.1", 8765, .25, .5, None)

    def must_not_call(*a, **kw):
        pytest.fail("oversized request reached model")

    app = create_app(store=object(), scorer=must_not_call, cfg=cfg, ask=must_not_call,
                     search=must_not_call)
    response = TestClient(app).post(path, content=b" " * (MAX_QUERY_BODY_BYTES + 1),
                                    headers={"Content-Type": "application/json"})
    assert response.status_code == 413
    assert response.json()["detail"] == "Query request body is too large."


@pytest.mark.parametrize("too_large", [False, True])
def test_chunked_bodies_are_bounded_and_accepted_body_is_replayed(too_large):
    sent, called, reads = [], [], []
    messages = iter([
        {"type": "http.request", "body": b"1234", "more_body": True},
        {"type": "http.request", "body": b"56" if too_large else b"5", "more_body": False},
    ])

    async def receive():
        reads.append(True)
        return next(messages)

    async def send(message):
        sent.append(message)

    async def application(scope, receive, send):
        called.append(await receive())

    scope = {"type": "http", "method": "POST", "path": "/ask", "headers": []}
    asyncio.run(QueryBodyLimit(application, max_bytes=5)(scope, receive, send))
    assert len(reads) == 2
    if too_large:
        assert called == [] and sent[0]["status"] == 413
    else:
        assert called == [{"type": "http.request", "body": b"12345", "more_body": False}]


def test_mounted_app_still_limits_body_before_model_work():
    from fastapi import FastAPI

    ep = Endpoint("u", "m")
    cfg = Config(ep, ep, ep, "unused", "127.0.0.1", 8765, .25, .5, None)
    outer = FastAPI()
    outer.mount("/vproc", create_app(store=object(), cfg=cfg, scorer=lambda *a: 1,
                                     search=lambda *a, **kw: []))
    response = TestClient(outer).post("/vproc/search", json={
        "query": "q", "padding": "x" * MAX_QUERY_BODY_BYTES,
    })
    assert response.status_code == 413
