import os
import re

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from vproc.answer.ask import ask_memory, search_memory
from vproc.config import load_config

# Filter values are interpolated into a LanceDB SQL-style .where() string. Memory/speaker
# ids are arbitrary filename stems / diarized names, so accept any short string and just
# neutralize the SQL-string breakout by doubling single quotes (as Store.delete_memory
# does). Reject only control characters and over-length values with HTTP 400.


class Query(BaseModel):
    question: str | None = None
    query: str | None = None
    project: str | None = None
    memory: str | None = None
    speaker: str | None = None


def _safe(value: str, field: str) -> str:
    if len(value) > 128 or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise HTTPException(status_code=400, detail=f"invalid {field} filter")
    return value.replace("'", "''")  # escape single quotes for the SQL-style filter


def build_where(q: Query) -> str | None:
    clauses = []
    if q.project:
        clauses.append(f"project_id = '{_safe(q.project, 'project')}'")
    if q.memory:
        clauses.append(f"memory_id = '{_safe(q.memory, 'memory')}'")
    if q.speaker:
        clauses.append(f"speaker = '{_safe(q.speaker, 'speaker')}'")
    return " AND ".join(clauses) if clauses else None


_VIDEO_TYPES = {"mp4": "video/mp4", "m4v": "video/mp4", "mov": "video/quicktime",
                "mkv": "video/x-matroska", "webm": "video/webm"}


def create_app(store=None, scorer=None, cfg=None, ask=ask_memory, search=search_memory) -> FastAPI:
    cfg = cfg or load_config()
    if store is None:
        from vproc.store.lancedb_store import Store
        store = Store(cfg.index_path)
    if scorer is None:
        from vproc.answer.faithfulness import HHEM
        scorer = HHEM(cfg.hhem_model).score

    mcp_app, lifespan = _build_mcp(store, cfg, scorer, ask, search)
    app = FastAPI(title="vproc", lifespan=lifespan)

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.post("/ask")
    def ask_route(q: Query):
        question = q.question or ""
        if not question.strip():
            raise HTTPException(status_code=400, detail="question is required")
        answer = ask(store, cfg, question, scorer, where=build_where(q))
        return answer.model_dump()

    @app.post("/search")
    def search_route(q: Query):
        query = q.query or q.question or ""
        if not query.strip():
            raise HTTPException(status_code=400, detail="query is required")
        evidence = search(store, cfg, query, where=build_where(q))
        return [e.model_dump() for e in evidence]

    @app.get("/api/memories")
    def api_memories():
        by_mem: dict[str, list[dict]] = {}
        for r in store.all_rows():
            if r.get("project_id", "default") == "default":
                by_mem.setdefault(r["memory_id"], []).append(r)
        return [
            {"memory_id": mid, "segment_count": len(rows),
             "duration_s": max(r["end_ts"] for r in rows),
             "speakers": sorted({r["speaker"] for r in rows})}
            for mid, rows in sorted(by_mem.items())
        ]

    @app.get("/api/memories/{memory_id}/segments")
    def api_segments(memory_id: str):
        # _safe validates length/control-chars; its escaped return isn't used here —
        # memory_rows filters in Python (no SQL), so escaping would break ids with quotes.
        _safe(memory_id, "memory")
        rows = store.memory_rows(memory_id, "default")
        if not rows:
            raise HTTPException(status_code=404, detail=f"no memory '{memory_id}'")
        rows.sort(key=lambda r: r["start_ts"])
        return [
            {"start_ts": r["start_ts"], "end_ts": r["end_ts"], "speaker": r["speaker"],
             "said_text": r["said_text"], "on_screen_text": r["on_screen_text"],
             "frame_name": os.path.basename(r["frame_path"]) if r.get("frame_path") else None}
            for r in rows
        ]

    # memory_id -> source_video path. source_video for a memory never changes within a
    # serve process (re-ingest replaces rows but keeps the same path), so a plain dict
    # with no invalidation is safe here and avoids a full store scan on every Range
    # request a <video> element makes while seeking.
    media_path_cache: dict[str, str] = {}

    @app.get("/api/media/{memory_id}")
    def api_media(memory_id: str):
        _safe(memory_id, "memory")
        path = media_path_cache.get(memory_id)
        if path is None:
            rows = store.memory_rows(memory_id, "default")
            if not rows:
                raise HTTPException(status_code=404, detail=f"no memory '{memory_id}'")
            path = media_path_cache[memory_id] = rows[0]["source_video"]
        if not os.path.exists(path):
            raise HTTPException(status_code=404,
                                detail=f"video file not found: {path} "
                                       "(ingested from a different directory?)")
        ctype = _VIDEO_TYPES.get(path.rsplit(".", 1)[-1].lower(), "application/octet-stream")
        from starlette.responses import FileResponse
        return FileResponse(path, media_type=ctype)

    @app.get("/api/frames/{memory_id}/{name}")
    def api_frame(memory_id: str, name: str):
        # The character-class regex alone is NOT sufficient: it allows dots, so a name of
        # ".." or "..png" (no slash needed) passes it — the explicit ".." check is what
        # actually blocks those. memory_id can't contain "/" either, but a bare ".."
        # segment (e.g. from "%2e%2e") would still resolve one level up via
        # os.path.join, so it gets the same explicit check.
        if not re.fullmatch(r"[A-Za-z0-9._-]+", name) or ".." in name or ".." in memory_id:
            raise HTTPException(status_code=404, detail="bad frame name")
        path = os.path.join(cfg.frames_dir, "default", memory_id, name)
        if not os.path.isfile(path):
            raise HTTPException(status_code=404, detail="no such frame")
        from starlette.responses import FileResponse
        return FileResponse(path, media_type="image/png")

    if mcp_app is not None:
        app.mount("/mcp", mcp_app)

        # Starlette's Mount only matches "/mcp/..." (with trailing slash); a bare "/mcp"
        # request normally falls through to the router's automatic redirect-slashes
        # handling. The catch-all static mount below matches everything (it's a Mount
        # at "/"), so it would swallow a bare "/mcp" as a FULL match before that
        # fallback ever runs, turning it into a 405 from StaticFiles. Reproduce the
        # redirect explicitly, registered (and thus matched) before the catch-all.
        @app.api_route("/mcp", methods=["GET", "POST", "DELETE"], include_in_schema=False)
        async def _mcp_trailing_slash_redirect():
            from starlette.responses import RedirectResponse
            return RedirectResponse(url="/mcp/", status_code=307)

    # Static UI mount LAST so every route registered above wins over it.
    # check_dir=False: a missing ui/ dir (e.g. a stripped-down install) shouldn't take
    # down the whole API at import time — requests under it just 404 instead.
    ui_dir = os.path.join(os.path.dirname(__file__), "ui")
    from starlette.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=ui_dir, html=True, check_dir=False), name="ui")

    return app


def _build_mcp(store, cfg, scorer, ask, search):
    """Build the MCP streamable-HTTP sub-app plus a FastAPI lifespan that runs its
    session manager (mounted sub-app lifespans are never executed otherwise). The tools
    are async and offload the blocking ask/search calls so they don't stall the event
    loop. Best-effort: on any import/setup failure, return (None, None) and REST still
    works with no lifespan requirement."""
    try:
        from contextlib import asynccontextmanager

        import anyio
        from mcp.server.fastmcp import FastMCP
        from mcp.server.transport_security import TransportSecuritySettings

        mcp = FastMCP("vproc")

        @mcp.tool()
        async def ask_memory_tool(question: str) -> dict:
            answer = await anyio.to_thread.run_sync(lambda: ask(store, cfg, question, scorer))
            return answer.model_dump()

        @mcp.tool()
        async def search_memory_tool(query: str) -> list:
            hits = await anyio.to_thread.run_sync(lambda: search(store, cfg, query))
            return [e.model_dump() for e in hits]

        mcp.settings.streamable_http_path = "/"  # mounted at /mcp -> endpoint is /mcp
        # This is a LAN service bound to 0.0.0.0 and reached by hostname, so the default
        # localhost-only DNS-rebinding guard would 421 every remote MCP client.
        mcp.settings.transport_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
        mcp_app = mcp.streamable_http_app()

        @asynccontextmanager
        async def lifespan(app):
            async with mcp.session_manager.run():
                yield

        return mcp_app, lifespan
    except Exception:
        return None, None
