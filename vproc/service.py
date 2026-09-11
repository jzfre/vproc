import logging
import os
import re
import unicodedata
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from openai import APIError, APITimeoutError
from pydantic import BaseModel

from vproc.answer.ask import ask_memory, search_memory
from vproc.config import load_config
from vproc.errors import IndexBusyError, IndexCompatibilityError

logger = logging.getLogger(__name__)

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
    if len(value) > 128 or any(unicodedata.category(c) == "Cc" for c in value):
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


def _query_text(value: str | None, field: str) -> str:
    if value is None or not value.strip():
        raise HTTPException(status_code=400, detail=f"{field} is required")
    if len(value) > 8192:
        raise HTTPException(status_code=400, detail=f"{field} must be at most 8192 characters")
    return value


def _backend(call):
    """Keep backend exceptions, response bodies, and credentials out of both APIs."""
    try:
        return call()
    except Exception as exc:
        if isinstance(exc, IndexCompatibilityError):
            status, detail = 409, (
                "Index embedding identity is missing or incompatible with the configured model. "
                "Use the original embedding model, or set a new VPROC_INDEX_PATH and re-ingest."
            )
        elif isinstance(exc, IndexBusyError):
            status, detail = 503, "Index is being updated. Retry after the current operation completes."
        elif isinstance(exc, APITimeoutError):
            status, detail = 504, "Model backend timed out. Check the backend and retry."
        elif isinstance(exc, APIError):
            status, detail = 503, "Model backend unavailable. Check its service and configuration, then retry."
        else:
            status, detail = 500, "Request failed. Check the service logs and retry."
        # SDK exception messages can contain response bodies or authenticated URLs.
        # MCP also logs surfaced tool errors, so only expose these fixed messages.
        logger.warning("Request failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=status, detail=detail) from None


_VIDEO_TYPES = {"mp4": "video/mp4", "m4v": "video/mp4", "mov": "video/quicktime",
                "mkv": "video/x-matroska", "webm": "video/webm"}


def create_app(store=None, scorer=None, cfg=None, ask=ask_memory, search=search_memory) -> FastAPI:
    cfg = cfg or load_config()
    if store is None:
        from vproc.store.lancedb_store import Store
        store = Store(cfg.index_path)
    if scorer is None:
        verifier = None
        verifier_lock = Lock()

        def scorer(premise, hypothesis):
            nonlocal verifier
            # Browsing/search need no local verifier. Serialize both its first load
            # and inference because REST and MCP may score on different threads.
            with verifier_lock:
                if verifier is None:
                    from vproc.answer.faithfulness import HHEM
                    verifier = HHEM(cfg.hhem_model)
                return verifier.score(premise, hypothesis)

    def ask_route(q: Query):
        question = _query_text(q.question, "question")
        where = build_where(q)
        return _backend(lambda: ask(store, cfg, question, scorer, where=where).model_dump())

    def search_route(q: Query):
        query = _query_text(q.query if q.query is not None else q.question, "query")
        where = build_where(q)
        return _backend(lambda: [e.model_dump() for e in search(store, cfg, query, where=where)])

    mcp_app, lifespan = _build_mcp(ask_route, search_route)
    app = FastAPI(title="vproc", lifespan=lifespan)
    app.post("/ask")(ask_route)
    app.post("/search")(search_route)

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.get("/api/status")
    def api_status():
        status = {"mcp_available": mcp_app is not None, "index_compatible": None}
        validate = getattr(store, "validate_embedding_model", None)
        if validate is not None:
            try:
                _backend(lambda: validate(cfg.embed.model))
                status["index_compatible"] = True
            except HTTPException as exc:
                if exc.status_code == 409:
                    status["index_compatible"] = False
                status["detail"] = exc.detail
        return status

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
        # memory_rows escapes its own SQL values; pass the original identifier.
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

    @app.get("/api/media/{memory_id}")
    def api_media(memory_id: str):
        _safe(memory_id, "memory")
        rows = store.memory_rows(memory_id, "default")
        if not rows:
            raise HTTPException(status_code=404, detail=f"no memory '{memory_id}'")
        path = rows[0]["source_video"]
        if not os.path.isfile(path):
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
        root = (Path(cfg.frames_dir).expanduser() / "default").resolve()
        path = (root / memory_id / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
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
    # An API-only install should return 404 for UI paths. check_dir=False merely
    # defers StaticFiles' missing-directory exception until the first request.
    ui_dir = os.path.join(os.path.dirname(__file__), "ui")
    from starlette.staticfiles import StaticFiles
    if os.path.isdir(ui_dir):
        app.mount("/", StaticFiles(directory=ui_dir, html=True), name="ui")

    return app


def _build_mcp(ask, search):
    """Build the MCP streamable-HTTP sub-app plus a FastAPI lifespan that runs its
    session manager (mounted sub-app lifespans are never executed otherwise). The tools
    are async and offload the blocking ask/search calls so they don't stall the event
    loop. Best-effort: on any import/setup failure, return (None, None) and REST still
    works with no lifespan requirement."""
    try:
        from contextlib import asynccontextmanager

        import anyio
        from mcp.server.fastmcp import FastMCP
        from mcp.server.fastmcp.exceptions import ToolError
        from mcp.server.transport_security import TransportSecuritySettings

        mcp = FastMCP("vproc")

        async def run_tool(call, q):
            try:
                return await anyio.to_thread.run_sync(lambda: call(q))
            except HTTPException as exc:
                raise ToolError(exc.detail) from None

        @mcp.tool()
        async def ask_memory_tool(question: str, project: str | None = None,
                                  memory: str | None = None, speaker: str | None = None) -> dict:
            return await run_tool(ask, Query(question=question, project=project, memory=memory, speaker=speaker))

        @mcp.tool()
        async def search_memory_tool(query: str, project: str | None = None,
                                     memory: str | None = None, speaker: str | None = None) -> list:
            return await run_tool(search, Query(query=query, project=project, memory=memory, speaker=speaker))

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
    except Exception as exc:
        logger.warning("MCP unavailable after setup failure (%s); REST remains available", type(exc).__name__)
        return None, None
