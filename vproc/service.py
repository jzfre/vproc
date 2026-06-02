import re

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from vproc.answer.ask import ask_memory, search_memory
from vproc.config import load_config

# Filter values are interpolated into a LanceDB SQL-style .where() string, so they must
# never contain control characters (notably a single quote, which would break out of the
# clause). Restrict to a safe, generous charset and reject anything else with HTTP 400.
_FILTER_RE = re.compile(r"[A-Za-z0-9_\-. ]{1,128}")


class Query(BaseModel):
    question: str | None = None
    query: str | None = None
    project: str | None = None
    memory: str | None = None
    speaker: str | None = None


def _safe(value: str, field: str) -> str:
    if not _FILTER_RE.fullmatch(value):
        raise HTTPException(status_code=400, detail=f"invalid {field} filter")
    return value


def build_where(q: Query) -> str | None:
    clauses = []
    if q.project:
        clauses.append(f"project_id = '{_safe(q.project, 'project')}'")
    if q.memory:
        clauses.append(f"memory_id = '{_safe(q.memory, 'memory')}'")
    if q.speaker:
        clauses.append(f"speaker = '{_safe(q.speaker, 'speaker')}'")
    return " AND ".join(clauses) if clauses else None


def create_app(store=None, scorer=None, cfg=None, ask=ask_memory, search=search_memory) -> FastAPI:
    cfg = cfg or load_config()
    if store is None:
        from vproc.store.lancedb_store import Store
        store = Store(cfg.index_path)
    if scorer is None:
        from vproc.answer.faithfulness import HHEM
        scorer = HHEM().score

    app = FastAPI(title="vproc")

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.post("/ask")
    def ask_route(q: Query):
        answer = ask(store, cfg, q.question or "", scorer, where=build_where(q))
        return answer.model_dump()

    @app.post("/search")
    def search_route(q: Query):
        evidence = search(store, cfg, q.query or q.question or "", where=build_where(q))
        return [e.model_dump() for e in evidence]

    _mount_mcp(app, store, cfg, scorer, ask, search)
    return app


def _mount_mcp(app, store, cfg, scorer, ask, search) -> None:
    """Expose the same logic as MCP tools at /mcp. Best-effort: if the installed
    mcp SDK's mounting API differs, REST still works and this is a no-op."""
    try:
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("vproc")

        @mcp.tool()
        def ask_memory_tool(question: str) -> dict:
            return ask(store, cfg, question, scorer).model_dump()

        @mcp.tool()
        def search_memory_tool(query: str) -> list:
            return [e.model_dump() for e in search(store, cfg, query)]

        app.mount("/mcp", mcp.streamable_http_app())
    except Exception:
        pass
