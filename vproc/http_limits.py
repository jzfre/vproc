"""Bound query bodies before JSON parsing; media streaming is unaffected."""
from starlette.responses import JSONResponse

MAX_QUERY_BODY_BYTES = 256 * 1024


class QueryBodyLimit:
    def __init__(self, app, max_bytes=MAX_QUERY_BODY_BYTES):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        path, root = scope.get("path", ""), scope.get("root_path", "")
        if root and (path == root or path.startswith(root + "/")):
            path = path[len(root):]
        if (scope["type"] != "http" or scope.get("method") != "POST"
                or path.rstrip("/") not in {"/ask", "/search", "/mcp"}):
            return await self.app(scope, receive, send)
        # Do not trust Content-Length: a streaming/chunked caller can omit it.
        # Buffer at most the cap, then replay to FastAPI/MCP's ordinary parser.
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body = message.get("body", b"")
            size += len(body)
            if size > self.max_bytes:
                return await JSONResponse(
                    {"detail": "Query request body is too large."}, status_code=413,
                )(scope, receive, send)
            chunks.append(body)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        consumed = False

        async def replay():
            nonlocal consumed
            if not consumed:
                consumed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)
