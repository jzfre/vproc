#!/usr/bin/env python3
"""Smoke-test an installed release, without using repository code or real models.

Example (from a checkout, after building the wheel):
    uv run --no-project --python 3.12 --with ./dist/vproc-0.1.0-py3-none-any.whl \
        python scripts/smoke_release.py

Requires ffmpeg and ffprobe on PATH; --http-only skips synthetic media ingestion.
All application data is temporary. Only loopback HTTP endpoints are used.
"""

import argparse
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
from threading import Thread
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener


TEXT = "Release smoke test: the meeting ships in September."
SOURCE_ROOT = Path(__file__).resolve().parents[1]
HTTP = build_opener(ProxyHandler({}))

# -I excludes the working directory, PYTHONPATH, and user site-packages. Validate
# the loaded module against distribution metadata before loading its real CLI entry.
BOOTSTRAP = """
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import vproc
dist = metadata.distribution('vproc')
module = Path(vproc.__file__).resolve()
expected = Path(dist.locate_file('vproc/__init__.py')).resolve()
editable = json.loads(dist.read_text('direct_url.json') or '{}').get('dir_info', {}).get('editable')
if editable or module != expected or module.is_relative_to(Path(sys.argv[1]) / 'vproc'):
    raise SystemExit('Smoke test requires an installed, non-editable vproc release; source import rejected.')
if len(sys.argv) == 2:
    print(json.dumps({'version': dist.version, 'module': str(module)}))
else:
    sys.argv = ['vproc', *sys.argv[2:]]
    entry = next(e for e in dist.entry_points if e.group == 'console_scripts' and e.name == 'vproc')
    entry.load()()
"""


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def cli(*args):
    return [sys.executable, "-I", "-c", BOOTSTRAP, str(SOURCE_ROOT), *args]


def run(command, cwd, env, timeout=60):
    result = subprocess.run(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=timeout)
    check(result.returncode == 0,
          f"Command failed: {' '.join(command[:2])}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def request(base, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = Request(base + path, data=data, headers={"Content-Type": "application/json"})
    with HTTP.open(req, timeout=5) as response:
        return response.headers.get_content_type(), response.read()


def request_json(base, path, payload=None):
    content_type, body = request(base, path, payload)
    check(content_type == "application/json", f"{path}: expected JSON, got {content_type}")
    return json.loads(body)


class ModelHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/v1/chat/completions":
            body = {"id": "smoke", "object": "chat.completion", "created": 0,
                    "model": "smoke-model", "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": TEXT}}]}
        elif self.path == "/v1/embeddings":
            body = {"object": "list", "model": "smoke-model", "data": [
                {"object": "embedding", "index": i, "embedding": [1.0, 0.0, 0.0]}
                for i, _ in enumerate(payload["input"])]}
        else:
            self.send_error(404)
            return
        encoded = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *args):
        pass


@contextmanager
def service(cwd, env):
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    with (cwd / "server.log").open("w+") as log:
        process = subprocess.Popen(cli("serve"), cwd=cwd, env={**env, "VPROC_PORT": str(port)},
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 45
            while True:
                check(process.poll() is None, "Server exited before becoming healthy")
                try:
                    if request_json(base, "/healthz") == {"ok": True}:
                        break
                except (URLError, TimeoutError, ConnectionError):
                    pass
                check(time.monotonic() < deadline, "Server did not become healthy within 45 seconds")
                time.sleep(0.1)
            yield base
        except BaseException:
            log.flush()
            log.seek(0)
            print(log.read()[-12000:], file=sys.stderr)
            raise
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def verify_http(base):
    for path, types, marker in (("/", {"text/html"}, b"<title>vproc</title>"),
                                ("/app.js", {"text/javascript", "application/javascript"}, b"fetch("),
                                ("/style.css", {"text/css"}, b"body")):
        content_type, body = request(base, path)
        check(content_type in types and marker in body, f"Missing or invalid packaged UI asset: {path}")
    status = request_json(base, "/api/status")
    check(status.get("mcp_available") is True, f"MCP setup failed: {status}")
    check(status.get("index_compatible") is True, f"Index compatibility check failed: {status}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--http-only", action="store_true", help="skip the FFmpeg ingest/persistence/search checks")
    args = parser.parse_args()
    if not args.http_only:
        check(shutil.which("ffmpeg") and shutil.which("ffprobe"),
              "ffmpeg and ffprobe must be on PATH (or use --http-only for UI/API checks only)")
    with tempfile.TemporaryDirectory(prefix="vproc-release-") as temporary:
        cwd = Path(temporary)
        env = {key: value for key, value in os.environ.items()
               if not key.upper().startswith("VPROC_") and key.upper() not in
               {"HF_TOKEN", "PYTHONPATH", "PYTHONHOME"}}
        env.update(VPROC_INDEX_PATH=str(cwd / "index.lance"), VPROC_FRAMES_DIR=str(cwd / "frames"),
                   VPROC_HOST="127.0.0.1", VPROC_DIARIZE_MODEL="", VPROC_SPEAKER_NAMING="off",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", NO_PROXY="127.0.0.1", no_proxy="127.0.0.1")
        with ThreadingHTTPServer(("127.0.0.1", 0), ModelHandler) as models:
            worker = Thread(target=models.serve_forever, daemon=True)
            worker.start()
            try:
                for endpoint in ("OCR", "EMBED", "GROUNDING", "TRANSCRIBE"):
                    env[f"VPROC_{endpoint}_BASE_URL"] = f"http://127.0.0.1:{models.server_port}/v1"
                    env[f"VPROC_{endpoint}_MODEL"] = "smoke-model"
                package = json.loads(run(cli(), cwd, env))
                print(f"Installed vproc {package['version']}: {package['module']}", flush=True)
                with service(cwd, env) as base:
                    verify_http(base)
                    check(request_json(base, "/api/memories") == [], "Fresh catalogue is not empty")
                print("PASS: isolated startup, packaged HTML/JS/CSS, empty catalogue, status and MCP setup", flush=True)
                if not args.http_only:
                    video = cwd / "release-smoke.mp4"
                    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-f", "lavfi",
                         "-i", "color=c=blue:s=64x64:r=10:d=2.5", "-c:v", "mpeg4", "-an", str(video)], cwd, env)
                    output = run(cli("ingest", str(video)), cwd, env, timeout=120)
                    check("ingested 1 segments" in output, f"Unexpected ingest result: {output}")
                    with service(cwd, env) as base:
                        verify_http(base)
                        memories = request_json(base, "/api/memories")
                        check(len(memories) == 1 and memories[0]["memory_id"] == video.stem
                              and memories[0]["segment_count"] == 1, f"Ingest did not persist: {memories}")
                        segments = request_json(base, f"/api/memories/{video.stem}/segments")
                        check(len(segments) == 1 and segments[0]["on_screen_text"] == TEXT,
                              f"OCR text did not persist: {segments}")
                        frame = segments[0]["frame_name"]
                        check(request(base, f"/api/frames/{video.stem}/{frame}")[1].startswith(b"\x89PNG"),
                              "Persisted frame is unavailable")
                        check(request(base, f"/api/media/{video.stem}")[1] == video.read_bytes(),
                              "Persisted video is unavailable")
                        evidence = request_json(base, "/search", {"query": "September", "memory": video.stem})
                        check(len(evidence) == 1 and TEXT in evidence[0]["text"]
                              and evidence[0]["memory_title"] == video.stem, f"Search failed: {evidence}")
                    print("PASS: synthetic FFmpeg ingest, OCR/embedding HTTP, restart persistence, media and search", flush=True)
            finally:
                models.shutdown()
                worker.join(timeout=5)


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
