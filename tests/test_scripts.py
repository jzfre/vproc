import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _run(script, url, query="release?"):
    env = {**os.environ, "VPROC_ASK_URL": url, "NO_PROXY": "127.0.0.1", "no_proxy": "127.0.0.1",
           "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"}
    return subprocess.run([str(SCRIPTS / script), query], env=env, text=True,
                          capture_output=True, timeout=5)


@pytest.fixture
def server():
    state = {"status": 200, "body": b"[]", "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            state["requests"].append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(state["body"])

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as httpd:
        worker = Thread(target=lambda: httpd.serve_forever(poll_interval=0.01))
        worker.start()
        try:
            yield f"http://127.0.0.1:{httpd.server_port}", state
        finally:
            httpd.shutdown()
            worker.join(timeout=5)


@pytest.mark.parametrize("script", ["ask.sh", "search.sh"])
@pytest.mark.parametrize("status", [409, 503, 504])
def test_scripts_show_server_error_detail_and_exit_nonzero(server, script, status):
    url, state = server
    state["status"] = status
    state["body"] = json.dumps({"detail": "Set a new VPROC_INDEX_PATH and re-ingest."}).encode()
    result = _run(script, url)
    assert result.returncode != 0
    assert result.stdout == ""
    assert "Set a new VPROC_INDEX_PATH and re-ingest." in result.stderr
    assert str(status) in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("script", ["ask.sh", "search.sh"])
def test_scripts_reject_non_json_response_without_traceback(server, script):
    url, state = server
    state["status"] = 502
    state["body"] = b"<html>upstream proxy failure</html>"
    result = _run(script, url)
    assert result.returncode != 0
    assert result.stdout == ""
    assert "502" in result.stderr
    assert "Traceback" not in result.stderr
    assert "<html>" not in result.stderr


@pytest.mark.parametrize("script", ["ask.sh", "search.sh"])
def test_scripts_report_unreachable_server_without_traceback(script):
    # Discover an unused loopback port, then close it so connection refusal is immediate.
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    result = _run(script, f"http://127.0.0.1:{port}")
    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr.strip()
    assert "Traceback" not in result.stderr


def test_ask_script_preserves_answer_format_and_json_quoting(server):
    url, state = server
    state["body"] = json.dumps({
        "answered": True, "abstained": False, "text": "Ships in July.",
        "claims": [{"text": "Ships in July", "citations": [{"memory_title": "standup", "start_ts": 12.8,
                                                            "speaker": "O'Brien"}]}],
    }).encode()
    query = 'Čo povedal "O\'Brien"?\n$HOME'
    result = _run("ask.sh", url, query)
    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout == "answered: True | abstained: False\nShips in July.\n  - Ships in July   [standup @ 12s (O'Brien)]\n"
    assert state["requests"] == [("/ask", {"question": query})]


def test_search_script_preserves_evidence_format(server):
    url, state = server
    state["body"] = json.dumps([{"start_ts": 10.2, "end_ts": 20.8, "speaker": "Tim", "text": "release"}]).encode()
    result = _run("search.sh", url)
    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout == "E1 [10-20s Tim] 'release'\n"
    assert state["requests"] == [("/search", {"query": "release?"})]
