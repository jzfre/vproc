#!/usr/bin/env python3
"""Exercise the bundled UI in real headless Chromium against synthetic media/API data.

Optional developer check (FFmpeg and Chromium system libraries must be installed;
on Linux, Playwright's `install-deps chromium` command installs browser libraries):
    uv run --no-project --with playwright python -m playwright install chromium
    uv run --no-project --with playwright python scripts/smoke_ui.py

No model calls, private media, or application index are used. Playwright is optional;
it is not an application dependency. Use --screenshot /tmp/vproc-ui.png to keep a capture.
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import tempfile
from threading import Thread
from urllib.parse import unquote, urlsplit

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
HOSTILE = '<img src=x onerror="window.__unexpected_script = true">'
MEMORIES = [
    {"memory_id": title, "duration_s": 6, "speakers": ["Ada"], "segment_count": 1}
    for title in ("alpha", "beta")
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="vproc-ui-smoke-") as directory:
        media = Path(directory) / "synthetic.webm"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=5:duration=6",
            "-c:v", "libvpx", "-b:v", "100k", "-an", str(media),
        ], check=True, timeout=30)
        requests = []
        errors = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send(self, data, media_type="application/json", status=200):
                payload = json.dumps(data).encode() if media_type == "application/json" else data
                self.send_response(status)
                self.send_header("Content-Type", media_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                path = unquote(urlsplit(self.path).path)
                if path == "/api/memories":
                    self.send(MEMORIES)
                elif path in ("/api/memories/alpha/segments", "/api/memories/beta/segments"):
                    self.send([{"start_ts": 1, "end_ts": 5, "speaker": "Ada", "said_text": HOSTILE,
                                "on_screen_text": "Release in July", "frame_name": None}])
                elif path in ("/api/media/alpha", "/api/media/beta"):
                    self.send(media.read_bytes(), "video/webm")
                elif path in ("/", "/app.js", "/style.css"):
                    name = "index.html" if path == "/" else path[1:]
                    mime = {"index.html": "text/html", "app.js": "text/javascript", "style.css": "text/css"}[name]
                    self.send((ROOT / "vproc" / "ui" / name).read_bytes(), mime)
                else:
                    self.send({"detail": "not found"}, status=404)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append((self.path, body))
                cite = {"memory_title": "beta", "start_ts": 2, "end_ts": 5, "speaker": "Ada"}
                if self.path == "/search":
                    self.send([{**cite, "text": "Release in July", "similarity": 0.9, "keyword_match": True}])
                elif self.path == "/ask":
                    self.send({"answered": True, "abstained": False, "claims": [
                        {"text": HOSTILE, "citations": [cite]}]})
                else:
                    self.send({"detail": "not found"}, status=404)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                try:
                    page = browser.new_page(viewport={"width": 1280, "height": 900})
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    page.goto(f"http://127.0.0.1:{server.server_port}/")
                    expect(page.locator("#memory-picker")).to_have_value("alpha")
                    expect(page.locator(".seg-text")).to_have_text(HOSTILE)
                    assert page.locator("#transcript img").count() == 0
                    page.wait_for_function("document.getElementById('video').readyState >= 1")
                    assert page.locator("#video").evaluate("v => v.duration") >= 5.9
                    page.locator(".seg-head").click()
                    page.wait_for_function("document.getElementById('video').currentTime >= 1")
                    page.locator("#video").evaluate("v => v.pause()")
                    expect(page.locator("#now-speaking")).to_contain_text("Ada")

                    page.select_option("#qa-mode", "search")
                    page.fill("#qa-input", "release")
                    page.click("#qa-go")
                    expect(page.locator(".hit")).to_have_count(1)
                    page.locator(".hit").click()
                    expect(page.locator("#memory-picker")).to_have_value("beta")
                    page.wait_for_function("document.getElementById('video').readyState >= 1 && "
                                           "document.getElementById('video').currentTime >= 2")
                    page.locator("#video").evaluate("v => v.pause()")

                    page.select_option("#qa-mode", "ask")
                    page.fill("#qa-input", "When is the release?")
                    page.click("#qa-go")
                    expect(page.locator(".claim-text")).to_have_text(HOSTILE)
                    assert page.locator("#qa-results img").count() == 0
                    assert page.evaluate("window.__unexpected_script") is None
                    assert [route for route, _ in requests] == ["/search", "/ask"], requests
                    assert all(body.get("project") == "default" for _, body in requests), requests
                    assert not errors, errors
                    if args.screenshot:
                        page.screenshot(path=str(args.screenshot), full_page=True)
                    print("PASS: Chromium media playback/seek, cross-meeting citation, search/ask project scope, "
                          "escaped transcript/model text, no JavaScript errors")
                finally:
                    browser.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    main()
