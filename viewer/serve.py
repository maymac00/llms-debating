#!/usr/bin/env python3
"""Standalone server for the deliberation viewer.

Serves ``viewer/index.html`` plus two tiny discovery endpoints so the page can
list the transcripts in ``outputs/`` and label agents by their display name.

Stdlib only — no dependency on the ``deliberation`` package. Run it and a
browser opens on the viewer:

    python viewer/serve.py
    python viewer/serve.py --port 9000 --no-browser

The viewer also works fully offline (open ``index.html`` and drag a ``.jsonl``
file in); this server just removes the manual step.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

VIEWER_DIR = Path(__file__).resolve().parent
ROOT = VIEWER_DIR.parent  # project root: serves /viewer and /outputs from here


def list_transcripts() -> list[dict[str, str]]:
    """Every ``*.jsonl`` under ``outputs/``, newest first, as {name, url}."""
    out = ROOT / "outputs"
    if not out.is_dir():
        return []
    files = sorted(out.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    items = []
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        items.append({"name": p.relative_to(out).as_posix(), "url": "/" + rel})
    return items


_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)


def agent_names() -> dict[str, str]:
    """Map cf_id -> display_name from ``agents/*/cf.md`` YAML frontmatter."""
    names: dict[str, str] = {}
    agents = ROOT / "agents"
    if not agents.is_dir():
        return names
    for cf in agents.glob("*/cf.md"):
        cf_id = cf.parent.name
        display = None
        try:
            m = _FRONTMATTER.match(cf.read_text(encoding="utf-8"))
        except OSError:
            m = None
        if m:
            for line in m.group(1).splitlines():
                key, _, val = line.partition(":")
                key, val = key.strip(), val.strip().strip("'\"")
                if key == "cf_id" and val:
                    cf_id = val
                elif key == "display_name" and val:
                    display = val
        names[cf_id] = display or cf_id
    return names


class Handler(SimpleHTTPRequestHandler):
    def _json(self, payload: object) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self) -> None:
        # This is a dev server: never let the browser cache the viewer or the
        # transcripts, so an edited index.html or a freshly-judged transcript
        # (e.g. one that just gained a verdict) always shows up on reload.
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        if self.path in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", "/viewer/index.html")
            self.end_headers()
            return
        if self.path == "/api/transcripts":
            return self._json(list_transcripts())
        if self.path == "/api/agents":
            return self._json(agent_names())
        return super().do_GET()

    def log_message(self, *args) -> None:  # keep the console quiet
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve the deliberation viewer.")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    handler = partial(Handler, directory=str(ROOT))
    port = args.port
    for _ in range(20):  # find a free port if the default is taken
        try:
            httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
            break
        except OSError:
            port += 1
    else:
        raise SystemExit("could not bind a port in range")

    url = f"http://127.0.0.1:{port}/viewer/index.html"
    print(f"Deliberation viewer: {url}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
