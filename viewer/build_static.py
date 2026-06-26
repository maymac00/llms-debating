#!/usr/bin/env python3
"""Bake one transcript into a single, self-contained HTML report.

The output embeds the transcript JSONL (and the agents' display names) inline, so
it needs no server, no network, and no separate data file — open it with a
double-click (``file://``) or send it as a single email attachment.

    python viewer/build_static.py outputs/transcript.jsonl          # -> outputs/transcript.html
    python viewer/build_static.py outputs/transcript.jsonl -o out.html

It reuses ``viewer/index.html`` as the template — the same rendering code as the
live viewer — so the static report can't drift from it. The only difference is a
small ``<script>`` injected ahead of the viewer's own, which pre-sets
``window.__TRANSCRIPT__`` / ``window.__AGENTS__``; the viewer renders those
immediately and skips its server-discovery path. Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VIEWER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VIEWER_DIR))  # reuse the server's display-name logic, no drift

from serve import agent_names  # noqa: E402


def _js(obj: object) -> str:
    """JSON-encode for embedding inside a ``<script>``.

    ``</`` is neutralised so transcript text can never accidentally close the
    surrounding ``<script>`` tag (the classic inline-JSON XSS/escape footgun).
    """
    return json.dumps(obj).replace("</", "<\\/")


def build(transcript_jsonl: str, template: str, names: dict[str, str]) -> str:
    """Return the template with the transcript + names baked in before its script."""
    inject = (
        "<script>\n"
        f"window.__TRANSCRIPT__ = {_js(transcript_jsonl)};\n"
        f"window.__AGENTS__ = {_js(names)};\n"
        "</script>\n"
    )
    idx = template.find("<script>")  # anchor: the viewer's own (first) script tag
    if idx == -1:
        raise SystemExit("template viewer/index.html has no <script> tag to anchor the injection")
    return template[:idx] + inject + template[idx:]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a self-contained HTML transcript report.")
    ap.add_argument("transcript", help="path to a transcript .jsonl")
    ap.add_argument(
        "-o", "--output", help="output .html path (default: the input path with a .html suffix)"
    )
    args = ap.parse_args()

    src = Path(args.transcript)
    if not src.is_file():
        raise SystemExit(f"no such transcript: {src}")

    template = (VIEWER_DIR / "index.html").read_text(encoding="utf-8")
    html = build(src.read_text(encoding="utf-8"), template, agent_names())

    out = Path(args.output) if args.output else src.with_suffix(".html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size // 1024} KiB) — open with file:// or email it")


if __name__ == "__main__":
    main()
