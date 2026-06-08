# Deliberation Viewer

A small, standalone web page for reading a finished debate the way it happened:
top to bottom, round by round, one message per agent turn. It shows only who is
speaking and what they proposed and argued — the agents' private reasoning steps
are kept out of the view (it does note how many model/tool calls a turn took).

It reads the JSONL transcripts written to `outputs/` (see `Transcript.to_jsonl`).
Nothing here imports the `deliberation` package; it's intentionally separate.

## Run it

```bash
python viewer/serve.py
```

This serves the page and opens it in your browser. It auto-discovers every
`*.jsonl` under `outputs/` (newest first) into the dropdown, and labels agents by
their `display_name` from `agents/*/cf.md`. Options:

```bash
python viewer/serve.py --port 9000   # pick a port (auto-bumps if taken)
python viewer/serve.py --no-browser  # don't auto-open
```

## Or fully offline

Open `viewer/index.html` directly in a browser and drag a `.jsonl` transcript
onto the page (or use *Open file…*). No server, no dependencies. Agent display
names aren't available this way, so speakers are labelled by their `cf_id`.

## Build a portable HTML file

To turn one transcript into a single self-contained `.html` file (scenario plus
every turn's proposal + justification, rendered exactly as the server shows
them), bake it in:

```bash
python viewer/build_static.py outputs/transcript.jsonl          # -> outputs/transcript.html
python viewer/build_static.py outputs/transcript.jsonl -o out.html
```

The result embeds the transcript and the agents' `display_name`s inline and
needs no server or network — share it or open it with `file://`. It reuses
`index.html` as the template, so the output can't drift from the live viewer.
