# Multi-Agent CF Deliberation (Minimal Core)

Several LLM agents, each grounded in a distinct **Conceptual Framework (CF)**
(virtue ethics, utilitarianism, Confucianism, …), deliberate on a policy
scenario over `T` rounds. The run produces a **Transcript**: a structured,
losslessly serialisable record of the discussion, designed so it can later be
scored and replayed as a reinforcement-learning rollout without regeneration.

## Install

```bash
conda activate deliberation        # or any Python >= 3.11 environment
pip install -e .
# optional extras:
pip install -e ".[dev]"            # pytest, mypy, ruff
pip install -e ".[local,pretty]"  # vLLM/OpenAI client, rich render()
```

API backends read credentials from environment variables (e.g.
`OPENAI_API_KEY`). Keys are never hard-coded.

## Run

```bash
python -m deliberation.run --config configs/default.yaml
```

This loads the scenario, builds each `CFAgent`, runs a round-robin debate,
writes a JSONL transcript to `output:`, and prints `transcript.render()`.

## Concepts

- **Turn** — one agent's contribution in a round: a **single backend call**.
  The agent sees its CF system prompt (skills composed in), the full
  deliberation so far, and the task prompt, and replies with one JSON object
  (`{"proposal": ..., "justification": ...}`). An on-demand skill/tool loop is
  a deferred design — see `implement_skills.md`.
- **Step** — one recorded call. Every LLM Step keeps the exact
  `messages_sent` and its `Completion` (with optional `logprobs`/`token_ids`),
  so the flattened sequence of LLM Steps is a valid multi-step rollout.
  (`kind="tool"` Steps exist only in transcripts recorded by the pre-refactor
  tool loop.)
- **Transcript** — `scenario` + `rounds`; round-trips losslessly through
  `to_jsonl` / `from_jsonl`, including all Steps.

## Adding a CF

Drop a folder under `agents/` with a `cf.md` (optional YAML frontmatter for
`cf_id` / `display_name`), optional `agent.yaml` (backend, ordered skills),
and optional `skills/*.md` — skill bodies are composed into the system prompt,
so they are always present in the agent's context. Add a config entry. No core
code change is needed.

## Swapping a backend

Edit the agent's `backend:` block in the config (`provider: litellm` or
`provider: vllm`). No code change. Backends are assigned per agent, so
heterogeneous backends are supported (not required).

## Layout

```
src/deliberation/
  models.py     Completion, StepLabel, Step, Turn, Round, Transcript + utilities
  backends.py   Backend (Protocol), LiteLLMBackend, VLLMBackend
  agent.py      CFAgent (single-call turn) + folder loader + reply parsing
  protocols.py  DebateProtocol (Protocol), RoundRobin
  run.py        CLI entry point
```

## Scope

In scope: the deliberation core only. **Out of scope** (clean seams left):
voting/aggregation, scorers, RL training.
