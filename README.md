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

- **Turn** — one agent's contribution in a round: an *agent-decided, capped
  trajectory of Steps*. Each iteration the agent chooses to call a tool (e.g.
  search the transcript) or to finalise its proposal, up to `max_calls` backend
  calls (default 5).
- **Step** — one call within a turn: an LLM generation (`kind="llm"`) or a
  pure-function tool execution (`kind="tool"`). Every LLM Step keeps the exact
  `messages_sent` and its `Completion` (with optional `logprobs`/`token_ids`),
  so the flattened sequence of LLM Steps is a valid multi-step rollout.
- **Transcript** — `scenario` + `rounds`; round-trips losslessly through
  `to_jsonl` / `from_jsonl`, including all Steps.

## Adding a CF

Drop a folder under `agents/` with a `cf.md` (optional YAML frontmatter for
`cf_id` / `display_name`), optional `agent.yaml` (backend, skills, turn config),
and optional `skills/*.md`. Add a config entry. No core code change is needed.

## Swapping a backend

Edit the agent's `backend:` block in the config (`provider: litellm` or
`provider: vllm`). No code change. Backends are assigned per agent, so
heterogeneous backends are supported (not required).

## Layout

```
src/deliberation/
  models.py     Completion, StepLabel, Step, Turn, Round, Transcript + utilities
  backends.py   Backend (Protocol), LiteLLMBackend, VLLMBackend
  tools.py      Tool + default pure-function transcript tools
  agent.py      CFAgent + folder loader + intra-turn loop + action parsing
  protocols.py  DebateProtocol (Protocol), RoundRobin
  run.py        CLI entry point
```

## Scope

In scope: the deliberation core only. **Out of scope** (clean seams left):
voting/aggregation, scorers, RL training.
