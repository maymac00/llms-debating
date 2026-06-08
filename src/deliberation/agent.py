"""CF agent, the ``agents/`` folder loader, and the capped intra-turn loop.

A turn is an *agent-decided, capped trajectory of Steps*: each iteration the
agent emits exactly one JSON action — either call a tool or finalise — and the
loop is bounded by ``max_calls`` backend (LLM) calls. Pure-function tool
executions do not count toward the cap. The loop always terminates with a
parsed proposal (a forced FINALISE on the last call guarantees it).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from .backends import Backend
from .models import Completion, Step, StepLabel, Transcript, Turn
from .tools import Tool, build_default_tools

logger = logging.getLogger(__name__)

DEFAULT_MAX_CALLS = 5


# --------------------------------------------------------------------------- #
# Action parsing — centralised in one helper.
# --------------------------------------------------------------------------- #
@dataclass
class ParsedAction:
    """The outcome of parsing a model reply into one structured action."""

    kind: Literal["tool", "final", "error"]
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    proposal: str | None = None
    justification: str | None = None
    error: str | None = None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first top-level JSON object from ``text``.

    Tolerates models that wrap JSON in prose or markdown fences by scanning for a
    balanced ``{ ... }`` span and parsing it.
    """
    stripped = text.strip()
    try:
        obj = json.loads(stripped)  # fast path: the whole reply is one JSON object
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    # Slow path: scan for the first balanced { ... }, ignoring braces in strings.
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:  # inside a string: only track escapes and the closing quote
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:  # balanced span closed: try to parse it
                    candidate = text[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break  # not valid JSON; retry from the next '{'
        start = text.find("{", start + 1)
    return None


def parse_action(text: str) -> ParsedAction:
    """Parse one model reply into a tool call, a finalise action, or an error."""
    obj = _extract_json_object(text)
    if obj is None:
        return ParsedAction(kind="error", error="no JSON object found")

    if "final" in obj:  # {"final": {"proposal": ..., "justification": ...}}
        final = obj["final"]
        if not isinstance(final, dict):
            return ParsedAction(kind="error", error="'final' must be an object")
        return ParsedAction(
            kind="final",
            proposal=str(final.get("proposal", "")),
            justification=str(final.get("justification", "")),
        )

    if obj.get("tool"):  # {"tool": "<name>", "input": {...}}
        tool_input = obj.get("input", {})
        if not isinstance(tool_input, dict):
            tool_input = {}  # ignore malformed input rather than crashing
        return ParsedAction(kind="tool", tool_name=str(obj["tool"]), tool_input=tool_input)

    return ParsedAction(kind="error", error="reply had neither 'tool' nor 'final'")


# --------------------------------------------------------------------------- #
# CFAgent
# --------------------------------------------------------------------------- #
class CFAgent:
    """An agent grounded in a Conceptual Framework, running the capped loop."""

    def __init__(
        self,
        cf_id: str,
        system_prompt: str,
        backend: Backend,
        *,
        tool_names: list[str] | None = None,
        max_calls: int = DEFAULT_MAX_CALLS,
    ) -> None:
        self.cf_id = cf_id
        self.system_prompt = system_prompt
        self.backend = backend
        # ``None`` means "all registered tools"; tools are built per-turn against
        # the live transcript, so only the allowed names are stored here.
        self.tool_names = tool_names
        self.max_calls = max_calls

    # -- tool binding ----------------------------------------------------
    def _build_tools(self, transcript: Transcript) -> list[Tool]:
        # Rebuilt each turn so the tools read the live transcript state.
        tools = build_default_tools(transcript)
        if self.tool_names is None:
            return tools  # all registered tools
        allowed = set(self.tool_names)
        return [t for t in tools if t.name in allowed]

    @property
    def tools(self) -> list[str] | None:
        """Names of tools available to this agent (``None`` = all)."""
        return self.tool_names

    # -- prompt construction --------------------------------------------
    def build_initial_messages(self, transcript: Transcript) -> list[dict[str, Any]]:
        """OpenAI-format setup: system prompt, shared context, and the task.

        The task user-message carries the scenario, the available tools (name,
        description, input schema), the finalise action, and the requirement to
        emit exactly one structured action per reply. The deliberation so far is
        included as messages via :meth:`Transcript.as_messages` (private steps
        excluded).
        """
        tools = self._build_tools(transcript)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        messages.extend(transcript.as_messages(self.cf_id))
        messages.append({"role": "user", "content": self._task_prompt(transcript, tools)})
        return messages

    def _task_prompt(self, transcript: Transcript, tools: list[Tool]) -> str:
        # Tool descriptions are injected here at loop setup, not into the static
        # system prompt — so the same agent can carry a different tool subset.
        tool_lines = "\n".join(
            f"- {t.name}: {t.description}\n  input schema: {json.dumps(t.schema)}"
            for t in tools
        )
        if not tool_lines:
            tool_lines = "(no tools available this turn)"
        n_prior = sum(len(rnd.turns) for rnd in transcript.rounds)
        if n_prior == 0:
            state = (
                "No one has spoken yet — you are opening the deliberation. There is "
                "no prior context, so make your opening proposal (the inspection "
                "tools would return nothing this round)."
            )
        else:
            state = (
                f"{n_prior} earlier turn(s) precede you, shown above as messages. "
                "Inspect them with the tools if it helps."
            )
        return (
            "You are taking your turn in a multi-agent policy deliberation. "
            "Reason from your conceptual framework.\n\n"
            "# Scenario\n"
            f"{transcript.scenario.strip()}\n\n"
            "# Deliberation so far\n"
            f"{state}\n\n"
            "# How to act\n"
            "Reply with EXACTLY ONE JSON object and nothing else. Choose one of:\n\n"
            "1. Call a tool to inspect the deliberation:\n"
            '   {"tool": "<name>", "input": { ... }}\n\n'
            "2. Finalise your contribution for this round:\n"
            '   {"final": {"proposal": "<one concrete policy, 1-2 sentences>", '
            '"justification": "<2-4 short points, grounded in your framework>"}}\n\n'
            "# Available tools\n"
            f"{tool_lines}\n\n"
            "Use tools to inform your decision and consider other agents' perspectives. When you "
            "finalise, the proposal and justification are shared with the other "
            "agents (your tool use and reasoning stay private).\n\n"
            "# Style — keep it tight and skimmable\n"
            "- Proposal: one concrete, specific policy.\n"
            '- Justification: Straightforward points, one line each as "- " bullets, each '
            "grounded in your framework; lead with the strongest.\n"
            "- Don't restate the scenario or re-quote other agents; but reference a prior "
            "point in a few words if you must.\n"
            "- Length is not quality — do "
            "NOT expand to match or out-do earlier turns. Brevity is never penalised."
        )

    # -- the capped, agent-decided loop ----------------------------------
    async def act(self, transcript: Transcript, round_idx: int) -> Turn:
        tools = self._build_tools(transcript)
        tools_by_name = {t.name: t for t in tools}
        messages = self.build_initial_messages(transcript)

        steps: list[Step] = []  # full intra-turn trajectory (llm + tool steps)
        proposal = ""
        justification = ""
        n_calls = 0  # backend (llm) calls so far; tool runs do not count
        cap_hit = False
        seen_tool_calls: set[tuple[str, str]] = set()  # (tool, input) already run this turn

        for call_index in range(self.max_calls):
            forced = call_index == self.max_calls - 1  # last allowed call
            if forced:
                # Force FINALISE and forbid tools so the turn always terminates.
                messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            "This is your final opportunity. You MUST finalise now. "
                            "Tools are not available. Reply ONLY with "
                            '{"final": {"proposal": "...", "justification": "..."}}'
                        ),
                    }
                ]

            sent = [dict(m) for m in messages]  # exact snapshot of this call's prompt
            t0 = time.perf_counter()
            completion = await self._generate(sent, round_idx, call_index)
            latency = time.perf_counter() - t0
            n_calls += 1

            action = parse_action(completion.text)  # tool / final / error
            base_meta: dict[str, Any] = {
                "latency_s": round(latency, 4),
                "call_index": call_index,
                "forced": forced,
            }

            # ----- finalise (volunteered) -----
            if action.kind == "final" and not forced:
                proposal = action.proposal or ""
                justification = action.justification or ""
                steps.append(
                    self._llm_step(StepLabel.FINALISE, sent, completion, base_meta, "ok")
                )
                logger.info("[%s r%d] finalised after %d call(s)", self.cf_id, round_idx, n_calls)
                break

            # ----- forced finalise (last call): final or fallback -----
            if forced:
                cap_hit = True
                if action.kind == "final":
                    proposal = action.proposal or ""
                    justification = action.justification or ""
                    status = "ok"
                else:
                    # Could not parse a clean final: keep the raw text as the
                    # proposal, leave justification empty (fallback).
                    proposal = completion.text.strip()
                    justification = ""
                    status = "fallback"
                meta = {**base_meta, "parse_status": status}
                steps.append(
                    self._llm_step(StepLabel.FINALISE, sent, completion, meta, status)
                )
                logger.info(
                    "[%s r%d] forced finalise (status=%s) at cap=%d",
                    self.cf_id,
                    round_idx,
                    status,
                    self.max_calls,
                )
                break

            # ----- tool call -----
            if action.kind == "tool" and action.tool_name in tools_by_name:
                label = StepLabel(action.tool_name)  # tool name == StepLabel value
                # Deciding llm Step, then the matching tool Step (same label).
                steps.append(self._llm_step(label, sent, completion, base_meta, "ok"))
                tool = tools_by_name[action.tool_name]
                tool_input = action.tool_input or {}
                tool_step, observation = self._run_tool(tool, label, tool_input)
                steps.append(tool_step)
                # The transcript is immutable during a turn, so an identical repeat
                # returns an identical result. Call that out so the agent stops
                # re-issuing the same query and moves on (a common round-1 loop).
                key = (action.tool_name, json.dumps(tool_input, sort_keys=True, ensure_ascii=False))
                if key in seen_tool_calls:
                    observation += (
                        "\n[note] You already ran this exact call this turn; the "
                        "result is identical and will stay so. Do not repeat it — "
                        "use a different input or finalise."
                    )
                seen_tool_calls.add(key)
                # Feed the result back as an observation and loop again.
                messages = messages + [{"role": "user", "content": observation}]
                continue

            # ----- parse failure / unknown tool (non-forced): re-prompt -----
            # No valid action was produced, so there is no "skill" label to
            # assign. We still record the llm Step (the trajectory must stay
            # replayable) and re-prompt with a correction; the loop still
            # advances toward the cap. We label it FINALISE — the only terminal
            # action in the vocabulary — and set metadata["parse_status"]="error"
            # so flow analysis can distinguish these recovery steps from genuine
            # finalisations.
            reason = (
                f"unknown tool '{action.tool_name}'"
                if action.kind == "tool"
                else (action.error or "could not parse a JSON action")
            )
            steps.append(
                self._llm_step(
                    StepLabel.FINALISE,
                    sent,
                    completion,
                    {**base_meta, "parse_status": "error", "error": reason},
                    "error",
                )
            )
            messages = messages + [
                {
                    "role": "user",
                    "content": (
                        f"Your last reply was not a valid action ({reason}). "
                        "Reply with EXACTLY ONE JSON object: either "
                        '{"tool": "<name>", "input": {...}} or '
                        '{"final": {"proposal": "...", "justification": "..."}}.'
                    ),
                }
            ]
            logger.debug("[%s r%d] corrective re-prompt: %s", self.cf_id, round_idx, reason)

        # Loop guaranteed to have set proposal (forced finalise on the last call).
        metadata = {"n_backend_calls": n_calls, "cap_hit": cap_hit, "max_calls": self.max_calls}
        return Turn(
            cf_id=self.cf_id,
            round_idx=round_idx,
            proposal=proposal,
            justification=justification,
            steps=steps,
            metadata=metadata,
        )

    async def _generate(
        self, messages: list[dict[str, Any]], round_idx: int, call_index: int
    ) -> Completion:
        logger.debug(
            "[%s r%d] llm call %d, messages=%r", self.cf_id, round_idx, call_index, messages
        )
        try:
            return await self.backend.generate(messages)
        except Exception as exc:  # add context, then surface
            raise RuntimeError(
                f"backend.generate failed for agent={self.cf_id} round={round_idx} "
                f"call_index={call_index}: {exc}"
            ) from exc

    def _llm_step(
        self,
        label: StepLabel,
        messages_sent: list[dict[str, Any]],
        completion: Completion,
        metadata: dict[str, Any],
        parse_status: str,
    ) -> Step:
        meta = dict(metadata)  # copy so callers' dicts aren't mutated
        meta.setdefault("parse_status", parse_status)
        meta.setdefault("backend", type(self.backend).__name__)
        return Step(
            kind="llm",
            label=label,
            messages_sent=messages_sent,
            completion=completion,
            metadata=meta,
        )

    def _run_tool(
        self, tool: Tool, label: StepLabel, tool_input: dict[str, Any]
    ) -> tuple[Step, str]:
        meta: dict[str, Any]
        try:
            result = tool.fn(**tool_input)
            meta = {"ok": True}
            observation = f"[tool {tool.name}] {result}"
        except Exception as exc:  # bad arguments etc.: record, re-prompt (don't drop)
            result = f"ERROR: {exc}"
            meta = {"ok": False, "error": str(exc)}
            observation = (
                f"[tool {tool.name}] {result}. Fix the input and try again, "
                "or finalise."
            )
        step = Step(
            kind="tool",
            label=label,
            tool_name=tool.name,
            tool_input=tool_input,
            tool_result=result,
            metadata=meta,
        )
        return step, observation


# --------------------------------------------------------------------------- #
# Folder loader
# --------------------------------------------------------------------------- #
def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split optional ``---`` YAML frontmatter from a markdown body."""
    if text.startswith("---"):
        parts = text.split("---", 2)  # ["", frontmatter, body]
        if len(parts) == 3:
            meta = yaml.safe_load(parts[1]) or {}
            if isinstance(meta, dict):
                return meta, parts[2].lstrip("\n")
    return {}, text  # no frontmatter: whole text is the body


def load_agent_meta(path: str | Path) -> dict[str, Any]:
    """Read an agent's ``agent.yaml`` (or return ``{}`` if absent)."""
    agent_yaml = Path(path) / "agent.yaml"
    if not agent_yaml.exists():
        return {}
    data = yaml.safe_load(agent_yaml.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _compose_skills(folder: Path, meta: dict[str, Any]) -> list[str]:
    """Return ordered skill bodies per ``agent.yaml.skills`` else lexical order."""
    skills_dir = folder / "skills"
    requested = meta.get("skills")
    bodies: list[str] = []
    if requested:  # explicit order from agent.yaml
        for name in requested:
            skill_file = skills_dir / f"{name}.md"
            if not skill_file.exists():
                raise FileNotFoundError(f"skill '{name}' not found at {skill_file}")
            bodies.append(skill_file.read_text(encoding="utf-8").strip())
    elif skills_dir.is_dir():  # fall back to all skills in lexical filename order
        for skill_file in sorted(skills_dir.glob("*.md")):
            bodies.append(skill_file.read_text(encoding="utf-8").strip())
    return bodies


def load_agent(path: str | Path, backend: Backend) -> CFAgent:
    """Build a :class:`CFAgent` from an ``agents/<name>/`` folder.

    System-prompt composition is pure and reproducible: the ``cf.md`` body,
    optionally followed by a ``## Capabilities`` section with the selected skill
    bodies in order. ``cf_id`` defaults to the folder name unless overridden in
    ``cf.md`` frontmatter or ``agent.yaml``. The ``backend`` is resolved by the
    caller (config overrides ``agent.yaml``) and injected here. Per-turn config
    (``max_calls``, tool subset) comes from ``agent.yaml`` and may be overridden
    by the caller afterward.
    """
    folder = Path(path)
    cf_file = folder / "cf.md"
    if not cf_file.exists():
        raise FileNotFoundError(f"missing cf.md in agent folder: {folder}")

    frontmatter, body = _split_frontmatter(cf_file.read_text(encoding="utf-8"))
    meta = load_agent_meta(folder)

    # Precedence: cf.md frontmatter > agent.yaml > folder name.
    cf_id = frontmatter.get("cf_id") or meta.get("cf_id") or folder.name

    skill_bodies = _compose_skills(folder, meta)
    system_prompt = body.strip()
    if skill_bodies:  # append the Capabilities section only when skills exist
        system_prompt += "\n\n## Capabilities\n" + "\n\n".join(skill_bodies)

    turn_cfg = meta.get("turn", {}) or {}
    max_calls = int(turn_cfg.get("max_calls", DEFAULT_MAX_CALLS))
    tool_names = turn_cfg.get("tools")  # None => all registered tools

    logger.info("loaded agent cf_id=%s from %s (max_calls=%d)", cf_id, folder, max_calls)
    return CFAgent(
        cf_id=cf_id,
        system_prompt=system_prompt,
        backend=backend,
        tool_names=tool_names,
        max_calls=max_calls,
    )
