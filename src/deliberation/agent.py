"""CF agent and the ``agents/`` folder loader.

A turn is a **single backend call**: the agent sees its CF system prompt (with
skills composed in), the full deliberation so far (via
:meth:`Transcript.as_messages`), and the task prompt; it replies with one JSON
object carrying its proposal and justification. There is no intra-turn tool
loop — see ``implement_skills.md`` for the deferred on-demand design.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import yaml

from .backends import Backend
from .models import Completion, Step, StepLabel, Transcript, Turn

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Reply parsing — centralised in one helper.
# --------------------------------------------------------------------------- #
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


def parse_proposal(text: str) -> tuple[str, str] | None:
    """Parse a model reply into ``(proposal, justification)``, or ``None``.

    Accepts the flat shape ``{"proposal": ..., "justification": ...}`` and, for
    tolerance, the legacy wrapper ``{"final": {...}}``.
    """
    obj = _extract_json_object(text)
    if obj is None:
        return None
    if isinstance(obj.get("final"), dict):  # legacy wrapper
        obj = obj["final"]
    if "proposal" in obj:
        return str(obj.get("proposal", "")), str(obj.get("justification", ""))
    return None


# --------------------------------------------------------------------------- #
# CFAgent
# --------------------------------------------------------------------------- #
class CFAgent:
    """An agent grounded in a Conceptual Framework; one backend call per turn."""

    def __init__(self, cf_id: str, system_prompt: str, backend: Backend) -> None:
        self.cf_id = cf_id
        self.system_prompt = system_prompt
        self.backend = backend

    # -- prompt construction --------------------------------------------
    def build_messages(self, transcript: Transcript) -> list[dict[str, Any]]:
        """OpenAI-format prompt: system prompt, shared context, and the task.

        The deliberation so far is included via :meth:`Transcript.as_messages`
        (proposal + justification only — private steps excluded).
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        messages.extend(transcript.as_messages(self.cf_id))
        messages.append({"role": "user", "content": self._task_prompt(transcript)})
        return messages

    def _task_prompt(self, transcript: Transcript) -> str:
        n_prior = sum(len(rnd.turns) for rnd in transcript.rounds)
        state = (
            "No one has spoken yet — you are opening the deliberation."
            if n_prior == 0
            else f"{n_prior} earlier turn(s) precede you, shown above as messages."
        )
        return (
            "You are taking your turn in a multi-agent policy deliberation. "
            "Reason from your conceptual framework.\n\n"
            "# Scenario\n"
            f"{transcript.scenario.strip()}\n\n"
            "# Deliberation so far\n"
            f"{state}\n\n"
            "# How to deliberate\n"
            "- Engage the strongest point made against your position, not the weakest.\n"
            "- Revise your proposal only if an argument genuinely persuades you — and say "
            "in a few words what moved you. Hold your position when your framework demands "
            "it: honest disagreement is more useful here than easy convergence, and the "
            "fact that others agree is not itself evidence that they are right.\n"
            "- Treat figures with care. If another agent asserts a number, ask whether it "
            "is sourced or assumed before building on it. Never state an invented "
            "statistic as fact — give your reasoning, mark estimates as estimates.\n\n"
            "# How to reply\n"
            "Reply with EXACTLY ONE JSON object and nothing else:\n"
            '{"proposal": "<one concrete policy, 1-2 sentences>", '
            '"justification": "<2-4 short points, grounded in your framework>"}\n\n'
            "Your proposal and justification are shared with the other agents.\n\n"
            "# Style — keep it tight and skimmable\n"
            "- Proposal: one concrete, specific policy.\n"
            '- Justification: straightforward points, one line each as "- " bullets, each '
            "grounded in your framework; lead with whatever this round actually turns on, "
            "not a fixed template.\n"
            "- Don't restate the scenario or re-quote other agents; reference a prior "
            "point in a few words if you must.\n"
            "- Length is not quality — do "
            "NOT expand to match or out-do earlier turns. Brevity is never penalised."
        )

    # -- the single-call turn ---------------------------------------------
    async def act(self, transcript: Transcript, round_idx: int) -> Turn:
        messages = self.build_messages(transcript)
        sent = [dict(m) for m in messages]  # exact snapshot of the prompt sent

        t0 = time.perf_counter()
        completion = await self._generate(sent, round_idx)
        latency = time.perf_counter() - t0

        parsed = parse_proposal(completion.text)
        if parsed is not None:
            proposal, justification = parsed
            parse_status = "ok"
        else:
            # No clean JSON: keep the raw text as the proposal so the turn (and
            # the trajectory) is never dropped.
            proposal, justification = completion.text.strip(), ""
            parse_status = "fallback"
            logger.warning("[%s r%d] unparseable reply; raw text kept", self.cf_id, round_idx)

        step = Step(
            kind="llm",
            label=StepLabel.FINALISE,
            messages_sent=sent,
            completion=completion,
            metadata={
                "latency_s": round(latency, 4),
                "parse_status": parse_status,
                "backend": type(self.backend).__name__,
            },
        )
        return Turn(
            cf_id=self.cf_id,
            round_idx=round_idx,
            proposal=proposal,
            justification=justification,
            steps=[step],
            metadata={"n_backend_calls": 1, "parse_status": parse_status},
        )

    async def _generate(self, messages: list[dict[str, Any]], round_idx: int) -> Completion:
        try:
            return await self.backend.generate(messages)
        except Exception as exc:  # add context, then surface
            raise RuntimeError(
                f"backend.generate failed for agent={self.cf_id} round={round_idx}: {exc}"
            ) from exc


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
    bodies in order — skills are always present in the agent's context.
    ``cf_id`` defaults to the folder name unless overridden in ``cf.md``
    frontmatter or ``agent.yaml``. The ``backend`` is resolved by the caller
    (config overrides ``agent.yaml``) and injected here.
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

    logger.info("loaded agent cf_id=%s from %s", cf_id, folder)
    return CFAgent(cf_id=cf_id, system_prompt=system_prompt, backend=backend)
