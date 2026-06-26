"""CF agent and the ``agents/`` folder loader.

A turn is a **single backend call**. The agent sees its CF system prompt (with
skills composed in), the full deliberation so far (via
:meth:`Transcript.as_messages`), and a task prompt whose shape depends on the
round:

* **Round 0 — PROPOSE.** The reply is one JSON object with the four Toulmin
  fields (``claim`` / ``grounds`` / ``cf_warrant`` / ``qualifier``).
* **Rounds ≥ 1 — CRITIQUE/DEFEND/REVISE (a "clash" turn).** One JSON object
  carrying the typed clash triple: ``critiques`` (each anchored to a verbatim
  quote of the target — the Direct Clash rule), ``defenses`` of the prior round's
  critiques, and a forced ``revision``.

There is no intra-turn *tool* loop. When a clash reply fails validation it is
re-sampled in place (bounded rejection sampling — the chosen hard-schema design),
which is regeneration of the same logical action, not a multi-step trajectory; an
unparseable/invalid reply still never drops the turn (it falls back to raw text).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import yaml

from .backends import Backend
from .models import (
    TOULMIN_COMPONENTS,
    AttackType,
    Completion,
    Critique,
    Defense,
    Revision,
    Step,
    StepLabel,
    ToulminProposal,
    Transcript,
    Turn,
)

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


def parse_toulmin(text: str) -> ToulminProposal | None:
    """Parse a round-0 reply into a :class:`ToulminProposal`, or ``None``.

    Requires all four Toulmin fields; tolerant of prose/fence wrapping via
    :func:`_extract_json_object`.
    """
    obj = _extract_json_object(text)
    if obj is None:
        return None
    if isinstance(obj.get("final"), dict):  # tolerate the legacy wrapper
        obj = obj["final"]
    if not all(k in obj for k in TOULMIN_COMPONENTS):
        return None
    return ToulminProposal(
        claim=str(obj["claim"]),
        grounds=str(obj["grounds"]),
        cf_warrant=str(obj["cf_warrant"]),
        qualifier=str(obj["qualifier"]),
    )


def _normalise_ws(text: str) -> str:
    return " ".join(text.split())


def is_verbatim(quoted: str, source: str | None) -> bool:
    """True if ``quoted`` is a (whitespace-/case-normalised) substring of ``source``.

    The Direct Clash anchor: a critique must quote a real span of the target's
    component. Normalising whitespace and case keeps the check usable without
    letting an agent paraphrase past it.
    """
    if source is None:
        return False
    needle = _normalise_ws(quoted).casefold()
    return bool(needle) and needle in _normalise_ws(source).casefold()


class ClashRejected(Exception):
    """A clash reply failed validation; the ``reason`` is fed back for re-sampling."""


# Shared output-style guidance, applied uniformly across agents so no CF reads as
# more verbose or ornate than another. Plain wording is required; framework
# vocabulary stays in the system prompt, not banned here. Kept in the shared task
# prompt (not per-agent skills) so the rule has a single source.
_STYLE_PLAIN = (
    "# Style — plain and easy to follow\n"
    "- Write in plain, everyday language and short sentences. Say it the simplest way "
    "that is still precise.\n"
    "- Technical terms from your OWN framework are fine where they carry real meaning; "
    "keep the rest of the wording simple, so anyone can follow you and the text is easy "
    "to read.\n"
    "- Don't reach for fancier or more complicated words to sound stronger — a clear "
    "point beats an ornate one. Length is not quality.\n"
    "- Use bullet points when you have several distinct points; they make you easier to "
    "read and to answer. Not required — use them where they help."
)

# Clash-only addition: lets agents reference a numbered point instead of re-quoting
# it, without weakening the Direct Clash anchor (the `quoted` field still needs the
# exact words).
_STYLE_NUMBERING = (
    "- You may number your points. You can refer to another agent's numbered point by "
    'its number (e.g. "Mill\'s point 2") instead of restating it in full — but a '
    "critique's `quoted` field still needs the exact words copied verbatim."
)


# --------------------------------------------------------------------------- #
# CFAgent
# --------------------------------------------------------------------------- #
class CFAgent:
    """An agent grounded in a Conceptual Framework; one backend call per turn."""

    def __init__(self, cf_id: str, system_prompt: str, backend: Backend) -> None:
        self.cf_id = cf_id
        self.system_prompt = system_prompt
        self.backend = backend

    # Bounded rejection sampling for clash turns: on a validation failure the
    # reason is fed back and the turn is re-sampled, up to this many extra calls.
    MAX_CLASH_RETRIES = 2

    # -- the single-call turn ---------------------------------------------
    async def act(self, transcript: Transcript, round_idx: int) -> Turn:
        """One backend call. Round 0 is a blind PROPOSE; rounds ≥ 1 are clash turns."""
        if round_idx == 0:
            return await self._propose(transcript, round_idx)
        return await self._clash(transcript, round_idx)

    def _build_messages(self, transcript: Transcript, task_prompt: str) -> list[dict[str, Any]]:
        """OpenAI-format prompt: system prompt, shared context, and the round's task.

        The deliberation so far is included via :meth:`Transcript.as_messages`
        (shared substance only — private steps excluded).
        """
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        messages.extend(transcript.as_messages(self.cf_id))
        messages.append({"role": "user", "content": task_prompt})
        return messages

    # -- round 0: blind Toulmin PROPOSE -----------------------------------
    async def _propose(self, transcript: Transcript, round_idx: int) -> Turn:
        sent = self._build_messages(transcript, self._propose_task_prompt(transcript))
        t0 = time.perf_counter()
        completion = await self._generate(sent, round_idx)
        latency = time.perf_counter() - t0

        toulmin = parse_toulmin(completion.text)
        if toulmin is None:  # never drop the turn: keep the raw text
            return self._fallback_turn(round_idx, completion, sent, 1, StepLabel.PROPOSE)

        step = self._llm_step(StepLabel.PROPOSE, sent, completion, "ok", latency)
        return Turn(
            cf_id=self.cf_id,
            round_idx=round_idx,
            proposal=toulmin.claim,
            justification=f"Warrant: {toulmin.cf_warrant} (Scope: {toulmin.qualifier})",
            toulmin=toulmin,
            steps=[step],
            metadata={"n_backend_calls": 1, "parse_status": "ok"},
        )

    def _propose_task_prompt(self, transcript: Transcript) -> str:
        return (
            "You are OPENING a multi-agent policy deliberation. This is the blind round: "
            "you cannot see the other agents yet, and they cannot see you. State your "
            "opening position, reasoning from your conceptual framework.\n\n"
            "# Scenario\n"
            f"{transcript.scenario.strip()}\n\n"
            "# Your task — a Toulmin-structured proposal\n"
            "Reply with EXACTLY ONE JSON object and nothing else:\n"
            '{"claim": "<one concrete, debatable policy, 1-2 sentences>", '
            '"grounds": "<the evidence or factual basis>", '
            '"cf_warrant": "<why the grounds license the claim, routed through your '
            "framework's core value>\", "
            '"qualifier": "<the scope/limit — where or for whom the claim does NOT hold>"}\n\n'
            "- The cf_warrant is what makes the proposal *yours* rather than generic — make "
            "the framework value explicit.\n"
            "- State the qualifier honestly: it is a real boundary (a natural attack "
            "surface later), not a hedge.\n"
            "- Treat figures with care: never state an invented statistic as fact; mark "
            "estimates as estimates.\n\n"
            f"{_STYLE_PLAIN}"
        )

    # -- rounds ≥ 1: CRITIQUE / DEFEND / REVISE ----------------------------
    async def _clash(self, transcript: Transcript, round_idx: int) -> Turn:
        base = self._build_messages(transcript, self._clash_task_prompt(transcript, round_idx))
        messages = base
        n_calls = 0
        completion: Completion | None = None
        sent: list[dict[str, Any]] = base
        latency = 0.0
        for attempt in range(self.MAX_CLASH_RETRIES + 1):
            sent = [dict(m) for m in messages]  # exact snapshot of this attempt's prompt
            t0 = time.perf_counter()
            completion = await self._generate(sent, round_idx)
            latency = time.perf_counter() - t0
            n_calls += 1
            try:
                critiques, defenses, revision = self._validate_clash(
                    completion.text, transcript, round_idx
                )
            except ClashRejected as rej:
                logger.warning(
                    "[%s r%d] clash rejected (attempt %d/%d): %s",
                    self.cf_id, round_idx, attempt + 1, self.MAX_CLASH_RETRIES + 1, rej,
                )
                # Re-sample with the failure fed back; keep the base prompt stable.
                messages = [
                    *base,
                    {"role": "assistant", "content": completion.text},
                    {"role": "user", "content": self._correction(str(rej))},
                ]
                continue
            return self._build_clash_turn(
                transcript, round_idx, critiques, defenses, revision,
                sent, completion, n_calls, latency,
            )

        # Exhausted retries: keep the raw text so the turn is never dropped.
        assert completion is not None
        return self._fallback_turn(round_idx, completion, sent, n_calls, StepLabel.CRITIQUE)

    def _clash_task_prompt(self, transcript: Transcript, round_idx: int) -> str:
        positions = self._positions_block(transcript)
        open_block = self._open_critiques_block(transcript, round_idx)
        return (
            "You are in a CLASH round. Your turn has three parts, in order: attack, then "
            "defend, then update.\n\n"
            "# Positions on the table (engage them — quote exactly)\n"
            f"{positions}\n\n"
            "# Critiques filed against you last round (answer each, or it counts as conceded)\n"
            f"{open_block}\n\n"
            "# How to engage — modest, not maximal\n"
            "- Attack the strongest version of a position, not a strawman, and quote the "
            "exact words you dispute.\n"
            "- Persuade on others' terms where you honestly can — point to ground their "
            "framework values too — and contest plainly where you cannot. Forced agreement "
            "is worse than honest disagreement; that others agree is not evidence they are "
            "right. Engage, don't perform hostility.\n"
            "- Treat figures with care: ask whether a number is sourced or assumed before "
            "building on it.\n\n"
            "# Reply — EXACTLY ONE JSON object and nothing else\n"
            "{\n"
            '  "critiques": [{"target_cf": "<another agent>", '
            '"target_component": "claim|grounds|cf_warrant|qualifier", '
            '"quoted": "<verbatim span copied EXACTLY from that component above>", '
            '"attack_type": "contest_grounds|contest_warrant|cf_value_conflict", '
            '"argument": "<your attack, grounded in your framework>"}],\n'
            '  "defenses": [{"critique_ref": "<id from the list above>", '
            '"rebuttal": "<your reply>"}],\n'
            '  "revision": {"delta": "<what changed and why, or null>", '
            '"revised_claim": "<new claim if delta, else null>", '
            '"rationale": "<if delta is null, why no critique warranted a change>"}\n'
            "}\n\n"
            "Rules:\n"
            "- At least one critique, each quoting a real span verbatim — a misquote is "
            "rejected and you will be asked again.\n"
            "- attack_type: contest_grounds = the evidence is wrong/insufficient/"
            "inapplicable; contest_warrant = grounds accepted but they do not license the "
            "claim; cf_value_conflict = coherent on its own terms but violates a value YOUR "
            "framework holds primary.\n"
            "- Answer every critique against you by its id, or it is recorded as conceded.\n"
            "- The revision is forced: either change your claim (non-null delta + "
            "revised_claim) or assert positively that no critique warranted a change "
            "(null delta + a real rationale). Silence is not an option.\n\n"
            f"{_STYLE_PLAIN}\n"
            f"{_STYLE_NUMBERING}"
        )

    def _positions_block(self, transcript: Transcript) -> str:
        lines: list[str] = []
        for cf_id in transcript.latest_proposals():
            if cf_id == self.cf_id:
                continue
            toulmin = transcript.latest_toulmin(cf_id)
            claim = transcript.current_claim(cf_id)
            if claim is None:
                continue
            lines.append(f"[{cf_id}] claim: {claim}")
            if toulmin is not None:
                lines.append(f"        grounds: {toulmin.grounds}")
                lines.append(f"        cf_warrant: {toulmin.cf_warrant}")
                lines.append(f"        qualifier: {toulmin.qualifier}")
        return "\n".join(lines) if lines else "(no other positions yet)"

    def _open_critiques_block(self, transcript: Transcript, round_idx: int) -> str:
        opens = transcript.open_critiques_against(self.cf_id, round_idx)
        if not opens:
            return "None — no one critiqued you last round."
        return "\n".join(
            f'[{c.ref}] (your {c.target_component}) "{c.quoted}" — {c.argument}' for c in opens
        )

    def _validate_clash(
        self, text: str, transcript: Transcript, round_idx: int
    ) -> tuple[list[Critique], list[Defense], Revision]:
        """Parse and validate a clash reply, or raise :class:`ClashRejected`.

        Enforces the Direct Clash rule (every critique quotes a real span of its
        target verbatim), a mandatory non-empty critique, and the forced revision.
        Invalid defenses are dropped (not fatal); everything else regenerates.
        """
        obj = _extract_json_object(text)
        if obj is None:
            raise ClashRejected("reply was not a JSON object")

        targets = {
            cf
            for cf in transcript.latest_proposals()
            if cf != self.cf_id and transcript.current_claim(cf) is not None
        }
        raw_crits = obj.get("critiques")
        if not isinstance(raw_crits, list):
            raise ClashRejected('reply is missing a "critiques" list')
        critiques: list[Critique] = []
        for i, item in enumerate(raw_crits):
            if not isinstance(item, dict):
                raise ClashRejected("each critique must be a JSON object")
            target_cf = str(item.get("target_cf", ""))
            component = str(item.get("target_component", ""))
            quoted = str(item.get("quoted", ""))
            argument = str(item.get("argument", ""))
            if target_cf not in targets:
                raise ClashRejected(
                    f'target_cf "{target_cf}" is not another agent on the table; '
                    f"valid targets: {sorted(targets)}"
                )
            if component not in TOULMIN_COMPONENTS:
                raise ClashRejected(
                    f'target_component "{component}" must be one of {list(TOULMIN_COMPONENTS)}'
                )
            try:
                attack_type = AttackType(str(item.get("attack_type", "")))
            except ValueError:
                raise ClashRejected(
                    f'attack_type "{item.get("attack_type")}" must be one of '
                    f"{[a.value for a in AttackType]}"
                ) from None
            if not is_verbatim(quoted, transcript.component_text(target_cf, component)):
                raise ClashRejected(
                    f"quoted text is not a verbatim span of {target_cf}'s {component} — "
                    "copy the exact words"
                )
            if not argument.strip():
                raise ClashRejected("each critique needs a non-empty argument")
            critiques.append(
                Critique(
                    critic_cf=self.cf_id,
                    target_cf=target_cf,
                    target_component=component,
                    quoted=quoted,
                    attack_type=attack_type,
                    argument=argument,
                    seq=i,
                )
            )
        if targets and not critiques:
            raise ClashRejected("you must file at least one critique against another agent")

        open_refs = {c.ref for c in transcript.open_critiques_against(self.cf_id, round_idx)}
        defenses: list[Defense] = []
        for item in obj.get("defenses") or []:
            if not isinstance(item, dict):
                continue
            ref = str(item.get("critique_ref", ""))
            rebuttal = str(item.get("rebuttal", ""))
            if ref in open_refs and rebuttal.strip():
                defenses.append(Defense(critique_ref=ref, rebuttal=rebuttal))
            else:  # drop a defense pointing at no open critique rather than fail the turn
                logger.warning(
                    "[%s r%d] dropping defense with unknown/empty ref %r",
                    self.cf_id, round_idx, ref,
                )

        rev_obj = obj.get("revision")
        if not isinstance(rev_obj, dict):
            raise ClashRejected('reply is missing a "revision" object')
        raw_delta = rev_obj.get("delta")
        delta = None if raw_delta in (None, "", "null") else str(raw_delta)
        raw_claim = rev_obj.get("revised_claim")
        revised_claim = str(raw_claim) if raw_claim and str(raw_claim) != "null" else None
        rationale = str(rev_obj.get("rationale", "") or "")
        if delta is None and not rationale.strip():
            raise ClashRejected("a null delta (no change) must be justified in 'rationale'")
        if delta is not None and not revised_claim:
            logger.warning(
                "[%s r%d] revision has a delta but no revised_claim; carrying claim forward",
                self.cf_id, round_idx,
            )
        return critiques, defenses, Revision(
            delta=delta, revised_claim=revised_claim, rationale=rationale
        )

    def _build_clash_turn(
        self,
        transcript: Transcript,
        round_idx: int,
        critiques: list[Critique],
        defenses: list[Defense],
        revision: Revision,
        sent: list[dict[str, Any]],
        completion: Completion,
        n_calls: int,
        latency: float,
    ) -> Turn:
        if revision.delta and revision.revised_claim:
            proposal = revision.revised_claim
        else:  # held position (or a delta with no new claim text): carry the standing claim
            proposal = transcript.current_claim(self.cf_id) or ""
        n_open = len(transcript.open_critiques_against(self.cf_id, round_idx))
        step = self._llm_step(StepLabel.CRITIQUE, sent, completion, "ok", latency)
        return Turn(
            cf_id=self.cf_id,
            round_idx=round_idx,
            proposal=proposal,
            justification=revision.delta or revision.rationale or "",
            critiques=critiques,
            defenses=defenses,
            revision=revision,
            steps=[step],
            metadata={
                "n_backend_calls": n_calls,
                "parse_status": "ok",
                "n_critiques": len(critiques),
                "n_open_against_me": n_open,
                "n_defended": len(defenses),
            },
        )

    # -- shared step / fallback helpers -----------------------------------
    def _llm_step(
        self,
        label: StepLabel,
        sent: list[dict[str, Any]],
        completion: Completion,
        parse_status: str,
        latency: float,
    ) -> Step:
        return Step(
            kind="llm",
            label=label,
            messages_sent=sent,
            completion=completion,
            metadata={
                "latency_s": round(latency, 4),
                "parse_status": parse_status,
                "backend": type(self.backend).__name__,
            },
        )

    def _fallback_turn(
        self,
        round_idx: int,
        completion: Completion,
        sent: list[dict[str, Any]],
        n_calls: int,
        label: StepLabel,
    ) -> Turn:
        """Keep the raw text as the proposal so a turn is never dropped."""
        logger.warning(
            "[%s r%d] reply unusable after %d call(s); raw text kept",
            self.cf_id, round_idx, n_calls,
        )
        step = self._llm_step(label, sent, completion, "fallback", 0.0)
        return Turn(
            cf_id=self.cf_id,
            round_idx=round_idx,
            proposal=completion.text.strip(),
            justification="",
            steps=[step],
            metadata={"n_backend_calls": n_calls, "parse_status": "fallback"},
        )

    @staticmethod
    def _correction(reason: str) -> str:
        return (
            f"Your previous reply was rejected: {reason}. Re-read the positions and "
            "critique ids above and reply with ONE corrected JSON object only — no prose."
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
