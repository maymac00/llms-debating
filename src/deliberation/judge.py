"""LLM-as-judge: the final, post-deliberation evaluation step.

The judge is an **evaluation / reporting layer**, not a debate participant and not
part of the RL-rollout spine. It reads a finished :class:`Transcript` through the
zero-API-spend pure reads, asks one model (configured separately from the debate
agents) to produce a structured :class:`Verdict`, and emits that as a standalone
JSON artefact. Two consequences follow from that placement:

* It stays cleanly outside the Transcript invariants — the verdict is *not* added
  to the transcript (which would silently drop on ``to_jsonl`` and break the
  lossless round-trip), it is a sibling artefact.
* Because transcripts are saved losslessly, judging can run standalone on any
  ``outputs/*.jsonl`` without re-running the debate (``python -m
  deliberation.judge --config <cfg> --transcript <jsonl>``).

The judge's mission, framed for a *pluralistic* policymaker:

1. Describe the debate and each CF's main points, and how they evolved.
2. Deliver a verdict — if the CFs converged, state the agreed policy; if not,
   present every CF's final proposal together with the main allegations it drew
   from the other CFs, so the policymaker can see each framework's concerns
   before choosing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from .agent import _extract_json_object, _split_frontmatter
from .backends import Backend
from .models import (
    CFSummary,
    Concern,
    Critique,
    DivergentPosition,
    Transcript,
    Verdict,
)

logger = logging.getLogger(__name__)

# The verdict data models now live in models.py (the Transcript embeds a Verdict).
# Re-exported here so ``from deliberation.judge import Verdict`` keeps working.
__all__ = [
    "CFSummary",
    "Concern",
    "DivergentPosition",
    "Judge",
    "Verdict",
    "concerns_ledger",
    "load_judge",
    "parse_verdict",
    "save_verdict",
]


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_verdict(text: str) -> Verdict | None:
    """Parse a judge reply into a :class:`Verdict`, or ``None`` if no JSON object.

    Tolerant of prose/fence wrapping (via :func:`_extract_json_object`) and of
    missing optional fields; only the absence of any JSON object yields ``None``
    (which the caller turns into a raw-text fallback).
    """
    obj = _extract_json_object(text)
    if obj is None:
        return None

    cf_summaries: list[CFSummary] = []
    for item in obj.get("cf_summaries") or []:
        if not isinstance(item, dict):
            continue
        cf_summaries.append(
            CFSummary(
                cf_id=str(item.get("cf_id", "")),
                opening=str(item.get("opening", "")),
                evolution=str(item.get("evolution", "")),
                final_position=str(item.get("final_position", "")),
            )
        )

    divergent: list[DivergentPosition] = []
    for item in obj.get("divergent_positions") or []:
        if not isinstance(item, dict):
            continue
        concerns: list[Concern] = []
        for c in item.get("concerns") or []:
            if not isinstance(c, dict):
                continue
            concerns.append(
                Concern(raised_by=str(c.get("raised_by", "")), summary=str(c.get("summary", "")))
            )
        divergent.append(
            DivergentPosition(
                cf_id=str(item.get("cf_id", "")),
                final_proposal=str(item.get("final_proposal", "")),
                concerns=concerns,
            )
        )

    raw_policy = obj.get("consensus_policy")
    consensus_policy = (
        str(raw_policy) if raw_policy not in (None, "", "null") else None
    )
    return Verdict(
        debate_summary=str(obj.get("debate_summary", "")),
        cf_summaries=cf_summaries,
        consensus=bool(obj.get("consensus")),
        consensus_policy=consensus_policy,
        divergent_positions=divergent,
    )


# --------------------------------------------------------------------------- #
# Context construction — pure reads of the finished transcript (no API spend)
# --------------------------------------------------------------------------- #
def concerns_ledger(transcript: Transcript) -> dict[str, list[Critique]]:
    """Every critique grouped by the CF it targets, across all rounds.

    The raw material for the "main allegations each proposal received" — the judge
    synthesises and ranks these rather than extracting them from prose.
    """
    ledger: dict[str, list[Critique]] = {}
    for rnd in transcript.rounds:
        for turn in rnd.turns:
            for c in turn.critiques:
                ledger.setdefault(c.target_cf, []).append(c)
    return ledger


def _final_positions(transcript: Transcript) -> dict[str, str]:
    """Each CF's standing claim at the end of the debate (revisions applied)."""
    latest = transcript.latest_proposals()
    return {cf: (transcript.current_claim(cf) or latest[cf]) for cf in latest}


def _final_positions_block(transcript: Transcript) -> str:
    lines = [f"[{cf}] {claim}" for cf, claim in _final_positions(transcript).items()]
    return "\n".join(lines) if lines else "(no positions on record)"


def _concerns_block(transcript: Transcript) -> str:
    ledger = concerns_ledger(transcript)
    if not ledger:
        return "(no critiques were filed)"
    parts: list[str] = []
    for target_cf, crits in ledger.items():
        parts.append(f"Against {target_cf}'s proposal:")
        for c in crits:
            status = (
                "conceded" if c.conceded
                else "answered" if c.conceded is False
                else "unanswered (final round)"
            )
            parts.append(
                f"  - {c.critic_cf} ({c.attack_type}, targets {c.target_component}; "
                f'{status}): quotes "{c.quoted}" — {c.argument}'
            )
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Judge
# --------------------------------------------------------------------------- #
class Judge:
    """Evaluates a finished deliberation in one backend call.

    ``system_prompt`` is the judge's role/mission (from its ``.md`` file); the
    precise JSON output schema lives in the task prompt here, co-located with
    :func:`parse_verdict` so the two cannot drift.
    """

    def __init__(self, system_prompt: str, backend: Backend) -> None:
        self.system_prompt = system_prompt
        self.backend = backend

    async def evaluate(self, transcript: Transcript) -> Verdict:
        sent = self._build_messages(transcript)
        completion = await self.backend.generate(sent)
        verdict = parse_verdict(completion.text)
        if verdict is None:  # never crash a run: keep the raw text
            logger.warning("judge reply unparseable; keeping raw text as the report")
            return Verdict(
                debate_summary=completion.text.strip(),
                parse_status="fallback",
                raw_text=completion.text,
            )
        return verdict

    def _build_messages(self, transcript: Transcript) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._task_prompt(transcript)},
        ]

    def _task_prompt(self, transcript: Transcript) -> str:
        return (
            "A multi-agent policy deliberation has finished. Each agent reasons from a "
            "distinct conceptual framework (CF). Below is the full record, then the "
            "final standing positions and every critique filed, grouped by the proposal "
            "it targets. Judge the debate as briefed.\n\n"
            "# Scenario\n"
            f"{transcript.scenario.strip()}\n\n"
            "# Full deliberation (round by round)\n"
            f"{transcript.render()}\n\n"
            "# Final standing position of each CF\n"
            f"{_final_positions_block(transcript)}\n\n"
            "# Critiques filed, grouped by the proposal they target\n"
            f"{_concerns_block(transcript)}\n\n"
            "# Your task\n"
            "Decide whether the CFs reached consensus on a single policy. Claims worded "
            "differently may still agree — judge the substance, not the wording.\n\n"
            "Reply with EXACTLY ONE JSON object and nothing else:\n"
            "{\n"
            '  "debate_summary": "<2-4 short paragraphs: what was argued, the main lines '
            'of clash, and how positions moved>",\n'
            '  "cf_summaries": [{"cf_id": "<framework>", '
            '"opening": "<its round-0 position>", '
            '"evolution": "<how it moved or held across the clash rounds, and why>", '
            '"final_position": "<where it ended>"}],\n'
            '  "consensus": <true|false>,\n'
            '  "consensus_policy": "<if consensus: the agreed policy in concrete terms; '
            'else null>",\n'
            '  "divergent_positions": [{"cf_id": "<framework>", '
            '"final_proposal": "<its final policy, concretely>", '
            '"concerns": [{"raised_by": "<another CF>", '
            '"summary": "<the main concern that CF would have if this proposal is '
            'chosen>"}]}]\n'
            "}\n\n"
            "Rules:\n"
            "- If consensus is true, give a concrete consensus_policy and leave "
            "divergent_positions empty.\n"
            "- If consensus is false, leave consensus_policy null and fill "
            "divergent_positions with EVERY CF's final proposal. For each, compile the "
            "main concerns the OTHER CFs raised against it — so a pluralistic policymaker "
            "can weigh each framework's objections before choosing. Draw the concerns from "
            "the critiques above; merge near-duplicates and keep the most substantive.\n"
            "- Be faithful to the record: attribute positions and concerns to the CF that "
            "actually held them. Do not invent figures or positions.\n"
            "- Write plainly, so a non-specialist policymaker can follow."
        )


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #
def load_judge(prompt_path: str | Path, backend: Backend) -> Judge:
    """Build a :class:`Judge` from a prompt ``.md`` file (optional frontmatter).

    The file body is the judge's system prompt; any ``---`` frontmatter (e.g. a
    ``display_name``) is parsed off and ignored here.
    """
    path = Path(prompt_path)
    if not path.exists():
        raise FileNotFoundError(f"judge prompt not found: {path}")
    _frontmatter, body = _split_frontmatter(path.read_text(encoding="utf-8"))
    logger.info("loaded judge prompt from %s", path)
    return Judge(system_prompt=body.strip(), backend=backend)


def save_verdict(verdict: Verdict, path: str | Path) -> None:
    """Write the verdict to ``path`` as pretty-printed JSON (the judge artefact)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(verdict.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# Standalone CLI: judge an already-saved transcript without re-running the debate
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Judge a finished deliberation transcript with an LLM judge."
    )
    parser.add_argument("--config", required=True, help="Config YAML with a 'judge:' block.")
    parser.add_argument(
        "--transcript", required=True, help="Path to the saved transcript JSONL to judge."
    )
    parser.add_argument(
        "--output", default=None, help="Where to write the verdict JSON (overrides the config)."
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level. Default: INFO.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Lazy import avoids a circular import (run imports judge for the auto-run path).
    from .run import build_judge, load_config

    config = load_config(args.config)
    built = build_judge(config)
    if built is None:
        raise SystemExit("config has no 'judge:' block; nothing to do")
    judge, default_output = built

    transcript = Transcript.from_jsonl(args.transcript)
    verdict = asyncio.run(judge.evaluate(transcript))

    # Embed the verdict in the transcript so the viewer can read it directly, and
    # write it straight back to the file it came from.
    transcript.verdict = verdict
    transcript.to_jsonl(args.transcript)
    logger.info(
        "embedded verdict in %s (parse_status=%s)", args.transcript, verdict.parse_status
    )

    output = args.output or default_output  # optional standalone JSON copy
    if output:
        save_verdict(verdict, output)
        logger.info("wrote verdict to %s", output)

    # A concise one-line summary, not the full JSON — printing the whole verdict
    # invites accidentally redirecting it (`>>`) onto the transcript file.
    if verdict.consensus:
        print(f"verdict: CONSENSUS — {verdict.consensus_policy or '(unstated)'}")
    else:
        print(
            f"verdict: NO CONSENSUS — {len(verdict.divergent_positions)} standing "
            f"proposal(s); verdict embedded in {args.transcript}"
        )


if __name__ == "__main__":
    main()
