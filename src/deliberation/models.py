"""Data models and transcript utilities.

The :class:`Transcript` is the central artefact of the system: it is both the
deliberation *context* (what an agent sees on its next turn, via
:meth:`Transcript.as_messages`) and a research record / future RL rollout. The
RL-readiness invariants live here: :class:`Completion` keeps optional
``logprobs``/``token_ids``; every LLM :class:`Step` keeps its exact
``messages_sent``; and (de)serialisation is lossless, including all Steps.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class Completion(BaseModel):
    """A single model response.

    ``logprobs`` and ``token_ids`` are reserved for backends that expose them
    (e.g. a local vLLM server); API backends leave them ``None``. They are never
    dropped, so the transcript stays usable as an RL rollout later.
    """

    text: str  # full raw model output
    logprobs: list[float] | None = None  # per-token logprobs, if the backend returns them
    token_ids: list[int] | None = None  # token ids, if the backend returns them
    usage: dict[str, Any] | None = None  # token counts etc.


class StepLabel(StrEnum):
    """Controlled vocabulary for the action a Step represents.

    Tied to the skill taxonomy so Steps are independently scoreable later. New
    members may be added without schema changes anywhere else.
    """

    # The typed-turn labels (PROPOSE / CRITIQUE) are what the current loop emits:
    # one llm Step per turn, labelled PROPOSE in the blind round 0 and CRITIQUE in
    # the clash rounds (the mandatory attack is a clash turn's primary act). DEFEND
    # and REVISE name the other components a clash turn carries — recorded as
    # structured Turn fields rather than separate Steps — and are reserved for
    # finer-grained per-step scoring later.
    PROPOSE = "propose"
    CRITIQUE = "critique"
    DEFEND = "defend"
    REVISE = "revise"
    # FINALISE is retained for the pre-typed-turn single-call loop. SEARCH and
    # LIST_PROPOSALS are kept so transcripts recorded by the pre-refactor tool
    # loop still deserialise (see implement_skills.md).
    FINALISE = "finalise"
    SEARCH = "search"
    LIST_PROPOSALS = "list_proposals"
    # --- reserved room to extend (skill taxonomy) ---
    ARGUMENT_CLASSIFICATION = "argument_classification"
    CONFLICT_DETECTION = "conflict_detection"


class AttackType(StrEnum):
    """How a :class:`Critique` attacks its target — the typed-tag the doc requires.

    ``cf_value_conflict`` is the genuinely CF-grounded attack: direct evidence the
    agent is reasoning from its framework rather than from the base-model prior.
    """

    CONTEST_GROUNDS = "contest_grounds"  # the evidence is wrong, insufficient, inapplicable
    CONTEST_WARRANT = "contest_warrant"  # grounds accepted but do not license the claim
    CF_VALUE_CONFLICT = "cf_value_conflict"  # coherent, but violates a value this CF holds primary


# Toulmin components an agent may quote and attack. Used by the Direct Clash
# validator to resolve which span of the target the ``quoted`` text must match.
TOULMIN_COMPONENTS = ("claim", "grounds", "cf_warrant", "qualifier")


class ToulminProposal(BaseModel):
    """A CF-legible opening proposal (round 0), structured as a Toulmin argument.

    ``cf_warrant`` is the field that makes the proposal CF-legible rather than
    generic: the inferential link from ``grounds`` to ``claim`` routed through the
    agent's framework value. ``qualifier`` states where the claim does *not* hold —
    a deliberate attack surface for later rounds.
    """

    claim: str
    grounds: str
    cf_warrant: str
    qualifier: str

    def component(self, name: str) -> str:
        """Return one component's text by name (one of :data:`TOULMIN_COMPONENTS`)."""
        return str(getattr(self, name))


class Critique(BaseModel):
    """One typed attack on another agent's proposal — the payload of a CRITIQUE.

    ``quoted`` must be a verbatim span of the target's ``target_component`` (the
    Direct Clash anchor); the agent loop rejects critiques that fail this. ``seq``
    is the critique's index within its author's turn, giving each a stable
    :attr:`ref` the defending agent cites in the next round.
    """

    critic_cf: str
    target_cf: str
    target_component: str
    quoted: str
    attack_type: AttackType
    argument: str
    seq: int = 0
    # None = not yet evaluated (e.g. the final round, which no later round answers);
    # True/False is set once the target's next round closes (dropped-argument rule).
    conceded: bool | None = None

    @property
    def ref(self) -> str:
        """Stable handle the defending agent references — unique within a round."""
        return f"{self.critic_cf}#{self.seq}"


class Defense(BaseModel):
    """A response to a critique levelled at this agent in the previous round.

    ``critique_ref`` matches a :attr:`Critique.ref`. Any open critique left without
    a matching :class:`Defense` at round end is marked ``conceded`` (the
    dropped-argument convention from competitive debate).
    """

    critique_ref: str
    rebuttal: str


class Revision(BaseModel):
    """The optional update step, with a forced ``delta``.

    ``delta`` is the per-agent drift signal: a non-null value states *what* changed
    and *why* (and carries the new claim in ``revised_claim``); a null ``delta`` is
    permitted but must be justified in ``rationale`` — converting silence into an
    observable decision.
    """

    delta: str | None = None
    revised_claim: str | None = None  # the new claim text when delta is non-null
    rationale: str = ""  # required justification when delta is null


class Step(BaseModel):
    """One call within a turn: an LLM generation or a tool execution.

    ``kind == "llm"`` steps are generation events and carry ``messages_sent``
    (the exact prompt) and ``completion``. ``kind == "tool"`` steps record a
    pure-function tool execution for flow analysis; they are not generation
    events.
    """

    kind: Literal["llm", "tool"]  # generation vs retrieval/computation
    label: StepLabel  # action selected (llm) or tool run (tool)
    messages_sent: list[dict[str, Any]] | None = None  # llm: exact prompt sent
    completion: Completion | None = None  # llm: model response
    tool_name: str | None = None  # tool: identifier
    tool_input: dict[str, Any] | None = None  # tool: arguments
    tool_result: str | None = None  # tool: serialised result
    metadata: dict[str, Any] = Field(default_factory=dict)  # latency, backend id, parse status…

    @model_validator(mode="after")
    def _check_kind_fields(self) -> Step:
        # Enforce the per-kind field contract so no half-formed step is stored.
        if self.kind == "llm":
            if self.messages_sent is None or self.completion is None:
                raise ValueError("llm steps require messages_sent and completion")
        elif self.kind == "tool":
            if self.tool_name is None:
                raise ValueError("tool steps require tool_name")
        return self


class Turn(BaseModel):
    """One agent's full contribution in a round.

    ``steps`` is the complete intra-turn trajectory (private cognition + the
    final call) and is **never** shared with other agents. The *shared substance*
    is the typed structure: ``toulmin`` (the round-0 opening), or the clash triple
    ``critiques`` / ``defenses`` / ``revision`` (rounds ≥ 1). ``proposal`` and
    ``justification`` remain a flat, human-readable rendering of that substance —
    kept for backward compatibility and the JSONL round-trip. All typed fields
    default empty, so transcripts predating the typed turn still deserialise.
    """

    cf_id: str
    round_idx: int
    proposal: str
    justification: str
    toulmin: ToulminProposal | None = None  # round 0: the blind opening
    critiques: list[Critique] = Field(default_factory=list)  # clash rounds: the attacks
    defenses: list[Defense] = Field(default_factory=list)  # clash rounds: replies to prior attacks
    revision: Revision | None = None  # clash rounds: the forced update step
    steps: list[Step] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def final_completion(self) -> Completion | None:
        """The completion of the last ``kind == "llm"`` step, if any."""
        for step in reversed(self.steps):  # walk backwards to the deciding call
            if step.kind == "llm":
                return step.completion
        return None


class Round(BaseModel):
    index: int
    turns: list[Turn] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Judge verdict — the post-deliberation report (see judge.py).
#
# These live here, not in judge.py, because the Transcript embeds the Verdict so
# the viewer (and any consumer) can read it straight off the transcript. The
# *evaluation* logic stays in judge.py; only the data shape lives here, alongside
# the rest of the transcript spine it now belongs to.
# --------------------------------------------------------------------------- #
class CFSummary(BaseModel):
    """One framework's arc through the debate."""

    cf_id: str
    opening: str  # the position it opened with (round 0)
    evolution: str  # how it moved (or held) across the clash rounds, and why
    final_position: str  # where it ended


class Concern(BaseModel):
    """One allegation a proposal drew from another framework."""

    raised_by: str  # the critic CF
    summary: str  # the concern, in the judge's words


class DivergentPosition(BaseModel):
    """A candidate policy and the concerns the other CFs hold about it.

    The pluralistic view: for each surviving proposal, what each *other* framework
    would object to if the policymaker chose it.
    """

    cf_id: str
    final_proposal: str
    concerns: list[Concern] = Field(default_factory=list)


class Verdict(BaseModel):
    """The judge's report on a finished deliberation.

    On a consensus, ``consensus_policy`` describes the agreed policy and
    ``divergent_positions`` is empty. Otherwise ``divergent_positions`` carries
    every CF's final proposal with the allegations it drew. ``parse_status`` is
    ``"fallback"`` (with ``raw_text`` kept) when the reply could not be parsed —
    the judge never crashes a run.
    """

    debate_summary: str
    cf_summaries: list[CFSummary] = Field(default_factory=list)
    consensus: bool = False
    consensus_policy: str | None = None
    divergent_positions: list[DivergentPosition] = Field(default_factory=list)
    parse_status: str = "ok"  # "ok" | "fallback"
    raw_text: str | None = None  # kept verbatim when parsing fails


class Transcript(BaseModel):
    """The deliberation record: a scenario, an ordered list of rounds, and — once
    the judge has run — its :class:`Verdict`."""

    scenario: str
    rounds: list[Round] = Field(default_factory=list)
    verdict: Verdict | None = None  # set by the judge as the final step; persisted

    # --- mutation -------------------------------------------------------
    def append(self, turn: Turn) -> None:
        """Append ``turn`` to the current/last round (creating round 0 if empty)."""
        if not self.rounds:
            self.rounds.append(Round(index=0))
        self.rounds[-1].turns.append(turn)

    def add_round(self, round: Round) -> None:
        """Append a (possibly empty or completed) round."""
        self.rounds.append(round)

    # --- access ---------------------------------------------------------
    def round(self, t: int) -> Round:
        """Return the round whose ``index`` is ``t``."""
        for rnd in self.rounds:  # match on .index, not list position
            if rnd.index == t:
                return rnd
        raise IndexError(f"no round with index {t}")

    def n_rounds(self) -> int:
        return len(self.rounds)

    def by_agent(self, cf_id: str) -> list[Turn]:
        """All turns produced by one agent, in chronological order."""
        return [
            turn
            for rnd in self.rounds
            for turn in rnd.turns
            if turn.cf_id == cf_id
        ]

    def _iter_turns(self) -> list[Turn]:
        return [turn for rnd in self.rounds for turn in rnd.turns]

    # --- pure analysis reads (zero API spend) ---------------------------
    def search(self, query: str, cf_id: str | None = None) -> list[Turn]:
        """Substring (case-insensitive) match over proposals + justifications.

        Pure function, no model call. ``cf_id`` optionally restricts the search
        to one agent's turns.
        """
        needle = query.casefold()
        results: list[Turn] = []
        for turn in self._iter_turns():
            if cf_id is not None and turn.cf_id != cf_id:
                continue
            haystack = f"{turn.proposal}\n{turn.justification}".casefold()
            if needle in haystack:
                results.append(turn)
        return results

    def latest_proposals(self) -> dict[str, str]:
        """Each agent's most recent proposal."""
        latest: dict[str, str] = {}
        for turn in self._iter_turns():  # chronological; later overwrites earlier
            latest[turn.cf_id] = turn.proposal
        return latest

    # --- typed-turn reads (zero API spend) ------------------------------
    def latest_toulmin(self, cf_id: str) -> ToulminProposal | None:
        """An agent's most recent :class:`ToulminProposal` (its round-0 opening)."""
        found: ToulminProposal | None = None
        for turn in self._iter_turns():
            if turn.cf_id == cf_id and turn.toulmin is not None:
                found = turn.toulmin
        return found

    def current_claim(self, cf_id: str) -> str | None:
        """An agent's standing claim: its Toulmin ``claim``, overridden by the most
        recent non-null :class:`Revision`'s ``revised_claim``."""
        claim: str | None = None
        for turn in self._iter_turns():
            if turn.cf_id != cf_id:
                continue
            if turn.toulmin is not None:
                claim = turn.toulmin.claim
            if turn.revision is not None and turn.revision.revised_claim:
                claim = turn.revision.revised_claim
        return claim

    def component_text(self, cf_id: str, component: str) -> str | None:
        """The current text of one Toulmin component for ``cf_id``.

        ``claim`` reflects the latest revision; the other components come from the
        agent's round-0 Toulmin (only the claim is revisable). Returns ``None`` if
        the agent has no opening on record.
        """
        if component == "claim":
            return self.current_claim(cf_id)
        toulmin = self.latest_toulmin(cf_id)
        if toulmin is None or component not in TOULMIN_COMPONENTS:
            return None
        return toulmin.component(component)

    def open_critiques_against(self, cf_id: str, round_idx: int) -> list[Critique]:
        """Critiques targeting ``cf_id`` raised in ``round_idx - 1``.

        These are the entries an agent must answer in its round-``round_idx``
        DEFEND step; any it leaves unanswered are marked conceded at round end.
        """
        prev = round_idx - 1
        if prev < 0:
            return []
        out: list[Critique] = []
        for rnd in self.rounds:
            if rnd.index != prev:
                continue
            for turn in rnd.turns:
                out.extend(c for c in turn.critiques if c.target_cf == cf_id)
        return out

    # --- context construction ------------------------------------------
    @staticmethod
    def _shared_body(turn: Turn) -> str:
        """The shared substance of one turn — the typed structure when present,
        else the flat proposal/justification. Private ``steps`` never appear here.
        """
        if turn.toulmin is not None:  # round 0: the Toulmin opening
            t = turn.toulmin
            return (
                f"Claim: {t.claim}\n"
                f"Grounds: {t.grounds}\n"
                f"CF-warrant: {t.cf_warrant}\n"
                f"Qualifier: {t.qualifier}"
            )
        if turn.critiques or turn.defenses or turn.revision is not None:  # clash turn
            parts: list[str] = []
            for c in turn.critiques:
                parts.append(
                    f"Critique → {c.target_cf} ({c.target_component}, {c.attack_type}): "
                    f'quotes "{c.quoted}" — {c.argument}'
                )
            for d in turn.defenses:
                parts.append(f"Defense of {d.critique_ref}: {d.rebuttal}")
            if turn.revision is not None:
                r = turn.revision
                if r.delta:
                    head = f"Revised claim: {r.revised_claim}" if r.revised_claim else "Revised"
                    parts.append(f"{head} — {r.delta}")
                else:
                    parts.append(f"Held position — {r.rationale}")
            return "\n".join(parts)
        return f"Proposal: {turn.proposal}\nJustification: {turn.justification}"

    def as_messages(self, cf_id: str) -> list[dict[str, Any]]:
        """OpenAI-format shared context for ``cf_id``'s next turn.

        Renders only each prior turn's shared substance (its typed structure, or
        the flat proposal/justification fallback) — private ``steps`` are excluded,
        so agents never see each other's scratchpads. A turn produced by ``cf_id``
        itself is rendered as an ``assistant`` message; every other turn is a
        ``user`` message tagged with its author.
        """
        messages: list[dict[str, Any]] = []
        for turn in self._iter_turns():
            # The reader's own turns are "assistant"; everyone else is "user".
            role = "assistant" if turn.cf_id == cf_id else "user"
            content = f"[{turn.cf_id} · round {turn.round_idx}]\n{self._shared_body(turn)}"
            messages.append({"role": role, "content": content})
        return messages

    def render(self) -> str:
        """Human-readable round-by-round timeline of each turn's shared substance."""
        lines: list[str] = [f"SCENARIO\n{self.scenario.strip()}\n"]
        for rnd in self.rounds:
            lines.append(f"{'=' * 70}\nROUND {rnd.index}\n{'=' * 70}")
            for turn in rnd.turns:
                lines.append(f"\n[{turn.cf_id}]")
                body = self._shared_body(turn)
                lines.extend(f"  {line}" for line in body.splitlines())
            lines.append("")
        return "\n".join(lines)

    # --- (de)serialisation: lossless round-trip incl. all steps ---------
    def to_jsonl(self, path: str) -> None:
        """Serialise to JSONL: a header line then one line per round.

        The header carries the scenario and, once the judge has run, the verdict —
        so the verdict travels with the transcript and any consumer (the viewer
        included) reads it straight off the header. Omitted when unset, keeping the
        header byte-identical to pre-judge transcripts.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        header: dict[str, Any] = {"scenario": self.scenario}
        if self.verdict is not None:
            header["verdict"] = self.verdict.model_dump()
        with p.open("w", encoding="utf-8") as f:
            f.write(json.dumps(header, ensure_ascii=False) + "\n")  # header
            for rnd in self.rounds:
                f.write(rnd.model_dump_json() + "\n")  # one round per line, steps included

    @staticmethod
    def from_jsonl(path: str) -> Transcript:
        """Inverse of :meth:`to_jsonl`; a lossless round-trip including all steps.

        Tolerates pre-judge transcripts whose header has no ``verdict`` key.
        """
        lines = [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            raise ValueError(f"empty transcript file: {path}")
        header = json.loads(lines[0])  # first line = scenario (+ optional verdict)
        rounds = [Round.model_validate_json(ln) for ln in lines[1:]]  # rest = rounds
        raw_verdict = header.get("verdict")
        verdict = Verdict.model_validate(raw_verdict) if raw_verdict else None
        return Transcript(scenario=header["scenario"], rounds=rounds, verdict=verdict)
