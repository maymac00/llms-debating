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

    # FINALISE is the only label the current single-call turn produces. SEARCH
    # and LIST_PROPOSALS are kept so transcripts recorded by the pre-refactor
    # tool loop still deserialise (see implement_skills.md).
    SEARCH = "search"
    LIST_PROPOSALS = "list_proposals"
    FINALISE = "finalise"
    # --- reserved room to extend (skill taxonomy) ---
    ARGUMENT_CLASSIFICATION = "argument_classification"
    CONFLICT_DETECTION = "conflict_detection"


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
    final call); ``proposal`` and ``justification`` are the only parts shared
    with other agents.
    """

    cf_id: str
    round_idx: int
    proposal: str
    justification: str
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


class Transcript(BaseModel):
    """The deliberation record: a scenario plus an ordered list of rounds."""

    scenario: str
    rounds: list[Round] = Field(default_factory=list)

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

    # --- context construction ------------------------------------------
    def as_messages(self, cf_id: str) -> list[dict[str, Any]]:
        """OpenAI-format shared context for ``cf_id``'s next turn.

        Renders only each prior turn's ``proposal`` + ``justification`` — private
        ``steps`` are excluded, so agents never see each other's scratchpads. A
        turn produced by ``cf_id`` itself is rendered as an ``assistant`` message;
        every other turn is a ``user`` message tagged with its author.
        """
        messages: list[dict[str, Any]] = []
        for turn in self._iter_turns():
            # The reader's own turns are "assistant"; everyone else is "user".
            role = "assistant" if turn.cf_id == cf_id else "user"
            content = (  # only the shared fields — never the private steps
                f"[{turn.cf_id} · round {turn.round_idx}]\n"
                f"Proposal: {turn.proposal}\n"
                f"Justification: {turn.justification}"
            )
            messages.append({"role": role, "content": content})
        return messages

    def render(self) -> str:
        """Human-readable round-by-round timeline of proposals + justifications."""
        lines: list[str] = [f"SCENARIO\n{self.scenario.strip()}\n"]
        for rnd in self.rounds:
            lines.append(f"{'=' * 70}\nROUND {rnd.index}\n{'=' * 70}")
            for turn in rnd.turns:
                lines.append(f"\n[{turn.cf_id}]")
                lines.append(f"  Proposal: {turn.proposal}")
                lines.append(f"  Justification: {turn.justification}")
            lines.append("")
        return "\n".join(lines)

    # --- (de)serialisation: lossless round-trip incl. all steps ---------
    def to_jsonl(self, path: str) -> None:
        """Serialise to JSONL: a header line (scenario) then one line per round."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"scenario": self.scenario}, ensure_ascii=False) + "\n")  # header
            for rnd in self.rounds:
                f.write(rnd.model_dump_json() + "\n")  # one round per line, steps included

    @staticmethod
    def from_jsonl(path: str) -> Transcript:
        """Inverse of :meth:`to_jsonl`; a lossless round-trip including all steps."""
        lines = [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            raise ValueError(f"empty transcript file: {path}")
        header = json.loads(lines[0])  # first line = scenario header
        rounds = [Round.model_validate_json(ln) for ln in lines[1:]]  # rest = rounds
        return Transcript(scenario=header["scenario"], rounds=rounds)
