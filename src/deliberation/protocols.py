"""Debate topologies behind a single protocol seam.

Only :class:`RoundRobin` is implemented; new topologies (TwoPhase, Moderated, …)
attach behind :class:`DebateProtocol` without touching the core.
"""

from __future__ import annotations

import logging
import random
from typing import Literal, Protocol, runtime_checkable

from .agent import CFAgent
from .models import Round, Transcript

logger = logging.getLogger(__name__)


@runtime_checkable
class DebateProtocol(Protocol):
    async def run(self, agents: list[CFAgent], scenario: str, T: int) -> Transcript: ...


class RoundRobin:
    """Agents act sequentially, in order, for ``T`` rounds.

    Round 0 is *blind*: every agent makes its opening proposal without seeing any
    other agent's round-0 turn, so first positions form independently rather than
    anchoring on whoever spoke first. From round 1 on, turns are appended as they
    are produced, so each agent's context includes earlier turns *from the same
    round*. Speaking order is a known deliberation confound, so it is exposed via
    ``order``/``seed`` rather than hidden: ``shuffle`` re-randomises the order
    each round using ``seed``.
    """

    def __init__(
        self, order: Literal["fixed", "shuffle"] = "fixed", seed: int | None = None
    ) -> None:
        self.order = order
        self.seed = seed
        self._rng = random.Random(seed)

    async def run(self, agents: list[CFAgent], scenario: str, T: int) -> Transcript:
        transcript = Transcript(scenario=scenario)
        for t in range(T):
            transcript.add_round(Round(index=t))  # open round t before any turns
            speaking_order = list(agents)
            if self.order == "shuffle":
                self._rng.shuffle(speaking_order)  # re-randomise per round (seeded)
            logger.info(
                "round %d speaking order: %s",
                t,
                [a.cf_id for a in speaking_order],
            )
            if t == 0:
                # Blind round: each agent acts against the still-empty round 0,
                # so no one sees another's opening proposal. Turns are collected
                # first and appended only once every agent has proposed.
                turns = [await agent.act(transcript, t) for agent in speaking_order]
                for turn in turns:
                    transcript.append(turn)
            else:
                # Sequential: each turn is appended before the next agent acts, so
                # later speakers already see earlier turns from this same round.
                for agent in speaking_order:
                    turn = await agent.act(transcript, t)
                    transcript.append(turn)
                # Dropped-argument convention: a critique from t-1 that its target
                # did not answer this round is now conceded.
                _mark_concessions(transcript, t)
        return transcript


def _mark_concessions(transcript: Transcript, defending_round: int) -> None:
    """Mark every critique from ``defending_round - 1`` conceded iff its target did
    not file a matching :class:`Defense` in ``defending_round``.

    Critiques in the final round are never reached here (no later round answers
    them), so they keep ``conceded=None`` — unevaluated rather than conceded.
    """
    prev = defending_round - 1
    if prev < 0:
        return
    defended = {
        d.critique_ref for turn in transcript.round(defending_round).turns for d in turn.defenses
    }
    for turn in transcript.round(prev).turns:
        for critique in turn.critiques:
            critique.conceded = critique.ref not in defended
