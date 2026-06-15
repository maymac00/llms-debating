"""Smoke tests with a scripted stub backend — no network.

Covers the single-call turn contract:
  (a) a clean JSON reply parses into proposal/justification and is recorded as
      one FINALISE llm Step with its exact prompt;
  (b) an unparseable reply falls back to the raw text as the proposal;
plus blind round 0, private-step exclusion from shared context, and a lossless
JSONL round-trip.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from deliberation.agent import CFAgent, parse_proposal
from deliberation.models import Completion, Round, Step, StepLabel, Transcript, Turn


class StubBackend:
    """Returns scripted replies in order; repeats the last reply if exhausted."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.calls = 0

    async def generate(self, messages: list[dict[str, Any]], **sampling: Any) -> Completion:
        # Every llm step must keep its exact prompt.
        assert messages, "messages_sent must be non-empty"
        text = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return Completion(text=text)


def _agent(backend: StubBackend) -> CFAgent:
    return CFAgent(cf_id="tester", system_prompt="You are a test agent.", backend=backend)


def _seed_transcript() -> Transcript:
    """A transcript with one prior turn carrying a private step."""
    t = Transcript(scenario="Test scenario about housing policy.")
    prior = Turn(
        cf_id="other",
        round_idx=0,
        proposal="Build more housing.",
        justification="Housing supply reduces rough sleeping.",
        steps=[
            Step(
                kind="llm",
                label=StepLabel.FINALISE,
                messages_sent=[{"role": "user", "content": "x"}],
                completion=Completion(text="..."),
            )
        ],
    )
    t.add_round(Round(index=0, turns=[prior]))
    return t


def test_single_call_turn() -> None:
    backend = StubBackend(
        [json.dumps({"proposal": "Housing First", "justification": "It works."})]
    )
    agent = _agent(backend)
    transcript = _seed_transcript()

    turn = asyncio.run(agent.act(transcript, round_idx=1))

    # Exactly one backend call, recorded as one FINALISE llm step.
    assert backend.calls == 1
    assert [(s.kind, s.label) for s in turn.steps] == [("llm", StepLabel.FINALISE)]
    assert turn.proposal == "Housing First"
    assert turn.justification == "It works."
    assert turn.metadata["n_backend_calls"] == 1
    assert turn.metadata["parse_status"] == "ok"

    # The step keeps the exact prompt, which includes the prior shared turn.
    step = turn.steps[0]
    assert step.messages_sent
    blob = json.dumps(step.messages_sent)
    assert "Build more housing." in blob  # shared context present
    assert step.completion is not None


def test_legacy_final_wrapper_accepted() -> None:
    assert parse_proposal(json.dumps({"final": {"proposal": "P", "justification": "J"}})) == (
        "P",
        "J",
    )
    # JSON wrapped in prose / fences still parses.
    assert parse_proposal('Sure!\n```json\n{"proposal": "P", "justification": "J"}\n```') == (
        "P",
        "J",
    )


def test_unparseable_reply_falls_back_to_raw_text() -> None:
    backend = StubBackend(["I refuse to emit JSON, but here is my opinion."])
    agent = _agent(backend)

    turn = asyncio.run(agent.act(Transcript(scenario="S"), round_idx=0))

    assert backend.calls == 1
    assert turn.proposal == "I refuse to emit JSON, but here is my opinion."
    assert turn.justification == ""
    assert turn.metadata["parse_status"] == "fallback"
    assert turn.steps[0].metadata["parse_status"] == "fallback"


def test_as_messages_excludes_private_steps() -> None:
    transcript = _seed_transcript()
    msgs = transcript.as_messages("tester")
    blob = json.dumps(msgs)
    # The prior turn's private step content ("x") must not leak.
    assert "Build more housing." in blob  # proposal is shared
    assert '"content": "x"' not in blob  # private step prompt is not


def test_round0_blind_then_sees_prior() -> None:
    """Round 0 is blind (no agent sees another's opening); round 1 sees them all."""
    from deliberation.protocols import RoundRobin

    def _proposer(name: str) -> CFAgent:
        backend = StubBackend(
            [json.dumps({"proposal": f"P-{name}", "justification": f"J-{name}"})]
        )
        return CFAgent(cf_id=name, system_prompt="t", backend=backend)

    agents = [_proposer("alice"), _proposer("bob")]
    transcript = asyncio.run(RoundRobin(order="fixed").run(agents, scenario="S", T=2))

    def _prompt_blob(turn: Any) -> str:
        return json.dumps(turn.steps[0].messages_sent)

    # Round 0: no agent's prompt contains another agent's round-0 proposal.
    r0 = transcript.round(0)
    for turn in r0.turns:
        blob = _prompt_blob(turn)
        for other in r0.turns:
            if other.cf_id != turn.cf_id:
                assert other.proposal not in blob  # blind to peers' openings

    # Round 1: each agent now sees both round-0 proposals.
    for turn in transcript.round(1).turns:
        blob = _prompt_blob(turn)
        assert "P-alice" in blob and "P-bob" in blob


def test_jsonl_roundtrip(tmp_path: Any) -> None:
    # Produce a real transcript via the loop, then round-trip it.
    backend = StubBackend(
        [json.dumps({"proposal": "Housing First", "justification": "It works."})]
    )
    agent = _agent(backend)
    transcript = _seed_transcript()
    turn = asyncio.run(agent.act(transcript, round_idx=1))
    transcript.append(turn)

    path = str(tmp_path / "t.jsonl")
    transcript.to_jsonl(path)
    restored = Transcript.from_jsonl(path)

    # Lossless including all steps.
    assert restored == transcript
    assert restored.rounds[-1].turns[-1].steps == turn.steps


def test_old_transcripts_with_tool_steps_still_load(tmp_path: Any) -> None:
    """Transcripts recorded by the pre-refactor tool loop must still deserialise."""
    t = Transcript(scenario="S")
    t.add_round(
        Round(
            index=0,
            turns=[
                Turn(
                    cf_id="old",
                    round_idx=0,
                    proposal="P",
                    justification="J",
                    steps=[
                        Step(
                            kind="tool",
                            label=StepLabel.SEARCH,
                            tool_name="search",
                            tool_input={"query": "housing"},
                            tool_result="{}",
                        )
                    ],
                )
            ],
        )
    )
    path = str(tmp_path / "old.jsonl")
    t.to_jsonl(path)
    assert Transcript.from_jsonl(path) == t
