"""Smoke tests with a scripted stub backend — no network.

Covers the core agent-loop contract:
  (a) search-once-then-finalise: SEARCH llm -> SEARCH tool -> FINALISE, in order;
  (b) never-finalise: exactly max_calls backend calls, a forced FINALISE, and a
      valid (possibly fallback) proposal;
plus messages_sent non-empty on every llm step, and a lossless JSONL round-trip.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from deliberation.agent import CFAgent
from deliberation.models import Completion, StepLabel, Transcript


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


def _agent(backend: StubBackend, **kw: Any) -> CFAgent:
    return CFAgent(
        cf_id="tester",
        system_prompt="You are a test agent.",
        backend=backend,
        **kw,
    )


def _seed_transcript() -> Transcript:
    """A transcript with one prior turn so search() has something to match."""
    t = Transcript(scenario="Test scenario about housing policy.")
    from deliberation.models import Round, Step, Turn

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


def test_search_then_finalise() -> None:
    backend = StubBackend(
        [
            json.dumps({"tool": "search", "input": {"query": "housing"}}),
            json.dumps({"final": {"proposal": "Housing First", "justification": "It works."}}),
        ]
    )
    agent = _agent(backend, max_calls=5)
    transcript = _seed_transcript()

    turn = asyncio.run(agent.act(transcript, round_idx=1))

    # SEARCH llm step -> SEARCH tool step -> FINALISE llm step, in order.
    assert [(s.kind, s.label) for s in turn.steps] == [
        ("llm", StepLabel.SEARCH),
        ("tool", StepLabel.SEARCH),
        ("llm", StepLabel.FINALISE),
    ]
    # The tool step immediately follows its deciding llm step.
    assert turn.steps[1].tool_name == "search"
    assert turn.steps[1].tool_input == {"query": "housing"}
    assert turn.proposal == "Housing First"
    assert turn.justification == "It works."
    assert turn.metadata["n_backend_calls"] == 2
    assert turn.metadata["cap_hit"] is False

    # messages_sent non-empty on every llm step.
    for step in turn.steps:
        if step.kind == "llm":
            assert step.messages_sent


def test_never_finalises_forces_finalise() -> None:
    # The stub always asks to search; it never volunteers a final.
    backend = StubBackend([json.dumps({"tool": "search", "input": {"query": "x"}})])
    max_calls = 3
    agent = _agent(backend, max_calls=max_calls)
    transcript = _seed_transcript()

    turn = asyncio.run(agent.act(transcript, round_idx=1))

    # Exactly max_calls backend (llm) calls.
    assert backend.calls == max_calls
    llm_steps = [s for s in turn.steps if s.kind == "llm"]
    assert len(llm_steps) == max_calls

    # Forced FINALISE on the last call, with a valid (fallback) proposal.
    last = turn.steps[-1]
    assert last.kind == "llm"
    assert last.label == StepLabel.FINALISE
    assert last.metadata["parse_status"] == "fallback"
    assert turn.proposal  # non-empty fallback (the raw text)
    assert turn.justification == ""
    assert turn.metadata["cap_hit"] is True

    # No turn exceeds max_calls backend calls.
    assert turn.metadata["n_backend_calls"] <= max_calls

    for step in llm_steps:
        assert step.messages_sent


def test_finalise_immediately() -> None:
    backend = StubBackend(
        [json.dumps({"final": {"proposal": "P", "justification": "J"}})]
    )
    agent = _agent(backend, max_calls=5)
    turn = asyncio.run(agent.act(Transcript(scenario="S"), round_idx=0))

    assert backend.calls == 1
    assert [(s.kind, s.label) for s in turn.steps] == [("llm", StepLabel.FINALISE)]
    assert turn.proposal == "P"
    assert turn.final_completion is not None


def test_as_messages_excludes_private_steps() -> None:
    transcript = _seed_transcript()
    msgs = transcript.as_messages("tester")
    blob = json.dumps(msgs)
    # The prior turn's private step content ("x") must not leak.
    assert "Build more housing." in blob  # proposal is shared
    assert '"content": "x"' not in blob  # private step prompt is not


def test_jsonl_roundtrip(tmp_path: Any) -> None:
    # Produce a real transcript via the loop, then round-trip it.
    backend = StubBackend(
        [
            json.dumps({"tool": "search", "input": {"query": "housing"}}),
            json.dumps({"final": {"proposal": "Housing First", "justification": "It works."}}),
        ]
    )
    agent = _agent(backend, max_calls=5)
    transcript = _seed_transcript()
    turn = asyncio.run(agent.act(transcript, round_idx=1))
    transcript.append(turn)

    path = str(tmp_path / "t.jsonl")
    transcript.to_jsonl(path)
    restored = Transcript.from_jsonl(path)

    # Lossless including all steps.
    assert restored == transcript
    assert restored.rounds[-1].turns[-1].steps == turn.steps
