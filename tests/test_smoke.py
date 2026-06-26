"""Smoke tests with a scripted stub backend — no network.

Covers the typed-turn contract:
  (a) round 0 is a blind Toulmin PROPOSE → one PROPOSE llm Step;
  (b) a clash turn (rounds ≥ 1) emits validated critiques/defenses/revision,
      with the Direct Clash rule enforced (verbatim quote required) and bounded
      rejection sampling on failure;
  (c) an unparseable/invalid reply falls back to raw text — a turn is never dropped;
  (d) the dropped-argument convention marks unanswered critiques conceded;
plus blind round 0, private-step exclusion from shared context, and a lossless
JSONL round-trip (including transcripts predating the typed turn).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from deliberation.agent import CFAgent, parse_proposal
from deliberation.judge import Judge, concerns_ledger, parse_verdict
from deliberation.models import (
    AttackType,
    Completion,
    Critique,
    Defense,
    Revision,
    Round,
    Step,
    StepLabel,
    ToulminProposal,
    Transcript,
    Turn,
)
from deliberation.protocols import RoundRobin, _mark_concessions


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


def _agent(backend: StubBackend, cf_id: str = "tester") -> CFAgent:
    return CFAgent(cf_id=cf_id, system_prompt="You are a test agent.", backend=backend)


# --- reply builders ------------------------------------------------------ #
def _toulmin_reply(claim: str, *, grounds: str = "g", warrant: str = "w", qual: str = "q") -> str:
    return json.dumps(
        {"claim": claim, "grounds": grounds, "cf_warrant": warrant, "qualifier": qual}
    )


def _critique(target: str, quoted: str, *, component: str = "claim",
              attack: str = "cf_value_conflict",
              argument: str = "it harms flourishing") -> dict[str, Any]:
    return {
        "target_cf": target,
        "target_component": component,
        "quoted": quoted,
        "attack_type": attack,
        "argument": argument,
    }


def _clash_reply(critiques: list[dict[str, Any]], *, defenses: list[dict[str, Any]] | None = None,
                 delta: str | None = None, revised_claim: str | None = None,
                 rationale: str = "my claim stands") -> str:
    return json.dumps(
        {
            "critiques": critiques,
            "defenses": defenses or [],
            "revision": {"delta": delta, "revised_claim": revised_claim, "rationale": rationale},
        }
    )


# --- transcript seeds ---------------------------------------------------- #
def _round0_step() -> Step:
    return Step(
        kind="llm",
        label=StepLabel.PROPOSE,
        messages_sent=[{"role": "user", "content": "x"}],
        completion=Completion(text="..."),
    )


def _seed_round0(claims: dict[str, str]) -> Transcript:
    """A transcript with a round-0 Toulmin proposal per agent in ``claims``."""
    t = Transcript(scenario="Test scenario about housing policy.")
    rnd = Round(index=0)
    for cf, claim in claims.items():
        rnd.turns.append(
            Turn(
                cf_id=cf,
                round_idx=0,
                proposal=claim,
                justification="j",
                toulmin=ToulminProposal(claim=claim, grounds="g", cf_warrant="w", qualifier="q"),
                steps=[_round0_step()],
            )
        )
    t.add_round(rnd)
    return t


def _seed_flat_turn() -> Transcript:
    """A transcript whose one prior turn has NO typed structure (the flat path)."""
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


# --- round 0: blind Toulmin PROPOSE -------------------------------------- #
def test_round0_propose_parses_toulmin() -> None:
    backend = StubBackend(
        [_toulmin_reply("Housing First", warrant="dignity", qual="absent austerity")]
    )
    turn = asyncio.run(_agent(backend).act(Transcript(scenario="S"), round_idx=0))

    assert backend.calls == 1
    assert [(s.kind, s.label) for s in turn.steps] == [("llm", StepLabel.PROPOSE)]
    assert turn.toulmin is not None
    assert turn.toulmin.claim == "Housing First"
    assert turn.proposal == "Housing First"  # flat proposal mirrors the claim
    assert "dignity" in turn.justification and "absent austerity" in turn.justification
    assert turn.metadata["parse_status"] == "ok"


def test_round0_unparseable_falls_back() -> None:
    backend = StubBackend(["I refuse to emit JSON."])
    turn = asyncio.run(_agent(backend).act(Transcript(scenario="S"), round_idx=0))

    assert turn.toulmin is None
    assert turn.proposal == "I refuse to emit JSON."
    assert turn.steps[0].label == StepLabel.PROPOSE
    assert turn.metadata["parse_status"] == "fallback"


# --- clash turns: CRITIQUE / DEFEND / REVISE ----------------------------- #
def test_clash_validates_direct_clash() -> None:
    transcript = _seed_round0({"other": "Ban cars downtown.", "tester": "Build housing now."})
    backend = StubBackend(
        [_clash_reply([_critique("other", "Ban cars downtown.")], rationale="My claim stands.")]
    )
    turn = asyncio.run(_agent(backend).act(transcript, round_idx=1))

    assert backend.calls == 1
    assert turn.steps[0].label == StepLabel.CRITIQUE
    assert len(turn.critiques) == 1
    c = turn.critiques[0]
    assert c.attack_type is AttackType.CF_VALUE_CONFLICT
    assert c.critic_cf == "tester" and c.target_cf == "other"
    assert c.ref == "tester#0"
    assert turn.revision is not None and turn.revision.delta is None
    assert turn.proposal == "Build housing now."  # null delta → standing claim carried
    assert turn.metadata["parse_status"] == "ok"


def test_clash_revision_changes_claim() -> None:
    transcript = _seed_round0({"other": "Ban cars downtown.", "tester": "Build housing now."})
    backend = StubBackend(
        [
            _clash_reply(
                [_critique("other", "Ban cars downtown.")],
                delta="Persuaded that supply alone is insufficient.",
                revised_claim="Build housing AND fund services.",
            )
        ]
    )
    turn = asyncio.run(_agent(backend).act(transcript, round_idx=1))

    assert turn.revision is not None and turn.revision.delta is not None
    assert turn.proposal == "Build housing AND fund services."


def test_clash_misquote_rejected_then_falls_back() -> None:
    transcript = _seed_round0({"other": "Ban cars downtown.", "tester": "Build housing now."})
    # Quote is not a verbatim span of other's claim → rejected on every attempt.
    backend = StubBackend([_clash_reply([_critique("other", "Legalise everything")])])
    agent = _agent(backend)
    turn = asyncio.run(agent.act(transcript, round_idx=1))

    assert backend.calls == agent.MAX_CLASH_RETRIES + 1  # initial + bounded retries
    assert turn.metadata["parse_status"] == "fallback"
    assert turn.critiques == []


def test_clash_resamples_then_accepts() -> None:
    transcript = _seed_round0({"other": "Ban cars downtown.", "tester": "Build housing now."})
    backend = StubBackend(
        [
            _clash_reply([_critique("other", "not a real quote")]),  # rejected
            _clash_reply([_critique("other", "Ban cars downtown.")]),  # accepted
        ]
    )
    turn = asyncio.run(_agent(backend).act(transcript, round_idx=1))

    assert backend.calls == 2
    assert len(turn.critiques) == 1
    assert turn.metadata["n_backend_calls"] == 2


def test_clash_null_delta_without_rationale_rejected() -> None:
    transcript = _seed_round0({"other": "Ban cars downtown.", "tester": "Build housing now."})
    backend = StubBackend(
        [_clash_reply([_critique("other", "Ban cars downtown.")], rationale="")]
    )
    agent = _agent(backend)
    turn = asyncio.run(agent.act(transcript, round_idx=1))

    # A null delta with no rationale is not an observable decision → rejected.
    assert backend.calls == agent.MAX_CLASH_RETRIES + 1
    assert turn.metadata["parse_status"] == "fallback"


# --- dropped-argument convention ----------------------------------------- #
def _clash_turn(cf_id: str, round_idx: int, *, critiques: list[Critique] | None = None,
                defenses: list[Defense] | None = None) -> Turn:
    return Turn(
        cf_id=cf_id,
        round_idx=round_idx,
        proposal="P",
        justification="",
        critiques=critiques or [],
        defenses=defenses or [],
        revision=Revision(delta=None, rationale="held"),
    )


def test_unanswered_critique_is_conceded() -> None:
    transcript = _seed_round0({"a": "Claim A.", "b": "Claim B."})
    crit = Critique(
        critic_cf="a", target_cf="b", target_component="claim", quoted="Claim B.",
        attack_type=AttackType.CONTEST_GROUNDS, argument="x", seq=0,
    )
    transcript.add_round(Round(index=1, turns=[_clash_turn("a", 1, critiques=[crit])]))
    # b does not defend in round 2.
    transcript.add_round(Round(index=2, turns=[_clash_turn("b", 2)]))

    _mark_concessions(transcript, 2)
    assert transcript.round(1).turns[0].critiques[0].conceded is True


def test_answered_critique_is_not_conceded() -> None:
    transcript = _seed_round0({"a": "Claim A.", "b": "Claim B."})
    crit = Critique(
        critic_cf="a", target_cf="b", target_component="claim", quoted="Claim B.",
        attack_type=AttackType.CONTEST_GROUNDS, argument="x", seq=0,
    )
    transcript.add_round(Round(index=1, turns=[_clash_turn("a", 1, critiques=[crit])]))
    defended = _clash_turn("b", 2, defenses=[Defense(critique_ref="a#0", rebuttal="no")])
    transcript.add_round(Round(index=2, turns=[defended]))

    _mark_concessions(transcript, 2)
    assert transcript.round(1).turns[0].critiques[0].conceded is False


# --- legacy / invariant guards ------------------------------------------- #
def test_legacy_final_wrapper_accepted() -> None:
    assert parse_proposal(json.dumps({"final": {"proposal": "P", "justification": "J"}})) == (
        "P",
        "J",
    )
    assert parse_proposal('Sure!\n```json\n{"proposal": "P", "justification": "J"}\n```') == (
        "P",
        "J",
    )


def test_as_messages_excludes_private_steps() -> None:
    transcript = _seed_flat_turn()
    msgs = transcript.as_messages("tester")
    blob = json.dumps(msgs)
    # The prior turn's shared proposal is visible; its private step content ("x") is not.
    assert "Build more housing." in blob
    assert '"content": "x"' not in blob


def test_as_messages_renders_toulmin_structure() -> None:
    transcript = _seed_round0({"other": "Ban cars downtown."})
    blob = json.dumps(transcript.as_messages("tester"))
    assert "Ban cars downtown." in blob and "CF-warrant" in blob


def test_round0_blind_then_sees_prior() -> None:
    """Round 0 is blind (no agent sees another's opening); round 1 sees them all."""

    def _proposer(name: str, target: str) -> CFAgent:
        backend = StubBackend(
            [
                _toulmin_reply(f"P-{name}"),
                _clash_reply([_critique(target, f"P-{target}")]),
            ]
        )
        return CFAgent(cf_id=name, system_prompt="t", backend=backend)

    agents = [_proposer("alice", "bob"), _proposer("bob", "alice")]
    transcript = asyncio.run(RoundRobin(order="fixed").run(agents, scenario="S", T=2))

    def _prompt_blob(turn: Any) -> str:
        return json.dumps(turn.steps[0].messages_sent)

    # Round 0: no agent's prompt contains another agent's round-0 claim.
    r0 = transcript.round(0)
    for turn in r0.turns:
        blob = _prompt_blob(turn)
        for other in r0.turns:
            if other.cf_id != turn.cf_id:
                assert other.proposal not in blob  # blind to peers' openings

    # Round 1: each agent now sees both round-0 claims, and clashes validly.
    for turn in transcript.round(1).turns:
        blob = _prompt_blob(turn)
        assert "P-alice" in blob and "P-bob" in blob
        assert len(turn.critiques) == 1
        assert turn.metadata["parse_status"] == "ok"


def test_jsonl_roundtrip(tmp_path: Any) -> None:
    transcript = _seed_round0({"other": "Ban cars downtown.", "tester": "Build housing now."})
    backend = StubBackend([_clash_reply([_critique("other", "Ban cars downtown.")])])
    turn = asyncio.run(_agent(backend).act(transcript, round_idx=1))
    transcript.add_round(Round(index=1, turns=[turn]))

    path = str(tmp_path / "t.jsonl")
    transcript.to_jsonl(path)
    restored = Transcript.from_jsonl(path)

    # Lossless including all steps and the typed clash structure.
    assert restored == transcript
    assert restored.rounds[-1].turns[-1].critiques == turn.critiques


# --- LLM-as-judge -------------------------------------------------------- #
def _verdict_reply(
    *,
    consensus: bool,
    consensus_policy: str | None = None,
    divergent: list[dict[str, Any]] | None = None,
) -> str:
    return json.dumps(
        {
            "debate_summary": "They argued about housing.",
            "cf_summaries": [
                {"cf_id": "a", "opening": "o", "evolution": "e", "final_position": "f"}
            ],
            "consensus": consensus,
            "consensus_policy": consensus_policy,
            "divergent_positions": divergent or [],
        }
    )


def test_judge_evaluate_consensus() -> None:
    transcript = _seed_round0({"a": "Build housing.", "b": "Build housing."})
    backend = StubBackend([_verdict_reply(consensus=True, consensus_policy="Build housing now.")])
    verdict = asyncio.run(Judge("you are a judge", backend).evaluate(transcript))

    assert backend.calls == 1
    assert verdict.parse_status == "ok"
    assert verdict.consensus is True
    assert verdict.consensus_policy == "Build housing now."
    assert verdict.divergent_positions == []


def test_judge_evaluate_no_consensus_compiles_concerns() -> None:
    transcript = _seed_round0({"a": "Ban cars.", "b": "Build housing."})
    backend = StubBackend(
        [
            _verdict_reply(
                consensus=False,
                divergent=[
                    {
                        "cf_id": "a",
                        "final_proposal": "Ban cars.",
                        "concerns": [{"raised_by": "b", "summary": "too costly"}],
                    }
                ],
            )
        ]
    )
    verdict = asyncio.run(Judge("you are a judge", backend).evaluate(transcript))

    assert verdict.consensus is False
    assert verdict.consensus_policy is None
    assert verdict.divergent_positions[0].cf_id == "a"
    assert verdict.divergent_positions[0].concerns[0].raised_by == "b"
    assert "costly" in verdict.divergent_positions[0].concerns[0].summary


def test_judge_unparseable_falls_back() -> None:
    transcript = _seed_round0({"a": "A.", "b": "B."})
    backend = StubBackend(["I will not emit JSON."])
    verdict = asyncio.run(Judge("j", backend).evaluate(transcript))

    assert verdict.parse_status == "fallback"
    assert verdict.raw_text == "I will not emit JSON."
    assert "I will not emit JSON." in verdict.debate_summary


def test_concerns_ledger_groups_critiques_by_target() -> None:
    transcript = _seed_round0({"a": "Claim A.", "b": "Claim B."})
    crit = Critique(
        critic_cf="a", target_cf="b", target_component="claim", quoted="Claim B.",
        attack_type=AttackType.CONTEST_GROUNDS, argument="x", seq=0,
    )
    transcript.add_round(Round(index=1, turns=[_clash_turn("a", 1, critiques=[crit])]))

    ledger = concerns_ledger(transcript)
    assert list(ledger) == ["b"]
    assert ledger["b"][0].critic_cf == "a"


def test_verdict_embeds_in_transcript_roundtrip(tmp_path: Any) -> None:
    transcript = _seed_round0({"a": "Ban cars.", "b": "Build housing."})
    backend = StubBackend(
        [
            _verdict_reply(
                consensus=False,
                divergent=[
                    {
                        "cf_id": "a",
                        "final_proposal": "Ban cars.",
                        "concerns": [{"raised_by": "b", "summary": "hurts the poor"}],
                    }
                ],
            )
        ]
    )
    transcript.verdict = asyncio.run(Judge("j", backend).evaluate(transcript))

    path = str(tmp_path / "t.jsonl")
    transcript.to_jsonl(path)
    restored = Transcript.from_jsonl(path)

    # The verdict travels with the transcript, losslessly.
    assert restored == transcript
    assert restored.verdict is not None
    assert restored.verdict.divergent_positions[0].concerns[0].raised_by == "b"


def test_transcript_without_verdict_roundtrips_and_omits_key(tmp_path: Any) -> None:
    transcript = _seed_round0({"a": "A.", "b": "B."})
    path = str(tmp_path / "t.jsonl")
    transcript.to_jsonl(path)

    # Header stays byte-compatible with pre-judge transcripts (no verdict key).
    header = json.loads(open(path, encoding="utf-8").readline())
    assert "verdict" not in header
    restored = Transcript.from_jsonl(path)
    assert restored == transcript and restored.verdict is None


def test_parse_verdict_handles_prose_wrapped_json() -> None:
    wrapped = 'Here is my verdict:\n```json\n' + _verdict_reply(
        consensus=True, consensus_policy="X"
    ) + "\n```"
    verdict = parse_verdict(wrapped)
    assert verdict is not None and verdict.consensus_policy == "X"


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
