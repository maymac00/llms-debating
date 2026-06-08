"""Tool abstraction and the shared default registry.

Tools are *pure functions* over the live :class:`Transcript` — zero API spend.
Each tool's ``name`` matches a :class:`StepLabel` member value (lowercased), so
an LLM Step that selects a tool and the tool Step that runs it share a label.
``build_default_tools`` binds the registry to a specific transcript so the
tools read its current state when invoked.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .models import Transcript


@dataclass
class Tool:
    """A model-selectable, pure-function operation over the transcript."""

    name: str  # matches a StepLabel member value, lowercased
    description: str  # shown to the model to enable selection
    schema: dict[str, Any]  # JSON schema of the input arguments
    fn: Callable[..., str]  # executes against the Transcript, returns serialised result


def build_default_tools(transcript: Transcript) -> list[Tool]:
    """Bind the shared default registry to ``transcript``.

    Returns a fresh list of tools whose functions read ``transcript`` at call
    time. Tools must not call a model.
    """

    # Both closures capture `transcript` and read its current state when called.
    def _search(query: str) -> str:
        turns = transcript.search(query)
        payload = [  # expose only the shared fields, as JSON for the model
            {
                "cf_id": t.cf_id,
                "round_idx": t.round_idx,
                "proposal": t.proposal,
                "justification": t.justification,
            }
            for t in turns
        ]
        result: dict[str, Any] = {"query": query, "matches": payload}
        if not payload:
            # An empty result is final for this turn (the transcript does not change
            # mid-turn). Say *why* it is empty so the model does not just retry.
            has_turns = any(rnd.turns for rnd in transcript.rounds)
            result["note"] = (
                "The deliberation has not started — there are no turns to search "
                "yet. Reason from your framework and make your proposal."
                if not has_turns
                else f"No turn so far mentions {query!r}; repeating returns the same. "
                "Try a different term or finalise."
            )
        return json.dumps(result, ensure_ascii=False)

    def _list_proposals() -> str:
        latest = transcript.latest_proposals()
        if not latest:  # a bare {} is ambiguous on its own — explain what it means
            return json.dumps(
                {
                    "proposals": {},
                    "note": "No proposals yet — no one has spoken. You are opening "
                    "the deliberation; make your proposal.",
                },
                ensure_ascii=False,
            )
        return json.dumps(latest, ensure_ascii=False)

    return [
        Tool(
            name="search",
            description=(
                "Search the deliberation transcript so far. Case-insensitive "
                "substring match over every turn's proposal and justification. "
                "Returns the matching turns as JSON. Early in the deliberation it "
                "may return no matches — that is expected, not an error to retry."
            ),
            schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Substring/keyword to look for.",
                    }
                },
                "required": ["query"],
            },
            fn=_search,
        ),
        Tool(
            name="list_proposals",
            description=(
                "List each agent's most recent proposal as a JSON object mapping "
                "cf_id to proposal text. Takes no arguments. Returns an empty set "
                "before anyone has proposed (e.g. the opening turn)."
            ),
            schema={"type": "object", "properties": {}, "required": []},
            fn=_list_proposals,
        ),
    ]
