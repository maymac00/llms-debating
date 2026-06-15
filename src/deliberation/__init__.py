"""Minimal core for multi-agent Conceptual-Framework policy deliberation."""

from __future__ import annotations

from .agent import CFAgent, load_agent, parse_proposal
from .backends import Backend, LiteLLMBackend, VLLMBackend
from .models import (
    Completion,
    Round,
    Step,
    StepLabel,
    Transcript,
    Turn,
)
from .protocols import DebateProtocol, RoundRobin

__all__ = [
    "Backend",
    "CFAgent",
    "Completion",
    "DebateProtocol",
    "LiteLLMBackend",
    "Round",
    "RoundRobin",
    "Step",
    "StepLabel",
    "Transcript",
    "Turn",
    "VLLMBackend",
    "load_agent",
    "parse_proposal",
]
