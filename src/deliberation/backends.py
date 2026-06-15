"""Inference backends.

A :class:`Backend` is a stateless (w.r.t. the debate) object exposing a single
``async generate`` call. Backends are assigned *per agent*, so heterogeneous
backends are supported. The default :class:`LiteLLMBackend` reaches any provider
LiteLLM supports (OpenAI/Anthropic/Gemini/Mistral/…); :class:`VLLMBackend`
targets an OpenAI-compatible vLLM server. Bounded retry with exponential
backoff is applied per ``generate`` call (so retries never count as extra
backend calls in the agent loop).
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .models import Completion

logger = logging.getLogger(__name__)

# Default sampling temperature applied by every backend unless the spec overrides
# it. Deliberately non-greedy: near-greedy decoding on a single shared model with
# a consensus-magnet scenario makes near-identical proposals across agents almost
# inevitable, collapsing the deliberation. Override per agent with ``temperature``
# in the backend spec (config or agent.yaml).
DEFAULT_TEMPERATURE = 0.7


@runtime_checkable
class Backend(Protocol):
    """Minimal inference interface; unchanged by the agent loop."""

    async def generate(self, messages: list[dict[str, Any]], **sampling: Any) -> Completion: ...


def _transient_exception_types() -> tuple[type[BaseException], ...]:
    """LiteLLM exception types worth retrying. Empty if LiteLLM is unavailable."""
    try:
        import litellm  # noqa: PLC0415
    except Exception:  # pragma: no cover - import guard
        return ()
    names = [
        "RateLimitError",
        "APIConnectionError",
        "Timeout",
        "APITimeoutError",
        "ServiceUnavailableError",
        "InternalServerError",
    ]
    found: list[type[BaseException]] = []
    source = getattr(litellm, "exceptions", litellm)
    for name in names:  # tolerate version drift: skip names this litellm lacks
        exc = getattr(source, name, None) or getattr(litellm, name, None)
        if isinstance(exc, type) and issubclass(exc, BaseException):
            found.append(exc)
    return tuple(found)


def _extract_usage(response: Any) -> dict[str, Any] | None:
    # Normalise litellm's usage object to a plain dict; tolerate either shape.
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    try:
        return dict(usage)
    except Exception:  # pragma: no cover - defensive
        return None


class LiteLLMBackend:
    """Default backend. Wraps ``litellm.acompletion`` for any supported provider.

    Credentials are read from environment variables by LiteLLM; keys are never
    hard-coded. ``logprobs``/``token_ids`` are left ``None`` unless the provider
    returns them. Sampling ``temperature`` defaults to :data:`DEFAULT_TEMPERATURE`
    when the spec does not set one.
    """

    def __init__(
        self,
        model: str,
        *,
        max_retries: int = 3,
        base_delay: float = 0.5,
        **default_sampling: Any,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay
        default_sampling.setdefault("temperature", DEFAULT_TEMPERATURE)
        self.default_sampling = default_sampling

    async def generate(self, messages: list[dict[str, Any]], **sampling: Any) -> Completion:
        import litellm  # noqa: PLC0415 - lazy import keeps tests network-free

        params = {**self.default_sampling, **sampling}  # per-call sampling overrides defaults
        transient = _transient_exception_types()
        # Retry lives inside one generate() so retries never count as extra
        # backend calls in the agent's per-turn cap.
        for attempt in range(self.max_retries):
            try:
                t0 = time.perf_counter()
                response = await litellm.acompletion(
                    model=self.model, messages=messages, **params
                )
                latency = time.perf_counter() - t0
                return self._to_completion(response, latency)
            except transient as exc:  # type: ignore[misc]  # only retry transient errors
                if attempt == self.max_retries - 1:
                    raise  # exhausted retries: surface the error
                delay = self.base_delay * (2**attempt)  # exponential backoff
                logger.warning(
                    "transient backend error on %s (attempt %d/%d): %s; retrying in %.1fs",
                    self.model,
                    attempt + 1,
                    self.max_retries,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
        raise RuntimeError("unreachable: retry loop exited without return")  # pragma: no cover

    def _to_completion(self, response: Any, latency: float) -> Completion:
        text = response.choices[0].message.content or ""  # first choice only
        return Completion(
            text=text,
            usage=_extract_usage(response),
            # API backends do not return token-level data; left None for RL later.
            logprobs=None,
            token_ids=None,
        )


class VLLMBackend:
    """Optional backend for an OpenAI-compatible vLLM server.

    Reached via ``base_url``. Implemented on top of LiteLLM's OpenAI-compatible
    path, so no extra dependency is required; it may populate
    ``logprobs``/``token_ids`` when the server returns them. If no server is
    reachable, ``generate`` raises like any other backend — the class and
    constructor signature exist regardless. Sampling ``temperature`` defaults to
    :data:`DEFAULT_TEMPERATURE` when the spec does not set one.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        api_key: str = "EMPTY",
        max_retries: int = 3,
        base_delay: float = 0.5,
        **default_sampling: Any,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.max_retries = max_retries
        self.base_delay = base_delay
        default_sampling.setdefault("temperature", DEFAULT_TEMPERATURE)
        self.default_sampling = default_sampling

    async def generate(self, messages: list[dict[str, Any]], **sampling: Any) -> Completion:
        import litellm  # noqa: PLC0415

        params = {**self.default_sampling, **sampling}
        transient = _transient_exception_types()
        # LiteLLM routes OpenAI-compatible servers via the ``openai/`` prefix.
        # A served model id may itself contain "/" (e.g. a HuggingFace repo id
        # like "QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ"), so we can't treat any
        # "/" as an existing provider prefix. Only skip prefixing when the route
        # is already OpenAI-compatible; otherwise prepend "openai/".
        model = (
            self.model
            if self.model.startswith(("openai/", "hosted_vllm/"))
            else f"openai/{self.model}"
        )
        for attempt in range(self.max_retries):  # same retry policy as LiteLLMBackend
            try:
                t0 = time.perf_counter()
                response = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    api_base=self.base_url,
                    api_key=self.api_key,
                    **params,
                )
                latency = time.perf_counter() - t0
                return self._to_completion(response, latency)
            except transient as exc:  # type: ignore[misc]
                if attempt == self.max_retries - 1:
                    raise
                delay = self.base_delay * (2**attempt)
                logger.warning(
                    "transient vLLM error on %s (attempt %d/%d): %s; retrying in %.1fs",
                    self.model,
                    attempt + 1,
                    self.max_retries,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
        raise RuntimeError("unreachable: retry loop exited without return")  # pragma: no cover

    def _to_completion(self, response: Any, latency: float) -> Completion:
        choice = response.choices[0]
        text = choice.message.content or ""
        logprobs, token_ids = _extract_token_data(choice)
        return Completion(
            text=text,
            logprobs=logprobs,
            token_ids=token_ids,
            usage=_extract_usage(response),
        )


def _extract_token_data(choice: Any) -> tuple[list[float] | None, list[int] | None]:
    """Best-effort extraction of per-token logprobs from an OpenAI-shaped choice."""
    lp = getattr(choice, "logprobs", None)
    if lp is None:
        return None, None
    content = getattr(lp, "content", None)  # OpenAI shape: logprobs.content[]
    if not content:
        return None, None
    try:
        logprobs = [float(tok.logprob) for tok in content]
    except Exception:  # pragma: no cover - defensive
        logprobs = None
    return logprobs, None  # token_ids not exposed by the OpenAI logprobs shape


# --------------------------------------------------------------------------- #
# Debug profile: log exactly what inference calls were made
# --------------------------------------------------------------------------- #
class InferenceLog:
    """Shared sink for :class:`LoggingBackend`: a monotonic call counter and an
    optional JSONL file.

    One instance is shared by every wrapped backend in a run, so call indices are
    global and ordered across agents. If ``path`` is given it is truncated at
    construction (one fresh log per run) and each call appends one JSON record.
    """

    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path) if path else None
        self._counter = itertools.count(1)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")  # fresh log each run

    def next_index(self) -> int:
        return next(self._counter)

    def write(self, record: dict[str, Any]) -> None:
        if self.path is None:
            return
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _preview(text: str, n: int = 80) -> str:
    """One-line, length-capped preview of a response for the console summary."""
    flat = " ".join(text.split())
    return flat[:n] + ("…" if len(flat) > n else "")


class LoggingBackend:
    """Backend decorator that records every ``generate`` call.

    Wraps any :class:`Backend` and logs the exact ``messages`` sent and the
    :class:`Completion` returned — plus model, latency and usage — without
    altering generation (it delegates and returns the inner result unchanged, so
    it satisfies the :class:`Backend` Protocol like any other backend).

    A one-line summary with a response *preview* goes to the logger at INFO (so it
    shows at the default log level); the full prompt and response go to DEBUG; and,
    when the shared :class:`InferenceLog` has a path, a complete JSON record per
    call is appended there. The exact, full record lives in the JSONL — the console
    is just a readable view.
    """

    def __init__(
        self, inner: Backend, *, label: str = "", sink: InferenceLog | None = None
    ) -> None:
        self.inner = inner
        self.label = label
        self.sink = sink or InferenceLog()
        self.model = getattr(inner, "model", "?")

    async def generate(self, messages: list[dict[str, Any]], **sampling: Any) -> Completion:
        idx = self.sink.next_index()  # assigned before the call so order = initiation order
        t0 = time.perf_counter()
        completion = await self.inner.generate(messages, **sampling)
        latency = time.perf_counter() - t0

        logger.info(
            "inference #%d [%s] %s | %d msg(s) | %.2fs | resp: %s",
            idx,
            self.label or "?",
            self.model,
            len(messages),
            latency,
            _preview(completion.text),
        )
        if logger.isEnabledFor(logging.DEBUG):  # full prompt+response only at DEBUG
            logger.debug("inference #%d messages_sent=%r", idx, messages)
            logger.debug("inference #%d response=%r", idx, completion.text)

        self.sink.write(
            {
                "call": idx,
                "label": self.label,
                "backend": type(self.inner).__name__,
                "model": self.model,
                "latency_s": round(latency, 4),
                "sampling": sampling or None,
                "usage": completion.usage,
                "messages_sent": messages,  # the exact prompt sent to the provider
                "response": completion.text,  # the exact text returned
            }
        )
        return completion
