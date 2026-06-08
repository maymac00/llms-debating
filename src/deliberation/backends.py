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
import logging
import time
from typing import Any, Protocol, runtime_checkable

from .models import Completion

logger = logging.getLogger(__name__)


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
    returns them.
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
    constructor signature exist regardless.
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
