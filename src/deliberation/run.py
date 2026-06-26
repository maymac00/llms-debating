"""Config-driven CLI entry point: run one deliberation end to end.

    python -m deliberation.run --config configs/default.yaml

One config fully determines one run (no hidden state). Per-agent ``backend`` and
``turn`` in the config override the agent's ``agent.yaml``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

from .agent import CFAgent, load_agent, load_agent_meta
from .backends import Backend, InferenceLog, LiteLLMBackend, LoggingBackend, VLLMBackend
from .judge import Judge, load_judge, save_verdict
from .protocols import DebateProtocol, RoundRobin

logger = logging.getLogger(__name__)

# Config values may reference environment variables as ${VAR} or ${VAR:-default},
# so secrets (e.g. a vLLM base_url with an internal IP) need not be committed.
_ENV_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def load_env_file() -> None:
    """Load a project ``.env`` into the process environment (best effort).

    Keys live in ``.env`` so the same file works in a terminal and in an IDE
    run config without manual exporting. Existing environment variables win
    (``override=False``): a value set in the real environment or an IDE
    run-config field takes precedence over ``.env``. Missing file or missing
    ``python-dotenv`` is tolerated silently so the CLI still runs.
    """
    try:
        from dotenv import find_dotenv, load_dotenv  # noqa: PLC0415
    except ImportError:  # pragma: no cover - dependency declared but tolerate absence
        logger.debug("python-dotenv not installed; skipping .env loading")
        return
    path = find_dotenv(usecwd=True)  # walk up from CWD (the usual repo-root invocation)
    if path:
        load_dotenv(path)
        logger.debug("loaded environment from %s", path)


def expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` references in a config.

    A bare ``${VAR}`` whose variable is unset (or empty) raises a clear error; use
    ``${VAR:-default}`` to supply a fallback. Strings without a reference and all
    non-string scalars pass through unchanged.
    """
    if isinstance(value, str):

        def _sub(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            resolved = os.environ.get(name)
            if resolved:
                return resolved
            if default is not None:
                return default
            raise ValueError(
                f"config references ${{{name}}} but environment variable {name!r} is "
                "not set; set it, or use a ${VAR:-default} fallback in the config"
            )

        return _ENV_VAR.sub(_sub, value)
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


def make_backend(spec: dict[str, Any]) -> Backend:
    """Construct a backend from a ``{provider, model, ...}`` spec.

    Adding a provider is the only code touch needed to support a new backend;
    swapping providers in the config requires none.
    """
    provider = spec.get("provider", "litellm")
    # Everything that isn't routing config is passed through as default sampling.
    sampling = {k: v for k, v in spec.items() if k not in {"provider", "model", "base_url"}}
    if provider == "litellm":
        return LiteLLMBackend(model=spec["model"], **sampling)
    if provider == "vllm":
        return VLLMBackend(model=spec["model"], base_url=spec["base_url"], **sampling)
    raise ValueError(f"unknown backend provider: {provider!r}")


def resolve_backend_spec(
    entry: dict[str, Any], run_default: dict[str, Any] | None
) -> dict[str, Any]:
    """Backend spec precedence: config entry > agent.yaml > run-config default."""
    if entry.get("backend"):
        return entry["backend"]  # 1. per-agent config entry
    agent_meta = load_agent_meta(entry["path"])
    if agent_meta.get("backend"):
        return agent_meta["backend"]  # 2. the agent's own agent.yaml
    if run_default:
        return run_default  # 3. top-level run-config default
    raise ValueError(
        f"no backend for agent {entry['path']}: set it in the config entry, "
        "the agent's agent.yaml, or as a top-level 'backend:' default"
    )


def make_sink(config: dict[str, Any]) -> InferenceLog | None:
    """The shared debug sink, if the debug profile is on; else ``None``.

    One sink is shared across every wrapped backend in a run (agents *and* the
    judge), so call indices stay global and ordered.
    """
    debug = config.get("debug") or {}
    return InferenceLog(debug.get("log_path")) if debug.get("log_inferences") else None


def build_agents(config: dict[str, Any], sink: InferenceLog | None = None) -> list[CFAgent]:
    run_default = config.get("backend")  # optional shared fallback backend
    if sink is None:  # standalone callers may not pass one; honour the config
        sink = make_sink(config)
    agents: list[CFAgent] = []
    for entry in config["agents"]:
        backend = make_backend(resolve_backend_spec(entry, run_default))
        if sink is not None:  # wrap so every inference call is recorded
            backend = LoggingBackend(backend, label=entry["path"], sink=sink)
        agents.append(load_agent(entry["path"], backend))
    return agents


def build_judge(
    config: dict[str, Any], sink: InferenceLog | None = None
) -> tuple[Judge, str | None] | None:
    """Build the judge and its output path from a top-level ``judge:`` block.

    Returns ``None`` when the config has no ``judge:`` block (judging is opt-in).
    The judge's backend honours the same debug-logging wrapper as the agents.
    """
    jc = config.get("judge")
    if not jc:
        return None
    backend = make_backend(jc["backend"])
    if sink is not None:
        backend = LoggingBackend(backend, label="judge", sink=sink)
    judge = load_judge(jc["prompt"], backend)
    output = jc.get("output")  # optional standalone JSON copy; verdict also embeds in transcript
    return judge, output


def build_protocol(config: dict[str, Any]) -> DebateProtocol:
    proto = config.get("protocol", {}) or {}
    ptype = proto.get("type", "round_robin")
    if ptype == "round_robin":
        return RoundRobin(order=proto.get("order", "fixed"), seed=proto.get("seed"))
    raise ValueError(f"unknown protocol type: {ptype!r}")


async def run_deliberation(config: dict[str, Any]) -> None:
    # One config fully determines one run.
    scenario = Path(config["scenario"]).read_text(encoding="utf-8")
    sink = make_sink(config)  # shared across agents and judge so call indices stay global
    agents = build_agents(config, sink)
    protocol = build_protocol(config)
    T = int(config["T"])

    logger.info("running %s with %d agents for T=%d", type(protocol).__name__, len(agents), T)
    transcript = await protocol.run(agents, scenario, T)

    output = config.get("output", "outputs/transcript.jsonl")
    transcript.to_jsonl(output)  # losslessly, including all steps
    logger.info("wrote transcript to %s", output)

    print(transcript.render())

    # Final step (opt-in): an LLM judge evaluates the finished transcript.
    built = build_judge(config, sink)
    if built is not None:
        judge, verdict_output = built
        logger.info("judging the deliberation")
        verdict = await judge.evaluate(transcript)
        # Embed the verdict in the transcript (so the viewer reads it directly) and
        # rewrite the transcript; the standalone JSON copy is optional.
        transcript.verdict = verdict
        transcript.to_jsonl(output)
        logger.info(
            "embedded verdict in %s (parse_status=%s)", output, verdict.parse_status
        )
        if verdict_output:
            save_verdict(verdict, verdict_output)
            logger.info("wrote verdict to %s", verdict_output)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a run config: pull in ``.env``, parse YAML, then expand ``${VAR}``.

    Shared by the run CLI and the standalone judge CLI so both resolve secrets
    and env references identically.
    """
    load_env_file()  # make .env keys reach LiteLLM (and ${VAR} expansion below)
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    return expand_env(config)  # resolve ${VAR} so secrets stay out of the config file


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a multi-agent CF deliberation.")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML config.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG logs full prompts). Default: INFO.",
    )
    parser.add_argument(
        "--debug-calls",
        action="store_true",
        help="Log every backend inference call (exact prompt + response) to the "
        "console and a JSONL file. Equivalent to a 'debug:' block in the config.",
    )
    parser.add_argument(
        "--debug-log",
        default=None,
        metavar="PATH",
        help="Where to write the inference-call JSONL (implies --debug-calls; "
        "default: outputs/inference_calls.jsonl).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)

    if args.debug_calls or args.debug_log:  # CLI flags turn the debug profile on
        debug = dict(config.get("debug") or {})
        debug["log_inferences"] = True
        debug["log_path"] = (
            args.debug_log or debug.get("log_path") or "outputs/inference_calls.jsonl"
        )
        config["debug"] = debug

    asyncio.run(run_deliberation(config))  # thin sync wrapper around the async run


if __name__ == "__main__":
    main()
