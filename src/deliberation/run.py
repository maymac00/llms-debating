"""Config-driven CLI entry point: run one deliberation end to end.

    python -m deliberation.run --config configs/default.yaml

One config fully determines one run (no hidden state). Per-agent ``backend`` and
``turn`` in the config override the agent's ``agent.yaml``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Any

import yaml

from .agent import CFAgent, load_agent, load_agent_meta
from .backends import Backend, LiteLLMBackend, VLLMBackend
from .protocols import DebateProtocol, RoundRobin

logger = logging.getLogger(__name__)


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


def build_agents(config: dict[str, Any]) -> list[CFAgent]:
    run_default = config.get("backend")  # optional shared fallback backend
    agents: list[CFAgent] = []
    for entry in config["agents"]:
        backend = make_backend(resolve_backend_spec(entry, run_default))
        agent = load_agent(entry["path"], backend)
        # Config 'turn' overrides agent.yaml (§9); applied after load_agent.
        turn_cfg = entry.get("turn", {}) or {}
        if "max_calls" in turn_cfg:
            agent.max_calls = int(turn_cfg["max_calls"])
        if "tools" in turn_cfg:
            agent.tool_names = turn_cfg["tools"]
        agents.append(agent)
    return agents


def build_protocol(config: dict[str, Any]) -> DebateProtocol:
    proto = config.get("protocol", {}) or {}
    ptype = proto.get("type", "round_robin")
    if ptype == "round_robin":
        return RoundRobin(order=proto.get("order", "fixed"), seed=proto.get("seed"))
    raise ValueError(f"unknown protocol type: {ptype!r}")


async def run_deliberation(config: dict[str, Any]) -> None:
    # One config fully determines one run.
    scenario = Path(config["scenario"]).read_text(encoding="utf-8")
    agents = build_agents(config)
    protocol = build_protocol(config)
    T = int(config["T"])

    logger.info("running %s with %d agents for T=%d", type(protocol).__name__, len(agents), T)
    transcript = await protocol.run(agents, scenario, T)

    output = config.get("output", "outputs/transcript.jsonl")
    transcript.to_jsonl(output)  # losslessly, including all steps
    logger.info("wrote transcript to %s", output)

    print(transcript.render())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a multi-agent CF deliberation.")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML config.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG logs full prompts). Default: INFO.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    asyncio.run(run_deliberation(config))  # thin sync wrapper around the async run


if __name__ == "__main__":
    main()
