"""Replay each recorded review opening under two system prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any, get_args

from bench.review import recordings, scoring

from agent6 import budget, config, kinds, terminal
from agent6.app import preflight, role_providers
from agent6.app.session import setup
from agent6.config import layer
from agent6.providers import anthropic, chatgpt, openai, provider_types, responses

HTTPProvider = (
    anthropic.AnthropicProvider
    | chatgpt.ChatGPTProvider
    | openai.OpenAIProvider
    | responses.ResponsesProvider
)


def replay(
    openings: list[recordings.Opening],
    provider: HTTPProvider,
    *,
    settings: kinds.ModelSettings,
    repeats: int,
    tool_choice: provider_types.ToolChoice = "auto",
) -> scoring.Results:
    """Write a JSON line per logical provider call and return the paired replies.

    Args:
        openings: The recorded first requests.
        provider: The configured HTTP provider, shared across arms.
        settings: The route and effort used to build the provider.
        repeats: The number of pairs per recorded review.
        tool_choice: Whether each arm may call the recorded tools.

    Returns:
        The replies by review, repetition and arm.

    Raises:
        ProviderError: A call failed. Its JSON line precedes the refusal.
        BudgetExceededError: The budget stopped the run.
    """
    results: scoring.Results = {}
    for opening in openings:
        pairs = results[opening.session_id] = []
        for repeat in range(1, repeats + 1):
            pair: dict[str, scoring.Reply] = {}
            for arm in scoring.ARMS if repeat % 2 else reversed(scoring.ARMS):
                system = opening.recorded if arm == "recorded" else opening.tree
                row = {
                    "session": opening.session_id,
                    "transcript": opening.transcript,
                    "repeat": repeat,
                    "arm": arm,
                    "model": settings.route.spec,
                    "requested_effort": settings.effort or "default",
                    "tool_choice": tool_choice,
                    "system_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
                }
                kwargs: dict[str, Any] = (
                    {"max_tokens": opening.max_tokens} if opening.max_tokens is not None else {}
                )
                try:
                    response = provider.call(
                        system=system,
                        messages=opening.messages,
                        tools=opening.tools,
                        tool_choice=tool_choice,
                        temperature=opening.temperature,
                        reasoning_effort=settings.effort,
                        **kwargs,
                    )
                except (provider_types.ProviderError, budget.BudgetExceededError) as exc:
                    failed = scoring.Reply("call failed", error=str(exc))
                    print(json.dumps({**row, **failed.fields()}), flush=True)
                    raise
                pair[arm] = scoring.score(response)
                print(json.dumps({**row, **pair[arm].fields()}), flush=True)
            pairs.append(pair)
    return results


def main(argv: list[str] | None = None) -> int:
    """Run the operator's replay and print the comparison to stderr."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "state_dir", type=pathlib.Path, help="state directory containing sessions/reviewers"
    )
    parser.add_argument(
        "--config", type=pathlib.Path, action="append", default=[], help="agent6 config overlay"
    )
    parser.add_argument(
        "--model", required=True, metavar="PROVIDER/MODEL", help="recorded reviewer route"
    )
    parser.add_argument(
        "--effort", required=True, choices=(*get_args(kinds.EffortLevel), "default")
    )
    parser.add_argument(
        "-n", "--repeats", type=int, default=1, help="calls per review per arm (default: 1)"
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="forbid tool calls but keep the recorded definitions",
    )
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    try:
        openings = recordings.read_reviews(args.state_dir)
        cfg = layer.load_effective(pathlib.Path.cwd(), tuple(args.config)).config
        route = cfg.model_route(args.model)
        entry = cfg.providers[route.provider]
        if isinstance(entry, config.ClaudeCodeProviderEntry):
            raise config.ConfigError("select an HTTP provider, not a local Claude Code process")
        for key in ("output_config", "reasoning", "reasoning_effort", "thinking"):
            if key in entry.extra_body:
                raise config.ConfigError(
                    f"remove providers.{route.provider}.extra_body.{key} for replay;"
                    " set reasoning effort with --effort only"
                )
        settings = kinds.ModelSettings(route, None if args.effort == "default" else args.effort)
        cfg = cfg.with_model_settings("reviewer", settings)
        if refusal := preflight.budget_preflight(cfg, [(route.provider, route.model)]):
            raise config.ConfigError(refusal)
        meter = setup.budget_tracker(cfg)
        try:
            provider = role_providers.build_role_provider(
                cfg, "reviewer", transcript_sink=None, budget=meter
            )
            assert isinstance(provider, HTTPProvider)
            results = replay(
                openings,
                provider,
                settings=settings,
                repeats=args.repeats,
                tool_choice="none" if args.no_tools else "auto",
            )
        finally:
            meter.settle()
        scoring.print_table(results)
    except (
        OSError,
        ValueError,
        config.ConfigError,
        provider_types.ProviderError,
        budget.BudgetExceededError,
    ) as exc:
        print(f"review replay: {terminal.scrub_terminal_controls(str(exc))}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
