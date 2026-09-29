# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 prompt` subcommands: inspect what the model receives."""

from __future__ import annotations

import json
import pathlib
from typing import Literal

from agent6 import paths, verify_infer
from agent6.config import layer
from agent6.harness import ModelExchange, model_exchange_for


def _cmd_prompt_show(
    config_path: pathlib.Path | None,
    *,
    mode: Literal["run", "plan", "ask", "agent"],
    as_json: bool = False,
) -> int:
    """Print everything the model receives on a session's first call here.

    The system prompt (its static blocks and the per-repo `<repo-priors>` block), the tool
    definitions the API's `tools` field carries, and the first user message, for this repo
    and the effective config in the given mode.

    Args:
        config_path: The `--config` file, if any.
        mode: The session kind whose prompt to show.
        as_json: Print one JSON object instead of text.

    Returns:
        The exit code, 0.
    """
    cwd = pathlib.Path.cwd()
    eff = layer.load_effective(cwd, config_path)
    cfg = eff.config
    if mode in ("run", "plan") and not cfg.harness.verify_command and cfg.harness.verify_infer:
        # The gate shapes the prompt, so infer it as a run does; the LLM tier spends, so not here.
        inferred = verify_infer.infer_verify_command(
            cwd, verify_infer.read_agents_md(cwd), llm_call=None
        )
        if inferred is not None:
            cfg = cfg.with_verify_command(inferred.argv)
    exchange = model_exchange_for(cfg, cwd, mode, state_dir=paths.state_dir(cwd))
    print(_as_json(exchange) if as_json else _as_text(exchange))
    return 0


def _as_json(x: ModelExchange) -> str:
    """Return the exchange as one indented JSON object."""
    return json.dumps(
        {
            "mode": x.mode,
            "system": x.system,
            "tools": [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in x.tools
            ],
            "first_message": x.first_message,
            "mcp_tools_pending": x.mcp_pending,
        },
        indent=2,
    )


def _as_text(x: ModelExchange) -> str:
    """Return the exchange as sectioned text."""
    out = [
        f"=== system prompt ({x.mode} mode, {len(x.system):,} chars) ===",
        x.system.rstrip(),
        "",
        f"=== tools ({len(x.tools)}; the API's `tools` field: name, description, input schema) ===",
    ]
    for t in x.tools:
        out.append(f"--- {t.name}")
        out.append(t.description)
        out.append("schema: " + json.dumps(t.input_schema, indent=2))
    if x.mcp_pending:
        out.append("--- (plus the tools of the enabled MCP servers, discovered at run start)")
    out += ["", "=== first user message ===", x.first_message]
    return "\n".join(out)
