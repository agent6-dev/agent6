# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 prompt` subcommands: inspect what the model receives."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from agent6.config.layer import load_effective, resolved_state_dir
from agent6.workflows import ModelExchange, model_exchange_for


def _cmd_prompt_show(
    config_path: Path | None,
    *,
    mode: Literal["run", "plan", "ask", "machine", "agent"],
    as_json: bool = False,
) -> int:
    """Print everything the model receives on a run's first call here, for
    THIS repo and the effective (layered) config, in the given mode: the system
    prompt (its static blocks and the per-repo `<repo-priors>` block), the tool
    definitions the API's `tools` field carries (name, description, input
    schema, exactly the list this config exposes), and the first user message
    around the task. `--json` prints the same as one object."""
    cwd = Path.cwd()
    eff = load_effective(cwd, config_path)
    exchange = model_exchange_for(eff.config, cwd, mode, state_dir=resolved_state_dir(cwd))
    print(_as_json(exchange) if as_json else _as_text(exchange))
    return 0


def _as_json(x: ModelExchange) -> str:
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
    out = [
        f"=== system prompt ({x.mode} mode, {len(x.system):,} chars) ===",
        x.system.rstrip(),
        "",
        f"=== tools ({len(x.tools)}; the API's `tools` field: name, description, input schema) ===",
    ]
    for t in x.tools:
        out.append(f"--- {t.name}")
        out.append(t.description)
        out.append("schema: " + json.dumps(t.input_schema, separators=(",", ":")))
    if x.mcp_pending:
        out.append("--- (plus the tools of the enabled MCP servers, discovered at run start)")
    out += ["", "=== first user message ===", x.first_message]
    return "\n".join(out)
