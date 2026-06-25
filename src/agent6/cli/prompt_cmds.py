# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 prompt` subcommands: inspect the assembled system prompt."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from agent6.config import ConfigError
from agent6.config_layer import load_effective
from agent6.workflows import system_prompt_for


def _cmd_prompt_show(
    config_path: Path | None,
    *,
    mode: Literal["run", "plan", "ask", "machine", "agent"],
) -> int:
    """Print the exact system prompt agent6 would send for THIS repo + the
    effective (layered) config, in the given mode. The static structural blocks
    are identical every run; the ``<repo-priors>`` block (repo map + AGENTS.md +
    recent commits) is assembled from the current repo. Useful for seeing what
    the worker actually receives, and as the basis for a custom prompt override."""
    cwd = Path.cwd()
    try:
        eff = load_effective(cwd, config_path)
    except ConfigError as exc:
        print(f"CONFIG ERROR:\n{exc}", file=sys.stderr)
        return 2
    print(system_prompt_for(eff.config, cwd, mode))
    return 0
