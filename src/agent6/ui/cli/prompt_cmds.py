# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 prompt` subcommands: inspect the assembled system prompt."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from agent6.ui.cli._common import load_config_or_exit
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
    eff = load_config_or_exit(cwd, config_path)
    if isinstance(eff, int):
        return eff
    print(system_prompt_for(eff.config, cwd, mode))
    return 0
