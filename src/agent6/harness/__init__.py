# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Run the agent loop over a session.

`Harness` (`loop.py`) drives one execution: the provider calls, the tool dispatch, the guards
and gates, and the snapshot a resume re-enters; the sibling `_name` modules hold its helpers.
`system_prompt_for` and `model_exchange_for` expose the run's first exchange to
`agent6 prompt show`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agent6.config import Config
from agent6.harness._context import load_repo_summary
from agent6.harness._dag_focus import initial_dag_hint
from agent6.harness._prompt_blocks import build_system_prompt, initial_instructions
from agent6.harness._toolset import tool_definitions
from agent6.kinds import IsolationLevel
from agent6.memory import decisions_path, decisions_text, memory_dir
from agent6.memory import index_text as memory_index_text
from agent6.providers import ToolDefinition
from agent6.sandbox.detect import IsolationUnavailableError, detect, resolve_isolation
from agent6.skills import ResolvedSkills
from agent6.tools.dispatch import ToolDispatcher

__all__ = [
    "ModelExchange",
    "model_exchange_for",
    "system_prompt_for",
]


def system_prompt_for(
    config: Config,
    root: Path,
    mode: Literal["run", "plan", "ask", "agent"] = "run",
    *,
    state_dir: Path | None = None,
) -> str:
    """Assemble the exact system prompt a run would send for this root, config and mode.

    The memory index, the decisions and the installed skills load on the loop's own rules:
    no recall in agent mode, skills in run mode only.

    Args:
        config: The run's config.
        root: The repo root.
        mode: The loop mode.
        state_dir: The per-repo state dir the memory and decisions live under.

    Returns:
        The system prompt text.
    """
    repo = load_repo_summary(root)
    # Agent mode assembles without repo context: one gate for every recall block.
    recall = None if mode == "agent" else state_dir
    return build_system_prompt(
        config=config,
        repo=repo,
        mode=mode,
        memory_index=memory_index_text(recall) if recall is not None else "",
        memory_dir_path=str(memory_dir(recall)) if recall is not None else "",
        decisions=decisions_text(recall) if recall is not None else "",
        decisions_path=str(decisions_path(recall)) if recall is not None else "",
        skills=_installed_skills(root, config, mode),
        isolation=_shown_isolation(config),
    )


@dataclass(frozen=True, slots=True)
class ModelExchange:
    """Hold everything the model receives on a run's first call, for `agent6 prompt show`.

    Attributes:
        mode: The loop mode.
        system: The system prompt.
        tools: The tool definitions, the API's `tools` field.
        first_message: The first user message with a placeholder for the task text.
        mcp_pending: Whether enabled MCP servers add tools at run start, unseen here.
    """

    mode: str
    system: str
    tools: tuple[ToolDefinition, ...]
    first_message: str
    mcp_pending: bool


def model_exchange_for(
    config: Config,
    root: Path,
    mode: Literal["run", "plan", "ask", "agent"] = "run",
    *,
    state_dir: Path | None = None,
) -> ModelExchange:
    """Build the exact exchange a run here would open with.

    The tool list comes from a dispatcher on this config, so what a run withholds is withheld
    here too.

    Args:
        config: The run's config.
        root: The repo root.
        mode: The loop mode.
        state_dir: The per-repo state dir.

    Returns:
        The system prompt, the tools and the first message.
    """
    system = system_prompt_for(config, root, mode, state_dir=state_dir)
    dispatcher = ToolDispatcher(
        root=root,
        config=config,
        mode="machine" if mode == "agent" else mode,
        state_dir=state_dir,
    )
    tools = tuple(tool_definitions(dispatcher, mode=mode))
    header = initial_instructions(
        mode, config.sandbox.run_commands, has_gate=bool(config.harness.verify_command)
    )
    hint = initial_dag_hint("<root task id>", mode, config.prompt.decompose == "on")
    return ModelExchange(
        mode=mode,
        system=system,
        tools=tools,
        first_message=f"TASK:\n<the task text>\n\n{header}{hint}",
        mcp_pending=config.mcp.enabled and any(s.enabled for s in config.mcp.servers.values()),
    )


def _shown_isolation(config: Config) -> IsolationLevel:
    """Return the isolation level a run here would resolve, "none" when the host cannot honor it."""
    try:
        return resolve_isolation(config.sandbox.isolation, detect())
    except IsolationUnavailableError:
        return "none"


def _installed_skills(
    root: Path, config: Config, mode: Literal["run", "plan", "ask", "agent"]
) -> ResolvedSkills | None:
    """Return the skills the loop would show: run mode only, None when nothing is installed."""
    if mode != "run":
        return None
    resolved = ToolDispatcher(root=root, config=config).resolved_skills()
    return resolved if (resolved.enabled or resolved.always) else None
