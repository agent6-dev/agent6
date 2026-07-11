# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The tool surface each loop mode exposes to the model.

Builds the ToolDefinition list from the dispatcher's availability plus the
mode's extras (run / plan / ask / machine / agent), and the read-only review
surface shared by the in-loop panel and `agent6 review`.
"""

from __future__ import annotations

from typing import Any, Literal

from agent6.providers import ToolDefinition
from agent6.tools.dispatch import ToolDispatcher, ToolError
from agent6.tools.schema import (
    ALL_TOOLS,
    ASK_EXTRA_TOOLS,
    LOOP_EXTRA_TOOLS,
    MACHINE_EXTRA_TOOLS,
    PLAN_EXTRA_TOOLS,
    ApplyEditInput,
    ApplyPatchInput,
    RunCommandInput,
    RunVerifyInput,
)
from agent6.workflows._review import ReviewDispatch

# The ONLY tools an explore-tier reviewer may use: read-only navigation, no
# edits/commits/run_command/dag/finish. Enforced both by what we expose AND by
# the dispatch wrapper (defense in depth).
READONLY_REVIEW_TOOLS = frozenset(
    {
        "read_file",
        "list_dir",
        "grep",
        "outline",
        "find_definition",
        "find_references",
        "agent6_docs",
    }
)


def tool_definitions(
    dispatcher: ToolDispatcher,
    *,
    mode: Literal["run", "plan", "ask", "machine", "agent"] = "run",
) -> list[ToolDefinition]:
    """Build the tool list exposed to the loop. Filters by what the
    dispatcher actually allows (e.g. run_command may be disabled).

    ``mode="plan"`` filters mutating tools
    (``apply_edit``/``apply_patch``) out of ``ALL_TOOLS`` and swaps
    ``LOOP_EXTRA_TOOLS`` for ``PLAN_EXTRA_TOOLS`` (drops
    ``finish_run``/``run_metric_command``, adds ``finish_planning``).
    ``mode="machine"`` (machine authoring) keeps only read-only navigation +
    ``finish_run`` so the agent's one job is to emit a `.asm.toml`.
    """
    available = set(dispatcher.available_tool_names())
    extras: tuple[type[Any], ...]
    if mode == "plan":
        extras = PLAN_EXTRA_TOOLS
    elif mode == "ask":
        extras = ASK_EXTRA_TOOLS
    elif mode in ("machine", "agent"):
        extras = MACHINE_EXTRA_TOOLS
    else:
        extras = LOOP_EXTRA_TOOLS
    base_tools: tuple[type[Any], ...] = ALL_TOOLS
    if mode in ("plan", "ask"):
        # Plan and ask are read-only; filter mutating tools even if the
        # dispatcher would otherwise allow them (the dispatcher's own
        # mode guard is the second line of defence).
        blocked = {ApplyEditInput.TOOL_NAME, ApplyPatchInput.TOOL_NAME}
        base_tools = tuple(cls for cls in ALL_TOOLS if cls.TOOL_NAME not in blocked)
    elif mode in ("machine", "agent"):
        # Authoring / machine agent-state: read-only navigation + finish_run
        # only, no edit/patch/verify/run_command. The deliverable is the
        # finish_run payload, not a file edit or a command run, and weak models
        # otherwise wander off editing the repo (observed live on Kimi K2.6).
        blocked = {
            ApplyEditInput.TOOL_NAME,
            ApplyPatchInput.TOOL_NAME,
            RunVerifyInput.TOOL_NAME,
            RunCommandInput.TOOL_NAME,
        }
        base_tools = tuple(cls for cls in ALL_TOOLS if cls.TOOL_NAME not in blocked)
    out: list[ToolDefinition] = []
    for cls in (*base_tools, *extras):
        if cls.TOOL_NAME not in available and cls not in extras:
            # Extras (finish_run / finish_planning / run_metric / dag_*) are
            # always exposed even though they're not in ALL_TOOLS.
            continue
        schema = cls.model_json_schema()
        schema.setdefault("type", "object")
        out.append(
            ToolDefinition(
                name=cls.TOOL_NAME,
                description=cls.TOOL_DESCRIPTION,
                input_schema=schema,
            )
        )
    # Any MCP tools the dispatcher's manager discovered get
    # appended verbatim. Names already carry the `mcp__<server>__`
    # prefix so they can never collide with built-in tool names.
    mgr = getattr(dispatcher, "_mcp_manager", None)
    if mgr is not None:
        for desc in mgr.descriptors():
            schema = dict(desc.input_schema)
            schema.setdefault("type", "object")
            out.append(
                ToolDefinition(
                    name=desc.qualified_name,
                    description=desc.description or f"MCP tool {desc.tool_name!r}",
                    input_schema=schema,
                )
            )
    return out


def build_readonly_review_tools(
    dispatcher: ToolDispatcher,
) -> tuple[list[ToolDefinition], ReviewDispatch]:
    """Read-only tool surface for explore-tier review seats: the navigation tools
    *dispatcher* exposes filtered to ``READONLY_REVIEW_TOOLS``, plus a dispatch
    wrapper that REFUSES anything outside the allowlist (so a reviewer can never
    edit, commit, run a command, or mutate the task graph). Shared by the in-loop
    panel and the post-hoc ``agent6 review`` path."""
    tools = [t for t in tool_definitions(dispatcher, mode="run") if t.name in READONLY_REVIEW_TOOLS]

    def dispatch(name: str, tool_input: dict[str, Any]) -> Any:
        if name not in READONLY_REVIEW_TOOLS:
            raise ToolError(f"review reviewer may not call {name!r} (read-only)")
        return dispatcher.dispatch(name, tool_input)

    return tools, dispatch
