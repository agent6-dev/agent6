# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Build the tool surface each mode exposes to the model.

The list comes from the dispatcher's availability plus the mode's extras; the read-only review
surface is shared by the in-loop panel and `agent6 review`.
"""

from __future__ import annotations

from typing import Any, Literal

from agent6 import kinds
from agent6.harness import _reviewer
from agent6.providers import ToolDefinition
from agent6.tools import dispatch as tools_dispatch
from agent6.tools import errors, results
from agent6.tools import schema as tools_schema

# The only tools a reviewer may use, enforced by the exposed list and by the dispatch wrapper.
READONLY_REVIEW_TOOLS = frozenset(
    {
        "read_file",
        "list_dir",
        "outline",
        "find_definition",
        "find_references",
    }
)


# The task-graph tools, offered only where a curator backs them.
DAG_TOOLS = (
    tools_schema.DagAddTaskInput,
    tools_schema.DagUpdateTaskInput,
    tools_schema.DagListTasksInput,
)


def tool_definitions(
    dispatcher: tools_dispatch.ToolDispatcher,
    *,
    mode: Literal["run", "plan", "ask", "machine", "agent"] = "run",
) -> list[ToolDefinition]:
    """Build the tool list the loop exposes in a mode.

    Args:
        dispatcher: The run's dispatcher, whose availability filters the mode's surface.
        mode: The loop mode.

    Returns:
        The tool definitions, with MCP tools appended in the editing modes only.
    """
    available = set(dispatcher.available_tool_names())
    surface = tools_schema.mode_tools(mode)
    out: list[ToolDefinition] = []
    for cls in (*surface.base, *surface.extras):
        if cls.TOOL_NAME not in available and cls not in surface.extras:
            continue  # the extras are exposed without being in ALL_TOOLS
        if dispatcher.tool_is_withheld(cls.TOOL_NAME):
            continue
        if cls in DAG_TOOLS and not dispatcher.dag_available:
            continue  # no curator: every task-graph call errors
        if (
            cls.TOOL_NAME == tools_schema.RunMetricInput.TOOL_NAME
            and not dispatcher.metric_configured()
        ):
            continue  # no metric: the tool could only answer that none is configured
        if (
            cls.TOOL_NAME == tools_schema.UseSkillInput.TOOL_NAME
            and not dispatcher.skills_available()
        ):
            continue  # no skills: the tool could only error
        out.append(
            ToolDefinition(
                name=cls.TOOL_NAME,
                description=cls.TOOL_DESCRIPTION,
                input_schema=tools_schema.wire_schema(cls),
            )
        )
    # MCP tools cannot be classified as read-only, so only an editing mode offers them.
    if kinds.session_kind(mode).edits:
        for desc in dispatcher.mcp_descriptors():
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
    dispatcher: tools_dispatch.ToolDispatcher,
) -> tuple[list[ToolDefinition], _reviewer.ReviewDispatch]:
    """Build the read-only tool surface a review seat gets.

    Args:
        dispatcher: The run's dispatcher.

    Returns:
        The navigation tools in `READONLY_REVIEW_TOOLS`, and a dispatch wrapper that refuses
        every other tool.
    """
    tools = [t for t in tool_definitions(dispatcher, mode="run") if t.name in READONLY_REVIEW_TOOLS]

    def dispatch(name: str, tool_input: dict[str, Any]) -> results.ToolResult:
        if name not in READONLY_REVIEW_TOOLS:
            raise errors.ToolError(f"review reviewer may not call {name!r} (read-only)")
        return dispatcher.dispatch(name, tool_input)

    return tools, dispatch
