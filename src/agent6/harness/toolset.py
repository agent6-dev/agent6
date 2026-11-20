# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Build the tool surface each mode exposes to the model.

The list comes from the dispatcher's availability plus the mode's extras.
"""

from __future__ import annotations

from agent6 import kinds
from agent6.config import Config
from agent6.providers import ToolDefinition
from agent6.tools import dispatch as tools_dispatch
from agent6.tools import schema as tools_schema

# The tools that start and steer children, offered together.
CHILD_TOOLS = (tools_schema.ParallelInput, tools_schema.SteerChildInput)
# The task-graph tools, offered only where a curator backs them.
DAG_TOOLS = (
    tools_schema.DagAddTaskInput,
    tools_schema.DagUpdateTaskInput,
    tools_schema.DagListTasksInput,
)


def tool_definitions(
    dispatcher: tools_dispatch.ToolDispatcher,
    *,
    mode: kinds.SessionMode = "run",
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
        if (cls in CHILD_TOOLS and not dispatcher.parallel_available()) or (
            cls is tools_schema.MergeChildInput and not dispatcher.merge_child_available()
        ):
            continue
        description = cls.TOOL_DESCRIPTION
        if cls is tools_schema.ListFindingsInput and dispatcher.session_id:
            description = description.replace("<session_id>", dispatcher.session_id)
        if cls is tools_schema.ParallelInput:
            description = parallel_description(dispatcher.config)
        if (
            cls in (tools_schema.ReadBackgroundInput, tools_schema.StopBackgroundInput)
            and dispatcher.parallel_available()
        ):
            # A default run reads neither note: its background tools reach commands alone.
            description += cls.CHILDREN_NOTE
        out.append(
            ToolDefinition(
                name=cls.TOOL_NAME,
                description=description,
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


def parallel_description(cfg: Config) -> str:
    """Return the `parallel` tool's description with the run's resolved roles and providers.

    Args:
        cfg: The run's config.

    Returns:
        The contract, the spec grammar, every role's route, effort and temperature with where
        an inherited value comes from, the provider names (never their settings), the child
        limits, and that a child edits like any run.
    """
    roles: list[str] = []
    for role in kinds.MODEL_ROLES:
        resolved = cfg.models.resolve(role)
        if resolved is None:
            continue
        effort = resolved.effort if resolved.effort is not None else "the backend's default"
        temperature = (
            f"{resolved.temperature:g}"
            if resolved.temperature is not None
            else "the provider's default"
        )
        inherited = [
            field
            for field in ("model", "effort", "temperature")
            if role != "default" and cfg.models.source_role(role, field) != role
        ]
        if role != "default" and len(inherited) == 3:
            origin = " (as default)"
        elif inherited:
            origin = f" ({' and '.join(inherited)} from default)"
        else:
            origin = ""
        roles.append(
            f"{role}: {resolved.provider}/{resolved.model}, effort {effort},"
            f" temperature {temperature}{origin}"
        )
    providers = ", ".join(sorted(cfg.providers)) or "none"
    return (
        tools_schema.ParallelInput.TOOL_DESCRIPTION
        + " Roles: "
        + "; ".join(roles)
        + f". Configured providers: {providers}. Each child is an ordinary run, a worker with"
        " this run's tools, guards and limits, that may edit its clone; the children share this"
        f" run's remaining budget, at most {cfg.harness.parallel.max_children} run at once"
        " (`harness.parallel.max_children`), and `read_session` reads a child's whole transcript."
    )
