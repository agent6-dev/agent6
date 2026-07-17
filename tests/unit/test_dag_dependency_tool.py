# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The add_dependency DAG tool: schema exposure + dispatch wiring.

Curator-level semantics (cycle rejection, journal op, focus gating on
depends_on) are covered by test_graph_curator.py and test_workflow.py; these
tests cover the LLM-facing layer added on top.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from agent6.config import Config, load_config
from agent6.graph.client import CuratorClientError, GraphClient
from agent6.graph.models import AddDependencyIntent, TaskNode
from agent6.tools.dispatch import ToolDispatcher, ToolError
from agent6.tools.schema import LOOP_EXTRA_TOOLS, PLAN_EXTRA_TOOLS, DagAddDependencyInput
from agent6.workflows import loop as loopmod

_VALID_TOML = """
[agent6]
config_version = 1
[providers.anthropic]
api_format = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
[models.worker]
provider = "anthropic"
model = "x"
[models.reviewer]
provider = "anthropic"
model = "x"
[workflow]
verify_command = ["true"]
"""

_A = "01" + "A" * 24
_B = "01" + "B" * 24


def _config(tmp_path: Path) -> Config:
    p = tmp_path / "agent6.toml"
    p.write_text(_VALID_TOML, encoding="utf-8")
    return load_config(p)


def _node(node_id: str, depends_on: tuple[str, ...]) -> TaskNode:
    now = datetime.now(tz=UTC)
    return TaskNode(
        id=node_id,
        parent_id=None,
        title="t",
        depends_on=depends_on,
        created_at=now,
        updated_at=now,
        created_by="worker",
    )


class _StubGraph:
    """Duck-typed GraphClient standing in for the curator connection."""

    def __init__(self, *, fail: str = "") -> None:
        self.fail = fail
        self.seen: list[AddDependencyIntent] = []

    def add_dependency(self, intent: AddDependencyIntent) -> TaskNode:
        self.seen.append(intent)
        if self.fail:
            raise CuratorClientError(self.fail)
        return _node(intent.id, (intent.depends_on,))


def test_add_dependency_in_run_and_plan_tool_lists(tmp_path: Path) -> None:
    assert DagAddDependencyInput in LOOP_EXTRA_TOOLS
    assert DagAddDependencyInput in PLAN_EXTRA_TOOLS
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    for mode in ("run", "plan"):
        names = {t.name for t in loopmod._tool_definitions(d, mode=mode)}  # pyright: ignore[reportPrivateUsage]
        assert "add_dependency" in names, mode
    for mode in ("ask", "machine", "agent"):
        names = {t.name for t in loopmod._tool_definitions(d, mode=mode)}  # pyright: ignore[reportPrivateUsage]
        assert "add_dependency" not in names, mode


def test_dispatch_add_dependency_roundtrip(tmp_path: Path) -> None:
    stub = _StubGraph()
    d = ToolDispatcher(
        root=tmp_path, config=_config(tmp_path), graph_client=cast(GraphClient, stub)
    )
    out = d.dispatch("add_dependency", {"id": _A, "depends_on": _B})
    assert out == {"id": _A, "title": "t", "depends_on": [_B]}
    assert stub.seen[0].id == _A and stub.seen[0].depends_on == _B


def test_dispatch_add_dependency_requires_curator(tmp_path: Path) -> None:
    d = ToolDispatcher(root=tmp_path, config=_config(tmp_path))
    with pytest.raises(ToolError, match="DAG curator not available"):
        d.dispatch("add_dependency", {"id": _A, "depends_on": _B})


def test_dispatch_add_dependency_surfaces_curator_rejection(tmp_path: Path) -> None:
    stub = _StubGraph(fail="would introduce cycle")
    d = ToolDispatcher(
        root=tmp_path, config=_config(tmp_path), graph_client=cast(GraphClient, stub)
    )
    with pytest.raises(ToolError, match="cycle"):
        d.dispatch("add_dependency", {"id": _A, "depends_on": _B})


def test_dispatch_add_dependency_validates_ids(tmp_path: Path) -> None:
    stub = _StubGraph()
    d = ToolDispatcher(
        root=tmp_path, config=_config(tmp_path), graph_client=cast(GraphClient, stub)
    )
    with pytest.raises(ToolError):
        d.dispatch("add_dependency", {"id": "short", "depends_on": _B})
    assert not stub.seen  # rejected at the schema, never reached the curator


def test_dag_prompt_blocks_mention_add_dependency() -> None:
    from agent6.prompts.loop import DAG_RULES_DECOMPOSE, DAG_RULES_OPTIONAL

    assert "add_dependency" in DAG_RULES_OPTIONAL
    assert "add_dependency" in DAG_RULES_DECOMPOSE
