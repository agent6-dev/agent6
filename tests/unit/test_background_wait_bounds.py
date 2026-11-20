# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Background reads share the run's command check-in bound."""

from __future__ import annotations

import pathlib
import time
from typing import Any

import pytest

from agent6.harness import child_work, toolset
from agent6.sessions import ipc
from agent6.tools import dispatch
from tests.sandbox import test_command_checkin as command_env
from tests.unit import test_children_background as child_env


@pytest.mark.parametrize("kind", ["command", "child", "next child"])
@pytest.mark.parametrize(
    ("checkin", "wait_s", "minimum", "capped"),
    [
        (0.2, 4, 0.15, True),
        (4, 0.2, 0.15, False),
        (0, 0.2, 0.15, False),
        (0.2, None, 0.15, True),
        (0, None, 0, False),
        (0.2, 0, 0, False),
    ],
    ids=["capped", "shorter", "disabled", "default", "disabled-default", "poll"],
)
def test_background_wait_uses_the_shorter_enabled_interval(
    tmp_path: pathlib.Path,
    kind: str,
    checkin: float,
    wait_s: float | None,
    minimum: float,
    capped: bool,
) -> None:
    """An explicit wait cannot bypass the check-in, and zero cannot suppress an explicit wait."""
    if kind == "command":
        d = command_env._dispatcher(tmp_path, checkin)
        d.dispatch("run_command", {"argv": ["sleep", "30"], "background": True})
        handle = "bg1"
        roster_key = "shells"
    else:
        repo = tmp_path / "repo"
        child_env._init_repo(repo)
        children = child_env._Children(repo, "run")
        harness = child_env._harness(repo, tmp_path / "run", None, children, max_turns=3)
        cfg = harness.dispatcher.config
        cfg = cfg.model_copy(
            update={"harness": cfg.harness.model_copy(update={"command_checkin_s": checkin})}
        )
        d = dispatch.ToolDispatcher(
            root=repo, config=cfg, isolation="none", session_dir=tmp_path / "run"
        )
        d.children = child_work.ChildWork(harness)
        child_env._start(harness.parallel, ("", "work"))
        handle = "run-p1-l1" if kind == "child" else ""
        roster_key = "children"
        if kind == "next child" and wait_s is None:
            minimum, capped = 0, False
    args: dict[str, Any] = {"id": handle}
    if wait_s is not None:
        args["wait_s"] = wait_s
    try:
        started = time.monotonic()
        result = d.dispatch("read_background", args).to_wire()
        elapsed = time.monotonic() - started

        assert minimum <= elapsed < 2, f"wait took {elapsed:.2f}s"
        assert "running" in result[roster_key][0]
        note = "Wait ended at harness.command_checkin_s = 0.2s."
        assert (note in result.get("output", "")) is capped
    finally:
        d.close()


@pytest.mark.parametrize("checkin", [0, 0.2, 5])
def test_background_tool_text_states_the_resolved_wait_bound(
    tmp_path: pathlib.Path, checkin: float
) -> None:
    """The model reads the interval actually bounding its waits, including a disabled cap."""
    d = command_env._dispatcher(tmp_path, checkin)
    try:
        description = next(
            t.description for t in toolset.tool_definitions(d) if t.name == "read_background"
        )
        assert f"`harness.command_checkin_s` ({checkin:g}s)" in description
        assert "Waits are bounded by that interval when it is above 0." in description
    finally:
        d.close()


def test_a_steer_does_not_claim_the_checkin_ended_a_wait(tmp_path: pathlib.Path) -> None:
    """An early operator interruption must not be reported as the check-in timeout."""
    d = command_env._dispatcher(tmp_path, 2)
    try:
        d.dispatch("run_command", {"argv": ["sleep", "30"], "background": True})
        ipc.submit_steer(tmp_path / "session", "inspect output")
        answer = d.dispatch("read_background", {"id": "bg1", "wait_s": 4}).to_wire()
        assert "harness.command_checkin_s" not in answer["output"]
        assert "running" in answer["shells"][0]
    finally:
        d.close()
