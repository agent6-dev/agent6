# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Slash commands describe existing handlers without importing them into the parser."""

from __future__ import annotations

import inspect
import pkgutil

from agent6 import directive


def test_each_command_names_one_handler() -> None:
    names = [row.name for row in directive.COMMANDS]
    assert len(names) == len(set(names))
    for row in directive.COMMANDS:
        assert row.name.startswith("/") and row.help.splitlines() == [row.help]
        assert row.accepts and row.accepts.isdisjoint(row.refuses)
        assert inspect.isfunction(pkgutil.resolve_name(row.handler)), row.handler


def test_a_task_is_not_a_steer() -> None:
    assert {row.name for row in directive.commands_for("start")} == {"/parallel", "/review"}
    assert {row.name for row in directive.commands_for("plan/ask")} == {"/review"}
    assert {row.name for row in directive.commands_for("child")} == {"/review"}
    assert not directive.commands_for("parent")
    for row in directive.COMMANDS:
        # A valid argument keeps a malformed directive from masquerading as a refusal.
        problem = directive.steer_problem(f"{row.name} argument")
        assert bool(problem) == ("start" in row.refuses), row.name


def test_skill_names_come_from_the_configured_skills() -> None:
    (skill,) = [row for row in directive.COMMANDS if row.name == "/<skill>"]
    assert {"steer", "resume", "pause", "script", "acp"} <= skill.accepts
    assert "acp start" not in skill.accepts
    assert skill not in directive.commands_for("pause")
