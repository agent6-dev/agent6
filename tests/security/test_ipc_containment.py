# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta

"""The bridge files a run writes stay inside the run.

Two of them name themselves after a string from outside: an approval scope,
which carries an MCP server name parsed out of a tool name the LLM chose, and
an answer id, which the web server takes from the request. Neither may steer
where the file lands.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest

from agent6.sessions.ipc import (
    approvals_dir,
    read_answer,
    session_allow_set,
    set_session_allow,
    set_session_deny,
    write_answer,
)

# What the write must survive. The invariant is the same for each: refuse, or
# land as one plain file directly in the approvals dir. Some of these are legal
# filenames once the caller's prefix or suffix is attached (`mcp..`,
# `...answer`) and are contained precisely because of it.
HOSTILE = [
    "../" * 6 + "tmp/agent6-ipc-escape",
    "..",
    "a/b",
    "/etc/agent6-ipc-escape",
    "",
    ".",
]


def _contained_plain_files(approvals: Path, tmp_path: Path) -> None:
    for child in approvals.iterdir():
        assert child.is_file(), f"the write made a {child}"
        assert child.parent == approvals
    assert not Path("/tmp/agent6-ipc-escape").exists()
    assert not Path("/etc/agent6-ipc-escape").exists()
    assert not (tmp_path.parent / "agent6-ipc-escape").exists()


@pytest.mark.parametrize("bad", HOSTILE)
def test_an_approval_scope_cannot_steer_where_a_grant_lands(tmp_path: Path, bad: str) -> None:
    """`mcp__../../../../tmp/x__t` parses to a server that is a path, and the
    scope becomes a filename: answering "allow all" on that prompt wrote the
    grant clean out of the run directory (observed landing in /tmp)."""
    approvals = approvals_dir(tmp_path)
    for write in (set_session_allow, set_session_deny):
        with contextlib.suppress(ValueError):
            write(tmp_path, f"mcp.{bad}")
    _contained_plain_files(approvals, tmp_path)


@pytest.mark.parametrize("bad", HOSTILE)
def test_an_answer_id_cannot_steer_where_an_answer_lands(tmp_path: Path, bad: str) -> None:
    """The web server answers whatever id the request names."""
    approvals = approvals_dir(tmp_path)
    with contextlib.suppress(ValueError):
        write_answer(tmp_path, bad, "yes")
    _contained_plain_files(approvals, tmp_path)


def test_a_separator_is_refused_rather_than_made_into_a_directory(tmp_path: Path) -> None:
    """The one hostile shape that stays inside the dir and still corrupts the
    layout: every marker is one file, so a scope is one file NAME."""
    with pytest.raises(ValueError, match="unsafe approval scope"):
        set_session_allow(tmp_path, "mcp.a/b")
    with pytest.raises(ValueError, match="unsafe answer id"):
        write_answer(tmp_path, "a/b", "yes")


def test_the_names_a_run_really_uses_still_work(tmp_path: Path) -> None:
    """The guard is a filename check, not a charset policy: every scope and id
    agent6 actually writes has to survive it."""
    for scope in ("command", "mcp.notes", "mcp.some-server_2"):
        set_session_allow(tmp_path, scope)
        assert session_allow_set(tmp_path, scope)
    write_answer(tmp_path, "approval-3", "yes")
    assert read_answer(tmp_path, "approval-3") == "yes"
