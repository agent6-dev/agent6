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
import pathlib

import pytest

from agent6.sessions import ipc

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


def _contained_plain_files(approvals: pathlib.Path, tmp_path: pathlib.Path) -> None:
    for child in approvals.iterdir():
        assert child.is_file(), f"the write made a {child}"
        assert child.parent == approvals
    assert not pathlib.Path("/tmp/agent6-ipc-escape").exists()
    assert not pathlib.Path("/etc/agent6-ipc-escape").exists()
    assert not (tmp_path.parent / "agent6-ipc-escape").exists()


@pytest.mark.parametrize("bad", HOSTILE)
def test_an_approval_scope_cannot_steer_where_a_grant_lands(
    tmp_path: pathlib.Path, bad: str
) -> None:
    """A scope that is a path cannot write a grant outside the run directory.

    `mcp__../../../../tmp/x__t` parses to a server that is a path, and the scope becomes a
    filename; answering "allow all" on that prompt would land the grant in /tmp.
    """
    approvals = ipc.approvals_dir(tmp_path)
    for write in (ipc.set_session_allow, ipc.set_session_deny):
        with contextlib.suppress(ValueError):
            write(tmp_path, f"mcp.{bad}")
    _contained_plain_files(approvals, tmp_path)


@pytest.mark.parametrize("bad", HOSTILE)
def test_an_answer_id_cannot_steer_where_an_answer_lands(tmp_path: pathlib.Path, bad: str) -> None:
    """The web server answers whatever id the request names."""
    approvals = ipc.approvals_dir(tmp_path)
    with contextlib.suppress(ValueError):
        ipc.write_answer(tmp_path, bad, "yes")
    _contained_plain_files(approvals, tmp_path)


def test_a_separator_is_refused_rather_than_made_into_a_directory(tmp_path: pathlib.Path) -> None:
    """The one hostile shape that stays inside the dir and still corrupts the layout.

    Every marker is one file, so a scope is one file name.
    """
    with pytest.raises(ValueError, match="unsafe approval scope"):
        ipc.set_session_allow(tmp_path, "mcp.a/b")
    with pytest.raises(ValueError, match="unsafe answer id"):
        ipc.write_answer(tmp_path, "a/b", "yes")


def test_the_names_a_run_really_uses_still_work(tmp_path: pathlib.Path) -> None:
    """The guard is a filename check, not a charset policy.

    Every scope and id agent6 writes has to survive it.
    """
    for scope in ("command", "mcp.notes", "mcp.some-server_2"):
        ipc.set_session_allow(tmp_path, scope)
        assert ipc.session_allow_set(tmp_path, scope)
    ipc.write_answer(tmp_path, "approval-3", "yes")
    assert ipc.read_answer(tmp_path, "approval-3") == "yes"
