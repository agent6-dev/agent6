# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Every directory a writer creates under the state base is 0700 whatever the process umask.

The base shields transcripts, memory and run history from other local users; a writer
that reached a fresh base through a plain `mkdir(parents=True)` (`agent6 init`, the first
command on a new machine) would leave it at the umask's 755 for good, since
`mkdir_for_real_user` never re-chmods a directory that exists.
"""

from __future__ import annotations

import os
import pathlib
from collections.abc import Callable, Iterator

import pytest

from agent6 import events, init, memory, paths
from agent6.graph import storage
from agent6.machine import journal
from agent6.providers import types
from agent6.sessions import ipc, layout
from agent6.tools import background


def _session(state: pathlib.Path) -> pathlib.Path:
    return layout.SessionLayout(state, "s1").session_dir


def _machine_lock(root: pathlib.Path) -> None:
    with journal.machine_lock(root):
        pass


WRITERS: dict[str, Callable[[pathlib.Path, pathlib.Path], object]] = {
    "init": lambda repo, state: init.init_workspace(repo),
    "memory add": lambda repo, state: memory.add(state, "note", "body"),
    "memory decision": lambda repo, state: memory.record_decision(
        state, question="q", answer="a", session="s1"
    ),
    "session layout": lambda repo, state: layout.SessionLayout(state, "s1").ensure(),
    "approvals dir": lambda repo, state: ipc.approvals_dir(_session(state)),
    "questions dir": lambda repo, state: ipc.questions_dir(_session(state)),
    "session grant": lambda repo, state: ipc.set_session_allow(_session(state), "command"),
    "frontend claim": lambda repo, state: ipc.register_frontend(_session(state), os.getpid()),
    "event sink": lambda repo, state: events.EventSink(
        layout.SessionLayout(state, "s1").logs_path
    ).emit("x"),
    "transcripts": lambda repo, state: types.TranscriptSink(
        layout.SessionLayout(state, "s1").transcripts_dir
    ),
    "background shells": lambda repo, state: background.BackgroundShells(_session(state)),
    "graph append": lambda repo, state: storage.append_jsonl(
        layout.SessionLayout(state, "s1").graph_dir / "g.jsonl", {"k": 1}
    ),
    "machine journal": lambda repo, state: journal.MachineJournal(
        layout.machines_root(state) / "m"
    ).ensure_dirs(),
    "machine lock": lambda repo, state: _machine_lock(layout.machines_root(state) / "m"),
    "machine source": lambda repo, state: journal.write_source(
        layout.machines_root(state) / "m", "x = 1\n"
    ),
    "machine stop": lambda repo, state: journal.write_stop_request(
        layout.machines_root(state) / "m"
    ),
}


@pytest.fixture
def umask_022() -> Iterator[None]:
    old = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(old)


def _open_dirs(base: pathlib.Path) -> list[str]:
    return [
        str(p.relative_to(base.parent))
        for p in (base, *base.rglob("*"))
        if p.is_dir() and (p.stat().st_mode & 0o777) != 0o700
    ]


@pytest.mark.parametrize("writer", sorted(WRITERS))
def test_every_dir_a_writer_creates_under_the_state_base_is_0700(
    writer: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, umask_022: None
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    state = paths.state_dir(repo)
    assert not paths.state_base().exists()
    WRITERS[writer](repo, state)
    assert paths.state_base().is_dir(), "the writer created nothing under the state base"
    assert _open_dirs(paths.state_base()) == []
