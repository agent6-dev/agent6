# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A run absorbed by `dirty_tree = "include"`, and a merge over an edited file.

Runs here are real `agent6 run` executions over a scripted model, in the base checkout.
"""

from __future__ import annotations

import pathlib
import subprocess as sp
from typing import Any

import pytest

from agent6 import git_ops, paths
from agent6.app import _session, providers
from agent6.providers import ProviderResponse
from agent6.sessions import manifest as sessions_manifest
from agent6.ui.cli import cli_main

_CONFIG = """
[sandbox]
isolation = "none"
run_commands = "no"

[models.worker]
provider = "p"
model = "m"

[providers.p]
api_format = "openai"
base_url = "http://127.0.0.1:1"
api_key_env = "PROBE_KEY"

[review]
trigger = "off"

[git]
dirty_tree = "include"
"""

_SEED = "x = 1\n\n\n\n\n\n\ny = 1\n"
_Step = tuple[str, tuple[dict[str, Any], ...]]


class _Scripted:
    """One canned ProviderResponse per call; the last repeats."""

    def __init__(self, script: list[_Step]) -> None:
        self.script = script
        self.calls = 0

    def call(self, **_kw: Any) -> ProviderResponse:
        text, tools = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return ProviderResponse(
            text=text,
            tool_uses=tools,
            stop_reason="tool_use" if tools else "end_turn",
            input_tokens=10,
            output_tokens=5,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            raw={"content": [*([{"type": "text", "text": text}] if text else []), *tools]},
        )


def _edit(path: str, content: str) -> _Step:
    tool = {
        "type": "tool_use",
        "id": "e1",
        "name": "apply_edit",
        "input": {"path": path, "old_string": "", "new_string": content, "kind": "overwrite"},
    }
    return ("", (tool,))


def _finish() -> _Step:
    tool = {"type": "tool_use", "id": "t1", "name": "finish_session", "input": {"summary": "done"}}
    return ("", (tool,))


def _none_isolation(*_a: Any, **_k: Any) -> str:
    return "none"


def _git(repo: pathlib.Path, *args: str) -> str:
    return sp.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _setup(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    cfg_home = tmp_path / "cfg"
    (cfg_home / "agent6").mkdir(parents=True)
    (cfg_home / "agent6" / "config.toml").write_text(_CONFIG, encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg_home))
    monkeypatch.setenv("PROBE_KEY", "sk-probe")
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "a.py").write_text(_SEED, encoding="utf-8")
    (repo / "b.py").write_text("z = 1\n", encoding="utf-8")
    _git(repo, "add", "a.py", "b.py")
    _git(repo, "commit", "-q", "-m", "seed")
    monkeypatch.chdir(repo)
    monkeypatch.setattr(_session, "select_isolation", _none_isolation)
    return repo


def _run(monkeypatch: pytest.MonkeyPatch, session_id: str, script: list[_Step]) -> None:
    prov = _Scripted(script)

    def _factory(*_a: Any, **_k: Any) -> _Scripted:
        return prov

    monkeypatch.setattr(providers, "build_role_provider", _factory)
    assert cli_main(["run", "--session-id", session_id, "t"]) == 0


def _manifest(repo: pathlib.Path, session_id: str) -> sessions_manifest.SessionManifest:
    return sessions_manifest.read_manifest(paths.state_dir(repo) / "sessions" / "runs" / session_id)


def test_merging_the_including_run_lands_the_run_it_absorbed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run whose work a later run took in by "include" reads merged once that run lands.

    The include was recorded nowhere, so the absorbed run stayed "unmerged" and its own merge
    conflicted with its work already on the base.
    """
    repo = _setup(tmp_path, monkeypatch)
    _run(monkeypatch, "inc-FIRST1", [_edit("a.py", _SEED.replace("x = 1", "x = 2")), _finish()])
    first_tip = git_ops.chain_tip(repo, git_ops.chain_ref_for("inc-FIRST1"))
    _run(monkeypatch, "inc-SECOND", [_edit("a.py", _SEED.replace("x = 1", "x = 3")), _finish()])

    included = _manifest(repo, "inc-SECOND").includes
    assert included is not None
    assert (included.session_id, included.tip) == ("inc-FIRST1", first_tip)
    assert cli_main(["sessions", "merge", "inc-SECOND"]) == 0
    capsys.readouterr()

    stamp = _manifest(repo, "inc-FIRST1").merged
    assert stamp is not None
    assert (stamp.into, stamp.tip, stamp.via) == ("main", first_tip, "inc-SECOND")
    assert stamp.sha == _git(repo, "rev-parse", "main")
    assert cli_main(["sessions", "merge", "inc-FIRST1"]) == 0
    assert "CONFLICT" not in capsys.readouterr().err
    assert _git(repo, "show", "main:a.py").splitlines()[0] == "x = 3"


def test_include_records_nothing_when_the_tree_holds_more_than_the_runs_tip(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An operator edit on top of the earlier run's work leaves that run unmerged."""
    repo = _setup(tmp_path, monkeypatch)
    _run(monkeypatch, "inc-FIRST2", [_edit("a.py", _SEED.replace("x = 1", "x = 2")), _finish()])
    (repo / "b.py").write_text("z = 2  # operator\n", encoding="utf-8")
    _run(monkeypatch, "inc-THIRD1", [_edit("c.py", "c = 1\n"), _finish()])

    assert _manifest(repo, "inc-THIRD1").includes is None
    assert cli_main(["sessions", "merge", "inc-THIRD1"]) == 0
    assert _manifest(repo, "inc-FIRST2").merged is None


def test_an_included_run_that_committed_again_is_not_stamped(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The include is checked again at the merge: a run that advanced since stays unmerged."""
    repo = _setup(tmp_path, monkeypatch)
    _run(monkeypatch, "inc-FIRST3", [_edit("a.py", _SEED.replace("x = 1", "x = 2")), _finish()])
    _run(monkeypatch, "inc-FOURTH", [_edit("c.py", "c = 1\n"), _finish()])
    assert _manifest(repo, "inc-FOURTH").includes is not None
    chain = git_ops.chain_ref_for("inc-FIRST3")
    later = _git(repo, "commit-tree", f"{chain}^{{tree}}", "-p", chain, "-m", "a later step")
    _git(repo, "update-ref", chain, later)

    assert cli_main(["sessions", "merge", "inc-FOURTH"]) == 0

    assert _manifest(repo, "inc-FIRST3").merged is None


def test_a_merge_over_an_edited_file_prints_the_commands_that_reconcile_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A merge over an uncommitted edit of a file the run changed says how to reconcile the two.

    The checkout kept its own copy, which undid the merged change in that file, and the one
    line printed gave no way to bring the copy and the merge together.
    """
    repo = _setup(tmp_path, monkeypatch)
    _run(monkeypatch, "edit-OVER01", [_edit("a.py", _SEED.replace("x = 1", "x = 2")), _finish()])
    before = _git(repo, "rev-parse", "main")
    # The checkout as a `run --worktree` leaves it: the base's file, plus the operator's edit.
    (repo / "a.py").write_text(_SEED.replace("y = 1", "y = 9  # mine"), encoding="utf-8")
    capsys.readouterr()

    assert cli_main(["sessions", "merge", "edit-OVER01"]) == 0

    said = capsys.readouterr()
    lines = (said.out + said.err).splitlines()
    assert any("your checkout keeps its own a.py" in line for line in lines)
    commands = [line.strip() for line in lines if line.strip().startswith("git ")]
    assert len(commands) == 2 and before[:12] in commands[0]
    for command in commands:
        sp.run(["sh", "-c", command], cwd=repo, check=True, capture_output=True)
    merged = _SEED.replace("x = 1", "x = 2").replace("y = 1", "y = 9  # mine")
    assert (repo / "a.py").read_text(encoding="utf-8") == merged
