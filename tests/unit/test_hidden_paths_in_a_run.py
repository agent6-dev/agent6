# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A hidden path inside the repository through a whole run: never stored, never stranded."""

from __future__ import annotations

import json
import pathlib
import subprocess
from typing import Any

import pytest

from agent6.app import _session, providers
from agent6.ui.cli import cli_main
from tests.unit import test_queued_question_lifecycle as lifecycle

_CONFIG = """
[sandbox]
isolation = "none"
run_commands = "allow"
hide_paths = [{hide}]

[git]
dirty_tree = "{dirty}"

[models.worker]
provider = "p"
model = "m"

[providers.p]
api_format = "openai"
base_url = "http://127.0.0.1:1"
api_key_env = "PROBE_KEY"

[review]
trigger = "off"
"""


def _git(root: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _in_store(root: pathlib.Path, content: str) -> bool:
    blob = subprocess.run(
        ["git", "-C", str(root), "hash-object", "--stdin"],
        input=content,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return (
        subprocess.run(["git", "-C", str(root), "cat-file", "-e", blob], check=False).returncode
        == 0
    )


def _command(id_: str, script: str) -> lifecycle._Step:  # pyright: ignore[reportPrivateUsage]
    tool = {
        "type": "tool_use",
        "id": id_,
        "name": "run_command",
        "input": {"argv": ["sh", "-c", script]},
    }
    return ("", (tool,))


def _setup(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, *, hide: tuple[str, ...], dirty: str
) -> pathlib.Path:
    repo = tmp_path / "repo"
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    cfg_home = tmp_path / "cfg"
    (cfg_home / "agent6").mkdir(parents=True)
    (cfg_home / "agent6" / "config.toml").write_text(
        _CONFIG.format(hide=", ".join(json.dumps(str(repo / h)) for h in hide), dirty=dirty),
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg_home))
    monkeypatch.setenv("PROBE_KEY", "sk-probe")
    monkeypatch.delenv("AGENT6_DETACHED_AWAY", raising=False)
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "secrets").mkdir()
    (repo / "secrets" / "key.txt").write_text("OLD-COMMITTED\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "seed")
    monkeypatch.chdir(repo)
    monkeypatch.setattr(_session, "select_isolation", lifecycle._none_isolation)  # pyright: ignore[reportPrivateUsage]
    return repo


def _run(monkeypatch: pytest.MonkeyPatch, session_id: str, steps: list[Any]) -> int:
    monkeypatch.setattr(
        providers, "build_role_provider", lifecycle._use(lifecycle._Scripted(steps))
    )  # pyright: ignore[reportPrivateUsage]
    return cli_main(["run", "--session-id", session_id, "task"])


def test_a_hidden_file_a_command_writes_reaches_no_commit_and_no_object(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The model writes `.env` and a hidden tracked file: the chain keeps neither's content."""
    repo = _setup(tmp_path, monkeypatch, hide=(".env", "secrets"), dirty="ask")
    edit = {
        "type": "tool_use",
        "id": "e1",
        "name": "apply_edit",
        "input": {"path": "a.py", "old_string": "x = 1\n", "new_string": "x = 2\n"},
    }
    rc = _run(
        monkeypatch,
        "hidden-RUN001",
        [
            _command(
                "c1", "printf 'TOKEN=abc\\n' > .env; printf 'NEW-SECRET-7777\\n' > secrets/key.txt"
            ),
            ("", (edit,)),
            lifecycle._finish(),  # pyright: ignore[reportPrivateUsage]
        ],
    )
    assert rc == 0
    assert (repo / ".env").read_text(encoding="utf-8") == "TOKEN=abc\n"
    tip = _git(repo, "rev-parse", "refs/agent6/hidden-RUN001/head")
    assert ".env" not in _git(repo, "ls-tree", "-r", "--name-only", tip).splitlines()
    assert _git(repo, "show", f"{tip}:secrets/key.txt") == "OLD-COMMITTED"
    assert not _in_store(repo, "TOKEN=abc\n")
    assert not _in_store(repo, "NEW-SECRET-7777\n")


def test_a_run_whose_only_write_is_a_hidden_file_is_not_stranded(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Nothing the run owns is left uncommitted: exit 0, no stranded warning, no stranded mark."""
    _setup(tmp_path, monkeypatch, hide=(".env",), dirty="ask")
    rc = _run(
        monkeypatch,
        "hidden-RUN002",
        [_command("c1", "printf 'TOKEN=abc\\n' > .env"), lifecycle._finish()],  # pyright: ignore[reportPrivateUsage]
    )
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "the run finished with no commit" not in out


def test_a_pre_run_stash_leaves_a_hidden_tracked_files_change_in_place(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The auto-stash takes the other changes; a hidden path's change stays in place, unstored."""
    repo = _setup(tmp_path, monkeypatch, hide=("secrets",), dirty="stash")
    (repo / "secrets" / "key.txt").write_text("OPERATOR-UNCOMMITTED-SECRET\n", encoding="utf-8")
    (repo / "a.py").write_text("x = 9\n", encoding="utf-8")
    assert _run(monkeypatch, "hidden-RUN003", [lifecycle._finish()]) == 0  # pyright: ignore[reportPrivateUsage]
    assert (repo / "secrets" / "key.txt").read_text(
        encoding="utf-8"
    ) == "OPERATOR-UNCOMMITTED-SECRET\n"
    assert not _in_store(repo, "OPERATOR-UNCOMMITTED-SECRET\n")
    assert _git(repo, "diff", "--name-only", "stash@{0}^", "stash@{0}").splitlines() == ["a.py"]
    assert _git(repo, "show", "stash@{0}:a.py") == "x = 9"
