from __future__ import annotations

import json
import pathlib
import shlex
from typing import Any

import pytest

from agent6 import paths
from agent6.sessions import manifest as sessions_manifest
from agent6.ui.cli import cli_main
from tests.unit import test_run_worktree as worktree_tests


@pytest.mark.needs_namespaces
@pytest.mark.parametrize("protect_git", ["auto", "on"])
def test_a_user_worktree_session_reads_git_but_cannot_write_metadata(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, protect_git: str
) -> None:
    """The start grants git reads, keeps the git dir read-only and hides the masked files."""
    repo = worktree_tests._repo(tmp_path, monkeypatch)
    checkout = tmp_path / "user-worktree"
    worktree_tests._git(repo, "worktree", "add", "--detach", str(checkout), "HEAD")
    git_dir = repo / ".git"
    admin = git_dir / "worktrees" / checkout.name
    hidden = admin / "private.txt"
    hidden.write_text("hidden-marker\n", encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(git_dir / "config-home"))
    monkeypatch.setenv("XDG_STATE_HOME", str(git_dir / "state-home"))
    paths.global_config_dir().mkdir(parents=True)
    paths.secrets_path().write_text("# secret-marker\n", encoding="utf-8")
    paths.secrets_path().chmod(0o600)
    config = worktree_tests._CONFIG.replace('isolation = "none"', 'isolation = "full"')
    config = config.replace(
        'run_commands = "deny"',
        f'run_commands = "allow"\nnetwork = "session"\nprotect_git = "{protect_git}"\n'
        f"hide_paths = [{json.dumps(str(hidden))}]",
    )
    paths.global_config_path().write_text(config, encoding="utf-8")
    monkeypatch.chdir(checkout)
    (checkout / "a.py").write_text("x = 3\n", encoding="utf-8")
    config += '\n[git]\ndirty_tree = "include"\n'
    paths.global_config_path().write_text(config, encoding="utf-8")
    index_before = (admin / "index").read_bytes()

    # Each mask names itself, so a failure says which one let something through.
    probe = "; ".join(
        f"{check} && echo {name}-held || echo {name}-OPEN"
        for name, check in (
            ("main-checkout", f"test ! -e {shlex.quote(str(repo / 'a.py'))}"),
            (
                "config-dir",
                f'test -z "$(ls -A {shlex.quote(str(paths.global_config_dir()))} 2>/dev/null)"',
            ),
            # The run's own `shells/logs` dir is bound by design, so no file may be visible.
            (
                "state-files",
                f'test -z "$(find {shlex.quote(str(paths.state_base()))} -type f 2>/dev/null)"',
            ),
            ("hidden-file", f"test ! -s {shlex.quote(str(hidden))}"),
        )
    )
    commands = (
        ["git", "status", "--porcelain"],
        ["git", "log", "-1", "--format=%s"],
        # An identity, so the commit reaches the write the read-only bind refuses.
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            "commit",
            "--allow-empty",
            "-m",
            "must not commit",
        ],
        ["sh", "-c", probe],
    )
    turns: list[tuple[dict[str, Any], ...]] = [
        ({"type": "tool_use", "id": f"cmd-{i}", "name": "run_command", "input": {"argv": argv}},)
        for i, argv in enumerate(commands)
    ]
    worktree_tests._script(monkeypatch, *turns)
    assert cli_main(["run", "--session-id", "jailed-linked", "inspect git"]) == 0
    session_dir = worktree_tests._run_dir(repo, "jailed-linked")
    saved = sessions_manifest.read_manifest(session_dir)
    assert saved.worktree is None and saved.worktree_git_dir == git_dir
    events = [json.loads(line) for line in (session_dir / "logs.jsonl").read_text().splitlines()]
    results = [e for e in events if e["type"] == "tool.result" and e["name"] == "run_command"]
    assert len(results) == 4
    status, log, commit, masks = results
    assert status["summary"].startswith("exit=0 "), status
    assert " M a.py" in status["stdout_tail"]
    assert log["summary"].startswith("exit=0 ") and log["stdout_tail"].strip() == "seed", log
    assert not commit["summary"].startswith("exit=0 "), commit
    assert "Read-only file system" in commit["stderr_tail"], commit
    assert "OPEN" not in masks["stdout_tail"] and masks["stdout_tail"].count("-held") == 4, masks
    assert (admin / "index").read_bytes() == index_before
    assert hidden.read_text(encoding="utf-8") == "hidden-marker\n"


@pytest.mark.needs_namespaces
def test_a_write_grant_inside_the_git_dir_refuses_the_start(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The jail protects paths under the workspace only, so the start refuses the grant by name."""
    repo = worktree_tests._repo(tmp_path, monkeypatch)
    checkout = tmp_path / "user-worktree"
    worktree_tests._git(repo, "worktree", "add", "--detach", str(checkout), "HEAD")
    admin = repo / ".git" / "worktrees" / checkout.name
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config-home"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state-home"))
    paths.global_config_dir().mkdir(parents=True)
    config = worktree_tests._CONFIG.replace('isolation = "none"', 'isolation = "full"')
    config = config.replace(
        'run_commands = "deny"',
        f'run_commands = "allow"\nnetwork = "session"\n'
        f"extra_write_paths = [{json.dumps(str(admin))}]",
    )
    paths.global_config_path().write_text(config, encoding="utf-8")
    monkeypatch.chdir(checkout)
    worktree_tests._script(monkeypatch)
    assert cli_main(["run", "--session-id", "jailed-linked-grant", "inspect git"]) == 2
    err = capsys.readouterr().err
    assert str(admin) in err and "inside the repository git dir" in err
