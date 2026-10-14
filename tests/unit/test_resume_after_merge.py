# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 resume` after the run's work landed, and after an operator commit on its branch.

Every resume here runs in the base checkout, the way a real one does: the run branch is never
checked out.
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
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "add", "a.py")
    _git(repo, "commit", "-q", "-m", "seed")
    monkeypatch.chdir(repo)
    monkeypatch.setattr(_session, "select_isolation", _none_isolation)
    return repo


def _play(monkeypatch: pytest.MonkeyPatch, argv: list[str], script: list[_Step]) -> int:
    prov = _Scripted(script)

    def _factory(*_a: Any, **_k: Any) -> _Scripted:
        return prov

    monkeypatch.setattr(providers, "build_role_provider", _factory)
    return cli_main(argv)


def _manifest(repo: pathlib.Path, session_id: str) -> sessions_manifest.SessionManifest:
    return sessions_manifest.read_manifest(paths.state_dir(repo) / "sessions" / "runs" / session_id)


def _commit_on_main(repo: pathlib.Path, name: str) -> None:
    (repo / name).write_text("operator\n", encoding="utf-8")
    _git(repo, "add", name)
    _git(repo, "commit", "-q", "-m", f"add {name}")


def _changed(repo: pathlib.Path, old: str, new: str) -> list[str]:
    return _git(repo, "diff", "--name-only", old, new).splitlines()


def test_a_resume_after_a_squash_merge_continues_from_head(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """A resumed run whose work landed commits, diffs and merges only its new work.

    The resume parented on the old chain tip and committed the checkout's whole tree, so its
    next commit took the commits the base gained since, `sessions diff` showed work that had
    already landed, and the note called committed content "changes no commit has".
    """
    repo = _setup(tmp_path, monkeypatch)
    run = ["run", "--session-id", "land-SQUASH", "t"]
    assert _play(monkeypatch, run, [_edit("z.txt", "1\n"), _finish()]) == 0
    assert cli_main(["sessions", "merge", "land-SQUASH"]) == 0
    _commit_on_main(repo, "r.md")
    head = _git(repo, "rev-parse", "HEAD")
    capfd.readouterr()

    resume = ["resume", "land-SQUASH", "--steer", "more"]
    assert _play(monkeypatch, resume, [_edit("y.txt", "1\n"), _finish()]) == 0

    assert "holds changes no commit has" not in capfd.readouterr().err
    base = _manifest(repo, "land-SQUASH").base_sha
    chain = git_ops.chain_ref_for("land-SQUASH")
    assert _git(repo, "rev-parse", f"{base}^{{tree}}") == _git(
        repo, "rev-parse", f"{head}^{{tree}}"
    )
    assert _changed(repo, base, chain) == ["y.txt"]
    assert cli_main(["sessions", "diff", "land-SQUASH", "--stat"]) == 0
    shown = capfd.readouterr().out
    assert "y.txt" in shown and "r.md" not in shown and "z.txt" not in shown
    assert cli_main(["sessions", "merge", "land-SQUASH"]) == 0
    assert _changed(repo, head, "main") == ["y.txt"]


def test_a_resumed_run_merges_cleanly_after_prune_delete_squashed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run resumed after its refs were pruned starts its chain on HEAD and lands cleanly.

    With the chain ref gone the resume parented on the run's original base, so the new chain
    held the landed work a second time and its merge conflicted with the squash on the base.
    """
    repo = _setup(tmp_path, monkeypatch)
    run = ["run", "--session-id", "land-PRUNED", "t"]
    assert _play(monkeypatch, run, [_edit("a.py", "x = 2\n"), _finish()]) == 0
    assert cli_main(["sessions", "merge", "land-PRUNED"]) == 0
    _commit_on_main(repo, "r.md")
    assert cli_main(["sessions", "prune", "--delete-squashed"]) == 0
    assert git_ops.chain_tip(repo, git_ops.chain_ref_for("land-PRUNED")) is None
    head = _git(repo, "rev-parse", "HEAD")

    resume = ["resume", "land-PRUNED", "--steer", "more"]
    assert _play(monkeypatch, resume, [_edit("a.py", "x = 3\n"), _finish()]) == 0
    capsys.readouterr()

    assert _manifest(repo, "land-PRUNED").base_sha == head
    assert cli_main(["sessions", "merge", "land-PRUNED"]) == 0
    assert "CONFLICT" not in capsys.readouterr().err
    assert _changed(repo, head, "main") == ["a.py"]
    assert _git(repo, "show", "main:a.py") == "x = 3"


def test_a_resume_keeps_its_chain_when_the_checkout_lacks_the_landing_commit(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Merged into a branch that is not checked out, the run continues on its own chain."""
    repo = _setup(tmp_path, monkeypatch)
    _git(repo, "branch", "other")
    run = ["run", "--session-id", "land-OTHER1", "t"]
    assert _play(monkeypatch, run, [_edit("z.txt", "1\n"), _finish()]) == 0
    base = _manifest(repo, "land-OTHER1").base_sha
    tip = git_ops.chain_tip(repo, git_ops.chain_ref_for("land-OTHER1"))
    assert cli_main(["sessions", "merge", "land-OTHER1", "--into", "other"]) == 0

    resume = ["resume", "land-OTHER1", "--steer", "more"]
    assert _play(monkeypatch, resume, [_edit("y.txt", "1\n"), _finish()]) == 0

    assert _manifest(repo, "land-OTHER1").base_sha == base
    chain = git_ops.chain_ref_for("land-OTHER1")
    assert _git(repo, "rev-parse", f"{chain}~1") == tip


def _commit_on_the_run_branch(repo: pathlib.Path, session_id: str) -> str:
    """Add a commit to `agent6/<id>` the way an operator in another worktree would."""
    branch = f"refs/heads/{git_ops.run_branch_for(session_id)}"
    tree = _git(repo, "rev-parse", f"{branch}^{{tree}}")
    sha = _git(repo, "commit-tree", tree, "-p", branch, "-m", "operator note")
    _git(repo, "update-ref", branch, sha)
    return sha


def test_resume_refuses_when_the_run_branch_holds_a_commit_the_chain_lacks(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A commit made on `agent6/<id>` refuses the resume, which would leave it behind.

    The resume read only the chain ref, so it committed past the operator's commit and the
    run's merge then landed without it, with no word at either step.
    """
    repo = _setup(tmp_path, monkeypatch)
    run = ["run", "--session-id", "land-BRANCH", "t"]
    assert _play(monkeypatch, run, [_edit("z.txt", "1\n"), _finish()]) == 0
    chain = git_ops.chain_ref_for("land-BRANCH")
    tip = git_ops.chain_tip(repo, chain)
    theirs = _commit_on_the_run_branch(repo, "land-BRANCH")
    capsys.readouterr()

    resume = ["resume", "land-BRANCH", "--steer", "more"]
    assert _play(monkeypatch, resume, [_edit("y.txt", "1\n"), _finish()]) == 2

    err = capsys.readouterr().err
    assert "agent6/land-BRANCH" in err and theirs[:12] in err
    assert "agent6 sessions merge land-BRANCH" in err and "--force" in err
    assert git_ops.chain_tip(repo, chain) == tip
    assert git_ops.branch_tip_sha(repo, "agent6/land-BRANCH") == theirs

    forced = [*resume, "--force"]
    assert _play(monkeypatch, forced, [_edit("y.txt", "1\n"), _finish()]) == 0
    assert "agent6/land-BRANCH stays at" in capsys.readouterr().err
    assert git_ops.branch_tip_sha(repo, "agent6/land-BRANCH") == theirs
    assert _git(repo, "rev-parse", f"{chain}~1") == tip


def test_a_resume_after_landing_an_operator_commit_on_the_run_branch_continues_from_head(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The refusal's first way out works: merge the branch with the commit, then resume."""
    repo = _setup(tmp_path, monkeypatch)
    run = ["run", "--session-id", "land-BOTH01", "t"]
    assert _play(monkeypatch, run, [_edit("z.txt", "1\n"), _finish()]) == 0
    _commit_on_the_run_branch(repo, "land-BOTH01")
    assert cli_main(["sessions", "merge", "land-BOTH01"]) == 0
    head = _git(repo, "rev-parse", "HEAD")

    resume = ["resume", "land-BOTH01", "--steer", "more"]
    assert _play(monkeypatch, resume, [_edit("y.txt", "1\n"), _finish()]) == 0

    chain = git_ops.chain_ref_for("land-BOTH01")
    assert _changed(repo, _manifest(repo, "land-BOTH01").base_sha, chain) == ["y.txt"]
    assert git_ops.branch_tip_sha(repo, "agent6/land-BOTH01") == git_ops.chain_tip(repo, chain)
    assert cli_main(["sessions", "merge", "land-BOTH01"]) == 0
    assert _changed(repo, head, "main") == ["y.txt"]


def test_a_fork_from_a_turn_before_the_source_landed_starts_at_that_turn(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fork cut before its source's base moved does not inherit the moved base.

    The source's base is the commit it continued from after landing; a fork of an earlier turn
    does not descend from it, so its diff against that base would read backwards.
    """
    repo = _setup(tmp_path, monkeypatch)
    run = ["run", "--session-id", "land-FORK01", "t"]
    assert _play(monkeypatch, run, [_edit("z.txt", "1\n"), _finish()]) == 0
    start = _manifest(repo, "land-FORK01").base_sha
    assert cli_main(["sessions", "merge", "land-FORK01"]) == 0
    _commit_on_main(repo, "r.md")
    resume = ["resume", "land-FORK01", "--steer", "more"]
    assert _play(monkeypatch, resume, [_edit("y.txt", "1\n"), _finish()]) == 0
    assert _manifest(repo, "land-FORK01").base_sha != start

    fork = ["fork", "land-FORK01", "--at-turn", "1", "--no-run", "--session-id", "land-FORK02"]
    assert cli_main(fork) == 0

    child = _manifest(repo, "land-FORK02")
    assert child.base_sha == child.forked_from_sha == start


def test_a_resumed_fork_of_a_squash_merged_run_shows_only_its_own_work(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """A fork resumed after its source landed diffs and lands only the fork's own work.

    The fork kept its source's base, so `sessions diff` showed the source's landed file
    while the fork's merge landed only its own.
    """
    repo = _setup(tmp_path, monkeypatch)
    run = ["run", "--session-id", "land-SRC001", "t"]
    assert _play(monkeypatch, run, [_edit("z.txt", "1\n"), _finish()]) == 0
    fork = ["fork", "land-SRC001", "--no-run", "--session-id", "land-SRC002"]
    assert cli_main(fork) == 0
    assert cli_main(["sessions", "merge", "land-SRC001"]) == 0
    landed = _git(repo, "rev-parse", "main")

    resume = ["resume", "land-SRC002", "--steer", "more"]
    assert _play(monkeypatch, resume, [_edit("y.txt", "1\n"), _finish()]) == 0
    capfd.readouterr()

    assert cli_main(["sessions", "diff", "land-SRC002", "--stat"]) == 0
    shown = capfd.readouterr().out
    assert "y.txt" in shown and "z.txt" not in shown
    assert cli_main(["sessions", "merge", "land-SRC002"]) == 0
    assert _changed(repo, landed, "main") == ["y.txt"]


def test_a_run_merged_into_another_branch_continues_there_once_it_is_checked_out(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The base branch follows the merge only onto the branch it landed on.

    A checkout on any other branch that holds the landing commit keeps the recorded base
    branch, so the run's next merge does not target a branch nothing was merged into.
    """
    repo = _setup(tmp_path, monkeypatch)
    _git(repo, "branch", "other")
    run = ["run", "--session-id", "land-INTO01", "t"]
    assert _play(monkeypatch, run, [_edit("z.txt", "1\n"), _finish()]) == 0
    assert cli_main(["sessions", "merge", "land-INTO01", "--into", "other"]) == 0
    (repo / "z.txt").unlink()
    _git(repo, "switch", "-q", "other")

    resume = ["resume", "land-INTO01", "--steer", "more"]
    assert _play(monkeypatch, resume, [_edit("y.txt", "1\n"), _finish()]) == 0

    manifest = _manifest(repo, "land-INTO01")
    assert manifest.base_branch == "other"
    assert _changed(repo, manifest.base_sha, git_ops.chain_ref_for("land-INTO01")) == ["y.txt"]

    assert cli_main(["sessions", "merge", "land-INTO01"]) == 0
    _git(repo, "switch", "-q", "-c", "side")
    assert _play(monkeypatch, resume, [_edit("x.txt", "1\n"), _finish()]) == 0
    manifest = _manifest(repo, "land-INTO01")
    assert manifest.base_branch == "other"
    assert _changed(repo, manifest.base_sha, git_ops.chain_ref_for("land-INTO01")) == ["x.txt"]
