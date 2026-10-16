# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 run --worktree`: the run works in its own linked worktree, the checkout untouched.

Driven through `cli_main` with a scripted provider over a real git repository, so the
worktree, the manifest, the chain and the sessions verbs are the real ones.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import pathlib
import shlex
import subprocess as sp
import time
from typing import Any

import pytest

from agent6 import budget as agent6_budget
from agent6 import errors, git_ops, kinds, paths
from agent6.app import (
    _session,
    fork_worktrees,
    parallel,
    preflight,
    providers,
    workspace_setup,
)
from agent6.app import lanes as app_lanes
from agent6.app import reporter as app_reporter
from agent6.app import run as run_lifecycle
from agent6.config import Config, layer
from agent6.harness import subrun
from agent6.providers import ProviderResponse
from agent6.sessions import ipc, lock
from agent6.sessions import manifest as sessions_manifest
from agent6.ui.cli import cli_main
from agent6.ui.cli import parallel as parallel_cmd
from agent6.ui.cli import run as cli_run
from agent6.viewmodel import listing


@pytest.fixture(autouse=True)
def _no_controlling_terminal(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run as CI does, with no controlling terminal: `rm` without `--yes` asks on one."""
    monkeypatch.setattr("agent6.portable.has_controlling_tty", lambda: False)


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
"""


class _Scripted:
    """One canned ProviderResponse per call; the last repeats."""

    def __init__(self, script: list[tuple[dict[str, Any], ...]]) -> None:
        self.script = script
        self.calls = 0

    def call(self, **_kw: Any) -> ProviderResponse:
        tools = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return ProviderResponse(
            text="",
            tool_uses=tools,
            stop_reason="tool_use",
            input_tokens=10,
            output_tokens=5,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            raw={"content": list(tools)},
        )


def _edit(path: str, content: str) -> tuple[dict[str, Any], ...]:
    return (
        {
            "type": "tool_use",
            "id": "e1",
            "name": "apply_edit",
            "input": {"path": path, "old_string": "", "new_string": content, "kind": "overwrite"},
        },
    )


_FINISH: tuple[dict[str, Any], ...] = (
    {"type": "tool_use", "id": "t1", "name": "finish_session", "input": {"summary": "done"}},
)


def _script(monkeypatch: pytest.MonkeyPatch, *turns: tuple[dict[str, Any], ...]) -> _Scripted:
    prov = _Scripted([*turns, _FINISH])

    def _factory(*_a: Any, **_k: Any) -> _Scripted:
        return prov

    monkeypatch.setattr(providers, "build_role_provider", _factory)
    return prov


def _git(repo: pathlib.Path, *args: str) -> str:
    return sp.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout


def _repo(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, *, config: str = _CONFIG
) -> pathlib.Path:
    """A committed repo, the process cwd, holding the operator's uncommitted work."""
    cfg_home = tmp_path / "cfg"
    (cfg_home / "agent6").mkdir(parents=True)
    (cfg_home / "agent6" / "config.toml").write_text(config, encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg_home))
    monkeypatch.setenv("PROBE_KEY", "sk-probe")
    monkeypatch.delenv("AGENT6_DETACHED_AWAY", raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "add", "a.py")
    _git(repo, "commit", "-q", "-m", "seed")
    (repo / "a.py").write_text("x = 2  # the operator's edit\n", encoding="utf-8")
    (repo / "notes.md").write_text("mine\n", encoding="utf-8")
    monkeypatch.chdir(repo)
    return repo


def _checkout_facts(repo: pathlib.Path) -> tuple[str, str, str, str]:
    """HEAD, the branch, the status and the stash list: what a run must not move."""
    return (
        _git(repo, "rev-parse", "HEAD"),
        _git(repo, "branch", "--show-current"),
        _git(repo, "status", "--porcelain"),
        _git(repo, "stash", "list"),
    )


def _worktrees(repo: pathlib.Path) -> list[str]:
    listed = _git(repo, "worktree", "list", "--porcelain").splitlines()
    return [line.removeprefix("worktree ") for line in listed if line.startswith("worktree ")]


def _run_dir(repo: pathlib.Path, session_id: str) -> pathlib.Path:
    return paths.state_dir(repo) / "sessions" / "runs" / session_id


def _nothing_was_created(repo: pathlib.Path, session_id: str) -> bool:
    return _worktrees(repo) == [str(repo)] and not _run_dir(repo, session_id).exists()


def test_the_run_works_in_its_own_worktree_and_the_checkout_stays_as_it_was(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`run --worktree` edits and commits in a linked worktree at HEAD, never in the checkout.

    The checkout's uncommitted edit raises no dirty-tree question, is not stashed and is not
    carried; the isolation preflight and the tool set get the worktree and its recorded git dir,
    as a resumed fork's do.
    """
    repo = _repo(tmp_path, monkeypatch)
    before = _checkout_facts(repo)
    seen: dict[str, Any] = {}
    real_isolation, real_tools = _session.select_isolation, _session.build_session_tools

    def _isolation(cfg: Any, **kw: Any) -> Any:
        seen["isolation"] = (kw["cwd"], kw["worktree_git_dir"])
        return real_isolation(cfg, **kw)

    def _tools(cfg: Any, **kw: Any) -> Any:
        seen["tools"] = (kw["cwd"], kw["worktree_git_dir"])
        return real_tools(cfg, **kw)

    monkeypatch.setattr(_session, "select_isolation", _isolation)
    monkeypatch.setattr(_session, "build_session_tools", _tools)
    _script(monkeypatch, _edit("new.txt", "made by the run\n"))

    assert cli_main(["run", "--worktree", "--session-id", "wt-AAAA01", "add new.txt"]) == 0
    out, err = capsys.readouterr()

    manifest = sessions_manifest.read_manifest(_run_dir(repo, "wt-AAAA01"))
    worktree = manifest.worktree
    assert worktree is not None and worktree.name == "wt-AAAA01"
    assert worktree.is_relative_to(paths.cache_dir() / "parallel")
    assert manifest.worktree_git_dir == (repo / ".git").resolve()
    assert manifest.base_branch == "main" and manifest.base_sha == before[0].strip()
    assert seen["isolation"] == (worktree, manifest.worktree_git_dir)
    assert seen["tools"] == (worktree, manifest.worktree_git_dir)

    assert (worktree / "new.txt").read_text(encoding="utf-8") == "made by the run\n"
    assert (worktree / "a.py").read_text(encoding="utf-8") == "x = 1\n"
    assert not (worktree / "notes.md").exists()
    tip = git_ops.chain_tip(repo, git_ops.chain_ref_for("wt-AAAA01"))
    assert tip is not None
    assert git_ops.tree_diff_paths(repo, manifest.base_sha, tip) == ["new.txt"]

    assert _checkout_facts(repo) == before
    assert not (repo / "new.txt").exists()
    assert (repo / "a.py").read_text(encoding="utf-8") == "x = 2  # the operator's edit\n"

    assert f"working in the worktree {worktree}" in err
    assert "agent6 sessions merge wt-AAAA01" in out


def test_resume_from_the_checkout_continues_in_the_worktree(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`agent6 resume <id>` typed in the checkout drives the run's worktree, not the checkout."""
    repo = _repo(tmp_path, monkeypatch)
    _script(monkeypatch, _edit("new.txt", "one\n"))
    assert cli_main(["run", "--worktree", "--session-id", "wt-RESU01", "add new.txt"]) == 0
    before = _checkout_facts(repo)
    worktree = sessions_manifest.read_manifest(_run_dir(repo, "wt-RESU01")).worktree
    assert worktree is not None

    capsys.readouterr()
    _script(monkeypatch, _edit("more.txt", "two\n"))
    assert cli_main(["resume", "wt-RESU01", "--steer", "add more.txt"]) == 0
    assert f"working in the worktree {worktree}" in capsys.readouterr().err

    assert pathlib.Path.cwd() == repo
    assert (worktree / "more.txt").read_text(encoding="utf-8") == "two\n"
    assert not (repo / "more.txt").exists()
    assert _checkout_facts(repo) == before
    tip = git_ops.chain_tip(repo, git_ops.chain_ref_for("wt-RESU01"))
    assert tip is not None
    assert git_ops.tree_diff_paths(repo, before[0].strip(), tip) == ["more.txt", "new.txt"]


@pytest.mark.parametrize("forked", [False, True], ids=["worktree-run", "fork"])
def test_resume_stash_restore_command_targets_the_sessions_worktree(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    forked: bool,
) -> None:
    """A worktree stash's restore command works from the origin without switching branches."""
    config = (
        _CONFIG
        + '\n[git]\ndirty_tree = "stash"\n\n[parallel]\nworkdir = '
        + json.dumps(str(tmp_path / "worktrees with spaces"))
        + "\n"
    )
    repo = _repo(tmp_path, monkeypatch, config=config)
    session_id = "wt-STAS01"
    _script(monkeypatch, _edit("new.txt", "one\n"))
    if forked:
        assert cli_main(["run", "--worktree", "--session-id", "wt-STAS00", "task"]) == 0
        _script(monkeypatch, _edit("fork.txt", "fork work\n"))
        assert cli_main(["fork", "wt-STAS00", "--session-id", session_id, "--steer", "task"]) == 0
    else:
        assert cli_main(["run", "--worktree", "--session-id", session_id, "task"]) == 0
    worktree = sessions_manifest.read_manifest(_run_dir(repo, session_id)).worktree
    assert worktree is not None
    before = _checkout_facts(repo)[:3]
    worktree_head = _git(worktree, "rev-parse", "HEAD")
    (worktree / "a.py").write_text("x = 3  # worktree draft\n", encoding="utf-8")
    capsys.readouterr()
    _script(monkeypatch, _edit("more.txt", "two\n"))

    assert cli_main(["resume", session_id, "--steer", "add more.txt"]) == 0

    err = capsys.readouterr().err
    entry = git_ops.find_stash(repo, git_ops.auto_stash_message(session_id))
    assert entry is not None
    restore = next(
        line.partition("restore them with: ")[2]
        for line in err.splitlines()
        if "restore them with: " in line
    )
    assert shlex.split(restore) == ["git", "-C", str(worktree), "stash", "apply", entry.sha]
    assert (worktree / "a.py").read_text(encoding="utf-8") == "x = 1\n"

    sp.run(shlex.split(restore), cwd=repo, check=True, capture_output=True)
    assert (worktree / "a.py").read_text(encoding="utf-8") == "x = 3  # worktree draft\n"
    assert (worktree / "more.txt").read_text(encoding="utf-8") == "two\n"
    assert _checkout_facts(repo)[:3] == before
    assert _git(worktree, "rev-parse", "HEAD") == worktree_head
    assert _git(worktree, "branch", "--show-current") == ""


def test_resume_auto_stash_pop_keeps_the_run_branch_checked_out_in_its_worktree(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Auto-restore stays in the worktree even when the run branch is checked out there."""
    repo = _repo(
        tmp_path,
        monkeypatch,
        config=_CONFIG + '\n[git]\ndirty_tree = "stash"\nauto_stash_pop = true\n',
    )
    session_id = "wt-STAS02"
    _script(monkeypatch, _edit("new.txt", "one\n"))
    assert cli_main(["run", "--worktree", "--session-id", session_id, "task"]) == 0
    worktree = sessions_manifest.read_manifest(_run_dir(repo, session_id)).worktree
    assert worktree is not None
    _git(worktree, "add", "new.txt")
    _git(worktree, "switch", "agent6/wt-STAS02")
    before = _checkout_facts(repo)[:3]
    (worktree / "a.py").write_text("x = 3  # worktree draft\n", encoding="utf-8")
    capsys.readouterr()
    _script(monkeypatch, _edit("more.txt", "two\n"))

    assert cli_main(["resume", session_id, "--steer", "add more.txt"]) == 0

    assert (worktree / "a.py").read_text(encoding="utf-8") == "x = 3  # worktree draft\n"
    assert (worktree / "more.txt").read_text(encoding="utf-8") == "two\n"
    assert _git(worktree, "branch", "--show-current").strip() == "agent6/wt-STAS02"
    assert _checkout_facts(repo)[:3] == before
    assert git_ops.find_stash(repo, git_ops.auto_stash_message(session_id)) is None
    assert f"restored your pre-run changes in the worktree {worktree}" in capsys.readouterr().err


def test_merge_lands_the_work_and_prune_removes_the_worktree(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`sessions merge <id>` lands a worktree run on the checkout's branch; prune then sweeps it.

    Before the merge prune keeps the worktree and says why.
    """
    repo = _repo(tmp_path, monkeypatch)
    _script(monkeypatch, _edit("new.txt", "made by the run\n"))
    assert cli_main(["run", "--worktree", "--session-id", "wt-MERG01", "add new.txt"]) == 0
    worktree = sessions_manifest.read_manifest(_run_dir(repo, "wt-MERG01")).worktree
    assert worktree is not None
    capsys.readouterr()

    assert cli_main(["sessions", "prune"]) == 0
    assert "kept wt-MERG01's worktree (unmerged)" in capsys.readouterr().out
    assert worktree.is_dir()

    assert cli_main(["sessions", "merge", "wt-MERG01"]) == 0
    capsys.readouterr()
    assert _git(repo, "show", "main:new.txt") == "made by the run\n"
    assert (repo / "a.py").read_text(encoding="utf-8") == "x = 2  # the operator's edit\n"

    assert cli_main(["sessions", "prune"]) == 0
    assert "removed wt-MERG01's worktree (merged)" in capsys.readouterr().out
    assert not worktree.exists()
    assert _worktrees(repo) == [str(repo)]


def test_a_worktree_run_resumed_after_its_merge_shows_only_its_unlanded_work(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """A worktree run resumed after its merge diffs, lists and lands the same new work.

    The worktree's HEAD never holds the landing commit, so the run kept the base it started
    on: `sessions diff` and `sessions commits` showed both executions while the next merge
    landed only the second.
    """
    repo = _repo(tmp_path, monkeypatch)
    _script(monkeypatch, _edit("one.txt", "1\n"))
    assert cli_main(["run", "--worktree", "--session-id", "wt-LAND01", "add one.txt"]) == 0
    chain = git_ops.chain_ref_for("wt-LAND01")
    first_tip = git_ops.chain_tip(repo, chain)
    assert first_tip is not None
    assert cli_main(["sessions", "merge", "wt-LAND01"]) == 0
    landed = _git(repo, "rev-parse", "main").strip()

    _script(monkeypatch, _edit("two.txt", "2\n"))
    assert cli_main(["resume", "wt-LAND01", "--steer", "add two.txt"]) == 0
    assert "earlier work landed on main" in capfd.readouterr().err

    manifest = sessions_manifest.read_manifest(_run_dir(repo, "wt-LAND01"))
    assert manifest.base_sha == first_tip and manifest.base_branch == "main"
    assert cli_main(["sessions", "diff", "wt-LAND01", "--stat"]) == 0
    shown = capfd.readouterr().out
    assert "two.txt" in shown and "one.txt" not in shown
    assert cli_main(["sessions", "commits", "wt-LAND01"]) == 0
    listed = capfd.readouterr().out
    assert listed.strip() and first_tip[:12] not in listed
    assert cli_main(["sessions", "merge", "wt-LAND01"]) == 0
    assert _git(repo, "diff", "--name-only", landed, "main").split() == ["two.txt"]


def test_rm_removes_the_worktree_with_the_record(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`sessions rm <id>` takes an unmerged worktree run's worktree with its history."""
    repo = _repo(tmp_path, monkeypatch)
    _script(monkeypatch, _edit("new.txt", "made by the run\n"))
    assert cli_main(["run", "--worktree", "--session-id", "wt-GONE01", "add new.txt"]) == 0
    worktree = sessions_manifest.read_manifest(_run_dir(repo, "wt-GONE01")).worktree
    assert worktree is not None
    capsys.readouterr()

    assert cli_main(["sessions", "rm", "wt-GONE01"]) == 0
    assert "its worktree" in capsys.readouterr().out
    assert not worktree.exists()
    assert _nothing_was_created(repo, "wt-GONE01")


def test_prune_typed_in_a_worktree_removes_it_and_finishes_the_sweep(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Prune ran its git in the worktree it had just removed, and crashed mid-sweep."""
    repo = _repo(tmp_path, monkeypatch)
    _script(monkeypatch)
    assert cli_main(["run", "--worktree", "--session-id", "wt-IDLE01", "look around"]) == 0
    _script(monkeypatch, _edit("new.txt", "made by the run\n"))
    assert cli_main(["run", "--worktree", "--session-id", "wt-HERE01", "add new.txt"]) == 0
    assert cli_main(["sessions", "merge", "wt-HERE01"]) == 0
    here = sessions_manifest.read_manifest(_run_dir(repo, "wt-HERE01")).worktree
    idle = sessions_manifest.read_manifest(_run_dir(repo, "wt-IDLE01")).worktree
    assert here is not None and idle is not None
    # The sweep takes the cwd's worktree first, so the other one is judged after it is gone.
    assert list(fork_worktrees.worktree_owners(paths.state_dir(repo))) == [here, idle]
    capsys.readouterr()
    monkeypatch.chdir(here)

    assert cli_main(["sessions", "prune"]) == 0
    out, err = capsys.readouterr()
    assert "removed wt-HERE01's worktree (merged)" in out
    assert "removed wt-IDLE01's worktree (committed nothing)" in out
    assert "kept agent6/wt-HERE01 (squash-merged into main" in out
    assert "deleted 0; kept 1 (1 merged, 0 unmerged)" in out
    assert f"this shell's directory is gone; cd {repo}" in err
    assert _worktrees(repo) == [str(repo)]


def test_rm_typed_in_the_worktree_removes_the_chain_ref_too(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Rm removed the record and the worktree, then crashed before the chain ref."""
    repo = _repo(tmp_path, monkeypatch)
    _script(monkeypatch, _edit("new.txt", "made by the run\n"))
    assert cli_main(["run", "--worktree", "--session-id", "wt-SELF01", "add new.txt"]) == 0
    worktree = sessions_manifest.read_manifest(_run_dir(repo, "wt-SELF01")).worktree
    assert worktree is not None
    capsys.readouterr()
    monkeypatch.chdir(worktree)

    assert cli_main(["sessions", "rm", "wt-SELF01"]) == 0
    out, err = capsys.readouterr()
    assert "removed wt-SELF01, its worktree and its chain ref" in out
    assert f"this shell's directory is gone; cd {repo}" in err
    assert git_ops.chain_tip(repo, git_ops.chain_ref_for("wt-SELF01")) is None
    assert _nothing_was_created(repo, "wt-SELF01")


def test_a_budget_stopped_lane_continues_with_the_reports_fork_command(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The report's fork command continues an imported capped lane after its clone is removed."""
    repo = _repo(tmp_path, monkeypatch)

    class CappedProvider:
        def __init__(self, tracker: agent6_budget.BudgetTracker) -> None:
            self.tracker = tracker

        def call(self, **_kw: Any) -> ProviderResponse:
            self.tracker.check()
            self.tracker.record(
                model="m",
                input_tokens=10,
                output_tokens=5,
                cache_read_tokens=0,
                cache_creation_tokens=0,
            )
            return _Scripted([_edit("lane.txt", "work\n")]).call()

    def provider(*_a: Any, budget: agent6_budget.BudgetTracker, **_kw: Any) -> CappedProvider:
        return CappedProvider(budget)

    monkeypatch.setattr(providers, "build_role_provider", provider)
    cfg = layer.load_effective(repo).config
    state = paths.state_dir(repo)
    lanes = app_lanes.build_lane_specs(
        "1",
        cfg=cfg,
        origin=repo,
        fanout_id="capped-fan",
        workdir_root=tmp_path / "lanes",
    )

    def spawn(spec: subrun.LaneSpec, task: str) -> subrun.LaneResult:
        subrun.clone_workspace(repo, spec.workdir)
        output, errors = io.StringIO(), io.StringIO()
        with (
            monkeypatch.context() as child,
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(errors),
        ):
            child.chdir(spec.workdir)
            child.setenv("AGENT6_PARALLEL_LINEAGE", f"capped-fan:capped-fan:{spec.lane}")
            assert (
                cli_main(
                    ["run", "--session-id", spec.session_id, "--max-tokens-fallback", "1", task]
                )
                == 3
            ), output.getvalue() + errors.getvalue()
        session_dir = paths.state_dir(spec.workdir) / "sessions" / "runs" / spec.session_id
        return subrun.LaneResult(spec, session_dir, f"agent6/{spec.session_id}", True, "")

    assert (
        parallel.run_parallel(
            "task",
            lanes,
            cfg=cfg,
            explicit_leaves=frozenset(),
            origin=repo,
            origin_state=state,
            runtime=parallel_cmd.lane_runtime(),
            spawner=spawn,
            fanout_id="capped-fan",
        )
        == 3
    )
    shown = capsys.readouterr().out
    hint = next(line for line in shown.splitlines() if "then continue with:" in line)
    command = hint.split("then continue with:", 1)[1].strip().split()
    assert command == ["agent6", "fork", "capped-fan-l1"]
    assert not lanes[0].workdir.exists()
    _script(monkeypatch, _FINISH)
    assert (
        cli_main([*command[1:], "--session-id", "continued", "--max-tokens-fallback", "100"]) == 0
    )
    manifest = sessions_manifest.read_manifest(_run_dir(repo, "continued"))
    assert manifest.parent_session_id == "capped-fan-l1"
    assert manifest.worktree is not None and manifest.worktree.is_dir()
    assert listing.summarize_session_dir(_run_dir(repo, "continued")).status == "finished"


def test_a_worktree_run_starts_while_another_run_drives_the_checkout(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The checkout's one-writer lock is another checkout's: a worktree run never parks on it."""
    repo = _repo(tmp_path, monkeypatch)
    held = lock.acquire_repo_writer(paths.state_dir(repo), repo, "other-run")
    assert held is not None
    try:
        _script(monkeypatch, _edit("new.txt", "made by the run\n"))
        assert cli_main(["run", "--worktree", "--session-id", "wt-BUSY01", "add new.txt"]) == 0
    finally:
        lock.release_single_writer(held)
    assert "PARKED" not in capsys.readouterr().err
    assert git_ops.chain_tip(repo, git_ops.chain_ref_for("wt-BUSY01")) is not None


def test_a_repository_with_no_commit_refuses(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """There is no commit to add a worktree at, and the refusal names the fix."""
    _repo(tmp_path, monkeypatch)
    unborn = tmp_path / "unborn"
    unborn.mkdir()
    _git(unborn, "init", "-q", "-b", "main")
    monkeypatch.chdir(unborn)

    assert cli_main(["run", "--worktree", "--session-id", "wt-UNBN01", "task"]) == 2
    err = capsys.readouterr().err
    assert "REFUSING: --worktree starts from HEAD's commit" in err and "git commit" in err
    assert _nothing_was_created(unborn, "wt-UNBN01")


def test_outside_a_git_repository_refuses(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The git wall every run hits comes first and names `git init`."""
    _repo(tmp_path, monkeypatch)
    plain = tmp_path / "plain"
    plain.mkdir()
    monkeypatch.chdir(plain)

    assert cli_main(["run", "--worktree", "task"]) == 2
    err = capsys.readouterr().err
    assert "is not a git repository" in err and "git init" in err


def test_parallel_and_worktree_together_refuse(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A fan-out's lanes already own a clone each; the pair is refused before anything runs."""
    repo = _repo(tmp_path, monkeypatch)

    assert cli_main(["run", "--worktree", "--parallel", "2", "task"]) == 2
    assert "REFUSING: --parallel cannot combine with --worktree" in capsys.readouterr().err
    assert _worktrees(repo) == [str(repo)]


def test_model_controlled_git_refuses(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A linked worktree's `.git` is read-only in the jail, so the model could not commit there."""
    model_git = _CONFIG.replace('run_commands = "no"', 'run_commands = "no"\nprotect_git = false')
    repo = _repo(tmp_path, monkeypatch, config=model_git + '\n[git]\ncontrol = "model"\n')

    assert cli_main(["run", "--worktree", "--session-id", "wt-MODL01", "task"]) == 2
    err = capsys.readouterr().err
    assert "REFUSING: --worktree runs in a linked worktree" in err
    assert '[git].control = "model"' in err and "without --worktree" in err
    assert "agent6 config set --repo git.control agent6" in err and "for this run" not in err
    assert _nothing_was_created(repo, "wt-MODL01")


def test_a_failed_worktree_add_leaves_nothing_behind(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A worktree git refuses to add ends the start: no session dir, no worktree record."""
    repo = _repo(tmp_path, monkeypatch)

    def _fail(_repo: pathlib.Path, _dest: pathlib.Path, _sha: str) -> None:
        raise errors.GitError("fatal: could not create the worktree")

    monkeypatch.setattr(git_ops, "add_worktree", _fail)

    assert cli_main(["run", "--worktree", "--session-id", "wt-FAIL01", "task"]) == 1
    err = capsys.readouterr().err
    assert "could not add the run's worktree" in err and "could not create the worktree" in err
    assert _nothing_was_created(repo, "wt-FAIL01")


def test_a_refusal_after_the_worktree_was_added_removes_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The isolation preflight runs on the added worktree; its refusal takes the worktree back."""
    repo = _repo(tmp_path, monkeypatch)
    added: list[pathlib.Path] = []

    def _refuse(_cfg: Any, **kw: Any) -> Any:
        added.append(kw["cwd"])
        assert (kw["cwd"] / ".git").is_file(), "the preflight sees the real worktree"
        raise preflight.SessionRefusedError(2)

    monkeypatch.setattr(_session, "select_isolation", _refuse)

    assert cli_main(["run", "--worktree", "--session-id", "wt-REFU01", "task"]) == 2
    assert "working in the worktree" not in capsys.readouterr().err
    assert added and not added[0].exists()
    assert _nothing_was_created(repo, "wt-REFU01")


def test_an_interrupt_at_the_isolation_question_removes_the_worktree(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No manifest names the worktree yet, so a Ctrl-C there would strand it past every sweep."""
    repo = _repo(tmp_path, monkeypatch)

    def _interrupt(_cfg: Any, **_kw: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(_session, "select_isolation", _interrupt)

    with pytest.raises(KeyboardInterrupt):
        run_lifecycle.run_task(
            layer.load_effective(repo, ()).config,
            "task",
            frontend=cli_run.session_frontend(),
            started_at=time.time(),
            session_id="wt-INTR01",
            worktree=True,
        )
    capsys.readouterr()
    assert _nothing_was_created(repo, "wt-INTR01")


def test_a_missing_provider_key_refuses_before_any_worktree(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A route that cannot run refuses with no worktree and no session dir."""
    keyed = _CONFIG.replace('api_format = "openai"', 'api_format = "anthropic"')
    repo = _repo(tmp_path, monkeypatch, config=keyed)
    monkeypatch.delenv("PROBE_KEY")

    assert cli_main(["run", "--worktree", "--session-id", "wt-NKEY01", "task"]) == 2
    assert "no API key for [providers.p]" in capsys.readouterr().err
    assert _nothing_was_created(repo, "wt-NKEY01")


def test_from_a_subdirectory_refuses(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A worktree is a whole tree: from a subdirectory the run would widen to all of it."""
    repo = _repo(tmp_path, monkeypatch)
    (repo / "sub").mkdir()
    monkeypatch.chdir(repo / "sub")

    assert cli_main(["run", "--worktree", "--session-id", "wt-SUBD01", "task"]) == 2
    err = capsys.readouterr().err
    assert "REFUSING: --worktree starts at the repository's top level" in err and str(repo) in err
    assert _nothing_was_created(repo, "wt-SUBD01")


def test_a_workdir_inside_the_checkout_refuses(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A worktree under the checkout would show up in its `git status`."""
    repo = _repo(tmp_path, monkeypatch, config=_CONFIG + '\n[parallel]\nworkdir = "wt"\n')
    before = _checkout_facts(repo)

    assert cli_main(["run", "--worktree", "--session-id", "wt-RELW01", "task"]) == 2
    assert "[parallel].workdir" in capsys.readouterr().err
    assert _checkout_facts(repo) == before
    assert _nothing_was_created(repo, "wt-RELW01")


def test_prune_removes_the_worktree_of_a_run_that_committed_nothing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no commit there is no merge to stamp, so prune would keep the empty worktree forever."""
    repo = _repo(tmp_path, monkeypatch)
    _script(monkeypatch)
    assert cli_main(["run", "--worktree", "--session-id", "wt-NOOP01", "look around"]) == 0
    assert git_ops.chain_tip(repo, git_ops.chain_ref_for("wt-NOOP01")) is None
    assert len(_worktrees(repo)) == 2
    capsys.readouterr()

    assert cli_main(["sessions", "prune"]) == 0
    assert "removed wt-NOOP01's worktree (committed nothing)" in capsys.readouterr().out
    assert _worktrees(repo) == [str(repo)]


def test_uncommitted_edits_are_named_in_the_worktree_and_survive_prune(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With `commit_per_step = false`, "the working tree" in the footer would name the checkout."""
    repo = _repo(tmp_path, monkeypatch, config=_CONFIG + "\n[git]\ncommit_per_step = false\n")
    _script(monkeypatch, _edit("new.txt", "made by the run\n"))
    assert cli_main(["run", "--worktree", "--session-id", "wt-NOCM01", "add new.txt"]) == 0
    worktree = sessions_manifest.read_manifest(_run_dir(repo, "wt-NOCM01")).worktree
    assert worktree is not None
    assert f"the run's edits are in the worktree {worktree}" in capsys.readouterr().out

    assert cli_main(["sessions", "prune"]) == 0
    out = capsys.readouterr().out
    assert "kept wt-NOCM01's worktree (holds work no commit has: new.txt)" in out
    assert (worktree / "new.txt").is_file()


def test_an_error_before_the_session_lock_removes_the_worktree(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A crash between the isolation preflight and the run's start would strand the worktree."""
    repo = _repo(tmp_path, monkeypatch)

    def _full(_session_dir: pathlib.Path) -> int | None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(lock, "acquire_single_writer", _full)

    with pytest.raises(OSError, match="No space left"):
        run_lifecycle.run_task(
            layer.load_effective(repo, ()).config,
            "task",
            frontend=cli_run.session_frontend(),
            started_at=time.time(),
            session_id="wt-FULL01",
            worktree=True,
        )
    capsys.readouterr()
    assert _worktrees(repo) == [str(repo)]


def test_a_refusal_before_the_manifest_removes_the_worktree(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A start refused after the worktree is added, before a manifest names it, removes it."""
    repo = _repo(tmp_path, monkeypatch)

    def _unwritable(_session_dir: pathlib.Path, _text: str) -> bool:
        return False

    monkeypatch.setattr(ipc, "submit_steer", _unwritable)

    rc = run_lifecycle.run_task(
        layer.load_effective(repo, ()).config,
        "task",
        frontend=cli_run.session_frontend(),
        started_at=time.time(),
        session_id="wt-STEE01",
        initial_steer="and then this",
        worktree=True,
    )
    assert rc == 2
    assert "could not write the initial steer request" in capsys.readouterr().err
    assert _worktrees(repo) == [str(repo)]


def test_a_worktree_the_cleanup_cannot_remove_is_named(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No manifest names it, so a silent failure would leave a worktree no sweep reports."""
    repo = _repo(tmp_path, monkeypatch)
    added: list[pathlib.Path] = []

    def _refuse(_cfg: Any, **kw: Any) -> Any:
        added.append(kw["cwd"])
        (kw["cwd"] / "junk.txt").write_text("left by a probe\n", encoding="utf-8")
        raise preflight.SessionRefusedError(2)

    monkeypatch.setattr(_session, "select_isolation", _refuse)

    assert cli_main(["run", "--worktree", "--session-id", "wt-KEPT01", "task"]) == 2
    err = capsys.readouterr().err
    assert f"the worktree {added[0]} stays: it holds work no commit has: junk.txt" in err
    assert f"git worktree remove {added[0]}" in err
    assert _worktrees(repo) == [str(repo), str(added[0])]


_COMMANDS = _CONFIG.replace('run_commands = "no"', 'run_commands = "yes"')


def _with_setup(repo: pathlib.Path, argv: str) -> None:
    """Write the repository's own config with a setup command."""
    path = paths.state_dir(repo) / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"[parallel]\nsetup_command = {argv}\n", encoding="utf-8")


def test_the_setup_command_runs_in_the_new_worktree_before_the_first_turn(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A worktree holds tracked files only: with no setup step its environment never existed.

    What the command leaves untracked is the environment, so the run's commits leave it out.
    """
    repo = _repo(tmp_path, monkeypatch, config=_COMMANDS)
    _with_setup(repo, '["sh", "-c", "echo ready > .setup-ran && echo synced"]')
    seen: list[bool] = []
    prov = _script(monkeypatch, _edit("new.txt", "made by the run\n"))
    real_call = prov.call

    def _call(**kw: Any) -> ProviderResponse:
        worktree = sessions_manifest.read_manifest(_run_dir(repo, "wt-SETU01")).worktree
        assert worktree is not None
        seen.append((worktree / ".setup-ran").exists())
        return real_call(**kw)

    monkeypatch.setattr(prov, "call", _call)

    assert cli_main(["run", "--worktree", "--session-id", "wt-SETU01", "task"]) == 0
    err = capsys.readouterr().err

    worktree = sessions_manifest.read_manifest(_run_dir(repo, "wt-SETU01")).worktree
    assert worktree is not None
    assert seen and seen[0]
    assert (worktree / ".setup-ran").read_text(encoding="utf-8") == "ready\n"
    assert not (repo / ".setup-ran").exists()
    assert "setup command: sh -c 'echo ready > .setup-ran && echo synced'" in err
    assert "synced" in err
    manifest = sessions_manifest.read_manifest(_run_dir(repo, "wt-SETU01"))
    tip = git_ops.chain_tip(repo, git_ops.chain_ref_for("wt-SETU01"))
    assert tip is not None
    assert git_ops.tree_diff_paths(repo, manifest.base_sha, tip) == ["new.txt"]


def test_a_failed_setup_command_ends_the_session_before_any_turn(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run going on after a failed setup would spend turns in a workspace with no environment."""
    repo = _repo(tmp_path, monkeypatch, config=_COMMANDS)
    _with_setup(repo, '["sh", "-c", "echo no lockfile >&2; exit 3"]')
    prov = _script(monkeypatch)

    assert cli_main(["run", "--worktree", "--session-id", "wt-SETU02", "task"]) == 1
    out, err = capsys.readouterr()

    assert prov.calls == 0
    assert "[agent6] LOOP: aborting -- the setup command failed (exit 3)" in out
    assert (
        "failed · setup command failed\n"
        "  the setup command failed (exit 3): sh -c 'echo no lockfile >&2; exit 3'\n"
        "    no lockfile\n"
    ) in out
    assert (
        "no turn ran, so there is nothing to resume: start the run again once the setup"
        " command works"
    ) in out
    assert "resume with:" not in out
    assert "setup command: sh -c" in err
    events = (_run_dir(repo, "wt-SETU02") / "logs.jsonl").read_text(encoding="utf-8")
    assert '"setup_command_failed"' in events
    # A resume of it says what happened in the reader's terms, with no internal path.
    assert cli_main(["resume", "wt-SETU02"]) == 2
    refused = capsys.readouterr().err
    assert "wt-SETU02 never reached its first turn" in refused
    assert "loop_state.json" not in refused


def test_a_run_in_the_checkout_runs_no_setup_command(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The checkout holds the operator's own environment; setup there would rebuild it."""
    repo = _repo(
        tmp_path,
        monkeypatch,
        config=_COMMANDS.replace("[sandbox]", '[git]\ndirty_tree = "include"\n\n[sandbox]'),
    )
    _with_setup(repo, '["sh", "-c", "echo ready > .setup-ran"]')
    _script(monkeypatch)

    assert cli_main(["run", "--session-id", "wt-SETU03", "task"]) == 0
    capsys.readouterr()

    assert not (repo / ".setup-ran").exists()


def test_a_lane_runs_the_setup_command_in_its_clone(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A lane's clone holds tracked files only, as a worktree does."""
    repo = _repo(
        tmp_path,
        monkeypatch,
        config=_COMMANDS.replace("[sandbox]", '[git]\ndirty_tree = "include"\n\n[sandbox]'),
    )
    _with_setup(repo, '["sh", "-c", "echo ready > .setup-ran"]')
    monkeypatch.setenv("AGENT6_PARALLEL_LINEAGE", "co:fan:1")
    _script(monkeypatch)

    assert cli_main(["run", "--session-id", "fan-l1", "task"]) == 0
    capsys.readouterr()

    assert (repo / ".setup-ran").read_text(encoding="utf-8") == "ready\n"


def _setup_events(repo: pathlib.Path, session_id: str) -> list[dict[str, Any]]:
    log = _run_dir(repo, session_id) / "logs.jsonl"
    events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    return [event for event in events if event["type"] == "workspace.setup"]


def test_the_setup_is_journaled_and_a_resume_never_runs_it_again(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A resume that ran setup again would rebuild the environment under the run's own work."""
    repo = _repo(tmp_path, monkeypatch, config=_COMMANDS)
    _with_setup(repo, '["sh", "-c", "echo ran >> .setup-ran"]')
    _script(monkeypatch, _edit("new.txt", "one\n"))
    assert cli_main(["run", "--worktree", "--session-id", "wt-SETU05", "add new.txt"]) == 0

    manifest = sessions_manifest.read_manifest(_run_dir(repo, "wt-SETU05"))
    assert manifest.workspace_set_up and manifest.worktree is not None
    (event,) = _setup_events(repo, "wt-SETU05")
    assert event["cmd"] == ["sh", "-c", "echo ran >> .setup-ran"]
    assert event["exit_code"] == 0 and event["duration_s"] >= 0

    capsys.readouterr()
    _script(monkeypatch, _edit("more.txt", "two\n"))
    assert cli_main(["resume", "wt-SETU05", "--steer", "add more.txt"]) == 0

    assert "setup command" not in capsys.readouterr().err
    assert (manifest.worktree / ".setup-ran").read_text(encoding="utf-8") == "ran\n"
    assert len(_setup_events(repo, "wt-SETU05")) == 1


def test_a_fork_starts_not_set_up_and_runs_the_setup_command_in_its_own_worktree(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A fork's worktree is new: its source's environment is in another directory."""
    repo = _repo(tmp_path, monkeypatch, config=_COMMANDS)
    _with_setup(repo, '["sh", "-c", "echo ran >> .setup-ran"]')
    _script(monkeypatch, _edit("new.txt", "one\n"))
    assert cli_main(["run", "--worktree", "--session-id", "wt-SETU06", "add new.txt"]) == 0

    assert cli_main(["fork", "wt-SETU06", "--session-id", "wt-FORK06", "--no-run"]) == 0
    forked = sessions_manifest.read_manifest(_run_dir(repo, "wt-FORK06"))
    assert forked.worktree is not None and not forked.workspace_set_up
    assert not (forked.worktree / ".setup-ran").exists()

    capsys.readouterr()
    _script(monkeypatch, _edit("more.txt", "two\n"))
    assert cli_main(["resume", "wt-FORK06", "--steer", "add more.txt"]) == 0

    assert "setup command: sh -c" in capsys.readouterr().err
    assert (forked.worktree / ".setup-ran").read_text(encoding="utf-8") == "ran\n"
    assert sessions_manifest.read_manifest(_run_dir(repo, "wt-FORK06")).workspace_set_up
    assert len(_setup_events(repo, "wt-FORK06")) == 1


def test_a_failed_setup_leaves_the_workspace_not_set_up(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A workspace stamped set up after a failed command would never be built."""
    repo = _repo(tmp_path, monkeypatch, config=_COMMANDS)
    _with_setup(repo, '["sh", "-c", "exit 3"]')
    _script(monkeypatch)

    assert cli_main(["run", "--worktree", "--session-id", "wt-SETU07", "task"]) == 1
    capsys.readouterr()

    assert not sessions_manifest.read_manifest(_run_dir(repo, "wt-SETU07")).workspace_set_up
    (event,) = _setup_events(repo, "wt-SETU07")
    assert event["exit_code"] == 3


def _commit_agents_md(repo: pathlib.Path, block: str) -> None:
    (repo / "AGENTS.md").write_text(
        f"# Agents\n\n## Setup command\n\n```bash\n{block}\n```\n", encoding="utf-8"
    )
    _git(repo, "add", "AGENTS.md")
    _git(repo, "commit", "-q", "-m", "agents")


def test_the_setup_command_is_inferred_from_the_agents_md_block(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A repository that states its setup in AGENTS.md needed a config line to have it run."""
    repo = _repo(tmp_path, monkeypatch, config=_COMMANDS)
    _commit_agents_md(repo, "echo from-agents-md > .setup-ran")
    _script(monkeypatch)

    assert cli_main(["run", "--worktree", "--session-id", "wt-SETU08", "task"]) == 0
    err = capsys.readouterr().err

    worktree = sessions_manifest.read_manifest(_run_dir(repo, "wt-SETU08")).worktree
    assert worktree is not None
    assert (worktree / ".setup-ran").read_text(encoding="utf-8") == "from-agents-md\n"
    assert (
        "setup_command not set; inferred from AGENTS.md: sh -c 'echo from-agents-md > .setup-ran'"
    ) in err
    (event,) = _setup_events(repo, "wt-SETU08")
    assert event["cmd"] == ["sh", "-c", "echo from-agents-md > .setup-ran"]


@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("deleted", [False, True])
def test_worktree_setup_is_inferred_from_its_commit_not_the_dirty_agents_md(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    committed: bool,
    deleted: bool,
) -> None:
    """An uncommitted setup block ran in a worktree whose AGENTS.md lacked that block."""
    repo = _repo(tmp_path, monkeypatch, config=_COMMANDS)
    agents = "## Verify command\n```sh\nsh -c 'exit 0'\n```\n"
    if committed:
        agents += "\n## Setup command\n```sh\necho committed > .setup-ran\n```\n"
    (repo / "AGENTS.md").write_text(agents, encoding="utf-8")
    _git(repo, "add", "AGENTS.md")
    _git(repo, "commit", "-q", "-m", "agents")
    if deleted:
        (repo / "AGENTS.md").unlink()
    else:
        (repo / "AGENTS.md").write_text(
            "## Setup command\n```sh\necho uncommitted; exit 3\n```\n", encoding="utf-8"
        )
    before = _checkout_facts(repo)
    _script(monkeypatch)
    assert cli_main(["run", "--worktree", "--session-id", "committed-setup", "task"]) == 0
    assert _checkout_facts(repo) == before
    worktree = sessions_manifest.read_manifest(_run_dir(repo, "committed-setup")).worktree
    assert worktree is not None
    assert (worktree / "AGENTS.md").read_text(encoding="utf-8") == agents
    assert len(_setup_events(repo, "committed-setup")) == int(committed)
    if committed:
        assert (worktree / ".setup-ran").read_text(encoding="utf-8") == "committed\n"
    assert "echo uncommitted" not in "".join(capsys.readouterr())


def test_a_configured_setup_command_outranks_the_agents_md_block(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The operator's config is the pin; the repository's block is the fallback."""
    repo = _repo(tmp_path, monkeypatch, config=_COMMANDS)
    _commit_agents_md(repo, "echo from-agents-md > .setup-ran")
    _with_setup(repo, '["sh", "-c", "echo configured > .setup-ran"]')
    _script(monkeypatch)

    assert cli_main(["run", "--worktree", "--session-id", "wt-SETU09", "task"]) == 0
    err = capsys.readouterr().err

    worktree = sessions_manifest.read_manifest(_run_dir(repo, "wt-SETU09")).worktree
    assert worktree is not None
    assert (worktree / ".setup-ran").read_text(encoding="utf-8") == "configured\n"
    assert "inferred from AGENTS.md" not in err


_FAKE_PYTHON = "#!/bin/sh\nexit 0\n"
_BUILD_VENV = (
    '["sh", "-c", "mkdir -p .venv/bin && printf \'#!/bin/sh\\\\nexit 0\\\\n\' > .venv/bin/python'
    ' && chmod +x .venv/bin/python"]'
)


def _untracked_venv(repo: pathlib.Path) -> None:
    """Give the checkout an environment git does not track, as a real `.venv` is."""
    (repo / ".git" / "info" / "exclude").write_text(".venv/\n", encoding="utf-8")
    (repo / ".venv" / "bin").mkdir(parents=True)
    python = repo / ".venv" / "bin" / "python"
    python.write_text(_FAKE_PYTHON, encoding="utf-8")
    python.chmod(0o755)


def _pin_gate(repo: pathlib.Path, extra: str = "") -> None:
    path = paths.state_dir(repo) / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'[harness]\nverify_command = [".venv/bin/python", "-m", "pytest", "-q"]\n{extra}',
        encoding="utf-8",
    )


def test_a_gate_on_an_untracked_path_with_no_setup_command_refuses_before_anything_is_created(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The run started, spent its turns, and ended `verify command unexecutable`."""
    repo = _repo(tmp_path, monkeypatch, config=_COMMANDS)
    _untracked_venv(repo)
    _pin_gate(repo)
    prov = _script(monkeypatch)

    assert cli_main(["run", "--worktree", "--session-id", "wt-GATE01", "task"]) == 2
    err = capsys.readouterr().err

    assert prov.calls == 0
    assert _nothing_was_created(repo, "wt-GATE01")
    assert "REFUSING: the verify gate runs `.venv/bin/python`" in err
    assert "git does not track" in err
    assert """agent6 config set --repo parallel.setup_command '["uv", "sync"]'""" in err
    assert "`## Setup command`" in err and "AGENTS.md" in err


def test_the_gate_is_inferred_in_the_checkout_and_carried_into_the_worktree(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Inferred in the worktree, where `.venv` is absent, the gate became `python3 -m pytest`."""
    repo = _repo(tmp_path, monkeypatch, config=_COMMANDS)
    (repo / "pyproject.toml").write_text('[project]\nname = "p"\nversion = "0"\n', encoding="utf-8")
    _git(repo, "add", "pyproject.toml")
    _git(repo, "commit", "-q", "-m", "manifest")
    _untracked_venv(repo)
    _with_setup(repo, _BUILD_VENV)
    _script(monkeypatch, _edit("new.txt", "one\n"))

    assert cli_main(["run", "--worktree", "--session-id", "wt-GATE02", "add new.txt"]) == 0
    err = capsys.readouterr().err

    manifest = sessions_manifest.read_manifest(_run_dir(repo, "wt-GATE02"))
    assert manifest.harness.verify_command == (".venv/bin/python", "-m", "pytest", "-q")
    assert manifest.harness.verify_origin == "inferred"
    assert "inferred from pyproject: .venv/bin/python -m pytest -q" in err
    assert manifest.worktree is not None and (manifest.worktree / ".venv/bin/python").exists()


def test_a_lane_stamps_the_gate_origin_its_coordinator_resolved(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lane's config carries the gate the coordinator inferred; `(configured)` was a lie."""
    repo = _repo(
        tmp_path, monkeypatch, config=_COMMANDS + '\n[harness]\nverify_command = ["true"]\n'
    )
    _git(repo, "commit", "-qam", "the operator's edit")
    _script(monkeypatch)
    monkeypatch.setenv("AGENT6_PARALLEL_LINEAGE", "co:fan:1")
    monkeypatch.setenv("AGENT6_LANE_VERIFY_ORIGIN", "inferred")

    assert cli_main(["run", "--session-id", "fan-l1", "task"]) == 0

    manifest = sessions_manifest.read_manifest(_run_dir(repo, "fan-l1"))
    assert manifest.harness.verify_command == ("true",)
    assert manifest.harness.verify_origin == "inferred"


def test_a_setup_command_that_leaves_the_gate_path_missing_ends_the_session_saying_so(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The setup command exited 0, and the run then ended `verify command unexecutable`."""
    repo = _repo(tmp_path, monkeypatch, config=_COMMANDS)
    _untracked_venv(repo)
    _pin_gate(repo, '\n[parallel]\nsetup_command = ["true"]\n')
    prov = _script(monkeypatch)

    assert cli_main(["run", "--worktree", "--session-id", "wt-GATE03", "task"]) == 1
    out = capsys.readouterr().out

    assert prov.calls == 0
    assert "failed · setup command failed" in out
    assert "the setup command ran (true), and the verify gate's `.venv/bin/python` is still" in out
    assert not sessions_manifest.read_manifest(_run_dir(repo, "wt-GATE03")).workspace_set_up


def test_a_fork_whose_gate_is_on_an_untracked_path_refuses_before_its_worktree_exists(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The fork was created, and its first execution ended `verify command unexecutable`."""
    in_checkout = _COMMANDS.replace("[sandbox]", '[git]\ndirty_tree = "include"\n\n[sandbox]')
    repo = _repo(tmp_path, monkeypatch, config=in_checkout)
    _untracked_venv(repo)
    _pin_gate(repo)
    _script(monkeypatch, _edit("new.txt", "one\n"))
    assert cli_main(["run", "--session-id", "wt-GATE04", "add new.txt"]) == 0
    capsys.readouterr()

    assert cli_main(["fork", "wt-GATE04", "--session-id", "wt-FORK04", "--no-run"]) == 2
    err = capsys.readouterr().err

    assert _nothing_was_created(repo, "wt-FORK04")
    assert "the verify gate runs `.venv/bin/python`" in err
    assert "a fork's new worktree" in err


def test_a_setup_on_resume_leaves_the_runs_own_files_in_its_commits(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every untracked file after a late setup was left out, the run's earlier files included.

    The chain commits without moving the worktree's index, so the run's own new files read
    untracked there; only what the setup command itself leaves is the environment.
    """
    repo = _repo(tmp_path, monkeypatch, config=_COMMANDS)
    _script(monkeypatch, _edit("new.txt", "one\n"))
    assert cli_main(["run", "--worktree", "--session-id", "wt-SETU10", "add new.txt"]) == 0

    _with_setup(repo, '["sh", "-c", "echo ready > .setup-ran"]')
    _script(monkeypatch, _edit("new.txt", "two\n"))
    assert cli_main(["resume", "wt-SETU10", "--steer", "rewrite new.txt"]) == 0
    assert "setup command: sh -c" in capsys.readouterr().err

    tip = git_ops.chain_tip(repo, git_ops.chain_ref_for("wt-SETU10"))
    assert tip is not None
    assert _git(repo, "show", f"{tip}:new.txt") == "two\n"
    assert ".setup-ran" not in _git(repo, "ls-tree", "--name-only", tip)


@pytest.mark.parametrize("unavailable", [False, True])
def test_dispatch_setup_keeps_timeout_and_wraps_jail_unavailability(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, unavailable: bool
) -> None:
    """Setup retains its jail timeout and reports jail refusal through prepare, not an abort."""
    from agent6.sandbox import jail
    from agent6.tools import dispatch

    cfg = Config.model_validate({"sandbox": {"run_commands": "yes"}}).with_setup_command(("true",))
    seen: list[kinds.JailPolicy] = []

    def run(policy: kinds.JailPolicy, **kwargs: object) -> kinds.CommandResult:
        seen.append(policy)
        if unavailable:
            raise jail.JailUnavailableError("namespace unavailable")
        return kinds.CommandResult(
            argv=policy.argv,
            returncode=124,
            stdout="",
            stderr="timed out",
            duration_s=policy.timeout_s,
        )

    monkeypatch.setattr(jail, "run_in_jail", run)
    dispatcher = dispatch.ToolDispatcher(root=tmp_path, config=cfg, isolation="none")
    prepared = workspace_setup.prepare(
        cfg,
        tmp_path,
        runner=dispatcher,
        reporter=app_reporter.STDIO_REPORTER,
    )
    assert prepared.problem is not None
    if unavailable:
        assert "namespace unavailable" in prepared.problem
        assert prepared.result is None
    else:
        assert seen[0].timeout_s > 0
        assert prepared.result is not None
        assert prepared.result.returncode == 124
        assert prepared.result.duration_s == seen[0].timeout_s


class _FailingSetup:
    """A dispatcher whose setup command exits 1 with a resolver error, at a given level."""

    def __init__(self, isolation: kinds.IsolationLevel) -> None:
        self.isolation: kinds.IsolationLevel = isolation

    def command_policy(self) -> str:
        return "yes"

    def run_setup(self, argv: tuple[str, ...]) -> kinds.CommandResult:
        return kinds.CommandResult(
            argv=argv,
            returncode=1,
            stdout="",
            stderr="error: dns error: failed to lookup address information",
            duration_s=0.1,
            exec_failed=False,
        )


@pytest.mark.parametrize(
    ("isolation", "network", "told"),
    [("full", "auto", True), ("full", "host", False), ("partial", "auto", False)],
)
def test_a_failed_setup_command_says_when_it_ran_with_no_route_out(
    tmp_path: pathlib.Path, isolation: kinds.IsolationLevel, network: str, told: bool
) -> None:
    """`uv sync` failed on a DNS error and nothing said the command had no route out.

    The line prints only where the run's commands have no route off the machine.
    """
    cfg = Config.model_validate(
        {"sandbox": {"network": network}, "parallel": {"setup_command": ["uv", "sync"]}}
    )
    dispatcher = _FailingSetup(isolation)

    found = workspace_setup.prepare(
        cfg,
        tmp_path,
        runner=dispatcher,
        reporter=app_reporter.STDIO_REPORTER,
    )

    assert found.problem is not None and found.problem.startswith(
        "the setup command failed (exit 1): uv sync"
    )
    pointer = (
        "the command ran with no route off this machine (sandbox.network), so it has to work"
        " offline. See docs/security.md."
    )
    assert (pointer in found.problem) is told


def test_the_no_setup_refusal_says_the_command_runs_in_the_sandbox(
    tmp_path: pathlib.Path,
) -> None:
    """The refusal's example was `uv sync`, which fails under the default network."""
    cfg = Config.model_validate(
        {"harness": {"verify_command": [".venv/bin/python", "-m", "pytest"]}}
    )
    dispatcher = _FailingSetup("full")

    text = workspace_setup.prepare(
        cfg,
        tmp_path,
        runner=dispatcher,
        reporter=app_reporter.STDIO_REPORTER,
    ).problem

    assert text is not None
    assert """agent6 config set --repo parallel.setup_command '["uv", "sync"]'""" in text
    assert (
        "it runs in the run's own sandbox: with no route off the machine it has to work"
        " offline (docs/security.md has the recipe)"
    ) in text


def test_a_gate_script_staged_but_not_committed_refuses_before_the_worktree_exists(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The worktree is cut from HEAD, which a staged addition is not part of.

    Read off the index, the check accepted the script and the run failed after the worktree
    and session existed.
    """
    repo = _repo(tmp_path, monkeypatch, config=_COMMANDS)
    _untracked_venv(repo)
    sp.run(["git", "add", "-f", ".venv/bin/python"], cwd=repo, check=True)
    _pin_gate(repo)
    prov = _script(monkeypatch)

    assert cli_main(["run", "--worktree", "--session-id", "wt-GATE02", "task"]) == 2

    assert prov.calls == 0
    assert _nothing_was_created(repo, "wt-GATE02")
    assert "REFUSING: the verify gate runs `.venv/bin/python`" in capsys.readouterr().err


def test_a_run_in_a_repo_with_a_name_that_is_not_utf8_completes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A latin-1 tracked name reaches the prompt escaped; raw, the snapshot JSON crashed the run."""
    repo = _repo(tmp_path, monkeypatch, config=_COMMANDS)
    (repo / os.fsdecode(b"caf\xe9.txt")).write_text("c\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "a latin-1 name")
    _script(monkeypatch, _edit("new.txt", "one\n"))

    assert cli_main(["run", "--session-id", "latin-1", "add new.txt"]) == 0
