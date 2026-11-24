# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""What `agent6 resume` does with edits made outside the run, between executions.

Each end records the tree the run left; the paths that differ from it at the next start are
the operator's and follow `git.dirty_tree`.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import shlex
import subprocess
import sys
from typing import Any

import pytest

from agent6 import paths
from agent6.app import resume_tree, role_providers
from agent6.app.session import runtime as session_runtime
from agent6.providers import provider_types
from agent6.sessions import ipc
from agent6.tools import operator_prompts, schema
from agent6.ui.cli import entry as cli_entry
from agent6.ui.cli import run
from agent6.vcs.git import ops as git_ops
from agent6.vcs.git import read as git_read

_CONFIG = """
[sandbox]
isolation = "none"
run_commands = "{run_commands}"

[models.default]
model = "p/m"

[providers.p]
backend = "openai"
base_url = "http://127.0.0.1:1"
api_key_env = "PROBE_KEY"

[review]
when = "never"

[git]
"""

_Step = tuple[str, tuple[dict[str, Any], ...]]


class _Scripted:
    """One canned ProviderResponse per call; the last repeats. A hook fires before each."""

    def __init__(self, script: list[_Step], *, before_call: Any = None) -> None:
        self.script = script
        self.calls = 0
        self.requests: list[list[dict[str, Any]]] = []  # the messages of each call
        self._before = before_call

    def call(self, **kw: Any) -> provider_types.ProviderResponse:
        if self._before is not None:
            self._before(self.calls)
        self.requests.append(list(kw.get("messages", ())))
        text, tools = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return provider_types.ProviderResponse(
            text=text,
            tool_uses=tools,
            stop_reason="tool_use" if tools else "end_turn",
            input_tokens=10,
            output_tokens=5,
            cache_read_tokens=0,
            cache_write_tokens=0,
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


def _use(prov: _Scripted) -> Any:
    """A build_role_provider stand-in that always yields `prov`."""

    def _factory(*_a: Any, **_k: Any) -> _Scripted:
        return prov

    return _factory


def _none_isolation(*_a: Any, **_k: Any) -> str:
    return "none"


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


def _setup(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    git_toml: str,
    *,
    run_commands: str = "deny",
) -> pathlib.Path:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    cfg_home = tmp_path / "cfg"
    (cfg_home / "agent6").mkdir(parents=True)
    (cfg_home / "agent6" / "config.toml").write_text(
        _CONFIG.format(run_commands=run_commands) + git_toml, encoding="utf-8"
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg_home))
    monkeypatch.setenv("PROBE_KEY", "sk-probe")
    monkeypatch.delenv("AGENT6_DETACHED_AWAY", raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "b.py").write_text("y = 1\n", encoding="utf-8")
    _git(repo, "add", "a.py", "b.py")
    _git(repo, "commit", "-q", "-m", "seed")
    monkeypatch.chdir(repo)
    monkeypatch.setattr(session_runtime, "select_isolation", _none_isolation)
    return repo


def _session_dir(repo: pathlib.Path, session_id: str) -> pathlib.Path:
    return paths.state_dir(repo) / "sessions" / "runs" / session_id


def _stopped_run(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, session_id: str, *steps: _Step
) -> None:
    """Stop at the turn boundary after the scripted steps."""
    session_dir = _session_dir(repo, session_id)

    def _stop_after_steps(call_index: int) -> None:
        if call_index == len(steps) - 1:
            ipc.request_stop(session_dir)

    prov = _Scripted([*steps, _finish()], before_call=_stop_after_steps)
    monkeypatch.setattr(role_providers, "build_role_provider", _use(prov))
    cli_entry.cli_main(["run", "--session-id", session_id, "task"])
    assert prov.calls == len(steps)


def _resume(monkeypatch: pytest.MonkeyPatch, session_id: str, script: list[_Step]) -> int:
    return _resume_recording(monkeypatch, session_id, script)[0]


def _resume_recording(
    monkeypatch: pytest.MonkeyPatch, session_id: str, script: list[_Step]
) -> tuple[int, _Scripted]:
    prov = _Scripted(script)
    monkeypatch.setattr(role_providers, "build_role_provider", _use(prov))
    return cli_entry.cli_main(["resume", session_id]), prov


def _user_texts(messages: list[dict[str, Any]]) -> list[str]:
    """Return the text blocks of the user messages, in order."""
    return [
        block["text"]
        for message in messages
        if message["role"] == "user"
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == "text"
    ]


def _answering_frontend(
    monkeypatch: pytest.MonkeyPatch, answer: str, *, can_ask: bool = True
) -> list[schema.UserQuestion]:
    """Make the CLI front-end answer every question with `answer`, and say whether it can ask."""
    asked: list[schema.UserQuestion] = []
    real = run.session_frontend

    def _frontend(config_paths: tuple[pathlib.Path, ...] = ()) -> object:
        fe = real(config_paths)

        def _questioner(_sd: pathlib.Path) -> operator_prompts.Questioner:
            def _ask(
                request: operator_prompts.QuestionRequest, /
            ) -> operator_prompts.QuestionAnswer:
                asked.extend(request.questions)
                return operator_prompts.QuestionAnswer(
                    tuple(answer for _ in request.questions), "stdin"
                )

            return _ask

        return dataclasses.replace(
            fe,
            capabilities=dataclasses.replace(fe.capabilities, can_ask=can_ask),
            build_questioner=_questioner,
        )

    monkeypatch.setattr(run, "session_frontend", _frontend)
    return asked


def _chain_file(repo: pathlib.Path, session_id: str, path: str) -> str:
    return _git(repo, "show", f"{git_read.chain_ref_for(session_id)}:{path}")


def test_session_end_records_the_tree_the_run_left(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The end event carries the worktree's tree id, so a resume can tell what changed since."""
    repo = _setup(tmp_path, monkeypatch, 'dirty_tree = "stash"\n')
    _stopped_run(repo, monkeypatch, "tree-END001", _edit("z.txt", "1\n"))
    events = [
        json.loads(line)
        for line in (_session_dir(repo, "tree-END001") / "logs.jsonl").read_text().splitlines()
    ]
    (end,) = [e for e in events if e["type"] == "session.stop"]
    tip = git_read.chain_tip(repo, git_read.chain_ref_for("tree-END001"))
    assert end["tree"] == git_read.worktree_tree(repo, tip, frozenset())
    assert _git(repo, "log", "--oneline", "-1", "--format=%s").strip() == "seed"  # no commit made


def test_resume_stashes_an_edit_made_outside_the_run_under_dirty_tree_stash(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Under `dirty_tree = "stash"` a resume sets the operator's edit aside, out of the chain.

    The resume had no record of the tree its last execution left, so the next chain commit took
    every difference from the chain tip as run work.
    """
    repo = _setup(tmp_path, monkeypatch, 'dirty_tree = "stash"\n')
    _stopped_run(repo, monkeypatch, "tree-STASH1", _edit("z.txt", "1\n"))
    (repo / "a.py").write_text("x = 2  # operator draft\n", encoding="utf-8")
    capsys.readouterr()

    assert _resume(monkeypatch, "tree-STASH1", [_edit("y.txt", "1\n"), _finish()]) == 0

    assert _chain_file(repo, "tree-STASH1", "a.py") == "x = 1\n"
    assert _chain_file(repo, "tree-STASH1", "y.txt") == "1\n"
    assert "agent6 auto-stash before run tree-STASH1" in _git(repo, "stash", "list")
    assert (repo / "z.txt").read_text(encoding="utf-8") == "1\n"  # the run's work stays
    err = capsys.readouterr().err
    assert "stashed for this execution as yours: a.py" in err
    assert "restore them with: git stash apply" in err  # the resumed execution's own end line


def test_a_resume_that_cannot_say_it_stashed_still_restores_the_stash(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resume whose stash note failed to print left the operator's edit stashed.

    The teardown ends the stash the stage recorded, and the note came between the stash and
    its record.
    """
    repo = _setup(tmp_path, monkeypatch, 'dirty_tree = "stash"\nauto_stash_pop = true\n')
    _stopped_run(repo, monkeypatch, "tree-NOTE01", _edit("z.txt", "1\n"))
    (repo / "a.py").write_text("x = 2\n", encoding="utf-8")
    stderr = sys.stderr

    class _FailsOneLine:
        def write(self, text: str) -> int:
            if "stashed for this execution" in text:
                raise OSError(5, "Input/output error")
            return stderr.write(text)

        def __getattr__(self, name: str) -> Any:
            return getattr(stderr, name)

    monkeypatch.setattr(sys, "stderr", _FailsOneLine())

    assert _resume(monkeypatch, "tree-NOTE01", [_finish()]) == 1

    assert (repo / "a.py").read_text(encoding="utf-8") == "x = 2\n"
    assert _git(repo, "stash", "list") == ""


def test_resume_stashes_an_outside_edit_to_a_name_git_quotes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The name came back quoted, so the stash left the real file out and said it had stashed it.

    The run's next commit then took the operator's edit as its own.
    """
    repo = _setup(tmp_path, monkeypatch, 'dirty_tree = "stash"\n')
    odd = 'notes\\draft "v1".txt'
    (repo / odd).write_text("base\n", encoding="utf-8")
    _git(repo, "add", "--", odd)
    _git(repo, "commit", "-q", "-m", "odd")
    _stopped_run(repo, monkeypatch, "tree-QUOTE1", _edit("z.txt", "1\n"))
    (repo / odd).write_text("operator edit\n", encoding="utf-8")
    capsys.readouterr()

    assert _resume(monkeypatch, "tree-QUOTE1", [_edit("y.txt", "1\n"), _finish()]) == 0

    assert _chain_file(repo, "tree-QUOTE1", odd) == "base\n"
    assert (repo / odd).read_text(encoding="utf-8") == "base\n"
    assert "agent6 auto-stash before run tree-QUOTE1" in _git(repo, "stash", "list")


def test_resume_stashes_a_staged_rename_without_taking_run_work(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A staged rename of an untouched file is stashed without a missing-source pathspec error."""
    repo = _setup(tmp_path, monkeypatch, 'dirty_tree = "stash"\n')
    session_id = "tree-MOVE01"
    monkeypatch.setattr(
        role_providers,
        "build_role_provider",
        _use(_Scripted([_edit("a.py", "x = 5\n"), _finish()])),
    )
    assert cli_entry.cli_main(["run", "--session-id", session_id, "task"]) == 0
    _git(repo, "mv", "b.py", "b.rst")
    (repo / "b.rst").write_text("y = 2  # operator draft\n", encoding="utf-8")
    monkeypatch.setattr(role_providers, "build_role_provider", _use(_Scripted([_finish()])))
    capsys.readouterr()

    assert cli_entry.cli_main(["resume", session_id, "--steer", "new work"]) == 0

    assert _chain_file(repo, session_id, "a.py") == "x = 5\n"
    assert _chain_file(repo, session_id, "b.py") == "y = 1\n"
    assert (repo / "a.py").read_text(encoding="utf-8") == "x = 5\n"
    assert (repo / "b.py").read_text(encoding="utf-8") == "y = 1\n"
    assert not (repo / "b.rst").exists()
    entry = git_read.find_stash(repo, git_ops.auto_stash_message(session_id))
    assert entry is not None
    assert "restore them with:" in capsys.readouterr().err

    assert git_ops.restore_stash(repo, entry)
    assert not (repo / "b.py").exists()
    assert (repo / "b.rst").read_text(encoding="utf-8") == "y = 2  # operator draft\n"
    assert (repo / "a.py").read_text(encoding="utf-8") == "x = 5\n"


@pytest.mark.parametrize("resuming", [False, True], ids=["run", "resume"])
def test_auto_stash_pop_restores_over_nonconflicting_run_work(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    resuming: bool,
) -> None:
    """A run or resume restores the operator's stash over the run's uncommitted work."""
    repo = _setup(tmp_path, monkeypatch, 'dirty_tree = "stash"\nauto_stash_pop = true\n')
    session_id = "tree-POP001"
    if resuming:
        monkeypatch.setattr(
            role_providers,
            "build_role_provider",
            _use(_Scripted([_edit("b.py", "y = 2\n"), _finish()])),
        )
        assert cli_entry.cli_main(["run", "--session-id", session_id, "task"]) == 0
    (repo / "a.py").write_text("x = 2  # operator draft\n", encoding="utf-8")
    capsys.readouterr()
    script = [_edit("b.py", "y = 5  # the run\n"), _finish()]

    monkeypatch.setattr(role_providers, "build_role_provider", _use(_Scripted(script)))
    if resuming:
        assert cli_entry.cli_main(["resume", session_id, "--steer", "new work"]) == 0
    else:
        assert cli_entry.cli_main(["run", "--session-id", session_id, "task"]) == 0

    assert (repo / "a.py").read_text(encoding="utf-8") == "x = 2  # operator draft\n"
    assert (repo / "b.py").read_text(encoding="utf-8") == "y = 5  # the run\n"
    assert _chain_file(repo, session_id, "a.py") == "x = 1\n"
    assert _chain_file(repo, session_id, "b.py") == "y = 5  # the run\n"
    assert _git(repo, "branch", "--show-current").strip() == "main"
    assert _git(repo, "show", "HEAD:b.py") == "y = 1\n"
    assert _git(repo, "stash", "list") == ""
    assert "restored your pre-run changes onto main" in capsys.readouterr().err


def test_auto_stash_pop_keeps_a_stash_that_would_overwrite_run_work(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failed apply preserves the stash and run work without claiming markers exist."""
    repo = _setup(tmp_path, monkeypatch, 'dirty_tree = "stash"\nauto_stash_pop = true\n')
    (repo / "a.py").write_text("x = 2  # operator draft\n", encoding="utf-8")
    monkeypatch.setattr(
        role_providers,
        "build_role_provider",
        _use(_Scripted([_edit("a.py", "x = 5\n"), _finish()])),
    )

    assert cli_entry.cli_main(["run", "--session-id", "tree-POP002", "task"]) == 0

    assert (repo / "a.py").read_text(encoding="utf-8") == "x = 5\n"
    entry = git_read.find_stash(repo, git_ops.auto_stash_message("tree-POP002"))
    assert entry is not None
    err = capsys.readouterr().err
    assert "stash is preserved" in err and f"git stash apply {entry.sha}" in err
    assert "resolve the markers" not in err
    assert _git(repo, "ls-files", "--unmerged") == ""


def test_resume_refuses_to_stash_a_file_the_run_also_changed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A stash would take the run's own work on the file with the operator's edit: refused."""
    repo = _setup(tmp_path, monkeypatch, 'dirty_tree = "stash"\n')
    _stopped_run(repo, monkeypatch, "tree-BOTH01", _edit("a.py", "x = 5  # the run\n"))
    tip = git_read.chain_tip(repo, git_read.chain_ref_for("tree-BOTH01"))
    (repo / "a.py").write_text("x = 6  # operator on top\n", encoding="utf-8")
    capsys.readouterr()

    assert _resume(monkeypatch, "tree-BOTH01", [_finish()]) == 2

    err = capsys.readouterr().err
    assert "a.py" in err and "the run also changed" in err
    assert _git(repo, "stash", "list") == ""
    assert (repo / "a.py").read_text(encoding="utf-8") == "x = 6  # operator on top\n"
    assert git_read.chain_tip(repo, git_read.chain_ref_for("tree-BOTH01")) == tip


@pytest.mark.parametrize("policy", ["stash", "ask"])
def test_resume_stash_refusal_counts_blockers_and_restores_from_the_run_branch(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    policy: str,
) -> None:
    """The refusal counts only blockers and its restore command keeps the run's edits."""
    repo = _setup(tmp_path, monkeypatch, f'dirty_tree = "{policy}"\n')
    session_id = "tree-UNDO01"
    monkeypatch.setattr(
        role_providers,
        "build_role_provider",
        _use(_Scripted([_edit("a.py", "x = 5\n"), _finish()])),
    )
    assert cli_entry.cli_main(["run", "--session-id", session_id, "task"]) == 0
    (repo / "a.py").write_text("x = 6  # operator on top\n", encoding="utf-8")
    (repo / "b.py").write_text("y = 2  # operator draft\n", encoding="utf-8")
    if policy == "ask":
        _answering_frontend(monkeypatch, "stash")
    monkeypatch.setattr(role_providers, "build_role_provider", _use(_Scripted([_finish()])))
    capsys.readouterr()

    assert cli_entry.cli_main(["resume", session_id, "--steer", "new work"]) == 2

    err = capsys.readouterr().err
    assert "1 file blocks stashing" in err
    assert "- a.py" in err and "- b.py" not in err
    assert "changed outside this run since it last ended:" not in err
    restore = next(
        line.strip() for line in err.splitlines() if line.strip().startswith("git restore")
    )
    assert shlex.split(restore) == [
        "git",
        "restore",
        "--source",
        "agent6/tree-UNDO01",
        "--",
        ":(top,literal)a.py",
    ]
    assert _git(repo, "show", "HEAD:a.py") == "x = 1\n"
    assert _git(repo, "stash", "list") == ""

    subprocess.run(shlex.split(restore), cwd=repo, check=True, capture_output=True)
    assert (repo / "a.py").read_text(encoding="utf-8") == "x = 5\n"
    assert (repo / "b.py").read_text(encoding="utf-8") == "y = 2  # operator draft\n"
    assert cli_entry.cli_main(["resume", session_id, "--steer", "new work"]) == 0
    assert _chain_file(repo, session_id, "a.py") == "x = 5\n"
    assert _chain_file(repo, session_id, "b.py") == "y = 1\n"


def test_resume_stash_refusal_restore_touches_only_the_files_it_names(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The printed `git restore` read `data[1].py` as a glob and took `data1.py` back too.

    The operator's edit to `data1.py`, a file the run never touched, was lost.
    """
    repo = _setup(tmp_path, monkeypatch, 'dirty_tree = "stash"\n')
    for name in ("data[1].py", "data1.py"):
        (repo / name).write_text("base\n", encoding="utf-8")
    _git(repo, "add", "--", "data[1].py", "data1.py")
    _git(repo, "commit", "-q", "-m", "data")
    session_id = "tree-GLOB01"
    monkeypatch.setattr(
        role_providers,
        "build_role_provider",
        _use(_Scripted([_edit("data[1].py", "run\n"), _finish()])),
    )
    assert cli_entry.cli_main(["run", "--session-id", session_id, "task"]) == 0
    (repo / "data[1].py").write_text("operator on top\n", encoding="utf-8")
    (repo / "data1.py").write_text("operator draft\n", encoding="utf-8")
    monkeypatch.setattr(role_providers, "build_role_provider", _use(_Scripted([_finish()])))
    capsys.readouterr()

    assert cli_entry.cli_main(["resume", session_id, "--steer", "new work"]) == 2

    err = capsys.readouterr().err
    restore = next(
        line.strip() for line in err.splitlines() if line.strip().startswith("git restore")
    )
    subprocess.run(shlex.split(restore), cwd=repo, check=True, capture_output=True)
    assert (repo / "data[1].py").read_text(encoding="utf-8") == "run\n"
    assert (repo / "data1.py").read_text(encoding="utf-8") == "operator draft\n"


def test_resume_stash_refusal_restores_work_the_run_never_committed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no run commit, the refusal's restore command uses the recorded end tree."""
    repo = _setup(tmp_path, monkeypatch, 'dirty_tree = "stash"\ncommit_per_turn = false\n')
    session_id = "tree-UNDO02"
    monkeypatch.setattr(
        role_providers,
        "build_role_provider",
        _use(_Scripted([_edit("a.py", "x = 5\n"), _finish()])),
    )
    assert cli_entry.cli_main(["run", "--session-id", session_id, "task"]) == 0
    events = [
        json.loads(line)
        for line in (_session_dir(repo, session_id) / "logs.jsonl").read_text().splitlines()
    ]
    (end,) = [e for e in events if e["type"] == "session.stop"]
    (repo / "a.py").write_text("x = 6  # operator on top\n", encoding="utf-8")
    monkeypatch.setattr(role_providers, "build_role_provider", _use(_Scripted([_finish()])))
    capsys.readouterr()

    assert cli_entry.cli_main(["resume", session_id, "--steer", "new work"]) == 2

    err = capsys.readouterr().err
    restore = next(
        line.strip() for line in err.splitlines() if line.strip().startswith("git restore")
    )
    assert shlex.split(restore) == [
        "git",
        "restore",
        "--source",
        end["tree"],
        "--",
        ":(top,literal)a.py",
    ]
    subprocess.run(shlex.split(restore), cwd=repo, check=True, capture_output=True)
    assert (repo / "a.py").read_text(encoding="utf-8") == "x = 5\n"
    assert cli_entry.cli_main(["resume", session_id, "--steer", "new work"]) == 0


def test_resume_refuses_an_outside_edit_nobody_can_answer_for(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Under `dirty_tree = "ask"` with nobody to answer, the resume refuses and names the file."""
    repo = _setup(tmp_path, monkeypatch, "")
    _stopped_run(repo, monkeypatch, "tree-ASK001", _edit("z.txt", "1\n"))
    tip = git_read.chain_tip(repo, git_read.chain_ref_for("tree-ASK001"))
    (repo / "a.py").write_text("x = 2\n", encoding="utf-8")
    capsys.readouterr()

    assert _resume(monkeypatch, "tree-ASK001", [_finish()]) == 2

    err = capsys.readouterr().err
    assert "a.py" in err and "no terminal and no front-end" in err
    assert git_read.chain_tip(repo, git_read.chain_ref_for("tree-ASK001")) == tip
    assert not (_session_dir(repo, "tree-ASK001") / "worker.pid").exists()


def test_resume_reads_the_away_mode_after_it_scanned_the_tree(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resume refused on the away mode it read before the scan for outside edits.

    The operator set `wait` while the scan ran. The resume said nobody could answer, though
    its question would have waited for `agent6 answer`.
    """
    repo = _setup(tmp_path, monkeypatch, "")
    _stopped_run(repo, monkeypatch, "tree-ASK003", _edit("z.txt", "1\n"))
    (repo / "a.py").write_text("x = 2\n", encoding="utf-8")
    asked = _answering_frontend(monkeypatch, "include", can_ask=False)
    scan = resume_tree.outside_edits

    def _scan_while_the_operator_sets_wait(*args: Any, **kwargs: Any) -> list[str]:
        found = scan(*args, **kwargs)
        ipc.set_away_mode(_session_dir(repo, "tree-ASK003"), "wait")
        return found

    monkeypatch.setattr(resume_tree, "outside_edits", _scan_while_the_operator_sets_wait)

    assert _resume(monkeypatch, "tree-ASK003", [_finish()]) == 0

    assert len(asked) == 1 and "a.py" in asked[0].question


@pytest.mark.parametrize(
    ("answer", "rc", "chain_a"),
    [
        ("stash", 0, "x = 1\n"),
        ("include", 0, "x = 2\n"),
        ("cancel", 2, "x = 1\n"),
    ],
)
def test_resume_asks_about_an_edit_made_outside_the_run(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    answer: str,
    rc: int,
    chain_a: str,
) -> None:
    """Under `dirty_tree = "ask"` a resume asks once, naming the file, and follows the answer."""
    repo = _setup(tmp_path, monkeypatch, "")
    _stopped_run(repo, monkeypatch, "tree-ASK002", _edit("z.txt", "1\n"))
    (repo / "a.py").write_text("x = 2\n", encoding="utf-8")
    asked = _answering_frontend(monkeypatch, answer)

    assert _resume(monkeypatch, "tree-ASK002", [_edit("y.txt", "1\n"), _finish()]) == rc

    assert len(asked) == 1
    assert "a.py" in asked[0].question and "z.txt" not in asked[0].question
    assert asked[0].options == ("stash", "include", "cancel")
    assert _chain_file(repo, "tree-ASK002", "a.py") == chain_a
    assert _git(repo, "stash", "list") == ""
    assert (repo / "a.py").read_text(encoding="utf-8") == "x = 2\n"
    if answer == "cancel":
        assert not (_session_dir(repo, "tree-ASK002") / "worker.pid").exists()


def test_model_git_resume_compares_its_uncommitted_edit_to_the_stop_tree(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A file tracked by the model's commit but ignored at the base vanished from resume's tree."""
    repo = _setup(tmp_path, monkeypatch, 'control = "model"\n', run_commands="allow")
    session_id = "tree-MODEL1"
    (repo / ".gitignore").write_text("*.log\n", encoding="utf-8")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-qm", "Ignore logs")
    move = {
        "type": "tool_use",
        "name": "run_command",
        "id": "move",
        "input": {"argv": ["sh", "-c", "git mv a.py draft.log && git commit -qm 'Track draft'"]},
    }
    _stopped_run(repo, monkeypatch, session_id, ("", (move,)), _edit("draft.log", "unfinished\n"))
    assert _git(repo, "show", "HEAD:draft.log") == "x = 1\n"
    draft = repo / "draft.log"
    assert draft.read_text() == "unfinished\n"
    capsys.readouterr()

    draft.write_text("operator\n", encoding="utf-8")
    rc, provider = _resume_recording(monkeypatch, session_id, [_finish()])
    assert rc == 2 and not provider.calls
    assert (
        "1 file changed outside this run since it last ended:\n- draft.log"
        in capsys.readouterr().err
    )
    assert draft.read_text() == "operator\n"

    draft.write_text("unfinished\n", encoding="utf-8")
    assert _resume(monkeypatch, session_id, [_finish()]) == 0
    note = "uncommitted from this run's last execution: draft.log"
    assert note in capsys.readouterr().err
    log = (_session_dir(repo, session_id) / "logs.jsonl").read_text()
    assert any(e.get("text") == note for e in map(json.loads, log.splitlines()))
    assert _git(repo, "status", "--short") == " M draft.log\n"


def test_resume_keeps_the_runs_own_uncommitted_work(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run stopped over work it never committed resumes with that work in place.

    "Any difference from the chain tip after a clean end is the operator's" would stash the
    run's own edits here: with `commit_per_turn = false` nothing of the run is committed.
    """
    repo = _setup(tmp_path, monkeypatch, 'dirty_tree = "stash"\ncommit_per_turn = false\n')
    _stopped_run(repo, monkeypatch, "tree-OWN001", _edit("a.py", "x = 5  # the run\n"))
    asked = _answering_frontend(monkeypatch, "cancel")
    capsys.readouterr()

    assert _resume(monkeypatch, "tree-OWN001", [_finish()]) == 0

    assert asked == []
    assert _git(repo, "stash", "list") == ""
    assert (repo / "a.py").read_text(encoding="utf-8") == "x = 5  # the run\n"


def test_model_git_resume_leaves_hidden_changes_out(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hidden file can differ between the stop tree and the commit seeding the resume."""
    repo = _setup(tmp_path, monkeypatch, 'control = "model"\n')
    hidden = frozenset({"b.py"})
    end_tree = git_read.worktree_tree(repo, "HEAD", hidden)
    (repo / "b.py").write_text("new committed version\n")
    _git(repo, "commit", "-qm", "Change hidden file", "--", "b.py")
    (repo / "b.py").write_text("private draft\n")

    assert resume_tree.outside_edits(repo, end_tree, "HEAD", hidden) == []


@pytest.mark.parametrize("change", ["directory", "tracked-pack-name", "index"])
def test_model_git_resume_refuses_unrecorded_changes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """A matching working-copy path cannot authorize a changed directory or staged content."""
    repo = _setup(tmp_path, monkeypatch, 'control = "model"\n')
    path = repo / (".agent6" if change == "tracked-pack-name" else "pkg") / "file.py"
    path.parent.mkdir()
    path.write_text("committed\n")
    _git(repo, "add", "-f", "--", str(path))
    _git(repo, "commit", "-qm", "Track file")
    sid = "tree-BND001"
    _stopped_run(repo, monkeypatch, sid, _edit("a.py", "run edit\n"))
    if change == "directory":
        path.unlink()
        path.parent.rmdir()
        path.parent.write_text("replacement\n")
    elif change == "index":
        path.write_text("operator staged\n")
        _git(repo, "add", "--", str(path))
        path.write_text("committed\n")
    else:
        path.write_text("operator\n")
    rc, provider = _resume_recording(monkeypatch, sid, [_finish()])
    assert rc == 2 and not provider.calls


def test_model_git_resume_names_a_missing_stage_measurement(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _setup(tmp_path, monkeypatch, 'control = "model"\n')
    sid = "tree-OLD001"
    _stopped_run(repo, monkeypatch, sid, _edit("a.py", "run edit\n"))
    log = _session_dir(repo, sid) / "logs.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    for row in rows:
        row.pop("index_tree", None)
    log.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _git(repo, "add", "a.py")
    capsys.readouterr()
    rc, provider = _resume_recording(monkeypatch, sid, [_finish()])
    assert rc == 2 and not provider.calls
    err = capsys.readouterr().err
    assert "no staged-tree record" in err and "a.py" in err
    assert "changed outside this run" not in err


def test_model_git_resume_distinguishes_staged_and_working_content(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _setup(tmp_path, monkeypatch, 'control = "model"\n', run_commands="allow")
    (repo / ".gitignore").write_text("*.log\n")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-qm", "Ignore logs")
    sid = "tree-STG001"
    stage = {
        "type": "tool_use",
        "name": "run_command",
        "id": "stage",
        "input": {
            "argv": [
                "sh",
                "-c",
                "echo first > draft.log && git add -f draft.log && echo second > draft.log",
            ]
        },
    }
    _stopped_run(repo, monkeypatch, sid, ("", (stage,)))
    draft = repo / "draft.log"
    draft.write_text("outside\n")
    rc, provider = _resume_recording(monkeypatch, sid, [_finish()])
    assert rc == 2 and not provider.calls
    draft.write_text("second\n")
    assert _resume(monkeypatch, sid, [_finish()]) == 0
    assert _git(repo, "show", ":draft.log") == "first\n"
    assert draft.read_text() == "second\n"


@pytest.mark.parametrize(("policy", "rc"), [("", 2), ('dirty_tree = "include"\n', 0)])
def test_a_run_killed_before_it_recorded_a_tree_treats_unwritten_changes_as_outside(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    policy: str,
    rc: int,
) -> None:
    """With no end tree, a tracked change no tool of the run wrote follows the dirty-tree policy.

    Taken silently into the run's next commit, another run's edits left in the checkout read as
    this run's work.
    """
    repo = _setup(tmp_path, monkeypatch, policy)
    _stopped_run(repo, monkeypatch, "tree-KILL01", _edit("z.txt", "1\n"))
    logs = _session_dir(repo, "tree-KILL01") / "logs.jsonl"
    kept = [
        ln for ln in logs.read_text(encoding="utf-8").splitlines() if '"session.stop"' not in ln
    ]
    logs.write_text("\n".join(kept) + "\n", encoding="utf-8")  # as a SIGKILL leaves it
    (repo / "a.py").write_text("x = 2\n", encoding="utf-8")  # never written by the run
    (repo / "z.txt").write_text("2\n", encoding="utf-8")  # the run's own file, unfinished
    capsys.readouterr()

    assert _resume(monkeypatch, "tree-KILL01", [_finish()]) == rc

    err = capsys.readouterr().err
    if rc == 2:
        assert "a.py" in err and "recorded no tree" in err and "z.txt" not in err
    else:
        assert "recorded no tree" not in err


def test_a_killed_runs_own_edit_to_a_committed_file_is_not_stashed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no end tree, a stash would take back the run's own work on a file it committed.

    The chain tip stands in for the missing tree: a path the run's commits changed blocks the
    stash, as it does after a clean end.
    """
    repo = _setup(tmp_path, monkeypatch, 'dirty_tree = "stash"\n')
    _stopped_run(repo, monkeypatch, "tree-KILL02", _edit("z.txt", "1\n"))
    # A file a run command wrote and the run committed: no tool result names it.
    (repo / "w.txt").write_text("1\n", encoding="utf-8")
    ref = git_read.chain_ref_for("tree-KILL02")
    assert git_ops.chain_commit(
        repo, "agent6 turn 2: w", ref=ref, fallback_parent=None, also_branch="agent6/tree-KILL02"
    )
    logs = _session_dir(repo, "tree-KILL02") / "logs.jsonl"
    kept = [
        ln for ln in logs.read_text(encoding="utf-8").splitlines() if '"session.stop"' not in ln
    ]
    logs.write_text("\n".join(kept) + "\n", encoding="utf-8")  # as a SIGKILL leaves it
    (repo / "w.txt").write_text("2\n", encoding="utf-8")  # edited again by a command, then killed
    capsys.readouterr()

    assert _resume(monkeypatch, "tree-KILL02", [_finish()]) == 2

    err = capsys.readouterr().err
    assert "w.txt" in err and "blocks stashing (the run also changed it)" in err
    assert (repo / "w.txt").read_text(encoding="utf-8") == "2\n"


def test_a_model_git_run_killed_before_it_recorded_a_tree_keeps_its_own_commits(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no end tree, every file the model had committed read as an edit made outside the run.

    The model's git leaves no chain, so the comparison fell back to the commit the run started
    from and a clean checkout was refused.
    """
    repo = _setup(tmp_path, monkeypatch, 'control = "model"\n', run_commands="allow")
    commit = {
        "type": "tool_use",
        "name": "run_command",
        "id": "commit",
        "input": {"argv": ["sh", "-c", "echo 'x = 2' > a.py && git commit -qam 'Change a'"]},
    }
    look = {
        "type": "tool_use",
        "name": "run_command",
        "id": "look",
        "input": {"argv": ["git", "status", "--short"]},
    }
    _stopped_run(repo, monkeypatch, "tree-KILL03", ("", (commit,)), ("", (look,)))
    assert _git(repo, "show", "HEAD:a.py") == "x = 2\n"
    assert _git(repo, "status", "--short") == ""
    logs = _session_dir(repo, "tree-KILL03") / "logs.jsonl"
    kept = [
        ln for ln in logs.read_text(encoding="utf-8").splitlines() if '"session.stop"' not in ln
    ]
    logs.write_text("\n".join(kept) + "\n", encoding="utf-8")  # as a SIGKILL leaves it
    (repo / "b.py").write_text("y = 2\n", encoding="utf-8")  # never written by the run
    capsys.readouterr()

    assert _resume(monkeypatch, "tree-KILL03", [_finish()]) == 2
    err = capsys.readouterr().err
    assert "1 file changed since this run's last commit" in err
    assert "- b.py" in err and "a.py" not in err

    (repo / "b.py").write_text("y = 1\n", encoding="utf-8")
    assert _resume(monkeypatch, "tree-KILL03", [_finish()]) == 0
    assert "recorded no tree" not in capsys.readouterr().err


def test_the_restore_recipe_names_the_file_as_it_is_not_as_it_prints(
    tmp_path: pathlib.Path,
) -> None:
    """A newline in a name reached the pathspec in its printed spelling, naming another file."""
    text = resume_tree._shared_refusal(tmp_path, ["a\nb.py"], end_tree="abc1234", run_branch=None)
    assert ":(top,literal)" not in text and "typed as the listing spells it" in text
    assert not [line for line in text.splitlines() if line.startswith("[agent6]")]


@pytest.mark.parametrize("policy", ['dirty_tree = "include"\n', ""])
def test_resume_names_the_files_the_operator_kept_in_the_tree_to_the_model(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    """An outside edit kept for the run is named to the model before its first call.

    Without the notice the model holds the file as it last read it and writes over the
    operator's change. The run's own file, unchanged, is not named.
    """
    repo = _setup(tmp_path, monkeypatch, policy)
    _stopped_run(repo, monkeypatch, "tree-TELL01", _edit("z.txt", "1\n"))
    (repo / "a.py").write_text("x = 2\n", encoding="utf-8")
    _answering_frontend(monkeypatch, "include")

    rc, prov = _resume_recording(monkeypatch, "tree-TELL01", [_finish()])

    assert rc == 0
    told = [t for t in _user_texts(prov.requests[0]) if "since the last execution" in t]
    assert len(told) == 1
    assert "a.py" in told[0] and "z.txt" not in told[0]
    assert (repo / "a.py").read_text(encoding="utf-8") == "x = 2\n"


def test_resume_says_nothing_to_the_model_about_a_stashed_outside_edit(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stashed edit is not in the tree the model works on, so the model does not hear of it."""
    repo = _setup(tmp_path, monkeypatch, 'dirty_tree = "stash"\n')
    _stopped_run(repo, monkeypatch, "tree-TELL02", _edit("z.txt", "1\n"))
    (repo / "a.py").write_text("x = 2\n", encoding="utf-8")

    rc, prov = _resume_recording(monkeypatch, "tree-TELL02", [_finish()])

    assert rc == 0
    assert not any("since the last execution" in t for t in _user_texts(prov.requests[0]))


@pytest.mark.parametrize(("answer", "rc"), [("include", 0), ("stash", 2)])
def test_resume_counts_a_file_the_run_created_and_someone_deleted(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    answer: str,
    rc: int,
) -> None:
    """A file the run created and lost after it ended is a change from outside.

    The resume asks about it; included, the model hears which file went; a stash is refused,
    since the run also changed it. The docs once promised untracked files were never counted.
    """
    repo = _setup(tmp_path, monkeypatch, "")
    _stopped_run(repo, monkeypatch, "tree-GONE01", _edit("z.txt", "1\n"))
    (repo / "z.txt").unlink()
    asked = _answering_frontend(monkeypatch, answer)
    capsys.readouterr()

    got, prov = _resume_recording(monkeypatch, "tree-GONE01", [_finish()])

    assert got == rc
    assert len(asked) == 1 and "z.txt" in asked[0].question
    if rc == 0:
        told = [t for t in _user_texts(prov.requests[0]) if "since the last execution" in t]
        assert len(told) == 1 and "z.txt" in told[0]
    else:
        assert "the run also changed it" in capsys.readouterr().err
