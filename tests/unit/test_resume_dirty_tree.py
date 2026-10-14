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
import subprocess as sp
from typing import Any

import pytest

from agent6 import git_ops, paths
from agent6.app import _session, providers
from agent6.providers import ProviderResponse
from agent6.sessions import ipc
from agent6.tools import operator_prompts, schema
from agent6.ui.cli import cli_main, run

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
"""

_Step = tuple[str, tuple[dict[str, Any], ...]]


class _Scripted:
    """One canned ProviderResponse per call; the last repeats. A hook fires before each."""

    def __init__(self, script: list[_Step], *, before_call: Any = None) -> None:
        self.script = script
        self.calls = 0
        self._before = before_call

    def call(self, **_kw: Any) -> ProviderResponse:
        if self._before is not None:
            self._before(self.calls)
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


def _use(prov: _Scripted) -> Any:
    """A build_role_provider stand-in that always yields `prov`."""

    def _factory(*_a: Any, **_k: Any) -> _Scripted:
        return prov

    return _factory


def _none_isolation(*_a: Any, **_k: Any) -> str:
    return "none"


def _git(repo: pathlib.Path, *args: str) -> str:
    return sp.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


def _setup(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, git_toml: str) -> pathlib.Path:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    cfg_home = tmp_path / "cfg"
    (cfg_home / "agent6").mkdir(parents=True)
    (cfg_home / "agent6" / "config.toml").write_text(_CONFIG + git_toml, encoding="utf-8")
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
    monkeypatch.setattr(_session, "select_isolation", _none_isolation)
    return repo


def _session_dir(repo: pathlib.Path, session_id: str) -> pathlib.Path:
    return paths.state_dir(repo) / "sessions" / "runs" / session_id


def _stopped_run(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch, session_id: str, step: _Step
) -> None:
    """Run one step, then end at the step boundary on an operator stop."""
    session_dir = _session_dir(repo, session_id)

    def _stop_during_the_first_call(call_index: int) -> None:
        if call_index == 0:
            ipc.request_stop(session_dir)

    prov = _Scripted([step, _finish()], before_call=_stop_during_the_first_call)
    monkeypatch.setattr(providers, "build_role_provider", _use(prov))
    cli_main(["run", "--session-id", session_id, "task"])
    assert prov.calls == 1  # stopped at the boundary after the step


def _resume(monkeypatch: pytest.MonkeyPatch, session_id: str, script: list[_Step]) -> int:
    prov = _Scripted(script)
    monkeypatch.setattr(providers, "build_role_provider", _use(prov))
    return cli_main(["resume", session_id])


def _answering_frontend(monkeypatch: pytest.MonkeyPatch, answer: str) -> list[schema.UserQuestion]:
    """Make the CLI front-end one that can ask, answering every question with `answer`."""
    asked: list[schema.UserQuestion] = []
    real = run.session_frontend

    def _frontend(config_path: pathlib.Path | None = None) -> object:
        fe = real(config_path)

        def _questioner(_sd: pathlib.Path) -> object:
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
            capabilities=dataclasses.replace(fe.capabilities, can_ask=True),
            build_questioner=_questioner,
        )

    monkeypatch.setattr(run, "session_frontend", _frontend)
    return asked


def _chain_file(repo: pathlib.Path, session_id: str, path: str) -> str:
    return _git(repo, "show", f"{git_ops.chain_ref_for(session_id)}:{path}")


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
    (end,) = [e for e in events if e["type"] == "session.end"]
    tip = git_ops.chain_tip(repo, git_ops.chain_ref_for("tree-END001"))
    assert end["tree"] == git_ops.worktree_tree(repo, tip, frozenset())
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


def test_resume_refuses_to_stash_a_file_the_run_also_changed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A stash would take the run's own work on the file with the operator's edit: refused."""
    repo = _setup(tmp_path, monkeypatch, 'dirty_tree = "stash"\n')
    _stopped_run(repo, monkeypatch, "tree-BOTH01", _edit("a.py", "x = 5  # the run\n"))
    tip = git_ops.chain_tip(repo, git_ops.chain_ref_for("tree-BOTH01"))
    (repo / "a.py").write_text("x = 6  # operator on top\n", encoding="utf-8")
    capsys.readouterr()

    assert _resume(monkeypatch, "tree-BOTH01", [_finish()]) == 2

    err = capsys.readouterr().err
    assert "a.py" in err and "the run also changed" in err
    assert _git(repo, "stash", "list") == ""
    assert (repo / "a.py").read_text(encoding="utf-8") == "x = 6  # operator on top\n"
    assert git_ops.chain_tip(repo, git_ops.chain_ref_for("tree-BOTH01")) == tip


def test_resume_refuses_an_outside_edit_nobody_can_answer_for(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Under `dirty_tree = "ask"` with nobody to answer, the resume refuses and names the file."""
    repo = _setup(tmp_path, monkeypatch, "")
    _stopped_run(repo, monkeypatch, "tree-ASK001", _edit("z.txt", "1\n"))
    tip = git_ops.chain_tip(repo, git_ops.chain_ref_for("tree-ASK001"))
    (repo / "a.py").write_text("x = 2\n", encoding="utf-8")
    capsys.readouterr()

    assert _resume(monkeypatch, "tree-ASK001", [_finish()]) == 2

    err = capsys.readouterr().err
    assert "a.py" in err and "no terminal and no front-end" in err
    assert git_ops.chain_tip(repo, git_ops.chain_ref_for("tree-ASK001")) == tip
    assert not (_session_dir(repo, "tree-ASK001") / "worker.pid").exists()


@pytest.mark.parametrize(
    ("answer", "rc", "chain_a", "stashed"),
    [
        ("stash", 0, "x = 1\n", True),
        ("include", 0, "x = 2\n", False),
        ("cancel", 2, "x = 1\n", False),
    ],
)
def test_resume_asks_about_an_edit_made_outside_the_run(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    answer: str,
    rc: int,
    chain_a: str,
    stashed: bool,
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
    assert bool(_git(repo, "stash", "list")) is stashed
    if answer == "cancel":
        assert (repo / "a.py").read_text(encoding="utf-8") == "x = 2\n"
        assert not (_session_dir(repo, "tree-ASK002") / "worker.pid").exists()


def test_resume_keeps_the_runs_own_uncommitted_work(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run stopped over work it never committed resumes with that work in place.

    "Any difference from the chain tip after a clean end is the operator's" would stash the
    run's own edits here: with `commit_per_step = false` nothing of the run is committed.
    """
    repo = _setup(tmp_path, monkeypatch, 'dirty_tree = "stash"\ncommit_per_step = false\n')
    _stopped_run(repo, monkeypatch, "tree-OWN001", _edit("a.py", "x = 5  # the run\n"))
    asked = _answering_frontend(monkeypatch, "cancel")
    capsys.readouterr()

    assert _resume(monkeypatch, "tree-OWN001", [_finish()]) == 0

    assert asked == []
    assert _git(repo, "stash", "list") == ""
    assert (repo / "a.py").read_text(encoding="utf-8") == "x = 5  # the run\n"
