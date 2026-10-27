# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A queued question through disk: asked, left pending at the end, answered, delivered once."""

from __future__ import annotations

import json
import pathlib
import subprocess
from typing import Any

import pytest

from agent6 import paths
from agent6.app import _session, providers
from agent6.providers import ProviderResponse
from agent6.ui.cli import cli_main
from agent6.viewmodel import listing

_CONFIG = """
[sandbox]
isolation = "none"
run_commands = "deny"

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

_Step = tuple[str, tuple[dict[str, Any], ...]]


class _Scripted:
    """One canned response per call, the last repeating; every call's messages are kept."""

    def __init__(self, script: list[_Step]) -> None:
        self.script = script
        self.seen: list[list[dict[str, Any]]] = []

    def call(self, **kw: Any) -> ProviderResponse:
        self.seen.append(list(kw.get("messages", ())))
        text, tools = self.script[min(len(self.seen) - 1, len(self.script) - 1)]
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


def _ask_queued() -> _Step:
    tool = {
        "type": "tool_use",
        "id": "q1",
        "name": "ask_user",
        "input": {
            "questions": [{"question": "Which DB?", "options": ["pg", "sqlite"]}],
            "wait": False,
        },
    }
    return ("", (tool,))


def _finish() -> _Step:
    tool = {"type": "tool_use", "id": "t1", "name": "finish_session", "input": {"summary": "done"}}
    return ("", (tool,))


def _use(prov: _Scripted) -> Any:
    def _factory(*_a: Any, **_k: Any) -> _Scripted:
        return prov

    return _factory


def _none_isolation(*_a: Any, **_k: Any) -> str:
    return "none"


def _setup(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    cfg_home = tmp_path / "cfg"
    (cfg_home / "agent6").mkdir(parents=True)
    (cfg_home / "agent6" / "config.toml").write_text(_CONFIG, encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg_home))
    monkeypatch.setenv("PROBE_KEY", "sk-probe")
    monkeypatch.delenv("AGENT6_DETACHED_AWAY", raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"], check=True)
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "a.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "seed"], check=True)
    monkeypatch.chdir(repo)
    monkeypatch.setattr(_session, "select_isolation", _none_isolation)
    return repo


def _dump(messages: list[dict[str, Any]]) -> str:
    """Return the conversation as JSON text, tool results included."""
    return json.dumps(messages)


def test_a_queued_question_is_answered_on_disk_and_delivered_once_across_resumes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Run, answer on disk, resume: the answer arrives once; a second resume repeats nothing."""
    repo = _setup(tmp_path, monkeypatch)
    session_id = "queued-LIFE01"
    log = paths.state_dir(repo) / "sessions" / "runs" / session_id / "logs.jsonl"

    first = _Scripted([_ask_queued(), _finish()])
    monkeypatch.setattr(providers, "build_role_provider", _use(first))
    assert cli_main(["run", "--session-id", session_id, "task"]) == 0
    assert len(first.seen) == 2
    assert "pending" in _dump(first.seen[1])
    events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    prompts = [e for e in events if e["type"] == "question.prompt"]
    assert [(p["id"], p["wait"]) for p in prompts] == [("question-1", False)]
    assert not [e for e in events if e["type"] == "question.answer"]

    capsys.readouterr()
    assert cli_main(["answer", session_id]) == 0
    listed = capsys.readouterr().out
    assert "question-1" in listed and "Which DB?" in listed
    assert listing.finished_needs_new_work(log.parent)  # finished: a bare resume refuses
    assert cli_main(["answer", session_id, "pg"]) == 0
    assert "resume" in capsys.readouterr().out
    assert not listing.finished_needs_new_work(log.parent)  # the saved answer is the new work

    second = _Scripted([_finish()])
    monkeypatch.setattr(providers, "build_role_provider", _use(second))
    assert cli_main(["resume", session_id]) == 0
    assert len(second.seen) == 1
    opening = _dump(second.seen[0])
    assert opening.count("OPERATOR ANSWERS") == 1
    assert "question-1" in opening and "pg" in opening
    events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    answers = [e for e in events if e["type"] == "question.answer"]
    assert [(a["id"], a["answers"], a["source"]) for a in answers] == [
        ("question-1", ["pg"], "late")
    ]

    third = _Scripted([_finish()])
    monkeypatch.setattr(providers, "build_role_provider", _use(third))
    assert cli_main(["resume", session_id, "--steer", "carry on"]) == 0
    again = _dump(third.seen[0])
    assert again.count("OPERATOR ANSWERS") == 1  # the history's copy, no second delivery
    events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len([e for e in events if e["type"] == "question.answer"]) == 1
    assert "QUESTIONS PENDING" not in again
