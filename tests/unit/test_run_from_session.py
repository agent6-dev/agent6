# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 run --from` source lineage through the real CLI lifecycle."""

from __future__ import annotations

import json
import pathlib
import subprocess
from typing import Any

import pytest

from agent6 import paths
from agent6.app import role_providers, session_runtime
from agent6.providers import ProviderResponse
from agent6.ui.cli import cli_main

_CONFIG = """
[sandbox]
isolation = "none"
run_commands = "deny"

[models.default]
model = "p/m"

[providers.p]
backend = "openai"
base_url = "http://127.0.0.1:1"
api_key_env = "PROBE_KEY"

[review]
trigger = "off"
"""


class _Finisher:
    def __init__(self) -> None:
        self.messages: list[object] = []
        self.answer_ask = False

    def call(self, **kwargs: Any) -> ProviderResponse:
        self.messages.append(kwargs["messages"])
        if self.answer_ask:
            return ProviderResponse(
                text="Use the source context.",
                tool_uses=(),
                stop_reason="end_turn",
                input_tokens=1,
                output_tokens=1,
                cache_read_tokens=0,
                cache_creation_tokens=0,
            )
        tool = {
            "type": "tool_use",
            "id": "finish-1",
            "name": "finish_session",
            "input": {"summary": "seed received"},
        }
        return ProviderResponse(
            text="",
            tool_uses=(tool,),
            stop_reason="tool_use",
            input_tokens=1,
            output_tokens=1,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            raw={"content": [tool]},
        )


def _setup(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[pathlib.Path, _Finisher]:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    config_home = tmp_path / "config"
    (config_home / "agent6").mkdir(parents=True)
    (config_home / "agent6" / "config.toml").write_text(_CONFIG, encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.setenv("PROBE_KEY", "test-key")

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "tracked.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=repo, check=True)
    monkeypatch.chdir(repo)

    def no_isolation(*_args: Any, **_kwargs: Any) -> str:
        return "none"

    provider = _Finisher()

    def fake_provider(*_args: Any, **_kwargs: Any) -> _Finisher:
        return provider

    monkeypatch.setattr(session_runtime, "select_isolation", no_isolation)
    monkeypatch.setattr(role_providers, "build_role_provider", fake_provider)
    return repo, provider


def _seed_source(repo: pathlib.Path, bucket: str, session_id: str) -> None:
    source = paths.state_dir(repo) / "sessions" / bucket / session_id
    source.mkdir(parents=True)
    mode = {"asks": "ask", "plans": "plan", "runs": "run"}[bucket]
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "version": 3,
                "session_id": session_id,
                "mode": mode,
                "user_task": f"source {mode} task",
            }
        ),
        encoding="utf-8",
    )
    (source / "logs.jsonl").write_text(
        json.dumps(
            {
                "type": "session.end",
                "reason": "finish_session",
                "turns": 1,
                "all_passed": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    if bucket == "plans":
        (source / "plan.md").write_text("# Plan: source plan\n\n1. Do it.\n", encoding="utf-8")
    if bucket == "asks":
        (source / "transcript.md").write_text(
            "# agent6 ask\n\n## Question\n\nsource ask task\n\n"
            "## Answer\n\nUse the indexed conversion table.\n",
            encoding="utf-8",
        )


@pytest.mark.parametrize("bucket", ["asks", "plans", "runs"])
def test_a_seeded_runs_manifest_records_its_resolved_source(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, bucket: str
) -> None:
    repo, provider = _setup(tmp_path, monkeypatch)
    source_id = f"source-{bucket}-AAA111"
    destination_id = f"seeded-{bucket}-BBB222"
    _seed_source(repo, bucket, source_id)
    source_prefix = source_id.removesuffix("AAA111")
    args = ["run", "--session-id", destination_id, "--from", source_prefix]
    if bucket != "plans":
        args.append("continue from the source")

    assert cli_main(args) == 0

    manifest = json.loads(
        (paths.state_dir(repo) / "sessions" / "runs" / destination_id / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["source_session_id"] == source_id

    prompt = json.dumps(provider.messages)
    if bucket == "plans":
        assert "# Plan: source plan" in prompt
        # The plan rides as composed context; the recorded task is its headline.
        assert manifest["user_task"] == "Execute the prepared plan: source plan"
    else:
        assert f"source {bucket.removesuffix('s')} task" in prompt
        assert "## Outcome / key events" in prompt
        assert "session.end reason=finish_session" in prompt
        assert "## Diff" in prompt
        if bucket == "asks":
            assert "Use the indexed conversion table." in prompt


def test_a_seeded_asks_manifest_records_its_resolved_source(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, provider = _setup(tmp_path, monkeypatch)
    source_id = "source-run-AAA111"
    _seed_source(repo, "runs", source_id)
    provider.answer_ask = True

    assert cli_main(["ask", "--from", "source-run-", "what should I do next?"]) == 0

    asks = list((paths.state_dir(repo) / "sessions" / "asks").iterdir())
    assert len(asks) == 1
    manifest = json.loads((asks[0] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_session_id"] == source_id
    prompt = json.dumps(provider.messages)
    assert "source run task" in prompt
    assert "what should I do next?" in prompt


@pytest.mark.parametrize("continuation", ["resume", "fork"])
def test_journal_keeps_operator_words_before_composition(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    continuation: str,
) -> None:
    """File expansion and a literal file example never replace or clip the journal's words."""
    repo, _provider = _setup(tmp_path, monkeypatch)
    (repo / "notes.md").write_text("context from disk")
    words = "Read @notes.md and keep this literal <file>example</file>.\n" + "detail " * 800
    session_id = "whole-prompt-AAA111"
    assert cli_main(["run", "--session-id", session_id, words]) == 0
    session = paths.state_dir(repo) / "sessions" / "runs" / session_id
    rows = [json.loads(line) for line in (session / "logs.jsonl").read_text().splitlines()]
    assert next(row for row in rows if row["type"] == "session.start")["user_task"] == words
    snapshot = json.loads((session / "loop_state.json").read_text())
    assert "context from disk" in snapshot["original_task"]
    assert snapshot["operator_task"] == words
    steer = "Keep <file>this example</file> too.\n" + "follow-up " * 600
    args = [continuation, session_id, "--steer", steer]
    if continuation == "fork":
        args += ["--session-id", "whole-fork-BBB222"]
    assert cli_main(args) == 0
    if continuation == "fork":
        session = session.parent / "whole-fork-BBB222"
    rows = [json.loads(line) for line in (session / "logs.jsonl").read_text().splitlines()]
    start = next(row for row in rows if row["type"] == "session.resume")
    assert start["user_task"] == words
    assert start["steer"] == steer
    assert next(row for row in rows if row["type"] == "loop.steer.injected")["text"] == steer
    capsys.readouterr()
    assert cli_main(["sessions", "show", session.name]) == 0
    shown = capsys.readouterr().out.replace("\n" + " " * 12, "\n")
    assert steer in shown
    if continuation == "resume":
        assert words in shown


def test_parked_submission_keeps_words_before_file_expansion(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A submission parked before its first journal event retains the original prompt."""
    from agent6.sessions import lock

    repo, _provider = _setup(tmp_path, monkeypatch)
    words = "Read @tracked.txt and keep <file>example</file>.\n" + "detail " * 800
    session_id = "parked-words-AAA111"
    held = lock.acquire_repo_writer(paths.state_dir(repo), repo, "busy-run")
    assert held is not None
    try:
        assert cli_main(["run", "--session-id", session_id, words]) == 2
    finally:
        lock.release_single_writer(held)
    assert cli_main(["resume", session_id]) == 0
    session = paths.state_dir(repo) / "sessions" / "runs" / session_id
    rows = [json.loads(line) for line in (session / "logs.jsonl").read_text().splitlines()]
    assert next(row for row in rows if row["type"] == "session.start")["user_task"] == words
