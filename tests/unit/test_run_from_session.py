# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 run --from` source lineage through the real CLI lifecycle."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

import agent6.app._session as session_mod
import agent6.app.run as run_mod
from agent6.paths import state_dir
from agent6.providers import ProviderResponse
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


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, _Finisher]:
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

    monkeypatch.setattr(run_mod, "select_isolation", no_isolation)
    monkeypatch.setattr(session_mod, "build_role_provider", fake_provider)
    return repo, provider


def _seed_source(repo: Path, bucket: str, session_id: str) -> None:
    source = state_dir(repo) / "sessions" / bucket / session_id
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
                "iterations": 1,
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bucket: str
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
        (state_dir(repo) / "sessions" / "runs" / destination_id / "manifest.json").read_text(
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, provider = _setup(tmp_path, monkeypatch)
    source_id = "source-run-AAA111"
    _seed_source(repo, "runs", source_id)
    provider.answer_ask = True

    assert cli_main(["ask", "--from", "source-run-", "what should I do next?"]) == 0

    asks = list((state_dir(repo) / "sessions" / "asks").iterdir())
    assert len(asks) == 1
    manifest = json.loads((asks[0] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_session_id"] == source_id
    prompt = json.dumps(provider.messages)
    assert "source run task" in prompt
    assert "what should I do next?" in prompt
