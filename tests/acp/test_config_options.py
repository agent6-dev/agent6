# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""An editor picks the model and the effort of a session's next prompts."""

from __future__ import annotations

import io
import json
import os
import pathlib
import subprocess
from typing import Any

import pytest

from agent6 import paths
from agent6.app import execution, run
from agent6.app import frontend as app_frontend
from agent6.app.session import runtime as session_runtime
from agent6.app.session import setup as session_setup
from agent6.config import Config
from agent6.harness import loop_snapshot
from agent6.models import cache
from agent6.sessions import manifest as sessions_manifest
from agent6.ui.acp import runner
from agent6.ui.acp import server as acp_server
from tests.acp import schema_check

_CONFIG = """
[agent6]
config_version = 1

[providers.anthropic]
backend = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"

[models.default]
model = "anthropic/default-model"

[models.worker]
model = "anthropic/worker-model"
effort = "low"
"""

_EFFORTS = ["inherit", "off", "low", "medium", "high", "xhigh", "max"]


class _Editor:
    """An ACP connection whose frames are read back from its output."""

    def __init__(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.config = tmp_path / "cfg.toml"
        self.config.write_text(_CONFIG, encoding="utf-8")
        # What each provider's cached listing offers, beside the roles' own models.
        self.cached = {"anthropic": ["other-model"]}

        def _cached(name: str) -> list[str]:
            return list(self.cached.get(name, ()))

        monkeypatch.setattr(cache, "cached_models", _cached)
        self._out = io.BytesIO()
        self.server = acp_server.ACPServer(stdin=io.BytesIO(), stdout=self._out)
        self.server.client_capabilities = app_frontend.FrontendCapabilities(can_ask=True)
        self.bridge = runner.RunBridge(server=self.server, config_paths=(self.config,))
        self.sessions = self.bridge.sessions()
        self.server.sessions = self.sessions

    def frames(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self._out.getvalue().splitlines()]

    def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send one request and return its reply frame."""
        line = json.dumps({"jsonrpc": "2.0", "id": 7, "method": method, "params": params})
        self.server._handle(line.encode())  # pyright: ignore[reportPrivateUsage]
        return self.frames()[-1]

    def new(self, cwd: pathlib.Path) -> tuple[str, list[dict[str, Any]]]:
        result = self.call("session/new", {"cwd": str(cwd)})["result"]
        assert not schema_check.errors("NewSessionResponse", result), result
        return result["sessionId"], result["configOptions"]

    def set(self, session_id: str, config_id: str, value: str) -> dict[str, Any]:
        params = {"sessionId": session_id, "configId": config_id, "value": value}
        assert not schema_check.errors("SetSessionConfigOptionRequest", params)
        return self.call("session/set_config_option", params)


def _repo(path: pathlib.Path) -> pathlib.Path:
    path.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    (path / "seed.txt").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "add", "seed.txt"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)
    return path


def _unconfined(*_a: object, **_k: object) -> str:
    return "none"


def _nothing(*_a: object, **_k: object) -> None:
    return None


def _current(options: list[dict[str, Any]]) -> dict[str, str]:
    return {option["id"]: option["currentValue"] for option in options}


def _recorded_run(repo: pathlib.Path, session_id: str, **manifest: Any) -> pathlib.Path:
    """Write a run an editor can resume: a manifest and a resume snapshot."""
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    session_dir = paths.state_dir(repo) / "sessions" / "runs" / session_id
    session_dir.mkdir(parents=True)
    record = {
        "version": 0,
        "session_id": session_id,
        "mode": "run",
        "user_task": "t",
        "base_sha": head,
        "base_branch": "main",
        "run_branch": f"agent6/{session_id}",
        **manifest,
    }
    (session_dir / "manifest.json").write_text(json.dumps(record), encoding="utf-8")
    snapshot = {
        "version": loop_snapshot.SNAPSHOT_VERSION,
        "system": "s",
        "messages": [],
        "tool_calls": 0,
        "next_turn": 1,
        "root_task_id": None,
        "original_task": "t",
        "verify_command": [],
    }
    (session_dir / "loop_state.json").write_text(json.dumps(snapshot), encoding="utf-8")
    return session_dir


def test_a_new_session_offers_the_model_and_the_effort_as_complete_lists(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The model list is the one every front-end's picker offers; the current route is in it."""
    editor = _Editor(tmp_path, monkeypatch)

    _session_id, options = editor.new(_repo(tmp_path / "repo"))

    model, effort = options
    assert (model["id"], model["category"], model["type"]) == ("model", "model", "select")
    assert model["currentValue"] == "anthropic/worker-model"
    assert [choice["value"] for choice in model["options"]] == [
        "anthropic/default-model",
        "anthropic/other-model",
        "anthropic/worker-model",
    ]
    assert (effort["id"], effort["category"], effort["type"]) == (
        "effort",
        "thought_level",
        "select",
    )
    assert effort["currentValue"] == "inherit"
    assert [choice["value"] for choice in effort["options"]] == _EFFORTS
    assert "modes" not in editor.frames()[-1]["result"]  # an ACP session only opens runs


def test_a_set_option_answers_with_the_complete_list(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    editor = _Editor(tmp_path, monkeypatch)
    session_id, before = editor.new(_repo(tmp_path / "repo"))

    result = editor.set(session_id, "effort", "high")["result"]

    assert not schema_check.errors("SetSessionConfigOptionResponse", result), result
    assert _current(result["configOptions"]) == {
        "model": "anthropic/worker-model",
        "effort": "high",
    }
    assert [o["options"] for o in result["configOptions"]] == [o["options"] for o in before]
    picked = editor.set(session_id, "model", "anthropic/other-model")["result"]
    assert _current(picked["configOptions"]) == {
        "model": "anthropic/other-model",
        "effort": "high",
    }
    back = editor.set(session_id, "effort", "inherit")["result"]
    assert _current(back["configOptions"])["effort"] == "inherit"


@pytest.mark.parametrize(
    ("config_id", "value", "why"),
    [
        ("mode", "run", "unknown config option 'mode'; this session offers: model, effort"),
        ("model", "anthropic/no-such", "is not one of the 3 values 'model' offers"),
        ("effort", "extreme", "is not one of the 7 values 'effort' offers"),
    ],
)
def test_an_unknown_option_or_value_is_refused_with_the_reason(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, config_id: str, value: str, why: str
) -> None:
    editor = _Editor(tmp_path, monkeypatch)
    session_id, _options = editor.new(_repo(tmp_path / "repo"))

    error = editor.set(session_id, config_id, value)["error"]

    assert error["code"] == -32602 and why in error["message"]
    session = editor.sessions.get({"sessionId": session_id})
    assert (session.model, session.effort) == ("", None)


def test_a_set_while_a_prompt_runs_is_refused(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pick never changes a running turn: it is refused, and the reply says when it applies."""
    editor = _Editor(tmp_path, monkeypatch)
    session_id, _options = editor.new(_repo(tmp_path / "repo"))
    session = editor.sessions.get({"sessionId": session_id})
    session.execution_live = True

    error = editor.set(session_id, "effort", "high")["error"]

    assert error["code"] == -32602
    assert "a prompt is running on this session" in error["message"]
    assert "applies from the next prompt" in error["message"]
    assert session.effort is None


def test_two_sessions_do_not_share_a_pick(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    editor = _Editor(tmp_path, monkeypatch)
    repo = _repo(tmp_path / "repo")
    first, _ = editor.new(repo)
    second, _ = editor.new(repo)

    editor.set(first, "model", "anthropic/other-model")
    editor.set(first, "effort", "max")

    untouched = editor.set(second, "effort", "inherit")["result"]["configOptions"]
    assert _current(untouched) == {"model": "anthropic/worker-model", "effort": "inherit"}
    assert _current(editor.set(first, "effort", "max")["result"]["configOptions"]) == {
        "model": "anthropic/other-model",
        "effort": "max",
    }


def _worker(cfg: Config) -> tuple[str, str | None]:
    """Return the route and effort the config builds the worker's provider with."""
    role = cfg.models.resolve("worker")
    assert role is not None
    return f"{role.provider}/{role.model}", role.effort


def test_a_new_run_starts_under_the_picked_model_and_effort(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The config handed to the run is the one its worker is built from."""
    editor = _Editor(tmp_path, monkeypatch)
    repo = _repo(tmp_path / "repo")
    started: list[tuple[tuple[str, str | None], object]] = []

    def _run_task(cfg: Config, _task: str, **kw: Any) -> int:
        started.append((_worker(cfg), kw.get("model")))
        return 0

    monkeypatch.setattr(run, "run_task", _run_task)
    plain, _ = editor.new(repo)
    picked, _ = editor.new(repo)
    editor.set(picked, "model", "anthropic/other-model")
    editor.set(picked, "effort", "high")

    for session_id in (plain, picked):
        session = editor.sessions.get({"sessionId": session_id})
        assert editor.bridge.run(session, "do the thing") == "end_turn"

    assert started == [
        (("anthropic/worker-model", "low"), None),
        (("anthropic/other-model", "high"), "anthropic/other-model"),
    ]


def test_a_resumed_run_is_built_with_the_picked_model_and_effort(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pick reaches the config the resumed execution runs, and no config file is written."""
    editor = _Editor(tmp_path, monkeypatch)
    repo = _repo(tmp_path / "repo")
    session_dir = _recorded_run(repo, "run-AAAA11")
    built: list[tuple[str, str | None]] = []

    def _execution(cfg: Config, *_a: object, **_k: object) -> execution.ExecutionEnd:
        built.append(_worker(cfg))
        return execution.ExecutionEnd(0)

    monkeypatch.setattr(execution, "run_execution", _execution)
    monkeypatch.setattr(session_runtime, "select_isolation", _unconfined)
    monkeypatch.setattr(session_setup, "check_provider_keys", _nothing)
    params = {"sessionId": "run-AAAA11", "cwd": str(repo)}
    opened = editor.call("session/resume", params)["result"]
    assert not schema_check.errors("ResumeSessionResponse", opened), opened
    assert _current(opened["configOptions"]) == {
        "model": "anthropic/worker-model",
        "effort": "inherit",
    }
    session = editor.sessions.get(params)

    assert editor.bridge.run(session, "carry on") == "end_turn"
    editor.set("run-AAAA11", "model", "anthropic/other-model")
    editor.set("run-AAAA11", "effort", "high")
    assert editor.bridge.run(session, "and again") == "end_turn"

    assert built == [("anthropic/worker-model", "low"), ("anthropic/other-model", "high")]
    # The record every surface reads names the pick as the execution's own, not a layer's.
    driver = sessions_manifest.read_manifest(session_dir).models.driver
    assert driver is not None
    assert (driver.model, driver.effort, driver.effort_source) == ("other-model", "high", "flag")
    assert editor.config.read_text(encoding="utf-8") == _CONFIG
    assert not list(pathlib.Path(os.environ["XDG_CONFIG_HOME"]).rglob("*.toml"))
    assert not list(paths.state_dir(repo).glob("*.toml"))


def test_a_loaded_session_shows_the_route_its_run_recorded(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resume replays the recorded route, so that is the one the picker shows."""
    editor = _Editor(tmp_path, monkeypatch)
    repo = _repo(tmp_path / "repo")
    recorded = {"driver": {"provider": "anthropic", "model": "recorded-model"}}
    _recorded_run(repo, "run-BBBB22", models=recorded)

    loaded = editor.call("session/load", {"sessionId": "run-BBBB22", "cwd": str(repo)})["result"]

    assert not schema_check.errors("LoadSessionResponse", loaded), loaded
    model = loaded["configOptions"][0]
    assert model["currentValue"] == "anthropic/recorded-model"
    assert "anthropic/recorded-model" in [choice["value"] for choice in model["options"]]


def test_an_option_agent6_changed_is_told_to_the_editor(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run's provider preflight caches the listing, so the model list grows without a set."""
    editor = _Editor(tmp_path, monkeypatch)
    session_id, before = editor.new(_repo(tmp_path / "repo"))

    def _run_task(*_a: object, **_k: object) -> int:
        editor.cached["anthropic"] = ["other-model", "newly-listed-model"]
        return 0

    monkeypatch.setattr(run, "run_task", _run_task)
    session = editor.sessions.get({"sessionId": session_id})

    assert editor.bridge.run(session, "do the thing") == "end_turn"
    assert editor.bridge.run(session, "once more") == "end_turn"

    told = [
        frame["params"]
        for frame in editor.frames()
        if frame.get("method") == "session/update"
        and frame["params"]["update"]["sessionUpdate"] == "config_option_update"
    ]
    assert len(told) == 1, "an unchanged list is not sent again"
    assert not schema_check.errors("SessionNotification", told[0]), told[0]
    options = told[0]["update"]["configOptions"]
    assert [o["id"] for o in options] == [o["id"] for o in before]
    assert "anthropic/newly-listed-model" in [c["value"] for c in options[0]["options"]]
    assert editor.set(session_id, "effort", "low")["result"]["configOptions"][0] == options[0]
