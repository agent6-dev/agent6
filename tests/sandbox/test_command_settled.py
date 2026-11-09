from __future__ import annotations

import json
import pathlib
import subprocess
from typing import Any
from unittest import mock

import pytest

from agent6 import paths, providers
from agent6.app import role_providers
from agent6.config import Config
from agent6.harness import guards, loop, loop_conversation, loop_state, run_chain
from agent6.tools import dispatch, results
from agent6.ui.cli import cli_main
from tests.unit.turn_context import turn_context


def _dispatcher(tmp_path: pathlib.Path, checkin: float) -> dispatch.ToolDispatcher:
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    session_dir = tmp_path / "session"
    session_dir.mkdir(exist_ok=True)
    cfg = Config.model_validate(
        {
            "sandbox": {"run_commands": "allow"},
            "harness": {"command_checkin_s": checkin},
        }
    )
    return dispatch.ToolDispatcher(
        root=root,
        config=cfg,
        isolation="none",
        session_dir=session_dir,
        use_jail_session=True,
    )


@pytest.mark.parametrize(
    ("action", "neutral"),
    [
        ("completed", False),
        ("failed", False),
        ("roster", False),
        ("stopped", False),
        ("exited_during_read", False),
        ("live_poll", True),
        ("handback", True),
        ("detached", True),
    ],
)
def test_settled_accounting_follows_the_actual_command_result(
    tmp_path: pathlib.Path, action: str, neutral: bool
) -> None:
    """A real live sibling does not shelter unrelated commands or ended reads from the guard."""
    d = _dispatcher(tmp_path, checkin=0.1 if action == "handback" else 30.0)
    wf = loop.Harness(
        chain=run_chain.RunChain(tmp_path / "repo"),
        config=Config(),
        provider=mock.MagicMock(),
        dispatcher=d,
        logger=lambda _line: None,
    )
    state = loop_state.LoopState(original_task="t", tool_calls=0)
    state.verify.note_pass()
    state.settled = guards.SettledGuard(tree="tree", idle=1)
    turn = loop_state.TurnState(
        number=3, resp=mock.MagicMock(), assistant=loop_conversation.AssistantMessage((), ())
    )
    try:
        sibling = d.dispatch("run_command", {"argv": ["/bin/sleep", "60"], "background": True})
        assert isinstance(sibling, results.ExecResult) and sibling.background_id
        name = "run_command"
        raw: dict[str, Any] = {"argv": ["/bin/true"]}
        if action == "failed":
            raw = {"argv": ["/bin/false"]}
        elif action == "roster":
            name, raw = "read_background", {}
        elif action == "live_poll":
            name, raw = "read_background", {"id": sibling.background_id, "wait_s": 0}
        elif action in ("handback", "detached"):
            raw = {"argv": ["/bin/sleep", "60"], "background": action == "detached"}
        elif action in ("stopped", "exited_during_read"):
            selected = d.dispatch("run_command", {"argv": ["/bin/sleep", "1"], "background": True})
            assert isinstance(selected, results.ExecResult)
            if action == "stopped":
                d.dispatch("stop_background", {"id": selected.background_id})
            name, raw = "read_background", {"id": selected.background_id, "wait_s": 5}
        outcome = d.dispatch(name, raw)
        wf._note_tool_effects(state, turn, name, outcome, raw)  # pyright: ignore[reportPrivateUsage]
        guards.verify_settled(turn, state, turn_context(tree_sha=lambda: "tree"))
        assert state.settled.idle == (1 if neutral else 2)
    finally:
        d.close()
    roster = d.dispatch("read_background", {}).to_wire()["shells"]
    assert roster and all("running" not in line for line in roster)


class _Scripted:
    def __init__(self, poll: bool) -> None:
        self.poll = poll
        self.calls = 0

    def call(self, **_kwargs: Any) -> providers.ProviderResponse:
        self.calls += 1
        assert self.calls <= 8, "the CLI requested a turn after the scripted finish"
        if self.calls == 1:
            name, raw = "apply_edit", {"path": "new.txt", "new_string": "work\n"}
        elif self.calls == 2:
            # Teardown must end this job, not a short sleep expiring under load.
            name, raw = "run_command", {"argv": ["/bin/sleep", "3600"], "background": True}
        elif self.calls == 8:
            name, raw = "finish_session", {"summary": "done"}
        elif self.poll:
            name, raw = "read_background", {"id": "bg1", "wait_s": 0}
        else:
            argv = ["git", "status", "--porcelain"] if self.calls % 2 else ["git", "diff"]
            name, raw = "run_command", {"argv": argv}
        tools = ({"type": "tool_use", "id": str(self.calls), "name": name, "input": raw},)
        return providers.ProviderResponse(
            text="",
            tool_uses=tools,
            stop_reason="tool_use",
            input_tokens=10,
            output_tokens=5,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            raw={"content": list(tools)},
        )


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.mark.parametrize(
    ("poll", "turns", "reason"),
    [(False, 5, "settled"), (True, 8, "finish_session")],
    ids=["unrelated-commands", "selected-live-polls"],
)
def test_cli_settles_only_unrelated_commands_beside_a_live_job(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    poll: bool,
    turns: int,
    reason: str,
) -> None:
    """A live job cannot shelter Git commands, but selected zero-wait polls stay neutral."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "seed")
    base = _git(repo, "rev-parse", "HEAD")
    cfg = tmp_path / "config" / "agent6"
    cfg.mkdir(parents=True)
    (cfg / "config.toml").write_text(
        '[sandbox]\nisolation = "none"\nrun_commands = "allow"\n'
        '[models.default]\nmodel = "p/m"\n[models.worker]\nmodel = "p/m"\n'
        '[providers.p]\nbackend = "openai"\nbase_url = "http://127.0.0.1:1"\n'
        'api_key_env = "PROBE_KEY"\n[review]\ntrigger = "off"\n'
        '[harness]\nverify_command = ["true"]\nverify_at_start = false\nverify_when = "never"\n'
        'settled_stop_after = 3\n[git]\ncontrol = "model"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg.parent))
    monkeypatch.setenv("PROBE_KEY", "test-key")
    monkeypatch.chdir(repo)
    provider = _Scripted(poll)

    def provider_factory(*_args: Any, **_kwargs: Any) -> _Scripted:
        return provider

    monkeypatch.setattr(role_providers, "build_role_provider", provider_factory)
    assert cli_main(["run", "--session-id", "settled", "write work, then inspect it"]) == 0
    out = capsys.readouterr().out
    session_dir = paths.state_dir(repo) / "sessions" / "runs" / "settled"
    events = [json.loads(line) for line in (session_dir / "logs.jsonl").read_text().splitlines()]
    stop = next(event for event in events if event["type"] == "session.stop")
    job = json.loads((session_dir / "shells" / "bg1" / "result.json").read_text())
    assert job["running"] is False and not job["error"]
    assert isinstance(job["returncode"], int) and job["returncode"] != 0
    assert provider.calls == turns
    assert stop["turns"] == turns and stop["reason"] == reason
    assert stop["verification"] == "unverified"
    assert "committ" not in stop["summary"]
    assert not any(event["type"] == "verify.end" for event in events)
    tool_results = [event for event in events if event["type"] == "tool.result"]
    assert len(tool_results) == turns and all(event["ok"] for event in tool_results)
    assert tool_results[1]["summary"].startswith("still running as bg1 ")
    if poll:
        assert [event["name"] for event in tool_results[2:]] == ["read_background"] * 5 + [
            "finish_session"
        ]
        assert all(event["summary"].startswith("[bg1] running:") for event in tool_results[2:-1])
    else:
        assert [event["name"] for event in tool_results[2:]] == ["run_command"] * 3
        assert all(event["summary"].startswith("exit=0 ") for event in tool_results[2:])
        assert "the worker settled; the verify never passed" in out
    assert (repo / "new.txt").read_text() == "work\n"
    assert _git(repo, "rev-parse", "HEAD") == base
    assert _git(repo, "status", "--porcelain") == "?? new.txt"
    assert "nothing verified the final tree" in out
    assert 'resume and run the gate:  agent6 resume settled --steer "verify"' in out
    assert "git was model-controlled" in out and "new untracked files: 1" in out
    assert "changes merged" not in out and "changes are on" not in out
