# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The one operator notify hook: `[notify].on_event` for a session's end and a machine's events."""

from __future__ import annotations

import json
import pathlib

import pytest

from agent6.app import notify_hook, reporter
from agent6.app.machine import build_machine_notify_hook
from agent6.config import Config, NotifyConfig, load_config


def _end(notify: NotifyConfig, tmp_path: pathlib.Path, **over: object) -> None:
    fields: dict[str, object] = dict(
        session_id="run-xyz",
        session_dir=tmp_path,
        ok=True,
        reason="finish_session",
        verification="passed",
        note=reporter.STDIO_REPORTER.note,
    )
    fields.update(over)
    notify_hook.session_end(notify, **fields)  # pyright: ignore[reportArgumentType]


def test_notify_noop_when_unconfigured(tmp_path: pathlib.Path) -> None:
    """An empty `on_event` is a no-op: no subprocess, no error."""
    _end(NotifyConfig(), tmp_path)


def test_notify_failure_does_not_raise(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failing argv (nonexistent binary) is noted, never raised."""
    _end(NotifyConfig(on_event=("/nonexistent/agent6-notify-binary",), timeout_s=5.0), tmp_path)
    assert "notify.on_event failed" in capsys.readouterr().err


def test_both_callers_run_the_same_hook(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two runners for one job drifted in both directions.

    The run hook swallowed a non-zero exit (a hook that fails silently stops notifying with nobody
    the wiser), and the machine hook let the hook's stdout into the parent's, under `agent6 acp`
    that is the JSON-RPC stream.
    """
    argv = ("sh", "-c", "echo HOOK_STDOUT; exit 3")
    _end(NotifyConfig(on_event=argv, timeout_s=5.0), tmp_path)
    cfg = Config.model_validate({"notify": {"on_event": list(argv)}})
    fire = build_machine_notify_hook(cfg, "machine-1", tmp_path)
    assert fire is not None
    fire("end", "done", "", "ok")

    captured = capsys.readouterr()
    assert captured.err.count("notify.on_event exited 3") == 2
    assert "HOOK_STDOUT" not in captured.out


def test_notify_ok_zero_when_failed(tmp_path: pathlib.Path) -> None:
    """ok=False sets AGENT6_SESSION_OK=0 and the level `failed`."""
    out = tmp_path / "ok.txt"
    argv = ("sh", "-c", f'printf "%s %s" "$AGENT6_SESSION_OK" "$AGENT6_NOTIFY_LEVEL" > {out}')
    _end(NotifyConfig(on_event=argv, timeout_s=5.0), tmp_path, ok=False, reason="provider_error")
    assert out.read_text(encoding="utf-8") == "0 failed"


_CFG_BODY = """
[agent6]
config_version = 1

[providers.anthropic]
backend = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"

[models.worker]
model = "anthropic/claude-sonnet-4-5"

[models.reviewer]
model = "anthropic/claude-opus-4-5"

[harness]
verify_command = ["true"]

[notify]
on_event = ["python3", "-c", "PLACEHOLDER"]
timeout_s = 10.0
"""

_DUMP_AGENT6_VARS = (
    "import json,os,sys; "
    "json.dump({k: v for k, v in os.environ.items() if k.startswith('AGENT6_')},"
    " open(sys.argv[1], 'w'))"
)


def _config(tmp_path: pathlib.Path, script: str, out: pathlib.Path) -> Config:
    body = _CFG_BODY.replace('"PLACEHOLDER"', f'"{script}", "{out}"')
    cfg_path = tmp_path / "agent6.toml"
    cfg_path.write_text(body, encoding="utf-8")
    return load_config(cfg_path)


def test_machine_notify_hook_fires_with_env(tmp_path: pathlib.Path) -> None:
    """A machine's notify and end carry the event, message and level plus the machine's facts."""
    out = tmp_path / "machine-notify.json"
    cfg = _config(tmp_path, _DUMP_AGENT6_VARS, out)
    hook = build_machine_notify_hook(cfg, "mymachine", tmp_path / "inst")
    assert hook is not None
    hook("notify", "poll", "attention needed", "warn")
    assert json.loads(out.read_text(encoding="utf-8")) == {
        "AGENT6_NOTIFY_EVENT": "machine.notify",
        "AGENT6_NOTIFY_MESSAGE": "attention needed",
        "AGENT6_NOTIFY_LEVEL": "warn",
        "AGENT6_MACHINE_ID": "mymachine",
        "AGENT6_MACHINE_DIR": str(tmp_path / "inst"),
        "AGENT6_MACHINE_STATE": "poll",
    }
    hook("end", "done", "finished", "ok")
    ended = json.loads(out.read_text(encoding="utf-8"))
    assert (ended["AGENT6_NOTIFY_EVENT"], ended["AGENT6_NOTIFY_LEVEL"]) == ("machine.end", "ok")


def test_machine_notify_hook_nonzero_exit_is_reported(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """check=False keeps the hook non-fatal, but a nonzero exit was discarded entirely.

    Notifications silently stopped arriving. The exit is named on stderr.
    """
    cfg = _config(tmp_path, "raise SystemExit(3)", tmp_path / "unused")
    hook = build_machine_notify_hook(cfg, "m", tmp_path / "inst")
    assert hook is not None
    hook("notify", "poll", "msg", "warn")
    assert "notify.on_event exited 3" in capsys.readouterr().err


def test_machine_notify_hook_none_when_unconfigured(tmp_path: pathlib.Path) -> None:
    body = _CFG_BODY.replace(
        '\n[notify]\non_event = ["python3", "-c", "PLACEHOLDER"]\ntimeout_s = 10.0\n', ""
    )
    cfg_path = tmp_path / "agent6.toml"
    cfg_path.write_text(body, encoding="utf-8")
    assert build_machine_notify_hook(load_config(cfg_path), "m", tmp_path) is None


def test_notify_in_config_loads(tmp_path: pathlib.Path) -> None:
    """The `[notify]` table round-trips through the config loader."""
    cfg = _config(tmp_path, "pass", tmp_path / "unused")
    assert cfg.notify.on_event == ("python3", "-c", "pass", str(tmp_path / "unused"))
    assert cfg.notify.timeout_s == 10.0


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"notify": {"on_complete": ["x"]}}, "[notify] on_complete was renamed: on_event = "),
        (
            {"machine": {"notify": {"on_event": ["x"]}}},
            "[machine.notify] moved: set [notify] on_event",
        ),
    ],
)
def test_the_old_hook_keys_are_refused_with_the_replacement(
    raw: dict[str, object], message: str
) -> None:
    """`[notify].on_complete` and `[machine.notify]` name their replacement, not an unknown key."""
    with pytest.raises(ValueError, match=r".*"):
        try:
            Config.model_validate(raw)
        except ValueError as exc:
            assert message in str(exc)
            raise


def test_a_prompt_opening_fires_session_waiting(tmp_path: pathlib.Path) -> None:
    """The listener fires `session.waiting` with the prompt's first line; other rows pass."""
    out = tmp_path / "waiting.json"
    listen = notify_hook.attention(
        NotifyConfig(on_event=("python3", "-c", _DUMP_AGENT6_VARS, str(out))),
        session_id="r1",
        session_dir=tmp_path,
        note=reporter.STDIO_REPORTER.note,
    )
    listen({"type": "model.call", "role": "worker"})
    assert not out.exists()
    listen({"type": "approval.prompt", "id": "approval-1", "prompt": "run `rm -rf build`?\nmore"})
    env = json.loads(out.read_text(encoding="utf-8"))
    assert env == {
        "AGENT6_NOTIFY_EVENT": "session.waiting",
        "AGENT6_NOTIFY_MESSAGE": "run `rm -rf build`?",
        "AGENT6_NOTIFY_LEVEL": "info",
        "AGENT6_SESSION_ID": "r1",
        "AGENT6_SESSION_DIR": str(tmp_path),
    }
    listen(
        {
            "type": "question.prompt",
            "id": "question-1",
            "questions": [{"question": "Which port?", "options": []}],
            "wait": False,
        }
    )
    env = json.loads(out.read_text(encoding="utf-8"))
    assert env["AGENT6_NOTIFY_MESSAGE"] == "queued question: Which port?"


def test_notify_hook_env_carries_no_secrets(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The notify hook's environment carries no secrets.

    The env is the minimal hook_env base, the AGENT6_* set docs/security.md promises; the operator's
    whole environment carries the provider bearer tokens `[providers.*].api_key_env` names, and a
    hook that logs or forwards its environment would carry a key with it.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-super-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-secret")
    out = tmp_path / "env.json"
    argv = (
        "python3",
        "-c",
        "import json,os,sys; json.dump({"
        "'key': os.environ.get('ANTHROPIC_API_KEY'), "
        "'or_key': os.environ.get('OPENROUTER_API_KEY'), "
        "'path': bool(os.environ.get('PATH')), "
        "'home': bool(os.environ.get('HOME')), "
        "'id': os.environ['AGENT6_SESSION_ID']"
        "}, open(sys.argv[1], 'w'))",
        str(out),
    )
    _end(NotifyConfig(on_event=argv, timeout_s=10.0), tmp_path, session_id="r1")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload == {"key": None, "or_key": None, "path": True, "home": True, "id": "r1"}


def test_machine_notify_hook_env_carries_no_secrets(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The machine hook goes through the same hook_env owner.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-super-secret")
    out = tmp_path / "env2.json"
    script = (
        "import json,os,sys; json.dump({"
        "'key': os.environ.get('ANTHROPIC_API_KEY'), "
        "'path': bool(os.environ.get('PATH')), "
        "'id': os.environ['AGENT6_MACHINE_ID']"
        "}, open(sys.argv[1], 'w'))"
    )
    hook = build_machine_notify_hook(_config(tmp_path, script, out), "m1", tmp_path / "inst")
    assert hook is not None
    hook("notify", "poll", "msg", "info")
    assert json.loads(out.read_text(encoding="utf-8")) == {"key": None, "path": True, "id": "m1"}


def test_hook_env_separates_deliberate_from_verified(tmp_path: pathlib.Path) -> None:
    """The hook env separates deliberate from verified.

    AGENT6_SESSION_OK says the agent stopped deliberately; AGENT6_SESSION_VERIFIED says what the
    gate said. A finish over a red verify is OK=1 VERIFIED=failed: a hook that wants "green" reads
    the second, because the first is true for a finish the verify never passed.
    """
    script = tmp_path / "hook.sh"
    out = tmp_path / "env.txt"
    script.write_text(
        f'#!/bin/sh\necho "$AGENT6_SESSION_OK $AGENT6_SESSION_VERIFIED" > {out}\n', encoding="utf-8"
    )
    script.chmod(0o755)
    _end(NotifyConfig(on_event=(str(script),)), tmp_path, session_id="r1", verification="failed")
    assert out.read_text(encoding="utf-8").strip() == "1 failed"


def test_the_hook_env_names_the_session_not_the_run(tmp_path: pathlib.Path) -> None:
    """The hook env names the session, not the run.

    `run` is the verb for the agentic coding loop; the session is the thing a hook is told about,
    and it may be a run, a plan, an ask or a machine.
    """
    out = tmp_path / "env.json"
    argv = ("python3", "-c", _DUMP_AGENT6_VARS, str(out))
    _end(NotifyConfig(on_event=argv), tmp_path, session_id="brave-oak-AAAAAA")
    env = json.loads(out.read_text(encoding="utf-8"))
    assert env == {
        "AGENT6_NOTIFY_EVENT": "session.stop",
        "AGENT6_NOTIFY_MESSAGE": "finish_session",
        "AGENT6_NOTIFY_LEVEL": "ok",
        "AGENT6_SESSION_ID": "brave-oak-AAAAAA",
        "AGENT6_SESSION_DIR": str(tmp_path),
        "AGENT6_SESSION_OK": "1",
        "AGENT6_SESSION_REASON": "finish_session",
        "AGENT6_SESSION_VERIFIED": "passed",
    }
