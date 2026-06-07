# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the post-run notify hook."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent6.cli.run import _fire_notify_hook  # pyright: ignore[reportPrivateUsage]
from agent6.config import NotifyConfig


def test_notify_noop_when_unconfigured(tmp_path: Path) -> None:
    """An empty `on_complete` tuple is a no-op (no subprocess, no error)."""
    notify = NotifyConfig()
    # Should return without raising, without doing anything.
    _fire_notify_hook(
        notify,
        run_id="abcdef0123456789",
        run_dir=tmp_path,
        ok=True,
        reason="finish_run",
    )


def test_notify_fires_with_env(tmp_path: Path) -> None:
    """When configured, the hook runs the argv with AGENT6_* env vars."""
    out = tmp_path / "notify-out.json"
    argv = (
        "python3",
        "-c",
        "import json,os,sys; "
        "json.dump({"
        "'id': os.environ['AGENT6_RUN_ID'], "
        "'ok': os.environ['AGENT6_RUN_OK'], "
        "'reason': os.environ['AGENT6_RUN_REASON'], "
        "'dir': os.environ['AGENT6_RUN_DIR']"
        "}, open(sys.argv[1], 'w'))",
        str(out),
    )
    notify = NotifyConfig(on_complete=argv, timeout_s=10.0)
    _fire_notify_hook(
        notify,
        run_id="run-xyz",
        run_dir=tmp_path,
        ok=True,
        reason="finish_run",
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload == {
        "id": "run-xyz",
        "ok": "1",
        "reason": "finish_run",
        "dir": str(tmp_path),
    }


def test_notify_failure_does_not_raise(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A failing argv (nonexistent binary) logs but does not raise."""
    notify = NotifyConfig(on_complete=("/nonexistent/agent6-notify-binary",), timeout_s=5.0)
    _fire_notify_hook(
        notify,
        run_id="run-xyz",
        run_dir=tmp_path,
        ok=False,
        reason="budget_exhausted",
    )
    captured = capsys.readouterr()
    assert "notify.on_complete failed" in captured.err


def test_notify_ok_zero_when_failed(tmp_path: Path) -> None:
    """ok=False sets AGENT6_RUN_OK=0."""
    out = tmp_path / "ok.txt"
    argv = (
        "sh",
        "-c",
        f'printf "%s" "$AGENT6_RUN_OK" > {out}',
    )
    notify = NotifyConfig(on_complete=argv, timeout_s=5.0)
    _fire_notify_hook(
        notify,
        run_id="r",
        run_dir=tmp_path,
        ok=False,
        reason="provider_error",
    )
    assert out.read_text(encoding="utf-8") == "0"


def test_notify_in_config_loads(tmp_path: Path) -> None:
    """[notify] section round-trips through the config loader."""
    from agent6.config import load_config

    body = """
[agent6]
config_version = 1

[providers.anthropic]
kind = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
prompt_caching = true

[models.worker]
provider = "anthropic"
model = "claude-sonnet-4-5"

[models.reviewer]
provider = "anthropic"
model = "claude-opus-4-5"

[sandbox]
profile = "auto"
network = "provider_only"
run_commands = "ask"
protect_git = true
protect_agent6 = true

[git]
allow_push = false
allow_force = false
allow_history_rewrite = false

[workflow]
verify_command = ["true"]

[budget]
max_input_tokens = 100000
max_output_tokens = 10000

[notify]
on_complete = ["notify-send", "agent6 done"]
timeout_s = 12.5
"""
    cfg_path = tmp_path / "agent6.toml"
    cfg_path.write_text(body, encoding="utf-8")
    cfg = load_config(cfg_path)
    assert cfg.notify.on_complete == ("notify-send", "agent6 done")
    assert cfg.notify.timeout_s == 12.5
