# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 exec --new`: the argument forms, the shell choice, the refusals and the policy built."""

from __future__ import annotations

import dataclasses
import os
import pathlib
import signal
from typing import Any

import pytest

from agent6 import child_env, kinds
from agent6.app import _setup
from agent6.config import Config
from agent6.sandbox import detect, jail, tool_paths
from agent6.tools import policy as tools_policy
from agent6.ui import cli
from agent6.ui.cli import exec_cmds


@pytest.fixture
def dispatched(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Record what the dispatcher hands `exec_new` and `exec_in_session`."""
    calls: dict[str, Any] = {}

    def _exec_new(cfg: Any, cwd: pathlib.Path, argv: tuple[str, ...]) -> int:
        calls.update(form="new", argv=argv)
        return 0

    def _exec_in_session(layout: Any, cfg: Any, cwd: pathlib.Path, argv: tuple[str, ...]) -> int:
        calls.update(form="session", argv=argv)
        return 0

    def _effective(*_args: Any, **_kwargs: Any) -> Any:
        return type("E", (), {"config": None, "explicit_leaves": frozenset({"sandbox.home"})})()

    def _resolve(target: str) -> object:
        return object()

    monkeypatch.setattr("agent6.ui.cli.exec_cmds.exec_new", _exec_new)
    monkeypatch.setattr("agent6.ui.cli.net_cmds.exec_in_session", _exec_in_session)
    monkeypatch.setattr("agent6.ui.cli._common.resolve_target", _resolve)
    monkeypatch.setattr("agent6.config.layer.load_effective", _effective)
    return calls


def test_new_takes_the_command_verbatim(dispatched: dict[str, Any]) -> None:
    assert cli.main(["exec", "--new", "ls", "-la"]) == 0
    assert dispatched == {
        "form": "new",
        "argv": ("ls", "-la"),
    }


def test_new_without_a_command_asks_for_the_shell(dispatched: dict[str, Any]) -> None:
    assert cli.main(["exec", "--new"]) == 0
    assert dispatched["form"] == "new" and dispatched["argv"] == ()


def test_new_drops_one_leading_separator(dispatched: dict[str, Any]) -> None:
    assert cli.main(["exec", "--new", "--", "git", "log", "--", "p"]) == 0
    assert dispatched["argv"] == ("git", "log", "--", "p")


def test_new_with_a_session_is_refused_naming_both_forms(
    dispatched: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["exec", "--new", "brave-otter", "--", "ls"]) == 2
    err = capsys.readouterr().err
    assert "agent6 exec --new -- CMD..." in err and "agent6 exec SESSION -- CMD..." in err
    assert dispatched == {}


def test_new_after_the_command_belongs_to_the_command(dispatched: dict[str, Any]) -> None:
    assert cli.main(["exec", "ls", "--new"]) == 0
    assert dispatched == {"form": "session", "argv": ("ls", "--new")}


def _shell_on_path(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, name: str) -> str:
    shell = tmp_path / name
    shell.write_text("#!/bin/sh\n", encoding="utf-8")
    shell.chmod(0o755)

    def _path() -> str:
        return str(tmp_path)

    monkeypatch.setattr(tool_paths, "jail_search_path", _path)
    return str(shell)


def test_shell_is_the_operators_when_the_jail_path_holds_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    on_path = _shell_on_path(tmp_path, monkeypatch, "zsh")
    monkeypatch.setenv("SHELL", "/somewhere/else/zsh")
    assert exec_cmds.resolve_shell() == on_path


def test_shell_falls_back_to_sh(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A `$SHELL` the jail's PATH lacks, or none at all, opens /bin/sh, never a 127."""
    _shell_on_path(tmp_path, monkeypatch, "zsh")
    monkeypatch.setenv("SHELL", "/opt/fish/bin/fish")
    assert exec_cmds.resolve_shell() == "/bin/sh"
    monkeypatch.delenv("SHELL")
    assert exec_cmds.resolve_shell() == "/bin/sh"


class _FakeNet:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A host where `full` works, with the spawn and the session network recorded."""
    seen: dict[str, Any] = {"nets": []}
    env = dataclasses.replace(
        detect.environment(),
        sandbox_available=True,
        seccomp_arch_supported=True,
        userns_supported=True,
        landlock_abi=4,
    )

    def _env() -> detect.Environment:
        return env

    def _run(policy: kinds.JailPolicy, session_net: Any = None) -> int:
        seen.update(policy=policy, session_net=session_net)
        seen["open_during_run"] = session_net is not None and not session_net.closed
        return 5

    def _open() -> _FakeNet:
        net = _FakeNet()
        seen["nets"].append(net)
        return net

    monkeypatch.delenv("AGENT6_DISABLE_SANDBOX", raising=False)
    monkeypatch.setattr(_setup, "detect_env", _env)
    monkeypatch.setattr(exec_cmds, "run_attached", _run)
    monkeypatch.setattr(jail.SessionNetwork, "open", staticmethod(_open))
    return seen


def test_new_builds_the_policy_a_runs_command_gets(
    tmp_path: pathlib.Path, host: dict[str, Any]
) -> None:
    """The policy is `jail_policy`'s for the same config, unbounded, on a network of its own."""
    cfg = Config.model_validate(
        {
            "sandbox": {
                "isolation": "full",
                "hide_paths": [str(tmp_path / "private")],
                "extra_read_paths": [str(tmp_path / "ro")],
                "memory_limit_mib": 512,
            }
        }
    )
    argv = ("ls", "-la")
    assert exec_cmds.exec_new(cfg, tmp_path, argv) == 5
    want = tools_policy.jail_policy(tmp_path, cfg, "full", argv, timeout_s=0.0)
    assert host["policy"] == want
    assert (want.isolation, want.network, want.timeout_s) == ("full", "session", 0.0)
    assert want.hide_paths == (tmp_path / "private",)
    (net,) = host["nets"]
    assert host["session_net"] is net and host["open_during_run"] and net.closed


def test_new_on_the_host_network_opens_no_session_network(
    tmp_path: pathlib.Path, host: dict[str, Any]
) -> None:
    cfg = Config.model_validate({"sandbox": {"isolation": "full", "network": "host"}})
    assert exec_cmds.exec_new(cfg, tmp_path, ("true",)) == 5
    assert host["policy"].network == "host"
    assert host["session_net"] is None and host["nets"] == []


def test_new_names_the_shell_and_the_isolation(
    tmp_path: pathlib.Path,
    host: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    shell = _shell_on_path(tmp_path, monkeypatch, "zsh")
    monkeypatch.setenv("SHELL", shell)
    cfg = Config.model_validate({"sandbox": {"isolation": "full", "network": "host"}})
    assert exec_cmds.exec_new(cfg, tmp_path, ()) == 5
    assert host["policy"].argv == (shell,)
    assert f"[agent6] {shell} in a fresh sandbox, isolation full." in capsys.readouterr().err


def test_new_refuses_what_a_run_refuses(
    tmp_path: pathlib.Path, host: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """An explicit network the level cannot give refuses as a run does, before any spawn."""
    cfg = Config.model_validate({"sandbox": {"isolation": "partial", "network": "session"}})
    assert exec_cmds.exec_new(cfg, tmp_path, ("true",)) == 2
    err = capsys.readouterr().err
    assert "REFUSING: sandbox.network = 'session' requires sandbox.isolation = 'full'" in err
    assert "policy" not in host


def test_new_refuses_a_workspace_under_a_hidden_path(
    tmp_path: pathlib.Path, host: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """`exec --new` built a jail for a workspace under a `sandbox.hide_paths` entry."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    cfg = Config.model_validate({"sandbox": {"isolation": "full", "hide_paths": [str(tmp_path)]}})
    assert exec_cmds.exec_new(cfg, workspace, ("true",)) == 2
    err = capsys.readouterr().err
    assert f"REFUSING: the workspace {str(workspace)!r} is under {str(tmp_path)!r}" in err
    assert "policy" not in host


def test_new_warns_about_a_grant_of_the_home_directory(
    tmp_path: pathlib.Path,
    host: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`exec --new` would run its command under a `$HOME` grant without the warning a run prints."""
    monkeypatch.setenv("HOME", str(tmp_path))
    workspace = tmp_path / "repo"
    workspace.mkdir()
    cfg = Config.model_validate(
        {"sandbox": {"isolation": "full", "extra_write_paths": [str(tmp_path)]}}
    )
    assert exec_cmds.exec_new(cfg, workspace, ("true",)) == 5
    err = capsys.readouterr().err
    assert f"[agent6] WARNING: sandbox.extra_write_paths grants {str(tmp_path)!r}" in err
    assert "your home directory: the run's commands and file tools can read and write" in err


def test_new_refuses_a_level_the_host_lacks(
    tmp_path: pathlib.Path,
    host: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _unavailable(_requested: str, _env: detect.Environment) -> kinds.IsolationLevel:
        raise detect.IsolationUnavailableError("full is not available here")

    monkeypatch.setattr(detect, "resolve_isolation", _unavailable)
    cfg = Config.model_validate({"sandbox": {"isolation": "full"}})
    assert exec_cmds.exec_new(cfg, tmp_path, ("true",)) == 2
    assert "REFUSING: full is not available here" in capsys.readouterr().err
    assert "policy" not in host


def test_new_unsandboxed_says_nothing_is_confined(
    tmp_path: pathlib.Path, host: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = Config.model_validate({"sandbox": {"isolation": "none"}})
    assert exec_cmds.exec_new(cfg, tmp_path, ("true",)) == 5
    assert host["policy"].isolation == "none" and host["nets"] == []
    assert "running UNSANDBOXED (sandbox.isolation = none)" in capsys.readouterr().err


def test_new_unsandboxed_gets_the_env_an_unsandboxed_command_gets(
    tmp_path: pathlib.Path, host: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under isolation none the command sees this process's env minus the provider keys."""
    monkeypatch.setenv("EXEC_NEW_PROBE", "visible")
    monkeypatch.setenv("EXEC_NEW_KEY", "secret")
    monkeypatch.setattr("agent6.child_env._provider_key_env", {"EXEC_NEW_KEY"})
    cfg = Config.model_validate({"sandbox": {"isolation": "none"}})
    assert exec_cmds.exec_new(cfg, tmp_path, ("true",)) == 5
    plain = tools_policy.jail_policy(tmp_path, cfg, "none", ("true",), timeout_s=0.0)
    want = child_env.without_provider_keys({**os.environ, **dict(plain.env)})
    assert dict(host["policy"].env) == want
    assert want["EXEC_NEW_PROBE"] == "visible" and "EXEC_NEW_KEY" not in want


class _Ended:
    """A command that already exited 0, as `run_attached` reads one."""

    def __init__(self) -> None:
        self.popen = self

    def wait(self) -> int:
        return 0

    def close(self) -> frozenset[int]:
        return frozenset()


@pytest.fixture
def on_a_terminal(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, ...]]:
    """Stdin and stdout read as terminals; the spawned argv is recorded and nothing runs."""
    spawned: list[tuple[str, ...]] = []

    def _spawn(policy: kinds.JailPolicy, **_stdio: Any) -> _Ended:
        spawned.append(policy.argv)
        return _Ended()

    def _relay(_master: int, _proc: Any) -> None:
        return None

    def _isatty(fd: int) -> bool:
        return fd in (0, 1)

    monkeypatch.setattr(jail, "spawn_in_jail", _spawn)
    monkeypatch.setattr("agent6.ui.cli.exec_cmds._relay", _relay)
    monkeypatch.setattr(os, "isatty", _isatty)
    return spawned


def test_on_a_terminal_the_command_gets_the_pty_as_its_controlling_terminal(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, on_a_terminal: list[tuple[str, ...]]
) -> None:
    """The command runs as given, on a pty that is its controlling terminal.

    A `setsid --ctty --wait` wrapper made job control depend on util-linux in the sandbox.
    `agent6 exec --new CMD 2> FILE` on a terminal writes the command's stderr to FILE.
    """
    stdio: list[dict[str, Any]] = []

    def _spawn(policy: kinds.JailPolicy, **given: Any) -> _Ended:
        on_a_terminal.append(policy.argv)
        stdio.append(given)
        return _Ended()

    monkeypatch.setattr(jail, "spawn_in_jail", _spawn)
    assert exec_cmds.run_attached(_unsandboxed(tmp_path, "true")) == 0
    assert on_a_terminal == [("/bin/sh", "-c", "true")]
    [given] = stdio
    assert given["controlling_terminal"] is True
    assert given["stdin"] == given["stdout"] and given["stdin"] is not None
    assert given["stderr"] is None


def _unsandboxed(tmp_path: pathlib.Path, script: str) -> kinds.JailPolicy:
    return kinds.JailPolicy(
        cwd=tmp_path, argv=("/bin/sh", "-c", script), isolation="none", timeout_s=0.0
    )


def test_run_attached_streams_through_the_callers_stdio(
    tmp_path: pathlib.Path, capfd: pytest.CaptureFixture[str]
) -> None:
    """Off a terminal the command writes the caller's stdout and stderr itself."""
    code = exec_cmds.run_attached(_unsandboxed(tmp_path, "echo out; echo err >&2; exit 7"))
    captured = capfd.readouterr()
    assert (code, captured.out, captured.err) == (7, "out\n", "err\n")


def test_run_attached_reports_a_signal_as_128_plus_it(tmp_path: pathlib.Path) -> None:
    before = signal.getsignal(signal.SIGTERM)
    assert exec_cmds.run_attached(_unsandboxed(tmp_path, "kill -TERM $$")) == 128 + signal.SIGTERM
    assert signal.getsignal(signal.SIGTERM) is before
