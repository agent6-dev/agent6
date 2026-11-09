# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Prepared control descriptions match command and host file-tool enforcement."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import tempfile
from collections.abc import Iterator

import pytest

from agent6 import kinds, paths
from agent6.app import confine, preflight, reporter, session_runtime
from agent6.config import Config
from agent6.sandbox import detect
from agent6.tools import dispatch, results, tool_errors


def _config(**sandbox: object) -> Config:
    return Config.model_validate(
        {"sandbox": {"network": "host", "run_commands": "allow", **sandbox}}
    )


@pytest.mark.needs_namespaces
@pytest.mark.parametrize("custom", [False, True], ids=["defaults", "overrides"])
def test_prepared_controls_match_policy_and_command_enforcement(
    tmp_path: pathlib.Path, custom: bool
) -> None:
    """Explicit controls change the prepared policy and command behavior together."""
    # /dev/tty is absent by default; stat it without accessing the host terminal.
    device = pathlib.Path("/dev/tty")
    if not device.is_char_device():
        pytest.skip("host has no /dev/tty")
    root, readable, writable = (tmp_path / name for name in ("repo", "read", "write"))
    for directory in (root, readable, writable):
        directory.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (readable / "value").write_text("read grant")
    hidden = root / "hidden"
    hidden.write_text("secret")
    overrides = {
        "network": "host",
        "home": "cache",
        "protect_git": "off",
        "extra_read_paths": [str(readable)],
        "extra_write_paths": [str(writable)],
        "extra_device_paths": [str(device)],
        "hide_paths": [str(hidden)],
    }
    cfg = _config(isolation="full", **(overrides if custom else {"network": "auto"}))
    access = session_runtime.prepare_local_execution(cfg, root, "full")
    try:
        session = access.session
        assert session is not None and session.target is not None
        policy, target = access.policy, session.target
        controls = {row.leaf.removeprefix("sandbox."): row for row in target.controls}
        for name in overrides:
            assert controls[name].configured == getattr(cfg.sandbox, name)
            assert controls[name].owner == "agent6"
            assert not controls[name].reason
        assert controls["network"].effective == policy.network == ("host" if custom else "session")
        assert controls["home"].effective == ("cache" if custom else "tmp")
        assert controls["protect_git"].effective == ("off" if custom else "on")
        assert ((root / ".git") in policy.extra_protect_paths) is not custom
        for name, path, granted in (
            ("extra_read_paths", readable, policy.extra_ro_paths),
            ("extra_write_paths", writable, policy.extra_rw_paths),
            ("extra_device_paths", device, policy.extra_device_paths),
            ("hide_paths", hidden, policy.hide_paths),
        ):
            expected = (str(path),) if custom else ()
            assert (
                controls[name].effective == tuple(str(p) for p in granted if p == path) == expected
            )
        home = pathlib.Path(dict(policy.env)["HOME"])
        expected_home = paths.jail_cache_home(root) if custom else pathlib.Path("/tmp/agent6-home")
        assert home == expected_home
        assert target.home == str(home)
        assert (home in policy.extra_rw_paths) is custom
        script = """
import json, os, pathlib, sys
root, readable, writable = map(pathlib.Path, sys.argv[1:])
def read(path):
    try:
        return path.read_text()
    except OSError:
        return None
def write(path):
    try:
        path.write_text("written")
        return True
    except OSError:
        return False
print(json.dumps({
    "network": os.readlink("/proc/self/ns/net"),
    "home": os.environ["HOME"],
    "home_write": write(pathlib.Path.home() / "control-probe"),
    "git_write": write(root / ".git" / "control-probe"),
    "read": read(readable / "value"),
    "read_write": write(readable / "value"),
    "write": write(writable / "value"),
    "hidden": read(root / "hidden"),
    "hidden_write": write(root / "hidden"),
    "device": pathlib.Path("/dev/tty").is_char_device(),
}))
"""
        outcome = session.run(
            ("/usr/bin/python3", "-c", script, str(root), str(readable), str(writable)),
            env=policy.env,
            timeout_s=5,
        )
        assert isinstance(outcome, kinds.CommandResult)
        assert outcome.returncode == 0, outcome.stderr
        observed = json.loads(outcome.stdout)
        host_network = str(pathlib.Path("/proc/self/ns/net").readlink())
        assert (observed["network"] == host_network) is custom
        assert observed["home"] == target.home
        assert observed["home_write"] is True
        assert observed["git_write"] is custom
        assert observed["read"] == ("read grant" if custom else None)
        assert observed["read_write"] is False
        assert observed["write"] is custom
        if custom:
            assert observed["hidden"] in (None, "")
        else:
            assert observed["hidden"] == "secret"
        assert observed["hidden_write"] is not custom
        assert observed["device"] is custom
        assert (readable / "value").read_text() == "read grant"
        assert hidden.read_text() == ("secret" if custom else "written")
        assert (writable / "value").exists() is custom
        assert (root / ".git" / "control-probe").exists() is custom
    finally:
        assert access.close() == ""
    if custom:
        assert (home / "control-probe").read_text() == "written"


@pytest.fixture
def scratch(monkeypatch: pytest.MonkeyPatch) -> Iterator[pathlib.Path]:
    """Keep grants and HOME outside /tmp, which partial grants without an explicit rule."""
    base = pathlib.Path("/var/tmp")
    if not (base.is_dir() and os.access(base, os.W_OK)):
        pytest.skip("/var/tmp is not writable here")
    with tempfile.TemporaryDirectory(prefix="agent6-controls-", dir=base) as directory:
        root = pathlib.Path(directory).resolve()
        (root / "operator-home").mkdir()
        monkeypatch.setenv("HOME", str(root / "operator-home"))
        monkeypatch.setenv("XDG_CACHE_HOME", str(root / "cache"))
        yield root


@pytest.mark.parametrize("isolation", ["partial", "none"])
@pytest.mark.parametrize("custom", [False, True], ids=["automatic", "cache-git-off"])
def test_lower_controls_distinguish_commands_from_file_tools(
    scratch: pathlib.Path, isolation: kinds.IsolationLevel, custom: bool
) -> None:
    """Lower isolation never presents host file-tool restrictions as command protection."""
    if isolation == "partial" and detect.probed_environment().landlock_abi < 3:
        pytest.skip("read-only truncation protection requires Landlock ABI 3")
    root, readable, writable, outside = (
        scratch / name for name in ("repo", "read", "write", "outside")
    )
    for directory in (root, readable, writable, outside):
        directory.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (readable / "value").write_text("read grant")
    (writable / "value").write_text("write grant")
    (outside / "value").write_text("outside")
    operator_home = scratch / "operator-home"
    (operator_home / "value").write_text("operator home")
    git_probe = root / ".git" / "control-probe"
    git_probe.write_text("git")
    # Partial refuses a hide inside a command grant; none leaves it command-accessible.
    hidden = (scratch if isolation == "partial" else writable) / "hidden"
    hidden.write_text("secret")
    cfg = _config(
        isolation=isolation,
        home="cache" if custom else "auto",
        protect_git="off" if custom else "auto",
        extra_read_paths=[str(readable)],
        extra_write_paths=[str(writable)],
        hide_paths=[str(hidden)],
    )
    assert confine.startup_refusal(cfg, isolation, root) is None
    access = session_runtime.prepare_local_execution(cfg, root, isolation)
    d = dispatch.ToolDispatcher(root=root, config=cfg, isolation=isolation, command_access=access)
    try:
        target = d.execution_target
        assert target is not None
        controls = {row.leaf.removeprefix("sandbox."): row for row in target.controls}
        unconfined = isolation == "none"
        file_owner = "host file tools" if unconfined else "agent6"
        grant_behaviour = (
            "host file-tool grants only; commands are unconfined"
            if unconfined
            else "resolved grants for commands and host file tools"
        )
        for name, path, granted in (
            ("extra_read_paths", readable, access.policy.extra_ro_paths),
            ("extra_write_paths", writable, access.policy.extra_rw_paths),
        ):
            assert controls[name].configured == controls[name].effective == (str(path),)
            assert path in granted
            assert controls[name].owner == file_owner
            assert controls[name].behaviour == grant_behaviour
        assert controls["hide_paths"].effective == (str(hidden),)
        assert access.policy.hide_paths == (hidden,)
        assert controls["hide_paths"].owner == file_owner
        assert controls["hide_paths"].behaviour == (
            "host file-tool denials only; commands are unconfined"
            if unconfined
            else "host file-tool denials; exposed command grants refuse"
        )
        assert controls["protect_git"].effective == "off"
        assert not access.policy.extra_protect_paths
        assert controls["protect_git"].owner == ("agent6" if custom else "host file tools")
        assert controls["protect_git"].behaviour == (
            "protection disabled" if custom else "file-tool protection only"
        )
        assert bool(controls["protect_git"].reason) is not custom
        cached = custom or isolation == "partial"
        home = paths.jail_cache_home(root) if cached else operator_home
        assert controls["home"].effective == ("cache" if cached else "host")
        assert target.home == dict(access.policy.env)["HOME"] == str(home)
        assert (home in access.policy.extra_rw_paths) is cached

        def edit(path: pathlib.Path, old: str) -> None:
            d.dispatch(
                "apply_edit", {"path": str(path), "old_string": old, "new_string": "file tool"}
            )

        read = d.dispatch("read_file", {"path": str(readable / "value")})
        assert isinstance(read, results.ReadFileResult) and read.content == "read grant"
        with pytest.raises(tool_errors.ToolError, match="granted path"):
            edit(readable / "value", "read grant")
        edit(writable / "value", "write grant")
        assert (writable / "value").read_text() == "file tool"
        with pytest.raises(tool_errors.ToolError, match="granted path"):
            d.dispatch("read_file", {"path": str(outside / "value")})
        with pytest.raises(tool_errors.ToolError, match="granted path"):
            edit(outside / "value", "outside")
        hidden_reason = "hidden from this run" if unconfined else "granted path"
        with pytest.raises(tool_errors.ToolError, match=hidden_reason):
            d.dispatch("read_file", {"path": str(hidden)})
        with pytest.raises(tool_errors.ToolError, match=hidden_reason):
            edit(hidden, "secret")
        if custom:
            edit(git_probe, "git")
            assert git_probe.read_text() == "file tool"
        else:
            with pytest.raises(tool_errors.ToolError, match=r"Refusing to write under \.git"):
                edit(git_probe, "git")
        script = """
import json, os, pathlib, sys
root, readable, writable, outside, hidden, operator_home = map(pathlib.Path, sys.argv[1:])
def read(path):
    try:
        return path.read_text()
    except OSError:
        return None
def write(path):
    try:
        path.write_text("command")
        return True
    except OSError:
        return False
print(json.dumps({
    "home": os.environ["HOME"],
    "home_write": write(pathlib.Path.home() / "control-probe"),
    "operator_home": read(operator_home / "value"),
    "git_write": write(root / ".git" / "control-probe"),
    "read": read(readable / "value"),
    "read_write": write(readable / "value"),
    "write": write(writable / "value"),
    "outside_read": read(outside / "value"),
    "outside_write": write(outside / "value"),
    "hidden_read": read(hidden),
    "hidden_write": write(hidden),
}))
"""
        outcome = d.dispatch(
            "run_command",
            {
                "argv": [
                    "/usr/bin/python3",
                    "-c",
                    script,
                    *map(str, (root, readable, writable, outside, hidden, operator_home)),
                ]
            },
        )
        assert isinstance(outcome, results.ExecResult) and outcome.returncode == 0
        observed = json.loads(outcome.stdout)
        assert observed["home"] == target.home
        assert observed["home_write"] is True
        assert observed["operator_home"] == ("operator home" if unconfined else None)
        assert observed["git_write"] is True
        assert observed["read"] == "read grant"
        assert observed["read_write"] is unconfined
        assert observed["write"] is True
        assert observed["outside_read"] == ("outside" if unconfined else None)
        assert observed["outside_write"] is unconfined
        assert observed["hidden_read"] == ("secret" if unconfined else None)
        assert observed["hidden_write"] is unconfined
        assert git_probe.read_text() == "command"
        assert (writable / "value").read_text() == "command"
        assert (readable / "value").read_text() == ("command" if unconfined else "read grant")
        assert hidden.read_text() == ("command" if unconfined else "secret")
    finally:
        d.close()
    assert (home / "control-probe").read_text() == "command"


@pytest.mark.parametrize("location", ["workspace", "read-grant", "write-grant"])
def test_partial_refuses_a_hide_exposed_to_commands(
    scratch: pathlib.Path, location: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Partial cannot mask a hidden path out of a command grant and refuses that setup."""
    if detect.probed_environment().landlock_abi < 1:
        pytest.skip("partial isolation requires Landlock")
    root, grant = scratch / "repo", scratch / "grant"
    root.mkdir()
    grant.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    hidden = (root if location == "workspace" else grant) / "hidden"
    hidden.write_text("secret")
    cfg = _config(
        isolation="partial",
        hide_paths=[str(hidden)],
        extra_read_paths=[str(grant)] if location == "read-grant" else [],
        extra_write_paths=[str(grant)] if location == "write-grant" else [],
    )
    refusal = confine.startup_refusal(cfg, "partial", root)
    assert refusal is not None and "hide_paths" in refusal and "full" in refusal
    with pytest.raises(preflight.SessionRefusedError):
        session_runtime.select_isolation(
            cfg,
            cwd=root,
            confirm_unconfined=lambda _isolation, _cfg: pytest.fail(
                "an unsupported hide reached confirmation"
            ),
            reporter=reporter.STDIO_REPORTER,
        )
    assert f"REFUSING: {refusal}" in capsys.readouterr().err
