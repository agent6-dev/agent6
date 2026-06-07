# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Machine tool egress: per-state allow_network, bundle validation, and the
running machine's files made read-only in run jails (immutability)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from agent6.cli.machine_cmds import (
    _machine_protect_paths,  # pyright: ignore[reportPrivateUsage]
    _validate_bundle,  # pyright: ignore[reportPrivateUsage]
)
from agent6.machine import MachineJournal, ToolState, drive, load_machine
from agent6.machine.engine import LiveWorld, ToolExecResult

# A two-tool machine: the first tool opts into the network, the second does not.
NET_MACHINE = """
machine = "netdemo"
version = 1
initial = "fetch"

[budget]
max_usd = 1.0
max_transitions = 100

[states.fetch]
kind = "tool"
command = ["scripts/fetch.sh"]
timeout_secs = 5
allow_network = "allow"
on = { ok = "store", nonzero = "stop_fail", timeout = "stop_fail" }

[states.store]
kind = "tool"
command = ["store"]
timeout_secs = 5
on = { ok = "stop_ok", nonzero = "stop_fail", timeout = "stop_fail" }

[states.stop_ok]
kind = "terminal"
status = "ok"
reason = "done"

[states.stop_fail]
kind = "terminal"
status = "failed"
reason = "failed"
"""


def _write(tmp_path: Path, text: str, name: str = "m.asm.toml") -> Path:
    f = tmp_path / name
    f.write_text(text, encoding="utf-8")
    return f


# --- ToolState.allow_network field -----------------------------------------


def test_tool_allow_network_defaults_auto(tmp_path: Path) -> None:
    text = NET_MACHINE.replace('allow_network = "allow"\n', "")
    spec = load_machine(_write(tmp_path, text))
    fetch = spec.states["fetch"]
    assert isinstance(fetch, ToolState)
    assert fetch.allow_network == "auto"


def test_tool_allow_network_roundtrips(tmp_path: Path) -> None:
    spec = load_machine(_write(tmp_path, NET_MACHINE))
    fetch = spec.states["fetch"]
    store = spec.states["store"]
    assert isinstance(fetch, ToolState) and fetch.allow_network == "allow"
    assert isinstance(store, ToolState) and store.allow_network == "auto"


# --- engine threads allow_network through to the World ----------------------


@dataclass
class _RecordingWorld:
    net_calls: list[tuple[tuple[str, ...], bool]]

    def run_tool(
        self, argv: tuple[str, ...], timeout_s: float, *, allow_network: bool = False
    ) -> ToolExecResult:
        self.net_calls.append((argv, allow_network))
        return ToolExecResult(exit_code=0, stdout="", timed_out=False)

    def run_agent(self, request: Any) -> Any:  # pragma: no cover - no agent states here
        raise AssertionError("no agent states")

    def now(self) -> float:
        return 1000.0

    def sleep_until(self, wake_epoch: float) -> Any:  # pragma: no cover
        return "tick"


def test_engine_passes_per_state_allow_network(tmp_path: Path) -> None:
    spec = load_machine(_write(tmp_path, NET_MACHINE))
    journal = MachineJournal(tmp_path / "inst")
    world = _RecordingWorld(net_calls=[])
    result = drive(spec, journal, world, live=True)
    assert result.status == "ok"
    # fetch opted in (True); store did not (False).
    assert world.net_calls == [(("scripts/fetch.sh",), True), (("store",), False)]


# --- LiveWorld (supervisor) honors the per-state allow_network flag --------
# The engine is the host-netns supervisor; whether an opt-in is permitted at
# all is gated at machine-run startup (sandbox.tool_network), so LiveWorld just
# passes the per-state flag straight through to the jail.


@dataclass
class _FakeJailResult:
    returncode: int = 0
    stdout: str = ""


def _patch_jail(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Capture every JailPolicy run_tool builds, without forking a real jail."""
    seen: list[Any] = []

    def fake_run_in_jail(policy: Any) -> _FakeJailResult:
        seen.append(policy)
        return _FakeJailResult()

    monkeypatch.setattr("agent6.machine.engine.run_in_jail", fake_run_in_jail)
    return seen


def test_liveworld_networked_tool_inherits_host_netns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _patch_jail(monkeypatch)
    world = LiveWorld(cwd=tmp_path, journal=MachineJournal(tmp_path / "i"), profile="strict")
    world.run_tool(("curl", "x"), 5.0, allow_network=True)
    assert seen[-1].allow_network is True


def test_liveworld_non_network_tool_is_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _patch_jail(monkeypatch)
    world = LiveWorld(cwd=tmp_path, journal=MachineJournal(tmp_path / "i"), profile="strict")
    world.run_tool(("true",), 5.0, allow_network=False)
    assert seen[-1].allow_network is False


def test_liveworld_passes_protect_paths_to_jail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _patch_jail(monkeypatch)
    guarded = (tmp_path / "m.asm.toml", tmp_path / "scripts")
    world = LiveWorld(
        cwd=tmp_path,
        journal=MachineJournal(tmp_path / "i"),
        profile="strict",
        protect_paths=guarded,
    )
    world.run_tool(("true",), 5.0)
    assert seen[-1].extra_protect_paths == guarded


# --- machine-file immutability (_machine_protect_paths) --------------------


def test_protect_paths_include_machine_file_and_scripts(tmp_path: Path) -> None:
    f = _write(tmp_path, NET_MACHINE)
    (tmp_path / "scripts").mkdir()
    got = _machine_protect_paths(f, tmp_path)
    assert f.resolve() in got
    assert (tmp_path / "scripts").resolve() in got


def test_protect_paths_skip_missing_scripts(tmp_path: Path) -> None:
    f = _write(tmp_path, NET_MACHINE)  # no scripts/ dir
    got = _machine_protect_paths(f, tmp_path)
    assert got == (f.resolve(),)


def test_protect_paths_exclude_machine_outside_cwd(tmp_path: Path) -> None:
    # A machine file outside the jail-mounted cwd isn't in the child's view, so
    # it isn't (and can't be) protected.
    outside = tmp_path.parent / "outside.asm.toml"
    outside.write_text(NET_MACHINE, encoding="utf-8")
    sub = tmp_path / "repo"
    sub.mkdir()
    assert _machine_protect_paths(outside, sub) == ()


# --- bundle / script-path validation ---------------------------------------


def test_bundle_ok_when_script_exists(tmp_path: Path) -> None:
    f = _write(tmp_path, NET_MACHINE)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "fetch.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    spec = load_machine(f)
    assert _validate_bundle(spec, f) == []


def test_bundle_flags_missing_script(tmp_path: Path) -> None:
    f = _write(tmp_path, NET_MACHINE)  # references scripts/fetch.sh, never created
    spec = load_machine(f)
    problems = _validate_bundle(spec, f)
    assert any("not found in bundle" in p for p in problems)


def test_bundle_flags_escaping_command_ref(tmp_path: Path) -> None:
    text = NET_MACHINE.replace(
        'command = ["scripts/fetch.sh"]', 'command = ["scripts/../../etc/x"]'
    )
    f = _write(tmp_path, text)
    spec = load_machine(f)
    problems = _validate_bundle(spec, f)
    assert any("escapes the bundle" in p for p in problems)


def test_bundle_flags_symlink_escape(tmp_path: Path) -> None:
    f = _write(tmp_path, NET_MACHINE)
    (tmp_path / "scripts").mkdir()
    outside = tmp_path.parent / "outside_secret"
    outside.write_text("secret", encoding="utf-8")
    (tmp_path / "scripts" / "fetch.sh").symlink_to(outside)
    spec = load_machine(f)
    problems = _validate_bundle(spec, f)
    assert any("outside the bundle" in p for p in problems)


def test_bundle_reports_circular_symlink_in_scripts(tmp_path: Path) -> None:
    # A circular symlink makes Path.resolve() raise RuntimeError; the validator
    # must report it as a problem, not crash.
    f = _write(tmp_path, NET_MACHINE)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "loop").symlink_to(tmp_path / "scripts" / "loop")
    spec = load_machine(f)
    problems = _validate_bundle(spec, f)  # must not raise
    assert any("loop" in p for p in problems)


def test_bundle_reports_circular_symlink_command_ref(tmp_path: Path) -> None:
    text = NET_MACHINE.replace('command = ["scripts/fetch.sh"]', 'command = ["scripts/loop"]')
    f = _write(tmp_path, text)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "loop").symlink_to(tmp_path / "scripts" / "loop")
    spec = load_machine(f)
    problems = _validate_bundle(spec, f)  # must not raise
    assert any("fetch" not in p and "loop" in p for p in problems)


def test_machine_check_fails_on_bad_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent6.cli import main

    text = NET_MACHINE.replace('command = ["scripts/fetch.sh"]', 'command = ["scripts/../escape"]')
    f = _write(tmp_path, text)
    monkeypatch.chdir(tmp_path)
    assert main(["machine", "check", str(f)]) == 1


def test_machine_check_passes_with_valid_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent6.cli import main

    f = _write(tmp_path, NET_MACHINE)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "fetch.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert main(["machine", "check", str(f)]) == 0
