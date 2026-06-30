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
    _machine_network_refusal,  # pyright: ignore[reportPrivateUsage]
    _machine_protect_paths,  # pyright: ignore[reportPrivateUsage]
    _resolve_network_refusal,  # pyright: ignore[reportPrivateUsage]
    _suggested_network_fix,  # pyright: ignore[reportPrivateUsage]
    _validate_bundle,  # pyright: ignore[reportPrivateUsage]
)
from agent6.config import Config
from agent6.machine import MachineJournal, ToolState, drive, load_machine
from agent6.machine.engine import LiveWorld, ToolExecResult
from agent6.types import CommandResult

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

TOOL_ONLY_MACHINE = """
machine = "toolonly"
version = 1
initial = "check"

[budget]
max_transitions = 5

[states.check]
kind = "tool"
command = ["true"]
timeout_secs = 5
on = { ok = "done", nonzero = "fail", timeout = "fail" }

[states.done]
kind = "terminal"
status = "ok"
reason = "checked"

[states.fail]
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


def test_liveworld_grants_data_dir_rw_and_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A machine's data dir is RW in every tool jail + exported as
    # $AGENT6_MACHINE_DATA_DIR, so a tool script can persist on hardened too.
    seen = _patch_jail(monkeypatch)
    data = tmp_path / "i" / "data"
    world = LiveWorld(
        cwd=tmp_path, journal=MachineJournal(tmp_path / "i"), profile="hardened", data_dir=data
    )
    world.run_tool(("true",), 5.0, allow_network=False)
    policy = seen[-1]
    assert data in policy.extra_rw_paths
    # Exported RELATIVE to cwd so it resolves inside a strict jail (cwd pivots to
    # /workspace there); the host abspath wouldn't exist in that jail.
    assert ("AGENT6_MACHINE_DATA_DIR", "i/data") in policy.env


def test_liveworld_no_data_dir_grants_no_extra_rw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _patch_jail(monkeypatch)
    world = LiveWorld(cwd=tmp_path, journal=MachineJournal(tmp_path / "i"), profile="hardened")
    world.run_tool(("true",), 5.0, allow_network=False)
    assert seen[-1].extra_rw_paths == ()
    assert all(k != "AGENT6_MACHINE_DATA_DIR" for k, _ in seen[-1].env)


def test_liveworld_disables_python_bytecode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _patch_jail(monkeypatch)
    world = LiveWorld(cwd=tmp_path, journal=MachineJournal(tmp_path / "i"), profile="hardened")
    world.run_tool(("python3", "-m", "unittest"), 5.0, allow_network=False)
    assert ("PYTHONDONTWRITEBYTECODE", "1") in seen[-1].env


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


def test_machine_run_refuses_escaping_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Security: `machine run` must re-validate the bundle, not only `check`. On a
    # profile that can't RO-bind the bundle, a `scripts/` symlink escaping it
    # would otherwise be executed; run must refuse before touching the world.
    from agent6.cli import main

    f = _write(tmp_path, NET_MACHINE)
    (tmp_path / "scripts").mkdir()
    outside = tmp_path.parent / "outside_secret_run"
    outside.write_text("secret", encoding="utf-8")
    (tmp_path / "scripts" / "fetch.sh").symlink_to(outside)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / ".state"))
    assert main(["machine", "run", str(f)]) == 1


def test_machine_run_validates_config_overlay_for_pure_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # B10: a pure wait/terminal machine has no agent/tool state, but its [config]
    # overlay must still be validated (and [machine] snapshot_keep honored). A
    # bogus overlay key now fails the run with CONFIG ERROR instead of being
    # silently ignored.
    from agent6.cli import main

    pure = (
        'machine = "pure"\nversion = 1\ninitial = "go"\n'
        "[budget]\nmax_transitions = 5\n"
        "[config.workflow]\nbogus_key = 42\n"
        '[states.go]\nkind = "wait"\nuntil = "2020-01-01T00:00:00Z"\n'
        'on = { tick = "done", signal = "done" }\n'
        '[states.done]\nkind = "terminal"\nstatus = "ok"\nreason = "x"\n'
    )
    f = _write(tmp_path, pure)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / ".state"))
    assert main(["machine", "run", str(f)]) == 2


def test_machine_run_keeps_tool_jail_strict_when_agent_egress_would_downgrade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent6.cli import main

    f = _write(tmp_path, TOOL_ONLY_MACHINE)
    seen_profiles: list[str] = []

    def fake_run_in_jail(policy: Any) -> CommandResult:
        seen_profiles.append(policy.profile)
        return CommandResult(
            argv=tuple(policy.argv),
            returncode=0,
            stdout="",
            stderr="",
            duration_s=0.0,
        )

    def fail_egress_probe(*_args: object) -> tuple[str, str | None]:
        pytest.fail("tool-only machines do not need the provider-egress downgrade")

    def select_strict(*_args: object) -> str:
        return "strict"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("agent6.cli.machine_cmds.select_profile", select_strict)
    monkeypatch.setattr(
        "agent6.cli.machine_cmds.resolve_strict_egress_viability", fail_egress_probe
    )
    monkeypatch.setattr("agent6.machine.engine.run_in_jail", fake_run_in_jail)

    assert main(["machine", "run", str(f)]) == 0
    assert seen_profiles == ["strict"]


# --- sandbox-conflict UX (suggest a fix, don't dead-end) --------------------


def _allow_tool(tmp_path: Path) -> ToolState:
    spec = load_machine(_write(tmp_path, NET_MACHINE))  # fetch sets allow_network="allow"
    fetch = spec.states["fetch"]
    assert isinstance(fetch, ToolState)
    return fetch


def test_suggested_network_fix_hardened(tmp_path: Path) -> None:
    # hardened can't isolate per-tool, so tools share the host net (+ agent open).
    fix = _suggested_network_fix(Config.model_validate({}), "hardened", [_allow_tool(tmp_path)])
    assert fix == {"sandbox.tool_network": "allow", "sandbox.agent_network": "open"}


def test_network_refusal_hardened_allow_tool_names_runnable_fix(tmp_path: Path) -> None:
    refusal = _machine_network_refusal(
        Config.model_validate({}), "hardened", [_allow_tool(tmp_path)]
    )
    assert refusal is not None
    assert "sandbox.tool_network = 'allow'" in refusal
    assert "sandbox.agent_network = 'open'" in refusal
    assert "only_explicit_states" not in refusal


def test_suggested_network_fix_strict(tmp_path: Path) -> None:
    # strict can single one tool out: explicit per-tool egress is the safe fix.
    fix = _suggested_network_fix(Config.model_validate({}), "strict", [_allow_tool(tmp_path)])
    assert fix == {"sandbox.tool_network": "only_explicit_states"}


def _plain_tool(tmp_path: Path) -> ToolState:
    # `store` has no allow_network -> defaults to "auto" (no network wanted).
    spec = load_machine(_write(tmp_path, NET_MACHINE))
    store = spec.states["store"]
    assert isinstance(store, ToolState)
    return store


def test_suggested_network_fix_plain_tool_hardened(tmp_path: Path) -> None:
    # Regression: on hardened EVERY tool (even one that wants no network) is
    # refused under tool_network="block" because no per-tool netns exists, and
    # the only config that runs it is tools sharing the host net (+ agent open).
    # Previously this returned None, contradicting the refusal's own advice.
    fix = _suggested_network_fix(Config.model_validate({}), "hardened", [_plain_tool(tmp_path)])
    assert fix == {"sandbox.tool_network": "allow", "sandbox.agent_network": "open"}


def test_suggested_network_fix_plain_tool_strict_is_noop(tmp_path: Path) -> None:
    # A plain no-network tool already runs on strict (its own empty netns), so
    # there is nothing to offer.
    fix = _suggested_network_fix(Config.model_validate({}), "strict", [_plain_tool(tmp_path)])
    assert fix is None


def test_suggested_network_fix_block_is_unfixable(tmp_path: Path) -> None:
    # allow_network="block" REQUIRES isolation only strict provides -> no config fix.
    text = NET_MACHINE.replace('allow_network = "allow"', 'allow_network = "block"')
    spec = load_machine(_write(tmp_path, text))
    fetch = spec.states["fetch"]
    assert isinstance(fetch, ToolState)
    assert _suggested_network_fix(Config.model_validate({}), "hardened", [fetch]) is None


def test_resolve_network_refusal_headless_prints_fix(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Non-interactive stdin (pytest): print the exact fix + simulate command,
    # exit 2, and NEVER relax a sandbox setting unattended.
    code = _resolve_network_refusal(
        tmp_path / "m.asm.toml",
        "a tool needs the network",
        Config.model_validate({}),
        "hardened",
        [_allow_tool(tmp_path)],
        tmp_path,
        {},
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "config set sandbox.tool_network allow" in err
    assert "config set sandbox.agent_network open" in err
    assert "machine test" in err
    # nothing was written
    assert not (tmp_path / ".agent6" / "config.toml").exists()


def test_resolve_network_refusal_unfixable_points_to_simulate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    text = NET_MACHINE.replace('allow_network = "allow"', 'allow_network = "block"')
    spec = load_machine(_write(tmp_path, text))
    fetch = spec.states["fetch"]
    assert isinstance(fetch, ToolState)
    code = _resolve_network_refusal(
        tmp_path / "m.asm.toml",
        "needs strict",
        Config.model_validate({}),
        "hardened",
        [fetch],
        tmp_path,
        {},
    )
    assert code == 2
    assert "machine test" in capsys.readouterr().err
