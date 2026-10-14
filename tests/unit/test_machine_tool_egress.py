# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Machine tool egress: per-state network, bundle validation, the bundle read-only in run jails."""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import pytest

from agent6 import paths
from agent6.app.machine import StateConfig, machine_protect_paths, run, validate_bundle
from agent6.config import Config
from agent6.machine import (
    MachineError,
    MachineJournal,
    ToolState,
    drive,
    engine,
    live_world,
    load_machine,
)

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
on = { ok = "store", nonzero = "stop_fail", timeout = "stop_fail" }

[states.fetch.config.sandbox]
network = "host"

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


def _write(tmp_path: pathlib.Path, text: str, name: str = "m.asm.toml") -> pathlib.Path:
    f = tmp_path / name
    f.write_text(text, encoding="utf-8")
    return f


# --- a tool state's network is its config table's -----------------------


def test_a_tool_state_sets_its_network_in_its_config_table(tmp_path: pathlib.Path) -> None:
    spec = load_machine(_write(tmp_path, NET_MACHINE))
    fetch = spec.states["fetch"]
    store = spec.states["store"]
    assert isinstance(fetch, ToolState) and fetch.config == {"sandbox": {"network": "host"}}
    assert isinstance(store, ToolState) and store.config == {}


def test_a_network_field_on_a_tool_state_fails_the_load(tmp_path: pathlib.Path) -> None:
    """One spelling: a state's network is `config.sandbox.network`, for tool and agent states."""
    text = NET_MACHINE.replace('command = ["store"]\n', 'command = ["store"]\nnetwork = "host"\n')
    with pytest.raises(MachineError, match="network"):
        load_machine(_write(tmp_path, text))


# --- the engine names each tool call's state to the World ----------------


@dataclasses.dataclass
class _RecordingWorld:
    state_calls: list[tuple[tuple[str, ...], str]]

    def run_tool(
        self,
        argv: tuple[str, ...],
        timeout_s: float,
        *,
        state: str = "",
        pass_env: tuple[str, ...] = (),
    ) -> engine.ToolExecResult:
        self.state_calls.append((argv, state))
        return engine.ToolExecResult(exit_code=0, stdout="", timed_out=False)

    def run_agent(self, request: Any) -> Any:  # pragma: no cover - no agent states here
        raise AssertionError("no agent states")

    def now(self) -> float:
        return 1000.0

    def sleep_until(self, wake_epoch: float | None) -> Any:  # pragma: no cover

        return engine.WaitWake("tick")

    def materialize_poke(self, payload: Any) -> None:  # pragma: no cover
        pass

    def notify(self, kind: str, state: str, message: str, level: str) -> None:  # pragma: no cover
        pass


def test_engine_names_the_state_of_each_tool_call(tmp_path: pathlib.Path) -> None:
    spec = load_machine(_write(tmp_path, NET_MACHINE))
    journal = MachineJournal(tmp_path / "inst")
    world = _RecordingWorld(state_calls=[])
    result = drive(spec, journal, world, live=True)
    assert result.status == "ok"
    assert world.state_calls == [(("scripts/fetch.sh",), "fetch"), (("store",), "store")]


# LiveWorld builds each call's policy from its state's config; the bounds are checked at startup.


@dataclasses.dataclass
class _FakeJailResult:
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


def _patch_jail(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Capture every JailPolicy run_tool builds, without forking a real jail."""
    seen: list[Any] = []

    def fake_run_in_jail(policy: Any) -> _FakeJailResult:
        seen.append(policy)
        return _FakeJailResult()

    monkeypatch.setattr("agent6.sandbox.jail.run_in_jail", fake_run_in_jail)
    return seen


def _world(
    tmp_path: pathlib.Path,
    isolation: str,
    *,
    cfg: Config | None = None,
    protect_paths: tuple[pathlib.Path, ...] = (),
    data_dir: pathlib.Path | None = None,
) -> live_world.LiveWorld:
    """A LiveWorld wired exactly as run_machine wires it, through the shared policy builder."""
    factory = run.machine_tool_policy_factory(
        {"": StateConfig(cfg or Config(), frozenset(), isolation)},  # type: ignore[arg-type]
        tmp_path,
        protect_paths=protect_paths,
        data_dir=data_dir,
    )
    return live_world.LiveWorld(
        cwd=tmp_path,
        journal=MachineJournal(tmp_path / "i"),
        tool_policy=factory,
        data_dir=data_dir,
    )


def test_machine_tool_jail_carries_operator_grants_and_protect_git(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The machine tool jail is built by the same policy builder as run commands.

    An operator's extra read and write grants, hide_paths and protect_git reach it.
    """
    seen = _patch_jail(monkeypatch)
    (tmp_path / ".git").mkdir()
    cfg = Config.model_validate(
        {
            "sandbox": {
                "extra_read_paths": ["/srv/ro"],
                "extra_write_paths": ["/srv/rw"],
                "hide_paths": ["/srv/secret"],
                "protect_git": True,
            }
        }
    )
    world = _world(tmp_path, "strict", cfg=cfg)
    world.run_tool(("true",), 5.0)
    policy = seen[-1]
    assert pathlib.Path("/srv/ro") in policy.extra_ro_paths
    assert pathlib.Path("/srv/rw") in policy.extra_rw_paths
    assert pathlib.Path("/srv/secret") in policy.hide_paths
    assert (tmp_path / ".git").resolve() in policy.extra_protect_paths


def test_liveworld_non_network_tool_is_isolated(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _patch_jail(monkeypatch)
    world = _world(tmp_path, "strict")
    world.run_tool(("true",), 5.0)
    assert seen[-1].network == "none"


def test_liveworld_tool_joins_the_host_network_when_its_config_says_host(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _patch_jail(monkeypatch)
    world = _world(tmp_path, "strict", cfg=Config.model_validate({"sandbox": {"network": "host"}}))
    world.run_tool(("true",), 5.0)
    assert seen[-1].network == "host"


def test_liveworld_refuses_a_state_with_no_resolved_config(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tool call never runs under another state's config, or the machine's, by fallback."""
    seen = _patch_jail(monkeypatch)
    world = _world(tmp_path, "strict")
    with pytest.raises(engine.EngineError, match="no config was resolved for state 'ghost'"):
        world.run_tool(("true",), 5.0, state="ghost")
    assert seen == []


def test_liveworld_grants_data_dir_rw_and_env(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The data dir is RW in every tool jail and exported as $AGENT6_MACHINE_DATA_DIR.
    seen = _patch_jail(monkeypatch)
    data = tmp_path / "i" / "data"
    world = _world(tmp_path, "hardened", data_dir=data)
    world.run_tool(("true",), 5.0)
    policy = seen[-1]
    assert data in policy.extra_rw_paths
    # Exported as the real host abspath on hardened; strict maps it to /rw<abspath>.
    assert ("AGENT6_MACHINE_DATA_DIR", str(data)) in policy.env


def test_liveworld_no_data_dir_grants_only_the_home(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _patch_jail(monkeypatch)
    world = _world(tmp_path, "hardened")
    world.run_tool(("true",), 5.0)
    # hardened's persistent HOME is the only extra grant: no data dir, no data grant.
    assert seen[-1].extra_rw_paths == (paths.jail_cache_home(),)
    assert all(k != "AGENT6_MACHINE_DATA_DIR" for k, _ in seen[-1].env)


def test_liveworld_disables_python_bytecode(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _patch_jail(monkeypatch)
    world = _world(tmp_path, "hardened")
    world.run_tool(("python3", "-m", "unittest"), 5.0)
    assert ("PYTHONDONTWRITEBYTECODE", "1") in seen[-1].env


def test_liveworld_passes_protect_paths_to_jail(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _patch_jail(monkeypatch)
    guarded = (tmp_path / "m.asm.toml", tmp_path / "scripts")
    world = _world(tmp_path, "strict", protect_paths=guarded)
    world.run_tool(("true",), 5.0)
    assert guarded[0] in seen[-1].extra_protect_paths
    assert guarded[1] in seen[-1].extra_protect_paths


def test_a_bundle_in_a_subdirectory_runs_its_own_scripts(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`machine check` resolves `scripts/` against the bundle, so the run does too.

    A bundle under `machines/` run from the repository root otherwise passes check and fails
    every tool step on a script the jail's cwd does not hold.
    """
    seen = _patch_jail(monkeypatch)
    bundle = tmp_path / "machines"
    (bundle / "scripts").mkdir(parents=True)
    factory = run.machine_tool_policy_factory(
        {"": StateConfig(Config(), frozenset(), "strict")},
        tmp_path,
        protect_paths=(),
        data_dir=None,
        bundle_dir=bundle,
    )
    world = live_world.LiveWorld(
        cwd=tmp_path, journal=MachineJournal(tmp_path / "i"), tool_policy=factory
    )
    world.run_tool(
        ("python3", "scripts/count.py", "./scripts/x.sh", "notes/scripts", "/bin/true"), 5.0
    )
    assert seen[-1].argv == (
        "python3",
        "machines/scripts/count.py",
        "machines/scripts/x.sh",
        "notes/scripts",
        "/bin/true",
    )


# --- machine-file immutability (_machine_protect_paths) --------------------


def test_protect_paths_include_machine_file_and_scripts(tmp_path: pathlib.Path) -> None:
    f = _write(tmp_path, NET_MACHINE)
    (tmp_path / "scripts").mkdir()
    got = machine_protect_paths(f, tmp_path)
    assert f.resolve() in got
    assert (tmp_path / "scripts").resolve() in got


def test_protect_paths_skip_missing_scripts(tmp_path: pathlib.Path) -> None:
    f = _write(tmp_path, NET_MACHINE)  # no scripts/ dir
    got = machine_protect_paths(f, tmp_path)
    assert got == (f.resolve(),)


def test_protect_paths_exclude_machine_outside_cwd(tmp_path: pathlib.Path) -> None:
    # A machine file outside the mounted cwd is not in the child's view, so cannot be protected.
    outside = tmp_path.parent / "outside.asm.toml"
    outside.write_text(NET_MACHINE, encoding="utf-8")
    sub = tmp_path / "repo"
    sub.mkdir()
    assert machine_protect_paths(outside, sub) == ()


# --- bundle / script-path validation ---------------------------------------


def test_bundle_ok_when_script_exists(tmp_path: pathlib.Path) -> None:
    f = _write(tmp_path, NET_MACHINE)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "fetch.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    spec = load_machine(f)
    assert validate_bundle(spec, f) == []


def test_bundle_flags_missing_script(tmp_path: pathlib.Path) -> None:
    f = _write(tmp_path, NET_MACHINE)  # references scripts/fetch.sh, never created
    spec = load_machine(f)
    problems = validate_bundle(spec, f)
    assert any("not found in bundle" in p for p in problems)


def test_bundle_flags_escaping_command_ref(tmp_path: pathlib.Path) -> None:
    text = NET_MACHINE.replace(
        'command = ["scripts/fetch.sh"]', 'command = ["scripts/../../etc/x"]'
    )
    f = _write(tmp_path, text)
    spec = load_machine(f)
    problems = validate_bundle(spec, f)
    assert any("escapes the bundle" in p for p in problems)


def test_bundle_flags_symlink_escape(tmp_path: pathlib.Path) -> None:
    f = _write(tmp_path, NET_MACHINE)
    (tmp_path / "scripts").mkdir()
    outside = tmp_path.parent / "outside_secret"
    outside.write_text("secret", encoding="utf-8")
    (tmp_path / "scripts" / "fetch.sh").symlink_to(outside)
    spec = load_machine(f)
    problems = validate_bundle(spec, f)
    assert any("outside the bundle" in p for p in problems)


def test_bundle_reports_circular_symlink_in_scripts(tmp_path: pathlib.Path) -> None:
    # A circular symlink makes Path.resolve() raise; the validator reports it as a problem.
    f = _write(tmp_path, NET_MACHINE)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "loop").symlink_to(tmp_path / "scripts" / "loop")
    spec = load_machine(f)
    problems = validate_bundle(spec, f)  # must not raise
    assert any("loop" in p for p in problems)


def test_bundle_reports_circular_symlink_command_ref(tmp_path: pathlib.Path) -> None:
    text = NET_MACHINE.replace('command = ["scripts/fetch.sh"]', 'command = ["scripts/loop"]')
    f = _write(tmp_path, text)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "loop").symlink_to(tmp_path / "scripts" / "loop")
    spec = load_machine(f)
    problems = validate_bundle(spec, f)  # must not raise
    assert any("fetch" not in p and "loop" in p for p in problems)


def test_machine_check_fails_on_bad_bundle(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent6.ui.cli import main

    text = NET_MACHINE.replace('command = ["scripts/fetch.sh"]', 'command = ["scripts/../escape"]')
    f = _write(tmp_path, text)
    monkeypatch.chdir(tmp_path)
    assert main(["machine", "check", str(f)]) == 1


def test_machine_check_passes_with_valid_bundle(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent6.ui.cli import main

    f = _write(tmp_path, NET_MACHINE)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "fetch.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    # The fetch state asks for the host network, so the operator's own config must grant it.
    assert main(["machine", "check", str(f)]) == 1
    paths.global_config_path().parent.mkdir(parents=True, exist_ok=True)
    paths.global_config_path().write_text('[sandbox]\nnetwork = "host"\n', encoding="utf-8")
    assert main(["machine", "check", str(f)]) == 0


def test_machine_run_refuses_escaping_bundle(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `machine run` re-validates the bundle: a `scripts/` symlink escaping it must refuse first.
    from agent6.ui.cli import main

    f = _write(tmp_path, NET_MACHINE)
    (tmp_path / "scripts").mkdir()
    outside = tmp_path.parent / "outside_secret_run"
    outside.write_text("secret", encoding="utf-8")
    (tmp_path / "scripts" / "fetch.sh").symlink_to(outside)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / ".state"))
    assert main(["machine", "run", str(f)]) == 1


def test_machine_run_validates_config_overlay_for_pure_machine(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A pure wait/terminal machine still validates its [config] overlay.
    from agent6.ui.cli import main

    pure = (
        'machine = "pure"\nversion = 1\ninitial = "go"\n'
        "[budget]\nmax_transitions = 5\n"
        "[config.harness]\nbogus_key = 42\n"
        '[states.go]\nkind = "wait"\nuntil = "2020-01-01T00:00:00Z"\n'
        'on = { tick = "done", signal = "done" }\n'
        '[states.done]\nkind = "terminal"\nstatus = "ok"\nreason = "x"\n'
    )
    f = _write(tmp_path, pure)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / ".state"))
    assert main(["machine", "run", str(f)]) == 2
