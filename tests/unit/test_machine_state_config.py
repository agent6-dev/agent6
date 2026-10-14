# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A machine state's own config table, and the sandbox bound every overlay sits under."""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any
from unittest import mock

import pytest

from agent6 import kinds, paths
from agent6.app import _setup, machine_agent
from agent6.app import reporter as app_reporter
from agent6.app.machine import (
    SandboxWidenedError,
    StateConfig,
    _states,
    machine_protect_paths,
    resolve_configs,
    run,
    sandbox_widening,
)
from agent6.config import Config, ConfigError, SandboxConfig
from agent6.machine import AgentExecResult, MachineError, load_machine
from agent6.sandbox import detect, jail

# Two tool states and an agent state; the machine narrows, `scan` and `judge` set their own.
MACHINE = """
machine = "bounded"
version = 1
initial = "scan"

[budget]
max_transitions = 10

[vars.agent]
verdict = { type = "v", default = {} }

[schemas.v]
ok = "bool"

[config.sandbox]
extra_read_paths = []
extra_write_paths = []

[states.scan]
kind = "tool"
command = ["true"]
timeout_secs = 5
on = { ok = "store", nonzero = "fail", timeout = "fail" }

[states.scan.config.sandbox]
extra_read_paths = ["/srv/inbox"]

[states.store]
kind = "tool"
command = ["true"]
timeout_secs = 5
on = { ok = "judge", nonzero = "fail", timeout = "fail" }

[states.judge]
kind = "agent"
prompt = "judge"
output_schema = "v"
capture = { finish_json = "verdict" }
timeout_secs = 60
on = { ok = "done", failed = "fail", timeout = "fail", budget_exhausted = "fail" }

[states.judge.config.sandbox]
extra_write_paths = ["/srv/outbox"]

[states.judge.config.harness]
max_iterations = 7

[states.done]
kind = "terminal"
status = "ok"
reason = "done"

[states.fail]
kind = "terminal"
status = "failed"
reason = "failed"
"""

OPERATOR = """
[providers.local]
api_format = "openai"
base_url = "http://127.0.0.1:9"

[models.worker]
provider = "local"
model = "test-model"

[sandbox]
extra_read_paths = ["/srv/inbox", "/opt/toolchain"]
extra_write_paths = ["/srv/outbox"]
"""


def _write(tmp_path: pathlib.Path, text: str, name: str = "m.asm.toml") -> pathlib.Path:
    f = tmp_path / name
    f.write_text(text, encoding="utf-8")
    return f


def _operator(text: str = OPERATOR) -> None:
    """Write the operator's own (global) config; conftest isolates the XDG dirs per test."""
    target = paths.global_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _cfg(**sandbox: Any) -> Config:
    return Config.model_validate({"sandbox": sandbox})


def _widening(operator: Config, state: Config, isolation: kinds.IsolationLevel = "strict") -> str:
    return "; ".join(sandbox_widening(operator, isolation, state, isolation))


# --- the grammar ------------------------------------------------------------------------------


def test_agent_and_tool_states_take_a_config_table(tmp_path: pathlib.Path) -> None:
    spec = load_machine(_write(tmp_path, MACHINE))
    assert spec.states["scan"].config == {"sandbox": {"extra_read_paths": ["/srv/inbox"]}}  # type: ignore[union-attr]
    assert spec.states["judge"].config["harness"] == {"max_iterations": 7}  # type: ignore[union-attr]
    assert spec.states["store"].config == {}  # type: ignore[union-attr]


def test_a_state_that_runs_nothing_takes_no_config_table(tmp_path: pathlib.Path) -> None:
    with pytest.raises(MachineError, match="config"):
        load_machine(
            _write(tmp_path, MACHINE + "\n[states.done.config.harness]\nmax_iterations = 3\n")
        )


@pytest.mark.parametrize(
    ("table", "needle"),
    [
        ('[states.judge.config.providers.x]\napi_format = "openai"\n', "providers"),
        ("[states.judge.config.mcp]\nenabled = true\n", "mcp"),
        ('[states.judge.config.machine]\npass_env = ["X"]\n', "machine"),
        ("[states.judge.config.machine]\nsnapshot_keep = 2\n", "read once for the whole machine"),
        ('[states.judge.config.parallel]\nworkdir = "/tmp/x"\n', "read once for the whole machine"),
        ('[states.judge.config.git.commit]\nname = "x"\n', "read once for the whole machine"),
        ("[states.judge.config.git]\nrun_repo_hooks = true\n", "git.run_repo_hooks"),
        (
            '[states.judge.config.prompt]\nsystem_prompt_file = "/etc/passwd"\n',
            "system_prompt_file",
        ),
    ],
)
def test_a_state_table_carries_no_operator_only_key(
    tmp_path: pathlib.Path, table: str, needle: str
) -> None:
    with pytest.raises(MachineError, match=needle):
        load_machine(_write(tmp_path, MACHINE + "\n" + table))


@pytest.mark.parametrize(
    ("table", "needle"),
    [
        ('[config.skills]\nextra_dirs = ["/tmp/skills"]\n', r"skills\.extra_dirs"),
        ('[config.parallel]\nworkdir = "/tmp/elsewhere"\n', r"parallel\.workdir"),
        ("[config.budget]\nallow_paid_credits = true\n", r"budget\.allow_paid_credits"),
        ('[states.judge.config.skills]\nextra_dirs = ["/tmp/skills"]\n', r"skills\.extra_dirs"),
        (
            "[states.judge.config.budget]\nallow_paid_credits = true\n",
            r"budget\.allow_paid_credits",
        ),
    ],
)
def test_an_overlay_reaches_neither_the_host_nor_the_operators_money(
    tmp_path: pathlib.Path, table: str, needle: str
) -> None:
    """A skill dir is read on the host, clones are made under the workdir, credits cost money."""
    with pytest.raises(MachineError, match=needle):
        load_machine(_write(tmp_path, MACHINE + "\n" + table))


def test_a_state_table_cannot_select_a_preset(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _operator()
    spec = load_machine(
        _write(
            tmp_path,
            MACHINE.replace(
                "max_iterations = 7",
                'max_iterations = 7\n\n[states.judge.config]\npreset = "ultra"',
            ),
        )
    )
    with pytest.raises(ConfigError, match="preset"):
        resolve_configs(spec, tmp_path, _setup.detect_env())


def test_a_knob_and_its_config_leaf_are_one_spelling(tmp_path: pathlib.Path) -> None:
    """Until the knobs fold into the table, a leaf set both ways is refused, never ordered."""
    both = MACHINE.replace('prompt = "judge"', 'prompt = "judge"\neffort = "high"') + (
        '\n[states.judge.config.models.worker]\neffort = "low"\n'
    )
    with pytest.raises(MachineError, match=r"`effort` and `config\.models\.worker\.effort`"):
        load_machine(_write(tmp_path, both))
    budget = MACHINE.replace('prompt = "judge"', 'prompt = "judge"\nmax_usd = 1.0') + (
        "\n[states.judge.config.budget]\nmax_usd = 2.0\n"
    )
    with pytest.raises(MachineError, match=r"`max_usd` and `config\.budget\.max_usd`"):
        load_machine(_write(tmp_path, budget))


# --- the bound, key by key --------------------------------------------------------------------


def test_every_sandbox_key_has_a_rule() -> None:
    """A new sandbox key fails here until it gets a rule."""
    assert set(_states._WITHIN) == set(SandboxConfig.model_fields)


def test_a_key_with_no_rule_must_equal_the_operators(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(_states._WITHIN, "memory_limit_mb")
    assert "sandbox.memory_limit_mb" in _widening(_cfg(), _cfg(memory_limit_mb=64))
    assert _widening(_cfg(memory_limit_mb=64), _cfg(memory_limit_mb=64)) == ""


@pytest.mark.parametrize(
    ("operator", "state"),
    [
        ({}, {}),
        ({"extra_read_paths": ["/a", "/b"]}, {"extra_read_paths": ["/a"]}),
        ({"extra_read_paths": ["/a"]}, {"extra_read_paths": []}),
        # Write implies read: a read entry may come from the operator's write list.
        ({"extra_write_paths": ["/w"]}, {"extra_write_paths": [], "extra_read_paths": ["/w"]}),
        # A path at or below a grant is within it, however either side spells it.
        ({"extra_read_paths": ["/a"]}, {"extra_read_paths": ["/a/"]}),
        ({"extra_read_paths": ["/a/"]}, {"extra_read_paths": ["/a"]}),
        ({"extra_read_paths": ["/a"]}, {"extra_read_paths": ["/a/sub"]}),
        ({"extra_write_paths": ["/w"]}, {"extra_write_paths": ["/w/sub/"]}),
        ({"extra_write_paths": ["/w"]}, {"extra_write_paths": [], "extra_read_paths": ["/w/sub"]}),
        ({"hide_paths": ["/h"]}, {"hide_paths": ["/h", "/more"]}),
        # Hiding an ancestor hides the operator's entry with it.
        ({"hide_paths": ["/h/secret"]}, {"hide_paths": ["/h"]}),
        ({"hide_paths": ["/h"]}, {"hide_paths": ["/h/"]}),
        ({"extra_device_paths": ["/dev/fuse"]}, {"extra_device_paths": ["/dev//fuse"]}),
        ({"run_commands": "yes"}, {"run_commands": "ask"}),
        ({"run_commands": "ask"}, {"run_commands": "no"}),
        ({"network": "host"}, {"network": "session"}),
        ({"network": "host"}, {"network": "auto"}),
        ({"network": "auto"}, {"network": "session"}),
        ({"memory_limit_mb": 0}, {"memory_limit_mb": 512}),
        ({"memory_limit_mb": 1024}, {"memory_limit_mb": 512}),
        ({"protect_git": False}, {"protect_git": True}),
        ({"home": "cache"}, {"home": "tmp"}),
        ({"fetch_hosts": ["*"]}, {"fetch_hosts": ["docs.python.org"]}),
        ({"fetch_hosts": ["a.example", "b.example"]}, {"fetch_hosts": ["a.example"]}),
        ({"isolation": "auto"}, {"isolation": "strict"}),
    ],
)
def test_an_overlay_may_narrow(operator: dict[str, Any], state: dict[str, Any]) -> None:
    assert _widening(_cfg(**operator), _cfg(**{**operator, **state})) == ""


@pytest.mark.parametrize(
    ("operator", "state", "leaf"),
    [
        ({}, {"extra_read_paths": ["/a"]}, "extra_read_paths"),
        ({"extra_read_paths": ["/a"]}, {"extra_read_paths": ["/a", "/b"]}, "extra_read_paths"),
        # A sibling that shares the grant's spelling as a prefix, and the grant's parent.
        ({"extra_read_paths": ["/a"]}, {"extra_read_paths": ["/ab"]}, "extra_read_paths"),
        ({"extra_read_paths": ["/a/sub"]}, {"extra_read_paths": ["/a"]}, "extra_read_paths"),
        ({"extra_read_paths": ["/a"]}, {"extra_write_paths": ["/a"]}, "extra_write_paths"),
        ({"extra_read_paths": ["/a"]}, {"extra_write_paths": ["/a/sub"]}, "extra_write_paths"),
        ({}, {"extra_device_paths": ["/dev/fuse"]}, "extra_device_paths"),
        # A device is granted by its own node, never by a directory above it.
        (
            {"extra_device_paths": ["/dev/dri"]},
            {"extra_device_paths": ["/dev/dri/card0"]},
            "extra_device_paths",
        ),
        ({"hide_paths": ["/h"]}, {"hide_paths": []}, "hide_paths"),
        ({"hide_paths": ["/h"]}, {"hide_paths": ["/other"]}, "hide_paths"),
        # Hiding a subdirectory leaves the rest of the operator's entry visible.
        ({"hide_paths": ["/h"]}, {"hide_paths": ["/h/sub"]}, "hide_paths"),
        ({"run_commands": "ask"}, {"run_commands": "yes"}, "run_commands"),
        ({"run_commands": "no"}, {"run_commands": "ask"}, "run_commands"),
        ({"network": "auto"}, {"network": "host"}, "network"),
        ({"network": "session"}, {"network": "host"}, "network"),
        ({"memory_limit_mb": 512}, {"memory_limit_mb": 0}, "memory_limit_mb"),
        ({"memory_limit_mb": 512}, {"memory_limit_mb": 1024}, "memory_limit_mb"),
        ({"protect_git": True}, {"protect_git": False}, "protect_git"),
        ({"home": "tmp"}, {"home": "cache"}, "home"),
        ({"fetch_hosts": ["a.example"]}, {"fetch_hosts": ["*"]}, "fetch_hosts"),
        ({}, {"fetch_hosts": ["a.example"]}, "fetch_hosts"),
    ],
)
def test_an_overlay_may_not_widen(
    operator: dict[str, Any], state: dict[str, Any], leaf: str
) -> None:
    problems = _widening(_cfg(**operator), _cfg(**{**operator, **state}))
    assert f"sandbox.{leaf} = " in problems and "wider than your config's" in problems


def test_a_path_is_compared_by_where_it_really_is(tmp_path: pathlib.Path) -> None:
    """A symlink inside a granted tree that points outside it is a grant of the outside."""
    granted, outside = tmp_path / "granted", tmp_path / "outside"
    (granted / "sub").mkdir(parents=True)
    outside.mkdir()
    (granted / "escape").symlink_to(outside)
    (tmp_path / "alias").symlink_to(granted)
    for leaf in ("extra_read_paths", "extra_write_paths"):
        op = _cfg(**{leaf: [str(granted)]})
        assert f"sandbox.{leaf}" in _widening(op, _cfg(**{leaf: [str(granted / "escape")]}))
        assert _widening(op, _cfg(**{leaf: [str(tmp_path / "alias" / "sub")]})) == ""
    # The operator hides the real directory; a state hiding only a link beside it hides less.
    op = _cfg(hide_paths=[str(outside)])
    assert _widening(op, _cfg(hide_paths=[str(granted / "escape")])) == ""
    assert "sandbox.hide_paths" in _widening(op, _cfg(hide_paths=[str(granted)]))


def test_isolation_is_compared_by_the_level_it_resolves_to() -> None:
    op, st = _cfg(isolation="strict"), _cfg(isolation="auto")
    assert sandbox_widening(op, "strict", st, "strict") == []
    assert "sandbox.isolation" in "; ".join(sandbox_widening(op, "strict", st, "hardened"))
    assert "sandbox.isolation" in "; ".join(
        sandbox_widening(_cfg(), "strict", _cfg(isolation="none"), "none")
    )


def test_network_is_compared_by_what_it_resolves_to() -> None:
    """Outside strict every child shares the host network, so `auto` there is not a widening."""
    assert _widening(_cfg(network="host"), _cfg(network="auto"), "hardened") == ""
    assert "sandbox.network" in _widening(_cfg(network="auto"), _cfg(network="host"), "strict")


# --- one config per state ---------------------------------------------------------------------


def test_the_machine_narrows_and_a_state_widens_back_up_to_the_operators(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _operator()
    configs = resolve_configs(
        load_machine(_write(tmp_path, MACHINE)), tmp_path, _setup.detect_env()
    )
    assert configs.operator.config.sandbox.extra_read_paths == ("/srv/inbox", "/opt/toolchain")
    assert configs.machine.config.sandbox.extra_read_paths == ()
    assert configs.states["scan"].config.sandbox.extra_read_paths == ("/srv/inbox",)
    assert configs.states["scan"].config.sandbox.extra_write_paths == ()
    # A state with no table of its own runs under the machine's.
    assert configs.states["store"] is configs.machine
    judge = configs.states["judge"].config
    assert (
        judge.sandbox.extra_write_paths == ("/srv/outbox",) and judge.sandbox.extra_read_paths == ()
    )
    assert judge.harness.max_iterations == 7
    assert configs.machine.config.harness.max_iterations != 7
    assert set(configs.states) == {"scan", "store", "judge"}


def test_a_state_wider_than_the_operator_is_refused_by_name(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _operator(OPERATOR.replace('extra_write_paths = ["/srv/outbox"]', "extra_write_paths = []"))
    spec = load_machine(_write(tmp_path, MACHINE))
    with pytest.raises(SandboxWidenedError) as refused:
        resolve_configs(spec, tmp_path, _setup.detect_env())
    said = str(refused.value)
    assert said.startswith("[states.judge.config] sets sandbox.extra_write_paths = ['/srv/outbox']")
    assert "wider than your config's []" in said and "agent6 config set --repo" in said


def test_the_ceiling_is_the_operators_layers_alone(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A grant the machine's own [config] writes is not a ceiling a state can then match."""
    monkeypatch.chdir(tmp_path)
    _operator(OPERATOR.replace('extra_read_paths = ["/srv/inbox", "/opt/toolchain"]', ""))
    widened = MACHINE.replace(
        "[config.sandbox]\nextra_read_paths = []",
        '[config.sandbox]\nextra_read_paths = ["/srv/inbox"]',
    )
    with pytest.raises(
        SandboxWidenedError, match=r"the machine's \[config\] sets sandbox\.extra_read_paths"
    ):
        resolve_configs(load_machine(_write(tmp_path, widened)), tmp_path, _setup.detect_env())


def test_a_config_flag_file_is_part_of_the_ceiling(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _operator(OPERATOR.replace('extra_write_paths = ["/srv/outbox"]', "extra_write_paths = []"))
    ops = _write(tmp_path, '[sandbox]\nextra_write_paths = ["/srv/outbox"]\n', name="ops.toml")
    spec = load_machine(_write(tmp_path, MACHINE))
    configs = resolve_configs(spec, tmp_path, _setup.detect_env(), config_path=ops)
    assert configs.states["judge"].config.sandbox.extra_write_paths == ("/srv/outbox",)
    # A state must not be able to rewrite its own ceiling: the file is read-only in every jail.
    assert ops.resolve() in machine_protect_paths(tmp_path / "m.asm.toml", tmp_path, ops)
    assert ops.resolve() not in machine_protect_paths(tmp_path / "m.asm.toml", tmp_path)


def test_the_operator_flags_apply_after_the_overlays(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _operator(OPERATOR + 'run_commands = "ask"\n')
    spec = load_machine(
        _write(tmp_path, MACHINE + '\n[states.store.config.sandbox]\nrun_commands = "no"\n')
    )
    configs = resolve_configs(spec, tmp_path, _setup.detect_env(), auto_approve=True)
    assert configs.states["scan"].config.sandbox.run_commands == "yes"
    # A state's own `no` stays: a flag never grants what the standing policy denied.
    assert configs.states["store"].config.sandbox.run_commands == "no"


# --- the two enforcement sites read the state's own config ------------------------------------


def test_each_tool_state_gets_its_own_jail_policy(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _operator(OPERATOR + 'network = "host"\n')
    machine = (
        MACHINE.replace("[config.sandbox]\n", '[config.sandbox]\nnetwork = "session"\n')
        + '\n[states.store.config.sandbox]\nnetwork = "host"\n'
    )
    configs = resolve_configs(
        load_machine(_write(tmp_path, machine)), tmp_path, _setup.detect_env()
    )
    build = run.machine_tool_policy_factory(
        configs.states, tmp_path, protect_paths=(), data_dir=None
    )
    scan, store = build("scan", ("true",), 5.0, ()), build("store", ("true",), 5.0, ())
    assert pathlib.Path("/srv/inbox") in scan.extra_ro_paths
    assert pathlib.Path("/srv/inbox") not in store.extra_ro_paths
    assert pathlib.Path("/opt/toolchain") not in scan.extra_ro_paths
    if configs.machine.isolation == "strict":
        assert (scan.network, store.network) == ("none", "host")
    with pytest.raises(jail.JailUnavailableError, match="no config was resolved for state 'ghost'"):
        build("ghost", ("true",), 5.0, ())


def test_each_agent_state_gets_its_own_overlay(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _operator()
    seen: dict[str, machine_agent.StateOverlay] = {}

    def fake_build(
        overlays: dict[str, machine_agent.StateOverlay], *_a: object, **_k: object
    ) -> Any:
        seen.update(overlays)

        def run_agent(*_a: object, **_k: object) -> AgentExecResult:
            return AgentExecResult(reason="finish_session", payload={"ok": True})

        return run_agent

    monkeypatch.setattr(machine_agent, "build_machine_agent_runner", fake_build)

    def ran(policy: kinds.JailPolicy) -> kinds.CommandResult:
        return kinds.CommandResult(
            argv=policy.argv, returncode=0, stdout="", stderr="", duration_s=0.0
        )

    monkeypatch.setattr(jail, "run_in_jail", ran)
    said: list[str] = []
    reporter = app_reporter.Reporter(out=said.append, err=said.append)
    assert run.run_machine(_write(tmp_path, MACHINE), reporter) == 0, said
    assert set(seen) == {"judge"}
    sandbox = seen["judge"].overlay["sandbox"]
    assert sandbox["extra_write_paths"] == ["/srv/outbox"] and sandbox["extra_read_paths"] == []
    assert seen["judge"].overlay["harness"]["max_iterations"] == 7


def test_an_agent_request_naming_no_state_ends_as_an_error(tmp_path: pathlib.Path) -> None:
    from agent6.machine import AgentRequest

    runner = machine_agent.build_machine_agent_runner(
        {"judge": machine_agent.StateOverlay({}, "none")}, tmp_path, tmp_path / "tr"
    )
    result = runner(AgentRequest(prompt="p", timeout_s=5.0, state_name="ghost"), None)
    assert result.reason == "error: no config was resolved for state 'ghost'"


# --- the refusal reaches the operator at check and at run ------------------------------------


def test_run_and_check_refuse_a_widening_before_anything_runs(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    _operator(OPERATOR.replace('extra_read_paths = ["/srv/inbox", "/opt/toolchain"]', ""))
    ran: list[kinds.JailPolicy] = []
    monkeypatch.setattr(jail, "run_in_jail", ran.append)
    mfile = _write(tmp_path, MACHINE)
    assert main(["machine", "check", str(mfile)]) == 1
    assert (
        "[states.scan.config] sets sandbox.extra_read_paths = ['/srv/inbox']"
        in capsys.readouterr().err
    )
    reporter = mock.MagicMock()
    assert run.run_machine(mfile, reporter) == 2
    said = " ".join(str(c.args[0]) for c in reporter.refuse.call_args_list)
    assert "[states.scan.config] sets sandbox.extra_read_paths = ['/srv/inbox']" in said
    assert ran == []


def test_check_prints_each_sandbox_overlay(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    _operator()
    assert main(["machine", "check", str(_write(tmp_path, MACHINE))]) == 0
    out = capsys.readouterr().out
    assert (
        "[config] sets the sandbox of every state: extra_read_paths = [], extra_write_paths = []"
        in out
    )
    assert "[states.scan] sets its own sandbox: extra_read_paths = ['/srv/inbox']" in out
    assert "[states.judge] sets its own sandbox: extra_write_paths = ['/srv/outbox']" in out


def test_a_state_this_host_cannot_honor_refuses_by_name(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each state's config passes the host checks a plain run's does, at startup."""
    monkeypatch.chdir(tmp_path)
    _operator()
    hardened = dataclasses.replace(_setup.detect_env(), userns_supported=False)
    if hardened.detected_isolation != "hardened":
        pytest.skip("this host cannot stand in for a hardened one")
    monkeypatch.setattr(_setup, "detect_env", lambda: hardened)
    mfile = _write(tmp_path, MACHINE + '\n[states.store.config.sandbox]\nnetwork = "session"\n')
    reporter = mock.MagicMock()
    assert run.run_machine(mfile, reporter) == 2
    said = " ".join(str(c.args[0]) for c in reporter.refuse.call_args_list)
    assert said.startswith("[states.store.config]: sandbox.network = 'session' requires the strict")


def test_resolve_configs_keeps_the_resolved_isolation_per_state(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _operator()
    env = _setup.detect_env()
    configs = resolve_configs(load_machine(_write(tmp_path, MACHINE)), tmp_path, env)
    expected = detect.resolve_isolation("auto", env)
    assert {s.isolation for s in configs.states.values()} == {expected}
    assert isinstance(configs.states["scan"], StateConfig)
