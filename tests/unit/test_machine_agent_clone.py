# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A mode="run" machine state works a fresh clone at the machine chain's tip.

The chain ref carries state-to-state continuation and the `agent6/machine-<id>` branch tracks the
same tip; the operator's checkout is never touched.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
from typing import Any, ClassVar, Literal

import pytest

from agent6 import errors, git_ops
from agent6.app import machine_agent as ma
from agent6.app import providers, reporter
from agent6.app.machine import StateConfig
from agent6.config import Config
from agent6.machine import AgentRequest
from agent6.providers import ProviderResponse
from agent6.sandbox import jail as sandbox_jail

BRANCH = "agent6/machine-m1"
CHAIN = git_ops.chain_ref_for("machine-m1")


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _origin(tmp_path: pathlib.Path) -> pathlib.Path:
    repo = tmp_path / "origin"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


_REAL_POPEN = subprocess.Popen


def _fake_popen(argv: list[str], **kw: Any) -> Any:
    """Fake only the machine-agent spawn; git's own Popen calls stay real."""
    if not any("machine_agent" in str(a) for a in argv):
        return _REAL_POPEN(argv, **kw)
    return _FakeChild(argv, **kw)


class _FakeChild:
    """Stand in for the machine-agent subprocess: commit one file in cwd, write result.json."""

    captured_env: ClassVar[dict[str, str]] = {}
    workdirs: ClassVar[list[pathlib.Path]] = []
    seen_cwds: ClassVar[list[pathlib.Path]] = []
    seen_files: ClassVar[list[set[str]]] = []

    def __init__(self, argv: list[str], **kw: Any) -> None:
        _FakeChild.captured_env = dict(kw.get("env") or {})
        req = json.loads(pathlib.Path(argv[-2]).read_text(encoding="utf-8"))
        # `root` is the execution root (a clone); `cwd` stays the checkout for the config reload.
        cwd = pathlib.Path(req["root"])
        _FakeChild.workdirs.append(cwd)
        _FakeChild.seen_cwds.append(pathlib.Path(req["cwd"]))
        _FakeChild.seen_files.append({e.name for e in cwd.iterdir()})
        seq = req["request"]["step_seq"]
        if req["request"]["mode"] == "run":
            # Mirror the nested loop: commit, advance the chain ref, leave HEAD.
            (cwd / f"work{seq}.txt").write_text(f"state {seq}\n", encoding="utf-8")
            _git(cwd, "add", "-A")
            _git(cwd, "commit", "-q", "-m", f"agent6 iter 1: state {seq}")
            _git(cwd, "update-ref", CHAIN, "HEAD")
        pathlib.Path(argv[-1]).write_text(
            json.dumps({"reason": "finish_session", "payload": None, "usd": 0.0}),
            encoding="utf-8",
        )
        self.pid = 4242
        self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        return 0


def _runner(origin: pathlib.Path, tmp_path: pathlib.Path):  # type: ignore[no-untyped-def]
    return ma.build_machine_agent_runner(
        _OVERLAYS,
        origin,
        tmp_path / "m1" / "agent_transcripts",
        (),
        None,
        machine_id="m1",
        clone_root=tmp_path / "clones",
    )


# The two states `_req` names, each under an empty overlay at isolation none.
_OVERLAYS = {f"s{seq}": ma.StateOverlay({}, "none") for seq in (0, 1)}


def _req(seq: int, mode: str = "run") -> AgentRequest:
    return AgentRequest(prompt="p", timeout_s=60.0, mode=mode, state_name=f"s{seq}", step_seq=seq)


def test_run_states_continue_the_machine_branch_and_never_touch_the_checkout(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    origin = _origin(tmp_path)
    monkeypatch.setattr(ma.subprocess, "Popen", _fake_popen)
    runner = _runner(origin, tmp_path)

    r1 = runner(_req(0), None)
    assert r1.reason == "finish_session"
    r2 = runner(_req(1), None)
    assert r2.reason == "finish_session"

    # State 1 built on state 0's tree, and the visible branch tracks the chain's tip.
    files = _git(origin, "ls-tree", "--name-only", BRANCH)
    assert "work0.txt" in files and "work1.txt" in files
    assert git_ops.chain_tip(origin, CHAIN) == git_ops.chain_tip(origin, BRANCH)
    # The operator's checkout never moved and holds neither file.
    assert _git(origin, "branch", "--show-current") == "main"
    assert not (origin / "work0.txt").exists()
    # Clones are gone once landed; the subprocess ran subordinate.
    assert not any((tmp_path / "clones").glob("state-*"))
    assert _FakeChild.captured_env["AGENT6_SUBRUN"] == "1"


def test_run_state_request_keeps_the_true_checkout_as_cwd_for_config(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run state's request keeps the operator's repo as `cwd`; `root` is the clone it works in."""
    origin = _origin(tmp_path)
    monkeypatch.setattr(ma.subprocess, "Popen", _fake_popen)
    _FakeChild.workdirs = []
    _FakeChild.seen_cwds = []
    runner = _runner(origin, tmp_path)

    assert runner(_req(0), None).reason == "finish_session"

    assert _FakeChild.seen_cwds[-1] == origin
    assert _FakeChild.workdirs[-1] != origin


def test_read_only_states_see_the_machine_tree_and_land_nothing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read-only judge in a machine with run states runs in a fresh clone at the chain tip.

    It commits nothing, so the branch does not move and its clone is cleaned up.
    """
    origin = _origin(tmp_path)
    monkeypatch.setattr(ma.subprocess, "Popen", _fake_popen)
    _FakeChild.workdirs = []
    _FakeChild.seen_files = []
    runner = _runner(origin, tmp_path)
    assert runner(_req(0), None).reason == "finish_session"
    tip_after_run = git_ops.chain_tip(origin, BRANCH)
    assert runner(_req(1, mode="agent"), None).reason == "finish_session"
    judge_dir = _FakeChild.workdirs[-1]
    assert judge_dir != origin
    assert "work0.txt" in _FakeChild.seen_files[-1]  # the run state's committed work
    assert git_ops.chain_tip(origin, BRANCH) == tip_after_run  # nothing landed
    assert not any((tmp_path / "clones").glob("state-*"))  # cleaned up


def test_without_a_machine_tree_read_only_states_run_in_place(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no machine_id or clone_root a read-only request runs in cwd."""
    origin = _origin(tmp_path)
    monkeypatch.setattr(ma.subprocess, "Popen", _fake_popen)
    _FakeChild.workdirs = []
    runner = ma.build_machine_agent_runner(
        _OVERLAYS, origin, tmp_path / "m1" / "agent_transcripts", (), None
    )
    assert runner(_req(0, mode="agent"), None).reason == "finish_session"
    assert _FakeChild.workdirs == [origin]
    assert not (tmp_path / "clones").exists()


def test_machine_tool_runner_runs_each_call_in_the_machine_tree(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tool state's jail mounts a clone of the chain tip, so it sees the run states' commits.

    The runner remaps the bundle protect paths to the clone and discards the tree after.
    """
    from agent6 import kinds
    from agent6.app.machine import run as machine_run

    origin = _origin(tmp_path)
    _git(origin, "checkout", "-q", "-b", "scratch")
    (origin / "work.txt").write_text("landed\n", encoding="utf-8")
    _git(origin, "add", "-A")
    _git(origin, "commit", "-q", "-m", "agent6 iter 1: work")
    _git(origin, "update-ref", CHAIN, "HEAD")
    _git(origin, "checkout", "-q", "main")
    _git(origin, "branch", "-D", "scratch")
    assert not (origin / "work.txt").exists()

    captured: dict[str, object] = {}

    def fake_jail(policy: kinds.JailPolicy) -> kinds.CommandResult:
        captured["cwd"] = policy.cwd
        captured["saw_work"] = (policy.cwd / "work.txt").exists()
        captured["protect"] = policy.extra_protect_paths
        return kinds.CommandResult(
            argv=policy.argv, returncode=0, stdout="", stderr="", duration_s=0.0
        )

    monkeypatch.setattr(sandbox_jail, "run_in_jail", fake_jail)
    runner = machine_run.machine_tool_runner(
        origin,
        "m1",
        tmp_path / "clones",
        {"check": StateConfig(Config(), "none")},
        reporter.STDIO_REPORTER,
        tool_policy=lambda state, argv, timeout, env: kinds.JailPolicy(cwd=origin, argv=argv),
    )
    policy = kinds.JailPolicy(cwd=origin, argv=("x",), extra_protect_paths=(origin / "scripts",))
    assert runner("check", policy, None).returncode == 0
    clone_cwd = captured["cwd"]
    assert isinstance(clone_cwd, pathlib.Path) and clone_cwd != origin
    assert captured["saw_work"] is True
    assert captured["protect"] == (clone_cwd / "scripts",)
    assert not clone_cwd.exists()  # scratch tree, discarded


@pytest.mark.parametrize("fail_setup", [False, True])
def test_tool_clone_runs_setup_before_reading_environment(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, fail_setup: bool
) -> None:
    """A tool clone runs the setup command already resolved by preflight."""
    from agent6 import kinds, machine
    from agent6.app.machine import run as machine_run

    origin = _origin(tmp_path)
    command = "echo broken >&2; exit 7" if fail_setup else "echo ready > environment.txt"
    cfg = Config.model_validate({"sandbox": {"run_commands": "yes"}}).with_setup_command(
        ("sh", "-c", command)
    )
    expected_setup = cfg.parallel.setup_command
    calls: list[tuple[str, ...]] = []

    def execute(policy: kinds.JailPolicy, **kwargs: object) -> kinds.CommandResult:
        calls.append(policy.argv)
        result = subprocess.run(
            policy.argv, cwd=policy.cwd, capture_output=True, text=True, check=False
        )
        return kinds.CommandResult(
            argv=policy.argv,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_s=2.5,
        )

    monkeypatch.setattr(sandbox_jail, "run_in_jail", execute)
    runner = machine_run.machine_tool_runner(
        origin,
        "m1",
        tmp_path / "clones",
        {"check": StateConfig(cfg, "none")},
        reporter.STDIO_REPORTER,
        tool_policy=lambda state, argv, timeout, env: kinds.JailPolicy(cwd=origin, argv=argv),
    )
    outcomes: list[kinds.CommandResult] = []

    def recorded_runner(
        state: str, policy: kinds.JailPolicy, events_log: pathlib.Path | None
    ) -> kinds.CommandResult:
        outcome = runner(state, policy, events_log)
        outcomes.append(outcome)
        return outcome

    spec_path = tmp_path / "m.asm.toml"
    spec_path.write_text(
        """
machine = "m1"
version = 1
initial = "check"
[budget]
max_transitions = 5
[states.check]
kind = "tool"
command = ["cat", "environment.txt"]
timeout_s = 5
on = { ok = "done", nonzero = "fail", timeout = "fail" }
[states.done]
kind = "terminal"
status = "ok"
reason = "ready"
[states.fail]
kind = "terminal"
status = "failed"
reason = "setup failed"
""",
        encoding="utf-8",
    )
    journal = machine.MachineJournal(tmp_path / "instance")
    failures: list[machine.ToolExecResult] = []
    world = machine.LiveWorld(
        cwd=origin,
        journal=journal,
        jail_runner=recorded_runner,
        tool_policy=lambda state, argv, timeout, env: kinds.JailPolicy(cwd=origin, argv=argv),
        on_tool_failure=lambda state, outcome: failures.append(outcome),
        state_log_root=tmp_path / "logs",
        state_log_keep=3,
    )
    result = machine.drive(machine.load_machine(spec_path), journal, world, live=True)
    assert result.status == ("failed" if fail_setup else "ok")
    assert calls.count(expected_setup) == 1
    assert len(calls) == (1 if fail_setup else 2)
    if fail_setup:
        assert "broken" in failures[0].stderr
        assert failures[0].exit_code == 7
        assert outcomes[0].argv == expected_setup
        assert outcomes[0].duration_s == 2.5
    logs = list((tmp_path / "logs").glob("*/logs.jsonl"))
    assert len(logs) == 1
    event = json.loads(logs[0].read_text())
    assert event["type"] == "workspace.setup"
    assert event["cmd"] == list(expected_setup)
    assert event["exit_code"] == (7 if fail_setup else 0)
    assert event["duration_s"] == 2.5
    assert not (origin / "environment.txt").exists()


def test_a_tool_states_setup_runs_without_its_pass_env(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setup runs the tree's code at the chain tip, so the tool's `pass_env` never reaches it.

    The tool command, the operator's argv, still receives the variables the state declares.
    """
    from agent6 import kinds
    from agent6.app.machine import run as machine_run

    origin = _origin(tmp_path)
    cfg = Config.model_validate({"sandbox": {"run_commands": "yes"}}).with_setup_command(
        ("sh", "-c", "true")
    )
    states = {"check": StateConfig(cfg, "none")}
    factory = machine_run.machine_tool_policy_factory(
        states, origin, protect_paths=(), data_dir=None, shared_writes=()
    )
    monkeypatch.setenv("X_TOKEN", "s3cret")
    seen: list[tuple[tuple[str, ...], bool]] = []

    def execute(policy: kinds.JailPolicy) -> kinds.CommandResult:
        seen.append((policy.argv, "X_TOKEN" in dict(policy.env)))
        return kinds.CommandResult(
            argv=policy.argv, returncode=0, stdout="", stderr="", duration_s=0.0
        )

    monkeypatch.setattr(sandbox_jail, "run_in_jail", execute)
    runner = machine_run.machine_tool_runner(
        origin, "m1", tmp_path / "clones", states, reporter.STDIO_REPORTER, tool_policy=factory
    )
    policy = factory("check", ("cat", "x"), 5.0, ("X_TOKEN",))
    assert runner("check", policy, None).returncode == 0
    assert seen == [(("sh", "-c", "true"), False), (("cat", "x"), True)]


@pytest.mark.parametrize("state", ["", "unknown"])
def test_tool_clone_unknown_state_is_an_engine_error(tmp_path: pathlib.Path, state: str) -> None:
    """An unknown state is a wiring error, never a synthetic command exit."""
    from agent6 import kinds, machine
    from agent6.app.machine import run as machine_run

    runner = machine_run.machine_tool_runner(
        tmp_path,
        "m1",
        tmp_path / "clones",
        {},
        reporter.STDIO_REPORTER,
        tool_policy=lambda state, argv, timeout, env: kinds.JailPolicy(cwd=tmp_path, argv=argv),
    )
    with pytest.raises(machine.EngineError, match="no config was resolved"):
        runner(state, kinds.JailPolicy(cwd=tmp_path, argv=("true",)), None)
    assert not (tmp_path / "clones").exists()


def test_tool_clone_host_failure_aborts_the_machine(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed clone is a host failure, not the tool state's nonzero edge."""
    from agent6 import kinds, machine
    from agent6.app.machine import run as machine_run
    from agent6.harness import subrun

    origin = _origin(tmp_path)

    def fail_clone(*args: object) -> None:
        raise subrun.SubrunError("clone unavailable")

    monkeypatch.setattr(ma, "clone_at_machine_chain", fail_clone)
    world = machine.LiveWorld(
        cwd=origin,
        journal=machine.MachineJournal(tmp_path / "instance"),
        jail_runner=machine_run.machine_tool_runner(
            origin,
            "m1",
            tmp_path / "clones",
            {"check": StateConfig(Config(), "none")},
            reporter.STDIO_REPORTER,
            tool_policy=lambda state, argv, timeout, env: kinds.JailPolicy(cwd=origin, argv=argv),
        ),
        tool_policy=lambda state, argv, timeout, env: kinds.JailPolicy(cwd=origin, argv=argv),
    )
    with pytest.raises(machine.EngineError, match="clone unavailable"):
        world.run_tool(("true",), 5, state="check")


def test_tool_setup_logs_share_numeric_state_log_pruning(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tool setup logs obey state_log_keep across the four-digit sequence boundary."""
    from agent6 import kinds, machine
    from agent6.app.machine import run as machine_run

    origin = _origin(tmp_path)
    cfg = Config.model_validate({"sandbox": {"run_commands": "yes"}}).with_setup_command(("true",))

    def execute(policy: kinds.JailPolicy) -> kinds.CommandResult:
        return kinds.CommandResult(
            argv=policy.argv, returncode=0, stdout="", stderr="", duration_s=0
        )

    monkeypatch.setattr(sandbox_jail, "run_in_jail", execute)
    world = machine.LiveWorld(
        cwd=origin,
        journal=machine.MachineJournal(tmp_path / "instance"),
        jail_runner=machine_run.machine_tool_runner(
            origin,
            "m1",
            tmp_path / "clones",
            {"check": StateConfig(cfg, "none")},
            reporter.STDIO_REPORTER,
            tool_policy=lambda state, argv, timeout, env: kinds.JailPolicy(cwd=origin, argv=argv),
        ),
        tool_policy=lambda state, argv, timeout, env: kinds.JailPolicy(cwd=origin, argv=argv),
        state_log_root=tmp_path / "states",
        state_log_keep=3,
    )
    for seq in (9998, 9999, 10000, 10001):
        world.run_tool(("true",), 5, state="check", step_seq=seq)
    assert sorted(p.name for p in (tmp_path / "states").iterdir()) == [
        "10000-check",
        "10001-check",
        "9999-check",
    ]


@pytest.mark.parametrize("failure", ["timeout", "missing", "jail", "success"])
def test_tool_setup_routes_its_outcome_without_checking_a_session_gate(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Setup exit outcomes take named edges, host failures abort, and tool clones have no gate."""
    from agent6 import kinds, machine
    from agent6.app.machine import run as machine_run

    origin = _origin(tmp_path)
    cfg = Config.model_validate(
        {
            "sandbox": {"run_commands": "yes"},
            "parallel": {"setup_command": ["setup"]},
            "harness": {"verify_command": [".missing-gate"]},
        }
    )
    calls: list[tuple[str, ...]] = []

    def execute(policy: kinds.JailPolicy) -> kinds.CommandResult:
        calls.append(policy.argv)
        if failure == "jail":
            raise sandbox_jail.JailUnavailableError("no namespace")
        return kinds.CommandResult(
            argv=policy.argv,
            returncode={"timeout": 124, "missing": 127}.get(failure, 0),
            stdout="",
            stderr="command not found" if failure == "missing" else "",
            duration_s=policy.timeout_s,
            exec_failed=failure == "missing",
        )

    monkeypatch.setattr(sandbox_jail, "run_in_jail", execute)
    runner = machine_run.machine_tool_runner(
        origin,
        "m1",
        tmp_path / "clones",
        {"check": StateConfig(cfg, "none")},
        reporter.STDIO_REPORTER,
        tool_policy=lambda state, argv, timeout, env: kinds.JailPolicy(cwd=origin, argv=argv),
    )
    policy = kinds.JailPolicy(cwd=origin, argv=("tool",), timeout_s=3)
    spec_path = tmp_path / "m.asm.toml"
    spec_path.write_text(
        """
machine = "m1"
version = 1
initial = "check"
[budget]
max_transitions = 5
[states.check]
kind = "tool"
command = ["tool"]
timeout_s = 3
on = { ok = "done", nonzero = "error", timeout = "timed_out" }
[states.done]
kind = "terminal"
status = "ok"
reason = "done"
[states.error]
kind = "terminal"
status = "failed"
reason = "error edge"
[states.timed_out]
kind = "terminal"
status = "failed"
reason = "timeout edge"
""",
        encoding="utf-8",
    )
    journal = machine.MachineJournal(tmp_path / "instance")
    failures: list[machine.ToolExecResult] = []
    world = machine.LiveWorld(
        cwd=origin,
        journal=journal,
        jail_runner=runner,
        tool_policy=lambda state, argv, timeout, env: policy,
        on_tool_failure=lambda state, outcome: failures.append(outcome),
    )
    if failure == "jail":
        with pytest.raises(machine.EngineError, match="no namespace"):
            machine.drive(machine.load_machine(spec_path), journal, world, live=True)
        return
    result = machine.drive(machine.load_machine(spec_path), journal, world, live=True)
    assert result.state == {"success": "done", "timeout": "timed_out"}.get(failure, "error")
    assert ("tool",) in calls if failure == "success" else ("tool",) not in calls


def test_invalid_utf8_result_routes_failed_instead_of_escaping(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Arbitrary result bytes from a killed child read like a missing result: the failed edge."""
    origin = _origin(tmp_path)

    class _BadResultChild:
        pid = 4242
        returncode = 0

        def __init__(self, argv: list[str], **_kwargs: object) -> None:
            pathlib.Path(argv[-1]).write_bytes(b"\xff\xfe")

        def wait(self, timeout: float | None = None) -> int:
            return 0

    def bad_result(argv: list[str], **kwargs: Any) -> Any:
        if any("machine_agent" in str(arg) for arg in argv):
            return _BadResultChild(argv, **kwargs)
        return _REAL_POPEN(argv, **kwargs)

    monkeypatch.setattr(ma.subprocess, "Popen", bad_result)
    runner = ma.build_machine_agent_runner(_OVERLAYS, origin, tmp_path / "transcripts")

    assert runner(_req(0, mode="agent"), None).reason == "error"


def test_spawn_failure_routes_failed_and_discards_the_unused_clone(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child that cannot start is a failed state, not an escaping host exception."""
    origin = _origin(tmp_path)

    def fail_spawn(argv: list[str] | tuple[str, ...], **kwargs: Any) -> Any:
        if any("machine_agent" in str(arg) for arg in argv):
            raise OSError("process table full")
        return _REAL_POPEN(argv, **kwargs)

    monkeypatch.setattr(ma.subprocess, "Popen", fail_spawn)
    result = _runner(origin, tmp_path)(_req(0), None)

    assert result.reason.startswith("error: machine agent failed to start:")
    assert not any((tmp_path / "clones").glob("state-*"))


def test_import_failure_keeps_the_clone_and_routes_failed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A state whose commits did not land routes failed; the clone is kept for adopt and prune."""
    origin = _origin(tmp_path)
    monkeypatch.setattr(ma.subprocess, "Popen", _fake_popen)

    def _boom(*_a: object, **_k: object) -> None:
        raise errors.GitError("refs locked")

    monkeypatch.setattr(git_ops, "fetch_branch", _boom)
    runner = _runner(origin, tmp_path)
    r = runner(_req(0), None)
    assert r.reason.startswith("import of")
    assert r.payload is None
    assert git_ops.chain_tip(origin, CHAIN) is None  # nothing landed
    assert list((tmp_path / "clones").glob("state-*"))  # evidence kept


class _Finishes:
    """A provider whose every call finishes the state."""

    calls = 0

    def call(self, **_kw: Any) -> ProviderResponse:
        self.calls += 1
        tools = (
            {"type": "tool_use", "id": "t1", "name": "finish_session", "input": {"summary": "ok"}},
        )
        return ProviderResponse(
            text="",
            tool_uses=tools,
            stop_reason="tool_use",
            input_tokens=1,
            output_tokens=1,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            raw={"content": list(tools)},
        )


def _state_in_a_clone(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    setup: list[str],
    mode: Literal["run", "agent", "machine"] = "run",
) -> tuple[ma.AgentExecResult, _Finishes, pathlib.Path, pathlib.Path]:
    """Run one run-mode state through `run_one` in a fresh clone, with a setup command."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("PROBE_KEY", "sk-probe")
    origin = _origin(tmp_path)
    clone = tmp_path / "clones" / "state-0000"
    ma.clone_at_machine_chain(origin, clone, CHAIN)
    prov = _Finishes()

    def _factory(*_a: Any, **_k: Any) -> _Finishes:
        return prov

    monkeypatch.setattr(providers, "build_role_provider", _factory)
    req = ma.MachineAgentRequest(
        cwd=origin,
        root=clone,
        overlay={
            "sandbox": {"isolation": "none", "run_commands": "yes"},
            "models": {"worker": {"provider": "p", "model": "m"}},
            "providers": {
                "p": {
                    "api_format": "openai",
                    "base_url": "http://127.0.0.1:1",
                    "api_key_env": "PROBE_KEY",
                }
            },
            "parallel": {"setup_command": setup},
        },
        isolation="none",
        transcript_dir=tmp_path / "m1" / "agent_transcripts",
        request=AgentRequest(prompt="p", timeout_s=60.0, mode=mode, state_name="s0", step_seq=0),
    )
    return ma.run_one(req), prov, origin, clone


@pytest.mark.parametrize("mode", ["run", "agent", "machine"])
def test_a_state_runs_the_setup_command_in_its_fresh_clone(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: Literal["run", "agent", "machine"],
) -> None:
    """A state's clone holds tracked files only: with no setup step it had no environment."""
    result, prov, origin, clone = _state_in_a_clone(
        tmp_path, monkeypatch, ["sh", "-c", "echo ready > .setup-ran"], mode
    )

    assert result.reason == "finish_session" and prov.calls == 1
    assert (clone / ".setup-ran").read_text(encoding="utf-8") == "ready\n"
    assert not (origin / ".setup-ran").exists()


@pytest.mark.parametrize("mode", ["run", "agent", "machine"])
def test_a_state_whose_setup_command_fails_makes_no_model_call(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: Literal["run", "agent", "machine"],
) -> None:
    """A state going on after a failed setup would spend its turns in an unbuilt clone."""
    result, prov, _origin_dir, _clone = _state_in_a_clone(
        tmp_path, monkeypatch, ["sh", "-c", "exit 3"], mode
    )

    assert result.reason == "setup_command_failed" and prov.calls == 0
