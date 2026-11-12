from __future__ import annotations

import asyncio
import json
import pathlib
import subprocess
from unittest import mock

import pytest

from agent6 import evidence, kinds, paths
from agent6.app import session_runtime, stamps
from agent6.config import Config, RoleModel
from agent6.harness import loop, run_chain
from agent6.sandbox import jail, jail_session
from agent6.sessions import event_log, journal, layout, manifest
from agent6.tools import operator_prompts
from agent6.ui.cli import run as cli_run
from agent6.ui.cli import sessions_show
from agent6.ui.tui import app as tui_app
from agent6.ui.web import review_routes
from agent6.viewmodel import receipt, wire
from tests.unit import test_harness

_ENVIRONMENT_COMMAND = (
    "/bin/sh",
    "-c",
    "set -eu; pwd -P; /usr/bin/id -u; printf '%s\\n' \"$HOME\"",
)


@pytest.fixture
def saved(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> layout.SessionLayout:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.chdir(root)
    for args in (("init", "-q"), ("commit", "-qm", "seed", "--allow-empty")):
        subprocess.run(("git", *args), check=True, capture_output=True)
    saved = layout.SessionLayout(paths.state_dir(root), "environment")
    saved.ensure()
    stamps.write_manifest(
        saved.manifest_path, manifest.SessionManifest(session_id=saved.session_id)
    )
    return saved


def _tools(
    saved: layout.SessionLayout,
    isolation: kinds.IsolationLevel = "none",
    *,
    verify_command: tuple[str, ...] = ("/bin/true",),
    verify_retries: int = 2,
) -> session_runtime.SessionTools:
    cfg = Config.model_validate(
        {
            "sandbox": {"isolation": isolation, "network": "auto", "run_commands": "allow"},
            "harness": {
                "verify_command": verify_command,
                "verify_retries": verify_retries,
                "verify_at_start": True,
                "verify_when": "finish",
                "command_checkin_s": 0,
                "max_turns": 2,
            },
            "review": {"trigger": "off"},
        }
    )
    return session_runtime.build_session_tools(
        cfg,
        cwd=pathlib.Path.cwd(),
        state_dir=saved.state_dir,
        layout=saved,
        isolation=isolation,
        mode="run",
        events=event_log.EventSink(saved.logs_path),
        prompts=operator_prompts.OperatorPrompts(),
        loop_log=lambda _line: None,
        mcp_manager=None,
        rm_role=RoleModel(provider="test", model="test"),
    )


@pytest.mark.parametrize(
    "isolation", [pytest.param("full", marks=pytest.mark.needs_namespaces), "partial", "none"]
)
def test_setup_baseline_and_repeated_executions_keep_their_target(
    saved: layout.SessionLayout, jail_bin: pathlib.Path, isolation: kinds.IsolationLevel
) -> None:
    """Successful checks of the same tree retain their distinct execution targets."""
    expected_tree = subprocess.check_output(("git", "rev-parse", "HEAD^{tree}"), text=True).strip()
    assert expected_tree
    for _ in range(2):
        tools = _tools(saved, isolation, verify_command=_ENVIRONMENT_COMMAND)
        dispatcher = tools.dispatcher
        try:
            dispatcher.run_setup(_ENVIRONMENT_COMMAND)
            provider = mock.Mock()
            provider.call.side_effect = [
                test_harness._tool_resp("run_verify_command"),
                test_harness._tool_resp("finish_session", {"summary": "checked"}, tool_id="done"),
            ]
            harness = loop.Harness(
                chain=run_chain.RunChain(
                    pathlib.Path.cwd(),
                    per_turn=False,
                    base_sha=subprocess.check_output(
                        ("git", "rev-parse", "HEAD"), text=True
                    ).strip(),
                ),
                config=tools.cfg,
                provider=provider,
                dispatcher=dispatcher,
                events=event_log.EventSink(saved.logs_path),
                max_turns=2,
            )
            result = harness.run("Check this tree")
            assert result.completed
            assert result.reason == "finish_session"
        finally:
            dispatcher.close()
    view = receipt.read_receipt(saved.session_dir, pathlib.Path.cwd())
    assert view.verifications
    for verification in view.verifications:
        assert verification.cmd == _ENVIRONMENT_COMMAND
        assert verification.outcome == "passed"
        assert verification.exit_code == 0
        assert verification.tree_before == verification.tree_after == expected_tree
    assert len(view.targets) == 2
    assert len({stamp.target_id for stamp in view.targets}) == 2
    assert len({stamp.execution_id for stamp in view.targets}) == 2
    assert len(view.setups) == 2
    for setup, stamp in zip(view.setups, view.targets, strict=True):
        assert setup.cmd == _ENVIRONMENT_COMMAND
        assert setup.exit_code == 0
        assert not setup.exec_failed
        assert setup.execution_id == stamp.execution_id
        assert setup.target_id == stamp.target_id
        attempts = [v for v in view.verifications if v.execution_id == stamp.execution_id]
        assert len(attempts) >= 2 and attempts[0].baseline
        assert all(v.target_id == stamp.target_id for v in attempts)
        for command in (setup, *attempts):
            assert command.stdout_tail.splitlines() == [
                str(stamp.target.command_root),
                stamp.target.user,
                stamp.target.home,
            ]
        assert [c.effective for c in stamp.target.controls if c.leaf == "sandbox.isolation"] == [
            isolation
        ]
        assert stamp.target.transport == "local"
        assert stamp.target.protocol_version == 4
        assert (
            stamp.target.launcher_instance is not None and len(stamp.target.launcher_instance) == 32
        )
    assert len({stamp.target.launcher_instance for stamp in view.targets}) == 2
    recorded = manifest.read_manifest(saved.session_dir)
    assert recorded.execution_targets == {stamp.target_id: stamp.target for stamp in view.targets}
    assert [job.outcome for job in view.jobs] == ["settled", "settled"]
    assert [env.outcome for env in view.environments] == ["not_applicable", "not_applicable"]


@pytest.fixture
def resumed(
    saved: layout.SessionLayout,
    monkeypatch: pytest.MonkeyPatch,
    jail_bin: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    request: pytest.FixtureRequest,
) -> layout.SessionLayout:
    """Run and resume through the console frontend, real command sessions and a saved snapshot."""
    import dataclasses
    import types
    import uuid

    from agent6 import providers
    from agent6.app import execution, reporter, role_providers

    origin = pathlib.Path.cwd()
    workspace = origin.parent / "worktree"
    subprocess.run(
        ("git", "worktree", "add", "-q", "--detach", str(workspace)),
        check=True,
        capture_output=True,
    )
    monkeypatch.chdir(workspace)
    cfg = Config.model_validate(
        {
            "models": {"default": {"model": "anthropic/claude-test"}},
            "sandbox": {"isolation": "none", "network": "host", "run_commands": "allow"},
            "workspace": {"setup_command": ["/bin/true"]},
            "git": {"commit_per_turn": False, "auto_merge": False},
            "prompt": {"decompose": "off"},
            "harness": {
                "verify_command": ["/bin/true"],
                "verify_at_start": True,
                "verify_when": "finish",
                "command_checkin_s": 0,
                "max_turns": 6,
            },
            "review": {"trigger": "off"},
        }
    )
    responses = iter(
        response
        for suffix in ("first", "resumed")
        for response in (
            test_harness._tool_resp("run_verify_command", tool_id=f"verify-{suffix}"),
            test_harness._tool_resp(
                "finish_session", {"summary": "checked"}, tool_id=f"finish-{suffix}"
            ),
        )
    )

    class ScriptedProvider:
        def call(self, **_kwargs: object) -> providers.ProviderResponse:
            return next(responses)

    def provider_factory(*_args: object, **_kwargs: object) -> ScriptedProvider:
        return ScriptedProvider()

    monkeypatch.setattr(role_providers, "build_role_provider", provider_factory)
    execution_ids = iter((uuid.UUID(int=1), uuid.UUID(int=2)))
    monkeypatch.setattr(execution, "uuid", types.SimpleNamespace(uuid4=lambda: next(execution_ids)))
    monkeypatch.setenv("AGENT6_FORCE_STREAM", "1")
    monkeypatch.delenv("AGENT6_STREAM_TO_LOG", raising=False)
    inputs = execution.ExecutionInputs(
        session_id=saved.session_id,
        mode="run",
        role="worker",
        explicit_leaves=frozenset(),
        isolation="none",
        tui_enabled=False,
        interactive=False,
        task="Check this tree",
        gate=lambda config, _budget, _available: config,
        chain_branch=None,
        base_sha=subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip(),
        untracked_at_start=frozenset(),
        resume_state_path=saved.session_dir / "loop_state.json",
        undo_forker=lambda: None,
        prompts=operator_prompts.OperatorPrompts(session_dir=saved.session_dir),
        ask_transcript_task=None,
        worktree_git_dir=origin / ".git",
    )
    stamps.write_session_manifest(
        saved,
        session_id=saved.session_id,
        user_task="Check this tree",
        base_sha=inputs.base_sha,
        base_branch="",
        run_branch=None,
        cfg=cfg,
        isolation="none",
        worktree=workspace,
        worktree_git_dir=origin / ".git",
    )
    mirror_failures: list[str] = []
    write_manifest = stamps.write_manifest

    def fail_resumed_target(path: pathlib.Path, record: manifest.SessionManifest) -> None:
        if len(record.execution_targets) > 1:
            mirror_failures.append("resumed target")
            raise OSError("injected resumed target mirror failure")
        write_manifest(path, record)

    if getattr(request, "param", False):
        monkeypatch.setattr(stamps, "write_manifest", fail_resumed_target)
    prior = b""
    for resuming in (False, True):
        surface = cli_run.session_frontend()
        try:
            end = execution.run_execution(
                cfg,
                saved,
                dataclasses.replace(
                    inputs,
                    task=None if resuming else inputs.task,
                    resuming=resuming,
                    resume_steer="Check again" if resuming else "",
                ),
                frontend=surface,
                reporter=reporter.Reporter(out=lambda _line: None, err=lambda _line: None),
                events=event_log.EventSink(saved.logs_path),
                transcript_sink=mock.Mock(),
                cwd=workspace,
                state_dir=saved.state_dir,
            )
        finally:
            surface.close_console_view()
        assert end.rc == 0
        assert inputs.resume_state_path.exists()
        assert saved.logs_path.read_bytes().startswith(prior)
        prior = saved.logs_path.read_bytes()
    if getattr(request, "param", False):
        assert mirror_failures == ["resumed target"]
        assert len(manifest.read_manifest(saved.session_dir).execution_targets) == 1
    return saved


def test_app_run_and_resume_preserve_execution_evidence(resumed: layout.SessionLayout) -> None:
    """App-assigned identities reach setup, baseline and the snapshot's resumed execution."""
    import uuid

    rows = [json.loads(line) for line in resumed.logs_path.read_text().splitlines()]
    starts = [row for row in rows if row["type"] in {"session.start", "session.resume"}]
    expected_ids = [uuid.UUID(int=1).hex, uuid.UUID(int=2).hex]
    assert [row["type"] for row in starts] == ["session.start", "session.resume"]
    assert [row["execution_id"] for row in starts] == expected_ids
    assert starts[1]["turn"] > 1
    view = receipt.read_receipt(resumed.session_dir, pathlib.Path.cwd())
    expected_tree = subprocess.check_output(("git", "rev-parse", "HEAD^{tree}"), text=True).strip()
    assert expected_tree and view.verifications
    for verification in view.verifications:
        assert verification.cmd == ("/bin/true",)
        assert verification.outcome == "passed"
        assert verification.exit_code == 0
        assert verification.tree_before == verification.tree_after == expected_tree
    assert [target.execution_id for target in view.targets] == expected_ids
    first_target, resumed_target = (target.target_id for target in view.targets)
    assert first_target != resumed_target
    assert len(view.setups) == 1
    assert (view.setups[0].execution_id, view.setups[0].target_id) == (
        expected_ids[0],
        first_target,
    )
    baselines = [record for record in view.verifications if record.baseline]
    assert len(baselines) == 1
    assert (baselines[0].execution_id, baselines[0].target_id) == (expected_ids[0], first_target)
    for execution_id, target_id in zip(expected_ids, (first_target, resumed_target), strict=True):
        attempts = [record for record in view.verifications if record.execution_id == execution_id]
        assert attempts and all(record.target_id == target_id for record in attempts)
    assert manifest.read_manifest(resumed.session_dir).execution_targets == {
        target.target_id: target.target for target in view.targets
    }


@pytest.mark.parametrize(
    "isolation", [pytest.param("full", marks=pytest.mark.needs_namespaces), "partial", "none"]
)
def test_machine_setup_records_its_one_shot_target(
    saved: layout.SessionLayout, jail_bin: pathlib.Path, isolation: kinds.IsolationLevel
) -> None:
    """One-shot setup records its command's cwd, user and HOME without a protocol identity."""
    from agent6.app import reporter
    from agent6.app.machine import workspace
    from agent6.tools import policy as tools_policy

    cfg = Config.model_validate(
        {"sandbox": {"isolation": isolation, "network": "host", "run_commands": "allow"}}
    ).with_setup_command(_ENVIRONMENT_COMMAND)
    result = workspace.tool_setup_failure(
        cfg,
        tools_policy.jail_policy(pathlib.Path.cwd(), cfg, isolation, ()),
        saved.logs_path,
        reporter.Reporter(out=lambda _line: None, err=lambda _line: None),
    )
    assert result is None
    view = receipt.read_receipt(saved.session_dir, pathlib.Path.cwd())
    assert len(view.setups) == len(view.targets) == 1
    setup, stamp = view.setups[0], view.targets[0]
    assert setup.cmd == _ENVIRONMENT_COMMAND
    assert setup.exit_code == 0 and not setup.exec_failed
    assert setup.target_id is not None
    assert setup.target_id == stamp.target_id
    assert setup.execution_id == stamp.execution_id
    assert setup.stdout_tail.splitlines() == [
        str(stamp.target.command_root),
        stamp.target.user,
        stamp.target.home,
    ]
    assert [c.effective for c in stamp.target.controls if c.leaf == "sandbox.isolation"] == [
        isolation
    ]
    assert stamp.target.protocol_version is None


def test_target_stamps_record_environment_policy_not_values(
    saved: layout.SessionLayout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The persisted target must not copy command-environment values into journals."""
    monkeypatch.setenv("CI", "environment-value-must-not-be-recorded")
    tools = _tools(saved)
    try:
        target = tools.dispatcher.execution_target
        assert target is not None
        assert dict(target.environment.variables)["CI"] == "environment-value-must-not-be-recorded"
    finally:
        tools.dispatcher.close()
    for path in (saved.manifest_path, saved.logs_path):
        assert "environment-value-must-not-be-recorded" not in path.read_text()
    view = receipt.as_dict(receipt.read_receipt(saved.session_dir, pathlib.Path.cwd()))
    assert "CI" in view["targets"][0]["target"]["environment"]["variable_names"]


def test_old_logs_do_not_borrow_a_later_environment(saved: layout.SessionLayout) -> None:
    """Historical setup and verification keep unknown target references even beside a new stamp."""
    saved.logs_path.write_text(
        json.dumps({"type": "workspace.setup", "cmd": ["true"], "exit_code": 0})
        + "\n"
        + json.dumps({"type": "verify.end", "cmd": ["true"], "exit_code": 0})
        + "\n"
    )
    tools = _tools(saved)
    tools.dispatcher.close()
    view = receipt.read_receipt(saved.session_dir, pathlib.Path.cwd())
    assert view.setups[0].target_id is None
    assert view.verifications[0].target_id is None
    assert "Environment: unknown" in receipt.render(view)


@pytest.fixture
def recorded(saved: layout.SessionLayout, jail_bin: pathlib.Path) -> layout.SessionLayout:
    tools = _tools(saved)
    tools.dispatcher.run_setup(("/bin/true",))
    tools.dispatcher.run_verify()
    tools.dispatcher.close()
    return saved


@pytest.mark.parametrize(
    "identities",
    [
        {},
        {"execution_id": "execution"},
        {"target_id": "target"},
        {"execution_id": None, "target_id": None},
        {"execution_id": "execution", "target_id": "target"},
    ],
)
def test_lifecycle_replay_preserves_optional_identities(
    recorded: layout.SessionLayout, identities: dict[str, str | None]
) -> None:
    """Omitted lifecycle identities stay unknown without discarding valid observations."""
    events = event_log.EventSink(recorded.logs_path)
    events.emit(journal.ExecutionJobs(outcome="settled", reason="", **identities))
    events.emit(
        journal.ExecutionEnvironment(
            outcome="left_running", reason="ownership unproved", **identities
        )
    )
    view = receipt.read_receipt(recorded.session_dir, pathlib.Path.cwd())
    assert not view.evidence_notes
    jobs, environment = view.jobs[-1], view.environments[-1]
    assert jobs.outcome == "settled"
    assert environment.outcome == "left_running"
    assert environment.reason == "ownership unproved"
    for observation in (jobs, environment):
        assert observation.execution_id == identities.get("execution_id")
        assert observation.target_id == identities.get("target_id")


def test_target_replay_preserves_omitted_execution_identity(recorded: layout.SessionLayout) -> None:
    """An unbound target stamp cannot borrow the execution of an earlier stamp."""
    target = next(
        row["target"]
        for line in recorded.logs_path.read_text().splitlines()
        if (row := json.loads(line))["type"] == "execution.target"
    )
    event_log.EventSink(recorded.logs_path).emit(
        journal.ExecutionTarget(target_id="unbound", target=target)
    )
    view = receipt.read_receipt(recorded.session_dir, pathlib.Path.cwd())
    assert not view.evidence_notes
    assert view.targets[-1].target_id == "unbound"
    assert view.targets[-1].execution_id is None
    assert view.targets[-1].target == view.targets[0].target


def test_receipt_replays_without_environment_probes(
    recorded: layout.SessionLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replay uses persisted targets even after the launcher and workspace disappear."""
    root = pathlib.Path.cwd()
    before = receipt.as_dict(receipt.read_receipt(recorded.session_dir, root))
    assert before["targets"][0]["target"]["host_root"] == str(root)
    assert not recorded.session_dir.is_relative_to(root)
    monkeypatch.chdir(root.parent)
    root.rename(root.with_name("removed-repo"))
    assert not root.exists()
    assert recorded.logs_path.is_file()
    monkeypatch.setattr(
        jail_session.JailSession, "open", mock.Mock(side_effect=AssertionError("probe"))
    )
    monkeypatch.setattr(jail, "run_in_jail", mock.Mock(side_effect=AssertionError("probe")))
    monkeypatch.setattr(subprocess, "Popen", mock.Mock(side_effect=AssertionError("probe")))
    assert receipt.as_dict(receipt.read_receipt(recorded.session_dir, root)) == before
    assert review_routes.receipt_payload(root, recorded.session_dir) == before


def test_separate_job_and_environment_observations_replay(saved: layout.SessionLayout) -> None:
    """Settled jobs do not imply that an attached environment stopped."""
    events = event_log.EventSink(saved.logs_path)
    events.emit(
        journal.ExecutionJobs(execution_id="e", target_id="t", outcome="settled", reason="")
    )
    events.emit(
        journal.ExecutionEnvironment(
            execution_id="e",
            target_id="t",
            outcome="left_running",
            reason="ownership unproved; run stop by hand",
        )
    )
    text = receipt.render(receipt.read_receipt(saved.session_dir, pathlib.Path.cwd()))
    assert "Run jobs settled" in text
    assert "Environment left running" in text
    assert "ownership unproved; run stop by hand" in text


@pytest.mark.parametrize("resumed", [False, True], indirect=True, ids=["mirrored", "stale-mirror"])
def test_console_run_and_resume_show_their_own_environment(
    resumed: layout.SessionLayout, capsys: pytest.CaptureFixture[str]
) -> None:
    """Live console starts read their target after stamping, without replay or a test policy."""
    text = capsys.readouterr().err
    blocks = [line for line in text.splitlines() if "Environment: " in line]
    targets = [
        row["target_id"]
        for line in resumed.logs_path.read_text().splitlines()
        if (row := json.loads(line))["type"] == "execution.target"
    ]
    assert len(targets) == len(set(targets)) == 2
    assert len(blocks) == 2, text
    executions = text.split("  ● passed")
    assert len(executions) == 3, text
    for index, output in enumerate(executions[:-1]):
        start, verification, _rest = output.partition("  verify: running")
        assert verification, output
        assert f"Environment: local (target {targets[index]})" in start
        assert targets[1 - index] not in start


@pytest.mark.parametrize("resumed", [False, True], indirect=True, ids=["mirrored", "stale-mirror"])
def test_cli_and_web_read_the_recorded_environment(
    resumed: layout.SessionLayout, capsys: pytest.CaptureFixture[str]
) -> None:
    """The web header selects the resumed target; session JSON preserves earlier references."""
    rows = [json.loads(line) for line in resumed.logs_path.read_text().splitlines()]
    targets = [row for row in rows if row["type"] == "execution.target"]
    first, latest = (row["target_id"] for row in targets)
    assert first != latest
    header = wire.session_snapshot(resumed.session_dir)["policy"]
    assert f"Environment: local (target {latest})" in header
    assert first not in header
    assert sessions_show._cmd_status(resumed.session_id) == 0
    text = capsys.readouterr().out
    assert first in text and latest in text and "Run jobs settled" in text
    assert sessions_show._cmd_status(resumed.session_id, as_json=True) == 0
    observations = json.loads(capsys.readouterr().out)["environment_evidence"]
    expected = [(row["execution_id"], row["target_id"]) for row in targets]
    for field in ("targets", "jobs", "environments", "setups"):
        assert [(row["execution_id"], row["target_id"]) for row in observations[field]] == (
            expected[:1] if field == "setups" else expected
        )


@pytest.mark.parametrize("resumed", [False, True], indirect=True, ids=["mirrored", "stale-mirror"])
def test_tui_header_reads_recorded_environment(resumed: layout.SessionLayout) -> None:
    """The rendered dashboard shows the resumed target within its header's height limit."""
    first, latest = (
        row["target_id"]
        for line in resumed.logs_path.read_text().splitlines()
        if (row := json.loads(line))["type"] == "execution.target"
    )
    assert first != latest

    async def scenario() -> None:
        app = tui_app.Agent6TUI(resumed.session_dir)
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await pilot.press("ctrl+d")
            await pilot.pause()
            screenshot = app.export_screenshot().replace("&#160;", " ")
            assert f"Environment: local (target {latest})" in screenshot
            assert f"Environment: local (target {first})" not in screenshot

    asyncio.run(scenario())


def test_a_target_record_written_as_launcher_version_reads_as_the_digest() -> None:
    """The field was renamed; records from before the rename keep their value."""
    stamp = {
        "transport": "container",
        "host_root": "/w",
        "command_root": "/w",
        "user": "1000",
        "home": "/home/u",
        "environment": {"inherit": False, "variable_names": []},
        "launcher_version": "sha256:ab",
    }
    record = evidence.TARGET.validate_python(
        {"target_id": "t1", "execution_id": "e1", "target": stamp}
    )
    assert record.target.launcher_digest == "sha256:ab"
    assert (
        evidence.TARGET.dump_python(record, mode="json")["target"]["launcher_digest"] == "sha256:ab"
    )
