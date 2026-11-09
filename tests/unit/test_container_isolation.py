# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`sandbox.isolation = "container"`: the table, the refusals and every surface that says so."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import subprocess

import pydantic
import pytest

from agent6 import kinds, paths
from agent6.app import confine, session_runtime, stamps
from agent6.app import reporter as app_reporter
from agent6.config import Config
from agent6.harness import prompt_blocks
from agent6.sandbox import container, detect
from agent6.sessions import layout, manifest
from agent6.tools import target_policy
from agent6.ui.cli import check_sandbox, exec_cmds, net_cmds
from tests.unit import test_container as fake_runtime

_AGENT_MACHINE = """\
machine = "agent-config"
version = 1
initial = "judge"

[budget]
max_transitions = 5

[schemas.verdict]
ok = "bool"

[vars.agent]
verdict = { type = "verdict", default = {} }

[states.judge]
kind = "agent"
prompt = "judge"
output_schema = "verdict"
capture = { finish_json = "verdict" }
timeout_s = 5
on = { ok = "done", failed = "done", budget_exhausted = "done", timeout = "done" }

[states.done]
kind = "terminal"
status = "ok"
reason = "done"
"""


def _cfg(**sandbox: object) -> Config:
    return Config.model_validate({"sandbox": {"isolation": "container", **sandbox}})


def test_the_container_table_validates_its_argv_and_workspace() -> None:
    """Every `[container]` leaf has a real default; argv words and the workspace path are checked.

    A relative or traversing workspace path, an empty argv word and an unknown key refuse.
    """
    cfg = Config.model_validate({})
    assert cfg.container.exec == () and cfg.container.runtime == ()
    assert cfg.container.workspace == "" and cfg.container.start == () and cfg.container.stop == ()
    cfg = Config.model_validate(
        {
            "container": {
                "exec": ["sudo", "-n", "docker", "exec", "-i", "agent6-dev"],
                "runtime": ["sudo", "-n", "docker"],
                "workspace": "/workspaces/agent6",
                "start": ["devcontainer", "up", "--workspace-folder", "."],
                "stop": ["docker", "stop", "agent6-dev"],
            }
        }
    )
    assert cfg.container.exec[0] == "sudo" and cfg.container.workspace == "/workspaces/agent6"
    with pytest.raises(pydantic.ValidationError, match="non-empty"):
        Config.model_validate({"container": {"exec": ["docker", ""]}})
    for bad in ("relative/path", "/a/../b", "/tab\there"):
        with pytest.raises(pydantic.ValidationError, match="absolute container path"):
            Config.model_validate({"container": {"workspace": bad}})
    with pytest.raises(pydantic.ValidationError):
        Config.model_validate({"container": {"image": "x"}})


def test_filling_the_table_alone_changes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A `[container]` table under a local isolation level selects no container and probes none."""
    cfg = Config.model_validate(
        {"sandbox": {"isolation": "none"}, "container": {"exec": ["docker", "exec", "-i", "x"]}}
    )

    def never(*_args: object, **_kwargs: object) -> container.ContainerDiscovery:
        raise AssertionError("a local level must not discover a container")

    monkeypatch.setattr(container, "discover", never)
    env = detect.probed_environment()
    assert detect.resolve_isolation(cfg.sandbox.isolation, env) == "none"
    assert stamps._policy_stamp(cfg, "none").isolation == "none"  # pyright: ignore[reportPrivateUsage]


def test_container_resolves_to_the_launchers_none_branch_without_host_probes() -> None:
    """The word selects the transport; the launcher inside runs its `none` branch."""
    env = detect.probed_environment()
    assert detect.resolve_isolation("container", env) == "none"
    offline = dataclasses.replace(
        env, sandbox_available=False, userns_supported=False, landlock_abi=0
    )
    assert detect.resolve_isolation("container", offline) == "none"


def test_the_manifest_keeps_the_word_container(tmp_path: pathlib.Path) -> None:
    """A container run's policy stamp says container, not the launcher's level inside."""
    stamp = stamps._policy_stamp(_cfg(), "none")  # pyright: ignore[reportPrivateUsage]
    assert stamp.isolation == "container" and stamp.network == "host"
    assert stamps._policy_stamp(_cfg(), "").isolation == "container"  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ("sandbox", "needle"),
    [
        ({"network": "session"}, "a container's network is the runtime's"),
        ({"protect_git": "on"}, "resolved to 'container'"),
        ({"home": "tmp"}, "has no place inside a container"),
        ({"home": "cache"}, "has no place inside a container"),
    ],
)
def test_an_explicit_control_a_container_cannot_honor_refuses(
    tmp_path: pathlib.Path, sandbox: dict[str, object], needle: str
) -> None:
    """Session networking, command-side git protection and a private HOME refuse by name."""
    refusal = confine.startup_refusal(_cfg(**sandbox), "none", tmp_path)
    assert refusal is not None and needle in refusal
    assert confine.startup_refusal(_cfg(home="host"), "none", tmp_path) is None
    assert confine.startup_refusal(_cfg(network="host"), "none", tmp_path) is None


def test_config_show_probes_nothing_for_a_container(monkeypatch: pytest.MonkeyPatch) -> None:
    """The listing names no runtime, starts nothing and resolves no `auto` against a probe."""

    def never() -> detect.Environment:
        raise AssertionError("config show must not probe the host for a container run")

    monkeypatch.setattr(detect, "probed_environment", never)
    assert "sandbox.isolation" not in confine.resolved_config_values(_cfg(network="auto"))


def test_the_start_warning_names_the_container_not_the_host(
    capsys: pytest.CaptureFixture[str], tmp_path: pathlib.Path
) -> None:
    """Under container isolation the gap warning says where the commands run."""
    env = detect.probed_environment()
    confine.warn_sandbox_gaps("none", env, _cfg(), root=tmp_path)
    err = capsys.readouterr().err
    assert "inside the selected container" in err and "UNSANDBOXED" not in err


def test_selection_skips_the_unconfined_confirm(tmp_path: pathlib.Path) -> None:
    """The container is the boundary the operator chose; no `none` confirmation fires."""

    def confirm(_level: kinds.IsolationLevel, _cfg: Config) -> bool:
        raise AssertionError("the unconfined confirm is for sandbox.isolation = none")

    reporter = app_reporter.Reporter(out=lambda _msg: None, err=lambda _msg: None)
    selected = session_runtime.select_isolation(
        _cfg(run_commands="allow"), cwd=tmp_path, confirm_unconfined=confirm, reporter=reporter
    )
    assert selected == "none"


def test_the_control_rows_name_the_container_as_the_owner() -> None:
    """Every sandbox leaf reads with its effective behaviour and enforcement owner."""
    rows = {c.leaf: c for c in target_policy.container_controls(_cfg(network="session"))}
    assert rows["sandbox.isolation"].effective == "container"
    assert rows["sandbox.network"].owner == "container" and rows["sandbox.network"].reason
    assert rows["sandbox.home"].effective == "host"
    assert rows["sandbox.protect_git"].owner == "host file tools"
    assert rows["sandbox.extra_write_paths"].owner == "host file tools"
    assert rows["sandbox.memory_limit_mib"].owner == "agent6"
    assert len(rows) == len(type(Config.model_validate({}).sandbox).model_fields)


def _discovery(
    mounts: tuple[tuple[str, str], ...], container_id: str | None
) -> container.ContainerDiscovery:
    return container.ContainerDiscovery(
        exec=("docker", "exec", "-i", "x"),
        target=kinds.ExecutionTarget(
            transport="container",
            host_root=pathlib.Path("/home/op/repo"),
            command_root=pathlib.Path("/work"),
            user="1000",
            home=None,
            container_id=container_id,
            environment=kinds.ExecutionEnvironment(variables=(), inherit=True),
        ),
        mounts=mounts,
    )


def test_a_hidden_path_a_mount_exposes_refuses() -> None:
    """An explicit hidden path reachable inside the container refuses; unseen mounts refuse too."""
    found = _discovery((("/home/op/repo", "/work"), ("/home/op/.ssh", "/root/.ssh")), "a" * 64)
    container.check_hidden_paths(found, ())
    container.check_hidden_paths(found, ("/home/op/other",))
    with pytest.raises(container.ContainerMappingError, match="exposed to commands"):
        container.check_hidden_paths(found, ("/home/op/.ssh/id_ed25519",))
    with pytest.raises(container.ContainerMappingError, match="exposed to commands"):
        container.check_hidden_paths(found, ("/home/op",))
    opaque = _discovery((), None)
    with pytest.raises(container.ContainerMappingError, match="cannot be inspected"):
        container.check_hidden_paths(opaque, ("/home/op/.ssh",))
    container.check_hidden_paths(opaque, ())


def test_exposures_name_host_paths_and_sockets_outside_the_workspace() -> None:
    """The start block warns about what the container mounts beyond the workspace."""
    found = _discovery(
        (
            ("/home/op/repo", "/work"),
            ("/var/run/docker.sock", "/var/run/docker.sock"),
            ("/home/op/.aws", "/home/vscode/.aws"),
        ),
        "a" * 64,
    )
    assert container.exposures(found) == (
        "socket /var/run/docker.sock mounted at /var/run/docker.sock",
        "host path /home/op/.aws mounted at /home/vscode/.aws",
    )


def test_the_mounted_root_is_the_repository_of_a_linked_worktree(tmp_path: pathlib.Path) -> None:
    """Discovery looks up the workspace the container mounts, not the worktree under it."""
    repo = tmp_path / "repo"
    assert session_runtime.mounted_root(repo / "wt", repo / ".git") == repo
    assert session_runtime.mounted_root(repo, None) == repo


def test_the_prompt_tells_the_model_both_roots() -> None:
    """Commands get the container path, file tools the host path, and no placeholder leaks."""
    repo = kinds.RepoSummary(
        root=pathlib.Path("/home/op/repo"),
        branch="main",
        head_sha="",
        file_count=0,
        top_level=(),
        agents_md="",
        recent_log="",
    )
    prompt = prompt_blocks.build_system_prompt(
        config=_cfg(),
        repo=repo,
        skills=None,
        isolation="none",
        container_roots=("/work", "/home/op/repo"),
    )
    assert "run_command runs inside a container" in prompt
    assert "`/work`" in prompt and "`/home/op/repo`" in prompt
    for mode in ("run", "plan", "ask", "read_only"):
        local = prompt_blocks.build_system_prompt(
            config=Config.model_validate({}), repo=repo, mode=mode, skills=None
        )
        assert "__CONTAINER_RULE__" not in local and "inside a container" not in local


def test_exec_refuses_under_container_isolation(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`agent6 exec` joins only a local jail; a container run names the runtime's own exec."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    assert exec_cmds.exec_new(_cfg(), tmp_path, ("true",)) == 2
    assert "runtime's own exec" in capsys.readouterr().err
    saved = layout.SessionLayout(paths.state_dir(tmp_path), "r1")
    saved.ensure()
    stamps.write_manifest(
        saved.manifest_path,
        manifest.SessionManifest(
            session_id="r1", policy=manifest.PolicyStamp(isolation="container", network="host")
        ),
    )
    assert net_cmds._launch_policy(saved, Config.model_validate({}), None) is None  # pyright: ignore[reportPrivateUsage]
    assert "runtime's own exec" in capsys.readouterr().err


def test_check_sandbox_inspects_the_container_and_starts_nothing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`check sandbox` names the container, user, workspace, refusals and controls."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (tmp_path / "bin").mkdir()
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.chdir(workspace)
    cli = fake_runtime.runtime(workspace)
    fake_runtime.ready(
        cli,
        fake_runtime.inspect_row(
            workspace,
            metadata=[{"remoteUser": "vscode"}],
            mounts=[
                fake_runtime.mount(workspace, "/work/project"),
                fake_runtime.mount("/var/run/docker.sock", "/var/run/docker.sock"),
            ],
        ),
    )
    rows = check_sandbox.check_sandbox_section(_cfg(network="session"))
    out = capsys.readouterr().out
    by_name = {row.name: row for row in rows}
    assert by_name["container.discovery"].status == "PASS"
    assert "user vscode" in by_name["container.discovery"].detail
    assert "/work/project" in by_name["container.discovery"].detail
    assert by_name["container.exposure"].status == "WARN"
    assert "docker.sock" in by_name["container.exposure"].detail
    assert by_name["sandbox.refusal"].status == "FAIL"
    assert "a container's network is the runtime's" in by_name["sandbox.refusal"].detail
    assert "sandbox.network: the container's network (container, not probed)" in out
    calls = [call["argv"][0] for call in cli.calls()]
    assert calls == ["ps", "inspect"]


def test_a_machine_refuses_under_container_isolation(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A machine's states run on this host; container isolation is for runs."""
    from agent6.app.machine import run as machine_run

    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    # agent6's private directories stay outside the checkout, as a run requires.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    explicit = tmp_path / "explicit.toml"
    explicit.write_text(
        '[providers.local]\nbackend = "openai"\nbase_url = "http://127.0.0.1:9"\n\n'
        '[models.default]\nmodel = "local/test-model"\n\n[sandbox]\nisolation = "container"\n',
        encoding="utf-8",
    )
    mfile = repo / "judge.asm.toml"
    mfile.write_text(_AGENT_MACHINE, encoding="utf-8")
    # `machine run` accepts a committed machine only.
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "base"],
        cwd=repo,
        check=True,
    )
    refused: list[str] = []
    reporter = app_reporter.Reporter(out=lambda _msg: None, err=refused.append)
    assert machine_run.run_machine(mfile, reporter, config_paths=(explicit,)) == 2
    assert any("is for runs" in line for line in refused), refused
    assert json.dumps(refused)  # the reporter received text, not objects
