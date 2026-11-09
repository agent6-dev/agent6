# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Container discovery queries runtimes without executing repository definitions."""

from __future__ import annotations

import builtins
import contextlib
import dataclasses
import io
import json
import os
import pathlib
import shlex
import signal
import sys
import time
import traceback
from typing import Any

import pytest

from agent6 import child_env
from agent6.sandbox import container
from agent6.sessions import process

_ID = "a" * 64
_OTHER_ID = "b" * 64


@dataclasses.dataclass(frozen=True, slots=True)
class Runtime:
    path: pathlib.Path
    log: pathlib.Path

    def replies(self, *rows: dict[str, Any], by_command: bool = False) -> None:
        self.path.write_text(
            f"#!{sys.executable}\n"
            "import json, os, pathlib, subprocess, sys, time\n"
            f"log = pathlib.Path({str(self.log)!r})\n"
            "calls = log.read_text().splitlines() if log.exists() else []\n"
            f"rows = {rows!r}\n"
            f"row = rows[int('inspect' in sys.argv) if {by_command!r} else len(calls)] "
            "if len(calls) < len(rows) else {}\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']) "
            "if row.get('child') else None\n"
            "with log.open('a') as stream:\n"
            "    stream.write(json.dumps({'argv': sys.argv[1:], 'env': dict(os.environ), "
            "'cwd': os.getcwd(), 'pid': os.getpid(), "
            "'child': child.pid if child else None}) + '\\n')\n"
            "if len(calls) >= len(rows):\n"
            "    sys.exit(79)\n"
            "time.sleep(row.get('sleep', 0))\n"
            "if row.get('flood'):\n"
            "    while True:\n"
            "        os.write(row['flood'], b'x' * 65536)\n"
            "sys.stdout.write(row.get('stdout', ''))\n"
            "sys.stderr.write(row.get('stderr', ''))\n"
            "sys.exit(row.get('exit', 0))\n",
            encoding="utf-8",
        )
        self.path.chmod(0o755)

    def calls(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.log.read_text().splitlines()]


@pytest.fixture
def workspace(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    root = tmp_path / "workspace with spaces;echo nope"
    root.mkdir()
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    monkeypatch.setenv("PATH", str(binary_dir))
    return root


def runtime(workspace: pathlib.Path, name: str = "docker") -> Runtime:
    return Runtime(workspace.parent / "bin" / name, workspace.parent / f"{name}.jsonl")


def mount(source: pathlib.Path | str, destination: str, kind: str = "bind") -> dict[str, Any]:
    return {"Type": kind, "Source": str(source), "Destination": destination, "RW": True}


def inspect_row(
    workspace: pathlib.Path,
    *,
    metadata: object = None,
    user: str = "1001:1001",
    mounts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    labels = {"devcontainer.local_folder": str(workspace)}
    if metadata is not None:
        labels["devcontainer.metadata"] = json.dumps(metadata)
    return {
        "Id": _ID,
        "State": {"Running": True, "Status": "running"},
        "Config": {"User": user, "Labels": labels, "Env": ["HOME=/root", "PATH=/bin"]},
        "Mounts": mounts if mounts is not None else [mount(workspace, "/work/project")],
        "Name": "/devcontainer",
    }


def ready(cli: Runtime, row: dict[str, Any]) -> None:
    cli.replies({"stdout": _ID + "\n"}, {"stdout": json.dumps([row])})


def ps_args(workspace: pathlib.Path, *, all_containers: bool = False) -> list[str]:
    return [
        "ps",
        *(["-a"] if all_containers else []),
        "--quiet",
        "--no-trunc",
        "--filter",
        f"label=devcontainer.local_folder={workspace}",
    ]


@pytest.mark.parametrize("name", ["docker", "podman"])
def test_discovers_each_runtime_with_exact_label_filter(workspace: pathlib.Path, name: str) -> None:
    cli = runtime(workspace, name)
    ready(cli, inspect_row(workspace, metadata=[{"remoteUser": "vscode"}]))

    found = container.discover(workspace, container.ContainerOptions())

    assert [call["argv"] for call in cli.calls()] == [
        [
            "ps",
            "--quiet",
            "--no-trunc",
            "--filter",
            f"label=devcontainer.local_folder={workspace}",
        ],
        ["inspect", "--type", "container", _ID],
    ]
    assert found.runtime == (str(cli.path),)
    assert found.exec == (str(cli.path), "exec", "-i", "--user", "vscode", _ID)
    assert found.target.host_root == workspace
    assert found.target.command_root == pathlib.Path("/work/project")
    assert found.target.container_id == _ID
    assert found.target.user == "vscode"
    assert found.target.transport == "container"
    assert found.target.home is None
    assert found.target.launcher_instance is None
    assert found.target.lifecycle_ref is None
    assert found.target.environment.inherit
    assert found.target.environment.variables == ()


@pytest.mark.parametrize("search_path", ["tools", "."])
@pytest.mark.parametrize("selection", ["automatic", "name", "relative"])
def test_relative_path_runtime_keeps_its_identity_after_cwd_change(
    workspace: pathlib.Path, monkeypatch: pytest.MonkeyPatch, search_path: str, selection: str
) -> None:
    cli = Runtime(workspace.parent / "tools/docker", workspace.parent / "docker.jsonl")
    cli.path.parent.mkdir()
    ready(cli, inspect_row(workspace))
    monkeypatch.chdir(workspace.parent if search_path == "tools" else cli.path.parent)
    monkeypatch.setenv("PATH", search_path)
    executable = (
        "docker" if selection == "name" else f"./{cli.path.relative_to(pathlib.Path.cwd())}"
    )
    args = () if selection == "automatic" else ("--context", "space;$(no-shell)")
    prefix = () if selection == "automatic" else (executable, *args)
    found = container.discover(workspace, container.ContainerOptions(runtime=prefix))
    assert found.runtime == (str(cli.path), *args)
    assert found.target.runtime_id == shlex.join(found.runtime)
    assert found.exec == (str(cli.path), *args, "exec", "-i", "--user", "1001:1001", _ID)
    assert cli.calls()[0]["argv"] == [*args, *ps_args(workspace)]
    assert all(call["cwd"] == "/" for call in cli.calls())


@pytest.mark.parametrize("link", ["symlink_to", "hardlink_to"])
def test_runtime_executable_aliases_are_queried_once(workspace: pathlib.Path, link: str) -> None:
    cli = runtime(workspace, "podman")
    cli.replies({"stdout": _ID}, {"stdout": json.dumps([inspect_row(workspace)])}, by_command=True)
    docker = cli.path.with_name("docker")
    getattr(docker, link)(cli.path)
    found = container.discover(workspace, container.ContainerOptions())
    assert found.runtime == (str(docker),)
    assert [call["argv"][0] for call in cli.calls()] == ["ps", "inspect"]


def test_default_searches_docker_then_podman(workspace: pathlib.Path) -> None:
    docker = runtime(workspace)
    docker.replies({"stdout": ""})
    podman = runtime(workspace, "podman")
    ready(podman, inspect_row(workspace))
    found = container.discover(workspace, container.ContainerOptions())
    assert found.runtime == (str(podman.path),)
    assert [c["argv"] for c in docker.calls()] == [ps_args(workspace)]


@pytest.mark.parametrize("across_runtimes", [False, True])
def test_ambiguous_running_matches_refuse_with_ids(
    workspace: pathlib.Path, across_runtimes: bool
) -> None:
    docker = runtime(workspace)
    if across_runtimes:
        docker.replies({"stdout": _ID})
        runtime(workspace, "podman").replies({"stdout": _OTHER_ID})
    else:
        docker.replies({"stdout": f"{_OTHER_ID}\n{_ID}\n{_ID}\n"})
    with pytest.raises(container.ContainerAmbiguousError) as error:
        container.discover(workspace, container.ContainerOptions())
    assert _ID in str(error.value) and _OTHER_ID in str(error.value)
    assert "container.exec" in str(error.value)
    assert len(docker.calls()) == 1


def test_missing_runtime_is_not_no_match(workspace: pathlib.Path) -> None:
    with pytest.raises(container.ContainerRuntimeUnavailableError, match=r"container\.runtime"):
        container.discover(workspace, container.ContainerOptions())


@pytest.mark.parametrize("phase", ["ps", "inspect", "stopped"])
def test_runtime_refusal_is_not_an_empty_result(workspace: pathlib.Path, phase: str) -> None:
    cli = runtime(workspace)
    refusal = {"exit": 1, "stderr": "permission denied connecting to docker.sock"}
    replies = {
        "ps": [refusal],
        "inspect": [{"stdout": _ID}, refusal],
        "stopped": [{"stdout": ""}, refusal],
    }
    cli.replies(*replies[phase])
    runtime(workspace, "podman").replies({"stdout": ""})
    with pytest.raises(container.ContainerRuntimeError) as error:
        container.discover(workspace, container.ContainerOptions())
    text = str(error.value)
    assert "permission denied" in text and "container.runtime" in text and "docker group" in text
    assert "sudo" not in cli.calls()[0]["argv"]


@pytest.mark.parametrize(
    "definition", [None, ".devcontainer.json", ".devcontainer/devcontainer.json"]
)
@pytest.mark.parametrize("stopped", [False, True])
def test_no_running_diagnostic_uses_stopped_labels_and_definition_existence_only(
    workspace: pathlib.Path, definition: str | None, stopped: bool
) -> None:
    if definition:
        path = workspace / definition
        path.parent.mkdir(exist_ok=True)
        path.write_text("not valid JSON: contents must not matter", encoding="utf-8")
    cli = runtime(workspace)
    cli.replies({"stdout": ""}, {"stdout": _ID if stopped else ""})
    with pytest.raises(container.ContainerNotFoundError) as error:
        container.discover(workspace, container.ContainerOptions())
    text = str(error.value)
    if stopped or definition:
        assert "VS Code" in text and "container.start" in text and "devcontainer up" in text
    else:
        assert "container.exec" in text and "no definition" in text
    if stopped:
        assert _ID in text
    assert [call["argv"] for call in cli.calls()] == [
        ps_args(workspace),
        ps_args(workspace, all_containers=True),
    ]


@pytest.mark.parametrize(
    ("metadata", "runtime_user", "expected"),
    [
        (None, "1001:1002", "1001:1002"),
        (None, "", "0"),
        ({"containerUser": "ignored"}, "runtime", "runtime"),
        ({"remoteUser": "remote", "containerUser": "ignored"}, "runtime", "remote"),
        ([{"remoteUser": "first"}, {"remoteUser": "last"}], "runtime", "last"),
        ([{"remoteUser": "first"}, {"containerUser": "ignored"}], "runtime", "first"),
    ],
)
def test_remote_user_precedence(
    workspace: pathlib.Path, metadata: object, runtime_user: str, expected: str
) -> None:
    cli = runtime(workspace)
    ready(cli, inspect_row(workspace, metadata=metadata, user=runtime_user))
    found = container.discover(workspace, container.ContainerOptions())
    assert found.target.user == expected
    assert found.exec[-3:] == ("--user", expected, _ID)


def test_remote_env_is_merged_data_not_host_expansion(
    workspace: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SECRET", "do-not-copy")
    cli = runtime(workspace)
    ready(
        cli,
        inspect_row(
            workspace,
            metadata=[
                {"remoteEnv": {"PATH": "${containerEnv:PATH}:/tools", "DROP": "old"}},
                {"remoteEnv": {"DROP": None, "VALUE": "${localEnv:SECRET}"}},
            ],
        ),
    )
    found = container.discover(workspace, container.ContainerOptions())
    assert found.remote_env == (
        ("DROP", None),
        ("PATH", "${containerEnv:PATH}:/tools"),
        ("VALUE", "${localEnv:SECRET}"),
    )
    assert found.target.environment.variables == ()


@pytest.mark.parametrize(
    "metadata",
    [
        "{broken",
        "null",
        "42",
        '[{"remoteUser": 42}]',
        '[{"remoteUser": null}]',
        '[{"remoteUser": ""}]',
        '[{"remoteUser": "--privileged"}]',
        '[{"remoteEnv": {"X": 42}}]',
        '[{"remoteEnv": {"BAD=KEY": "x"}}]',
    ],
)
def test_malformed_metadata_refuses_without_echoing_values(
    workspace: pathlib.Path, metadata: str
) -> None:
    row = inspect_row(workspace)
    row["Config"]["Labels"]["devcontainer.metadata"] = metadata
    cli = runtime(workspace)
    ready(cli, row)
    with pytest.raises(container.ContainerMetadataError, match=r"devcontainer\.metadata"):
        container.discover(workspace, container.ContainerOptions())


@pytest.mark.parametrize("damage", ["json", "multiple", "id", "running", "mount", "user", "label"])
def test_untrusted_inspect_is_validated(workspace: pathlib.Path, damage: str) -> None:
    row = inspect_row(workspace)
    if damage == "id":
        row["Id"] = _OTHER_ID
    elif damage == "running":
        row["State"]["Running"] = False
    elif damage == "mount":
        row["Mounts"][0]["Destination"] = "relative"
    elif damage == "user":
        row["Config"]["User"] = 42
    elif damage == "label":
        row["Config"]["Labels"]["devcontainer.local_folder"] = "/other/workspace"
    output = "{" if damage == "json" else json.dumps([row, row] if damage == "multiple" else [row])
    cli = runtime(workspace)
    cli.replies({"stdout": _ID}, {"stdout": output})
    with pytest.raises(container.ContainerMetadataError):
        container.discover(workspace, container.ContainerOptions())


@pytest.mark.parametrize("bad_id", ["--help", "abc", "x" * 64, '{"Id": "fake"}'])
def test_ps_ids_cannot_become_runtime_options(workspace: pathlib.Path, bad_id: str) -> None:
    cli = runtime(workspace)
    cli.replies({"stdout": bad_id})
    with pytest.raises(container.ContainerMetadataError):
        container.discover(workspace, container.ContainerOptions())
    assert len(cli.calls()) == 1


def test_mount_source_aliases_and_consistent_nested_binds(workspace: pathlib.Path) -> None:
    alias = workspace.parent / "alias"
    alias.symlink_to(workspace, target_is_directory=True)
    cli = runtime(workspace)
    ready(
        cli,
        inspect_row(
            workspace,
            mounts=[
                mount(workspace.parent, "/parent"),
                mount(alias, f"/parent/{workspace.name}"),
            ],
        ),
    )
    found = container.discover(workspace, container.ContainerOptions())
    assert found.target.command_root == pathlib.Path("/parent") / workspace.name


@pytest.mark.parametrize("source", ["relative", "/host\u0000path", ""])
@pytest.mark.parametrize("override", [None, pathlib.Path("/override")])
def test_invalid_bind_sources_are_rejected_at_inspect_boundary(
    workspace: pathlib.Path, source: str, override: pathlib.Path | None
) -> None:
    cli = runtime(workspace)
    ready(cli, inspect_row(workspace, mounts=[mount(source, "/work")]))
    with pytest.raises(container.ContainerMetadataError, match="inspect"):
        container.discover(workspace, container.ContainerOptions(workspace=override))


def test_lexical_and_resolved_label_queries_deduplicate_ids(workspace: pathlib.Path) -> None:
    alias = workspace.parent / "alias"
    alias.symlink_to(workspace, target_is_directory=True)
    cli = runtime(workspace)
    cli.replies({"stdout": _ID}, {"stdout": _ID}, {"stdout": json.dumps([inspect_row(workspace)])})
    found = container.discover(alias, container.ContainerOptions())
    assert found.target.command_root == pathlib.Path("/work/project")
    assert [c["argv"] for c in cli.calls()] == [
        ps_args(alias),
        ps_args(workspace),
        ["inspect", "--type", "container", _ID],
    ]


@pytest.mark.parametrize("mount_case", ["missing", "conflict", "shadow", "volume", "prefix"])
def test_unmapped_or_conflicting_mounts_refuse(workspace: pathlib.Path, mount_case: str) -> None:
    mounts = {
        "missing": [],
        "conflict": [mount(workspace, "/one"), mount(workspace, "/two")],
        "shadow": [
            mount(workspace.parent, "/parent"),
            mount("/other", f"/parent/{workspace.name}"),
        ],
        "volume": [mount(workspace, "/work", "volume")],
        "prefix": [mount(str(workspace)[:-1], "/work")],
    }
    cli = runtime(workspace)
    ready(cli, inspect_row(workspace, mounts=mounts[mount_case]))
    with pytest.raises(container.ContainerMappingError, match=r"container\.workspace"):
        container.discover(workspace, container.ContainerOptions())


def test_workspace_override_precedes_missing_or_ambiguous_mount_mapping(
    workspace: pathlib.Path,
) -> None:
    cli = runtime(workspace)
    ready(cli, inspect_row(workspace, mounts=[mount(workspace, "/one"), mount(workspace, "/two")]))
    found = container.discover(
        workspace, container.ContainerOptions(workspace=pathlib.Path("/chosen"))
    )
    assert found.target.command_root == pathlib.Path("/chosen")


@pytest.mark.parametrize(
    "prefix",
    [
        ("docker", "exec", "-i", "--user", "vscode", "agent6-dev"),
        ("sudo", "-n", "docker", "exec", "-i", "agent6-dev"),
        ("podman", "exec", "-i", "NAME"),
        ("devcontainer", "exec", "--workspace-folder", "DIR with spaces"),
        ("custom", "", "  untouched;$(no-shell)  "),
    ],
)
@pytest.mark.parametrize("override", [None, pathlib.Path("/different path")])
def test_manual_prefix_is_opaque_and_bypasses_all_discovery(
    workspace: pathlib.Path,
    prefix: tuple[str, ...],
    override: pathlib.Path | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> Any:
        pytest.fail("manual prefix must bypass runtime and definition discovery")

    monkeypatch.setattr(container.shutil, "which", forbidden)
    monkeypatch.setattr(container.subprocess, "Popen", forbidden)
    monkeypatch.setattr(pathlib.Path, "exists", forbidden)
    found = container.discover(
        workspace,
        container.ContainerOptions(exec=prefix, runtime=("must-not-run",), workspace=override),
    )
    assert found.exec == prefix
    assert found.target.command_root == (override or workspace)
    assert found.runtime == ()
    assert found.target.container_id is None and found.target.user is None


def test_explicit_runtime_prefix_is_used_for_queries_and_derived_exec(
    workspace: pathlib.Path,
) -> None:
    cli = runtime(workspace, "sudo")
    ready(cli, inspect_row(workspace))
    prefix = (str(cli.path), "-n", "docker")
    found = container.discover(workspace, container.ContainerOptions(runtime=prefix))
    assert [c["argv"] for c in cli.calls()] == [
        ["-n", "docker", *ps_args(workspace)],
        ["-n", "docker", "inspect", "--type", "container", _ID],
    ]
    assert found.exec == (*prefix, "exec", "-i", "--user", "1001:1001", _ID)


def test_explicit_missing_runtime_refuses_without_fallback(workspace: pathlib.Path) -> None:
    runtime(workspace).replies({"stdout": _ID})
    with pytest.raises(container.ContainerRuntimeUnavailableError):
        container.discover(workspace, container.ContainerOptions(runtime=("missing-cli",)))


@pytest.mark.parametrize("override", [pathlib.Path("relative"), pathlib.Path("/work/../other")])
def test_workspace_override_must_be_an_absolute_container_path(
    workspace: pathlib.Path, override: pathlib.Path
) -> None:
    with pytest.raises(container.ContainerMappingError):
        container.discover(
            workspace, container.ContainerOptions(exec=("manual",), workspace=override)
        )


@pytest.mark.parametrize("name", [".devcontainer.json", ".devcontainer/devcontainer.json"])
def test_discovery_never_opens_definitions_or_runs_lifecycle_fields(
    workspace: pathlib.Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    marker = workspace.parent / "lifecycle-ran"
    command = [sys.executable, "-c", f"open({str(marker)!r}, 'w').write('unsafe')"]
    metadata = {
        key: command
        for key in (
            "initializeCommand",
            "onCreateCommand",
            "updateContentCommand",
            "postCreateCommand",
            "postStartCommand",
            "postAttachCommand",
        )
    }
    definition = workspace / name
    definition.parent.mkdir(exist_ok=True)
    definition.write_text(json.dumps(metadata), encoding="utf-8")
    cli = runtime(workspace)
    ready(cli, inspect_row(workspace, metadata=metadata))
    original_open = builtins.open
    original_io_open = io.open

    def guard_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        assert str(file) != str(definition), "repository definition contents opened"
        return original_open(file, *args, **kwargs)

    def guard_io_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        assert str(file) != str(definition), "repository definition contents opened"
        return original_io_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guard_open)
    monkeypatch.setattr(io, "open", guard_io_open)
    found = container.discover(workspace, container.ContainerOptions())
    assert found.target.user == "1001:1001"
    assert not marker.exists()
    assert [c["argv"] for c in cli.calls()] == [
        ps_args(workspace),
        ["inspect", "--type", "container", _ID],
    ]
    cli.log.unlink()
    cli.replies({"stdout": ""}, {"stdout": ""})
    with pytest.raises(container.ContainerNotFoundError):
        container.discover(workspace, container.ContainerOptions())
    assert not marker.exists()
    assert [c["argv"] for c in cli.calls()] == [
        ps_args(workspace),
        ps_args(workspace, all_containers=True),
    ]


def test_invalid_metadata_diagnostics_do_not_expose_values(workspace: pathlib.Path) -> None:
    cli = runtime(workspace)
    ready(cli, inspect_row(workspace, metadata={"remoteEnv": {"X": ["private-value"]}}))
    with pytest.raises(container.ContainerMetadataError) as error:
        container.discover(workspace, container.ContainerOptions())
    assert "private-value" not in "".join(traceback.format_exception(error.value))


def test_host_runtime_environment_is_curated(
    workspace: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SECRET", "private")
    monkeypatch.setenv("DOCKER_HOST", "unix:///operator.sock")
    monkeypatch.setenv("CONTAINER_HOST", "unix:///podman.sock")
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1234")
    monkeypatch.setenv("DOCKER_CONTEXT", "provider-secret")
    monkeypatch.setattr(child_env, "_provider_key_env", {"DOCKER_CONTEXT"})
    cli = runtime(workspace)
    ready(cli, inspect_row(workspace))
    container.discover(workspace, container.ContainerOptions())
    call = cli.calls()[0]
    assert call["cwd"] == "/"
    assert "SECRET" not in call["env"] and "DOCKER_CONTEXT" not in call["env"]
    assert call["env"]["DOCKER_HOST"] == "unix:///operator.sock"
    assert call["env"]["CONTAINER_HOST"] == "unix:///podman.sock"
    assert call["env"]["XDG_RUNTIME_DIR"] == "/run/user/1234"


@pytest.mark.parametrize("failure", ["timeout", "stdout", "stderr"])
def test_runtime_queries_are_time_and_output_bounded(
    workspace: pathlib.Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    cli = runtime(workspace)
    row = {"sleep": 30} if failure == "timeout" else {"flood": 1 if failure == "stdout" else 2}
    cli.replies({**row, "child": True})
    monkeypatch.setattr(container, "QUERY_TIMEOUT_S", 1)
    monkeypatch.setattr(container, "MAX_QUERY_BYTES", 4096)
    started = time.monotonic()
    try:
        with pytest.raises(container.ContainerRuntimeError, match=r"timed out|output limit"):
            container.discover(workspace, container.ContainerOptions())
        assert time.monotonic() - started < 3
        call = cli.calls()[0]
        pids = (call["pid"], call["child"])
        deadline = time.monotonic() + 2
        while any(process.pid_alive(pid) for pid in pids) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not [pid for pid in pids if process.pid_alive(pid)], "runtime processes survived"
        with pytest.raises(ProcessLookupError):
            os.kill(call["pid"], 0)  # The client is reaped, not merely a zombie.
    finally:
        if cli.log.exists():
            for pid in (cli.calls()[0]["pid"], cli.calls()[0]["child"]):
                if process.pid_alive(pid):
                    os.kill(pid, signal.SIGKILL)
                with contextlib.suppress(ChildProcessError):
                    os.waitpid(pid, 0)
