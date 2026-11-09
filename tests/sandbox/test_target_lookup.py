# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import shutil
import tempfile

import pytest

from agent6 import kinds
from agent6.sandbox import detect, jail, jail_session
from agent6.tools import verify_infer


def test_executable_lookup_reads_the_prepared_environment(tmp_path: pathlib.Path) -> None:
    """An executable found only on the target PATH is available to inference."""
    tool = tmp_path / "agent6-target-only"
    tool.write_text("#!/bin/sh\nexit 0\n")
    tool.chmod(0o755)
    assert shutil.which(tool.name) is None
    policy = kinds.JailPolicy(
        cwd=tmp_path,
        argv=(),
        isolation="none",
        env=(("PATH", f"{tmp_path}:/usr/bin:/bin"), ("HOME", str(tmp_path))),
    )
    session = jail_session.JailSession.open(policy)
    try:
        assert session.executable_available(tool.name)
        assert not session.executable_available("agent6-missing-tool")
        result = session.run((tool.name,), env=policy.env)
        assert isinstance(result, kinds.CommandResult) and result.ok
    finally:
        session.close()


@pytest.mark.parametrize(
    "isolation", [pytest.param("full", marks=pytest.mark.needs_namespaces), "partial"]
)
def test_lookup_respects_target_filesystem_visibility(isolation: kinds.IsolationLevel) -> None:
    """Host-visible PATH entries do not bypass mounts or Landlock during target lookup."""
    if isolation == "partial" and detect.probed_environment().landlock_abi < 1:
        pytest.skip("partial isolation requires Landlock")
    # Partial grants /tmp wholesale; neither the excluded tools nor an ancestor may be granted.
    base = pathlib.Path("/var/tmp")
    if not (base.is_dir() and os.access(base, os.W_OK)):
        pytest.skip("/var/tmp is not writable here")
    with tempfile.TemporaryDirectory(prefix="agent6-target-lookup-", dir=base) as directory:
        root = pathlib.Path(directory).resolve()
        workspace, excluded = root / "workspace", root / "excluded"
        workspace.mkdir()
        excluded.mkdir()
        unavailable, available = excluded / "host-only", workspace / "target-tool"
        for tool in (unavailable, available, excluded / available.name):
            tool.write_text("#!/bin/sh\nprintf 'ran\\n'\n")
            tool.chmod(0o755)
        search_path = f"{excluded}:{workspace}:/usr/bin:/bin"
        assert shutil.which(unavailable.name, path=search_path) == str(unavailable)
        assert shutil.which(available.name, path=search_path) == str(excluded / available.name)
        policy = kinds.JailPolicy(
            cwd=workspace, argv=(), isolation=isolation, env=(("PATH", search_path),)
        )
        session = jail_session.JailSession.open(policy)
        try:
            assert not session.executable_available(unavailable.name)
            # An inaccessible match must not stop lookup before the later accessible entry.
            assert session.executable_available(available.name)
            denied = session.run((unavailable.name,), env=policy.env)
            assert isinstance(denied, kinds.CommandResult) and denied.exec_failed
            allowed = session.run((available.name,), env=policy.env)
            assert isinstance(allowed, kinds.CommandResult) and allowed.ok
            assert allowed.stdout == "ran\n"
        finally:
            assert session.close() == ""


@pytest.mark.parametrize(
    "isolation", [pytest.param("full", marks=pytest.mark.needs_namespaces), "partial", "none"]
)
@pytest.mark.parametrize("readable_alternative", [False, True])
def test_execute_only_lookup_is_indeterminate_not_missing(
    tmp_path: pathlib.Path, isolation: kinds.IsolationLevel, readable_alternative: bool
) -> None:
    """Read denial cannot discard an executable gate or hide a later readable PATH match."""
    if os.geteuid() == 0:
        pytest.skip("root can read execute-only files")
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    tool = first / "true"
    shutil.copyfile("/bin/true", tool)
    assert tool.read_bytes()[:4] == b"\x7fELF"
    tool.chmod(0o111)
    assert not os.access(tool, os.R_OK)
    if readable_alternative:
        alternative = second / "true"
        shutil.copyfile("/bin/true", alternative)
        alternative.chmod(0o755)
    search_path = f"{first}:{second}"
    assert shutil.which("true", path=search_path) == str(tool)
    policy = kinds.JailPolicy(
        cwd=tmp_path, argv=(), isolation=isolation, env=(("PATH", search_path),)
    )
    session = jail_session.JailSession.open(policy)
    try:
        result = session.run(("true",), env=policy.env)
        assert isinstance(result, kinds.CommandResult) and result.ok
        if readable_alternative:
            inferred = verify_infer.infer_verify_command(
                tmp_path, "Verify: true", available=session.executable_available
            )
            assert inferred.argv == ("true",)
        else:
            with pytest.raises(jail.JailUnavailableError, match=r"execute-only.*indeterminate"):
                verify_infer.infer_verify_command(
                    tmp_path, "Verify: true", available=session.executable_available
                )
        explicit = verify_infer.infer_verify_command(
            tmp_path, "Verify: ./first/true", available=session.executable_available
        )
        assert explicit.argv == ("./first/true",)
        result = session.run(explicit.argv, env=policy.env)
        assert isinstance(result, kinds.CommandResult) and result.ok
    finally:
        assert session.close() == ""


@pytest.mark.parametrize(
    "isolation", [pytest.param("full", marks=pytest.mark.needs_namespaces), "partial", "none"]
)
@pytest.mark.parametrize(
    ("script_mode", "shell_available", "expected"),
    [
        pytest.param(None, True, None, id="absent"),
        pytest.param(0o644, False, None, id="no-shell"),
        pytest.param(0o644, True, ("sh", "verify.sh"), id="shell"),
        pytest.param(0o755, False, ("./verify.sh",), id="executable"),
    ],
)
def test_unused_npm_lookup_cannot_block_verify_sh(
    tmp_path: pathlib.Path,
    isolation: kinds.IsolationLevel,
    script_mode: int | None,
    shell_available: bool,
    expected: tuple[str, ...] | None,
) -> None:
    """Execute-only npm matters only when inference reaches its package.json candidate."""
    if os.geteuid() == 0:
        pytest.skip("root can read execute-only files")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    npm = bin_dir / "npm"
    shutil.copyfile("/bin/true", npm)
    assert npm.read_bytes()[:4] == b"\x7fELF"
    npm.chmod(0o111)
    assert not os.access(npm, os.R_OK)
    if shell_available:
        (bin_dir / "sh").symlink_to("/bin/sh")
    if script_mode is not None:
        script = tmp_path / "verify.sh"
        script.write_text("#!/bin/sh\nprintf verified\n")
        script.chmod(script_mode)
    (tmp_path / "package.json").write_text('{"scripts": {"test": "node --test"}}')
    policy = kinds.JailPolicy(
        cwd=tmp_path, argv=(), isolation=isolation, env=(("PATH", str(bin_dir)),)
    )
    session = jail_session.JailSession.open(policy)
    lookups: list[str] = []

    def available(name: str) -> bool:
        lookups.append(name)
        return session.executable_available(name)

    try:
        if expected is None:
            with pytest.raises(
                jail.JailUnavailableError, match=r"npm: execute-only.*indeterminate"
            ):
                verify_infer.infer_verify_command(tmp_path, "", available=available)
            assert lookups[-1] == "npm"
        else:
            inferred = verify_infer.infer_verify_command(tmp_path, "", available=available)
            assert (inferred.argv, inferred.source, inferred.passed_over) == (
                expected,
                "verify.sh",
                (),
            )
            assert lookups == [expected[0]]
            result = session.run(inferred.argv, env=policy.env)
            assert isinstance(result, kinds.CommandResult) and result.ok
            assert result.stdout == "verified"
    finally:
        assert session.close() == ""


@pytest.mark.parametrize("script", ["exit 0", "cd ."])
def test_inference_rejects_builtin_only_direct_argv(tmp_path: pathlib.Path, script: str) -> None:
    """A builtin is not a direct executable when npm cannot wrap the test script."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "sh").symlink_to("/bin/sh")
    (tmp_path / "package.json").write_text(json.dumps({"scripts": {"test": script}}))
    policy = kinds.JailPolicy(
        cwd=tmp_path, argv=(), isolation="none", env=(("PATH", str(bin_dir)),)
    )
    session = jail_session.JailSession.open(policy)
    try:
        assert not session.executable_available("npm")
        inferred = verify_infer.infer_verify_command(
            tmp_path, "", available=session.executable_available
        )
        assert not inferred.argv
        assert not session.executable_available(script.split(maxsplit=1)[0])
        wrapped = verify_infer.infer_verify_command(
            tmp_path, f"Verify: sh -c '{script}'", available=session.executable_available
        )
        assert wrapped.argv == ("sh", "-c", script)
        result = session.run(wrapped.argv, env=policy.env)
        assert isinstance(result, kinds.CommandResult) and result.ok
    finally:
        session.close()


def test_lookup_does_not_execute_a_path_shadowed_shell(tmp_path: pathlib.Path) -> None:
    """The fixed probe cannot execute a checkout's sh before command approval."""
    marker = tmp_path / "unapproved"
    shell = tmp_path / "sh"
    shell.write_text("#!/bin/sh\nprintf compromised > unapproved\nexit 0\n")
    shell.chmod(0o755)
    policy = kinds.JailPolicy(
        cwd=tmp_path, argv=(), isolation="none", env=(("PATH", str(tmp_path)),)
    )
    session = jail_session.JailSession.open(policy)
    try:
        available = session.executable_available("agent6-missing-tool")
        assert not marker.exists(), "lookup executed the workspace shell"
        assert not available
    finally:
        session.close()


@pytest.mark.parametrize("search_path", ["bin", "other:bin", ":other", "other:", ""])
def test_lookup_checks_executable_files_even_with_builtin_names(
    tmp_path: pathlib.Path, search_path: str
) -> None:
    """Relative and empty PATH entries resolve files without executing a builtin or candidate."""
    bin_dir = tmp_path / "bin" if "bin" in search_path else tmp_path
    bin_dir.mkdir(exist_ok=True)
    (tmp_path / "other").mkdir()
    tool = bin_dir / "exit"
    tool.write_text("#!/bin/sh\nprintf ran > candidate-ran\n")
    tool.chmod(0o755)
    (bin_dir / "directory").mkdir()
    not_executable = bin_dir / "unexecutable"
    not_executable.write_text("no execute bit")
    policy = kinds.JailPolicy(cwd=tmp_path, argv=(), isolation="none", env=(("PATH", search_path),))
    session = jail_session.JailSession.open(policy)
    try:
        assert session.executable_available("exit")
        assert not session.executable_available("if")
        assert not session.executable_available("unexecutable")
        assert not session.executable_available("directory")
        assert not (tmp_path / "candidate-ran").exists()
    finally:
        session.close()


@pytest.mark.parametrize(
    "probe,timed_out", [("exit 126", False), ('kill -TERM "$$"', False), ("sleep 30", True)]
)
def test_abnormal_lookup_refuses_inference(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, probe: str, timed_out: bool
) -> None:
    """A failed probe cannot discard a repository gate as though its executable were absent."""
    policy = kinds.JailPolicy(cwd=tmp_path, argv=(), isolation="none")
    session = jail_session.JailSession.open(policy)
    real_run = jail_session.JailSession.run
    outcomes: list[kinds.CommandResult] = []

    def run(
        self: jail_session.JailSession,
        argv: tuple[str, ...],
        *,
        env: tuple[tuple[str, str], ...],
        timeout_s: float,
    ) -> kinds.CommandResult:
        result = real_run(self, argv, env=env, timeout_s=0.1 if timed_out else timeout_s)
        assert isinstance(result, kinds.CommandResult)
        outcomes.append(result)
        return result

    monkeypatch.setattr(jail_session, "_EXECUTABLE_LOOKUP", probe)
    monkeypatch.setattr(jail_session.JailSession, "run", run)
    try:
        with pytest.raises(jail.JailUnavailableError, match="lookup"):
            verify_infer.infer_verify_command(
                tmp_path, "Verify: target-tool check", available=session.executable_available
            )
        assert len(outcomes) == 1
        assert not outcomes[0].exec_failed
        assert outcomes[0].timed_out is timed_out
    finally:
        session.close()


@pytest.mark.parametrize("returncode", [0, 1])
def test_timed_out_lookup_refuses_even_with_a_normal_exit_code(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, returncode: int
) -> None:
    """A timed-out outcome overrides both found and absent exit codes."""
    policy = kinds.JailPolicy(cwd=tmp_path, argv=(), isolation="none")
    session = jail_session.JailSession.open(policy)
    real_run = jail_session.JailSession.run

    def run(
        self: jail_session.JailSession,
        argv: tuple[str, ...],
        *,
        env: tuple[tuple[str, str], ...],
        timeout_s: float,
    ) -> kinds.CommandResult:
        result = real_run(self, argv, env=env, timeout_s=timeout_s)
        assert isinstance(result, kinds.CommandResult)
        return dataclasses.replace(result, returncode=returncode, timed_out=True)

    monkeypatch.setattr(jail_session.JailSession, "run", run)
    try:
        with pytest.raises(jail.JailUnavailableError, match="lookup timed out"):
            session.executable_available("sh")
    finally:
        session.close()


@pytest.mark.parametrize(
    "isolation", [pytest.param("full", marks=pytest.mark.needs_namespaces), "partial", "none"]
)
@pytest.mark.parametrize(
    "shell_mode", [None, 0o111, 0o755], ids=["missing", "execute-only", "readable"]
)
def test_shell_builtin_inference_ignores_an_execute_only_namesake(
    tmp_path: pathlib.Path, isolation: kinds.IsolationLevel, shell_mode: int | None
) -> None:
    """A wrapped builtin needs its shell, not an external namesake's read permission."""
    if os.geteuid() == 0:
        pytest.skip("root can read execute-only files")
    namesake = tmp_path / "cd"
    shutil.copyfile("/bin/false", namesake)
    assert namesake.read_bytes()[:4] == b"\x7fELF"
    namesake.chmod(0o111)
    assert not os.access(namesake, os.R_OK)
    if shell_mode is not None:
        shell = tmp_path / "sh"
        shutil.copyfile("/bin/sh", shell)
        shell.chmod(shell_mode)
    policy = kinds.JailPolicy(
        cwd=tmp_path, argv=(), isolation=isolation, env=(("PATH", str(tmp_path)),)
    )
    session = jail_session.JailSession.open(policy)
    try:
        with pytest.raises(jail.JailUnavailableError, match=r"cd: execute-only.*indeterminate"):
            verify_infer.infer_verify_command(
                tmp_path, "Verify: cd .", available=session.executable_available
            )
        result = session.run(("sh", "-c", "cd ."), env=policy.env)
        assert isinstance(result, kinds.CommandResult)
        assert result.ok is (shell_mode is not None)
        if shell_mode == 0o111:
            with pytest.raises(jail.JailUnavailableError, match=r"sh: execute-only.*indeterminate"):
                verify_infer.infer_verify_command(
                    tmp_path, "Verify: sh -c 'cd .'", available=session.executable_available
                )
        else:
            inferred = verify_infer.infer_verify_command(
                tmp_path, "Verify: sh -c 'cd .'", available=session.executable_available
            )
            if shell_mode is None:
                assert result.exec_failed
                assert not inferred.argv
                assert "sandbox has no `sh`" in inferred.passed_over[0][1]
            else:
                assert inferred.argv == ("sh", "-c", "cd .")
                assert not inferred.passed_over
    finally:
        assert session.close() == ""


def test_shell_wrapped_builtin_needs_a_shell_on_the_target_path(tmp_path: pathlib.Path) -> None:
    """The probe's trusted interpreter does not grant an inferred bare sh command."""
    policy = kinds.JailPolicy(
        cwd=tmp_path, argv=(), isolation="none", env=(("PATH", str(tmp_path)),)
    )
    session = jail_session.JailSession.open(policy)
    try:
        inferred = verify_infer.infer_verify_command(
            tmp_path, "Verify: sh -c 'cd .'", available=session.executable_available
        )
        assert not inferred.argv
    finally:
        session.close()
