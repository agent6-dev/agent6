from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time
import traceback

import pytest

from agent6 import kinds
from agent6.sandbox import container, container_install, jail, jail_session
from tests.sandbox import conftest


def test_explicit_launcher_validation_cannot_skip(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit launcher rejected by the validator must fail, not skip the suite."""
    binary = jail.require_jail_binary().resolve()
    monkeypatch.setenv("AGENT6_JAIL_BIN", str(binary))

    def refuse() -> tuple[bytes, str]:
        raise jail.JailUnavailableError("validator regression")

    monkeypatch.setattr(container_install, "launcher", refuse)
    try:
        selected = inspect.unwrap(conftest.static_launcher)(monkeypatch)
    except pytest.skip.Exception:
        pytest.fail("fixture skipped an explicitly provisioned launcher")
    assert selected == binary
    discovery = container.discover(
        tmp_path, container.ContainerOptions(exec=("/must-not-run-bootstrap",))
    )
    with pytest.raises(container.ContainerError, match="validator regression"):
        container.prepare(discovery)


def test_nonroot_environment_installation_and_capture(
    target: conftest.Target, static_launcher: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bootstrap and launcher preserve a non-root user's HOME and toolchain."""
    assert os.getuid() != 0, "run the environment pin as a non-root user"
    expected_version = "sha256:" + hashlib.sha256(static_launcher.read_bytes()).hexdigest()
    monkeypatch.setenv("HOME", "/host-home-must-not-cross")
    monkeypatch.setenv("PATH", "/host-path-must-not-cross")
    monkeypatch.setenv("TMPDIR", "/host-tmp-must-not-cross")
    with conftest.opened(target) as session:
        assert session.target is not None
        assert session.target.user == f"{os.getuid()}:{os.getgid()}"
        assert session.target.home == str(target.home)
        assert session.target.lifecycle_ref is None
        assert session.target.launcher_instance
        assert session.target.launcher_version == expected_version
        assert session.executable_available("inside-only")
        assert not session.executable_available("not-a-real-executable")
        result = session.run(
            ("/bin/sh", "-c", 'printf "%s\\n" "$HOME" "$PATH" "$TMPDIR"; inside-only')
        )
        assert isinstance(result, kinds.CommandResult)
        assert result.returncode == 0, result
        home, path, temp, inside = result.stdout.splitlines()
        assert home == str(target.home)
        assert path == f"{target.tool_dir}:/usr/bin:/bin"
        assert inside == "inside"
        assert temp.startswith("/tmp/a6.") and len(temp) < 40
        assert pathlib.Path(temp).stat().st_uid == os.getuid()
        assert pathlib.Path(temp).stat().st_mode & 0o777 == 0o700
        handle = session.start_background(
            ("/bin/sh", "-c", "sleep 0.25; printf captured; sleep 30")
        )
        deadline = time.monotonic() + 5
        while True:
            capture = session.read_background(handle, offset=0)
            if capture.data == b"captured" or time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        assert capture.data == b"captured" and not capture.capture_error
        assert not session.stop_background(handle).error
    assert not pathlib.Path(temp).exists()
    assert not list(target.discovery.target.host_root.glob(".agent6-probe-*"))


def test_launcher_version_tracks_payload_changes_at_the_same_path(
    target: conftest.Target,
    static_launcher: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build identity hashes the transferred bytes, not the selected path or a constant."""
    binary = static_launcher.read_bytes()
    selected = tmp_path / "selected-launcher"
    monkeypatch.setenv("AGENT6_JAIL_BIN", str(selected))
    versions: set[str | None] = set()
    # ELF permits trailing data; both payloads must install and execute successfully.
    for trailer in (b"", b"\0agent6-build-identity-pin\0"):
        payload = binary + trailer
        selected.write_bytes(payload)
        selected.chmod(0o755)
        with conftest.opened(target) as session:
            result = session.run(("/bin/echo", "payload-ran"))
            assert isinstance(result, kinds.CommandResult)
            assert result.returncode == 0 and result.stdout == "payload-ran\n", result
            assert session.target is not None
            version = session.target.launcher_version
            assert version == "sha256:" + hashlib.sha256(payload).hexdigest()
            versions.add(version)
    assert len(versions) == 2


@pytest.mark.parametrize("mask", [0o002, 0o022, 0o027, 0o077])
def test_installation_preserves_the_target_umask(target: conftest.Target, mask: int) -> None:
    """The private installation mask must not change command-created workspace permissions."""
    prefix = (
        *target.discovery.exec,
        "/bin/sh",
        "-ec",
        'umask "$1"; shift; exec "$@"',
        "target-umask",
        f"{mask:03o}",
    )
    discovery = dataclasses.replace(target.discovery, exec=prefix)
    with conftest.opened(dataclasses.replace(target, discovery=discovery)) as session:
        temp = pathlib.Path(session.probe().environment["TMPDIR"])
        assert temp.stat().st_mode & 0o777 == 0o700
        result = session.run(
            ("/bin/sh", "-ec", "umask; printf generated > generated; mkdir generated-dir")
        )
        assert isinstance(result, kinds.CommandResult) and result.returncode == 0, result
        root = discovery.target.host_root
        observed = (
            int(result.stdout.strip(), 8),
            (root / "generated").stat().st_mode & 0o777,
            (root / "generated-dir").stat().st_mode & 0o777,
        )
        assert observed == (mask, 0o666 & ~mask, 0o777 & ~mask)
    assert not temp.exists()


@pytest.mark.needs_namespaces
def test_requested_user_owns_the_installation(
    host_tmp_path: pathlib.Path, static_launcher: pathlib.Path
) -> None:
    """Dropping explicit user selection must not pass as the namespace's default root user."""
    uid, gid = 1000, 1001
    mapping = ("unshare", "-U", "--map-root-user", "--map-auto")
    selected = (*mapping, "--setuid", str(uid), "--setgid", str(gid))
    available = subprocess.run(
        (*selected, "/bin/true"), check=False, capture_output=True, text=True
    )
    if available.returncode != 0:
        pytest.skip("subordinate UID/GID mappings unavailable: " + available.stderr.strip())
    host_tmp_path.chmod(0o755)
    root, home = host_tmp_path / "workspace", host_tmp_path / "home"
    root.mkdir()
    home.mkdir()
    default = subprocess.check_output(
        (*mapping, "/bin/sh", "-ec", "id -u; id -g"), text=True
    ).splitlines()
    assert default == ["0", "0"]
    discovery = container.discover(
        root,
        container.ContainerOptions(
            exec=(
                *selected,
                "/usr/bin/env",
                "-i",
                "HOME=" + str(home),
                "PATH=/usr/bin:/bin",
            )
        ),
    )
    session = container.prepare(discovery)
    try:
        conftest.assert_selected_user(session, uid, gid)
    finally:
        assert not session.close()


@pytest.mark.needs_namespaces
def test_executable_lookup_uses_the_target_filesystem(target: conftest.Target) -> None:
    """A host-side lookup must not substitute for the prepared target's filesystem view."""
    prefix = (
        "unshare",
        "-Ur",
        "-m",
        "/bin/sh",
        "-ec",
        'mount -t tmpfs tmpfs "$1"; shift; exec "$@"',
        "target-tools",
        str(target.tool_dir),
        *target.discovery.exec,
    )
    discovery = dataclasses.replace(target.discovery, exec=prefix)
    with conftest.opened(dataclasses.replace(target, discovery=discovery)) as session:
        conftest.assert_target_only_executable(session, str(target.tool_dir))


@pytest.mark.needs_namespaces
def test_prepared_execution_replays_container_receipt(target: conftest.Target) -> None:
    """Container observations survive execution journaling and receipt replay."""
    root = target.discovery.target.host_root
    for argv in (("init", "-q"), ("commit", "-qm", "seed", "--allow-empty")):
        subprocess.run(("git", "-C", str(root), *argv), check=True, capture_output=True)
    destination = target.home.parent / "command workspace"
    destination.mkdir()
    prefix = (
        "unshare",
        "-Ur",
        "-m",
        "/bin/sh",
        "-ec",
        'mount --bind "$1" "$2"; shift 2; exec "$@"',
        "receipt-target",
        str(root),
        str(destination),
        *target.discovery.exec,
    )
    discovery = container.discover(
        root, container.ContainerOptions(exec=prefix, workspace=destination)
    )
    with conftest.opened(dataclasses.replace(target, discovery=discovery)) as session:
        print(conftest.assert_execution_receipt(session))


@pytest.mark.needs_namespaces
@pytest.mark.parametrize("readonly", [False, True])
def test_different_roots_prove_both_directions(target: conftest.Target, readonly: bool) -> None:
    """A read-only bind mount cannot complete the two-way nonce exchange."""
    destination = target.home.parent / "command workspace"
    destination.mkdir()
    prepared = target.home / "mount-prepared"
    script = 'mount --bind "$1" "$2"; '
    if readonly:
        script += 'mount -o remount,bind,ro "$2"; '
    script += ': > "$3"; shift 3; exec "$@"'
    prefix = (
        "unshare",
        "-Ur",
        "-m",
        "/bin/sh",
        "-ec",
        script,
        "mount-test",
        str(target.discovery.target.host_root),
        str(destination),
        str(prepared),
        *target.discovery.exec,
    )
    discovery = container.discover(
        target.discovery.target.host_root,
        container.ContainerOptions(exec=prefix, workspace=destination),
    )
    if readonly:
        with pytest.raises(
            container.ContainerError, match=r"mapping.*Read-only file system \(os error 30\)"
        ):
            container.prepare(discovery)
    else:
        session = container.prepare(discovery)
        try:
            result = session.run(("/bin/pwd",))
            assert isinstance(result, kinds.CommandResult)
            assert result.stdout.strip() == str(destination)
        finally:
            session.close()
    assert not list(target.discovery.target.host_root.glob(".agent6-probe-*"))
    assert prepared.is_file(), "mount setup must complete before the launcher starts"
    assert not list(destination.iterdir())


def test_mismatched_mapping_refuses(target: conftest.Target) -> None:
    discovery = dataclasses.replace(
        target.discovery,
        target=dataclasses.replace(target.discovery.target, command_root=target.home),
    )
    with pytest.raises(container.ContainerError, match=r"mapping|mount"):
        container.prepare(discovery)
    assert not list(target.discovery.target.host_root.glob(".agent6-probe-*"))


def test_symlinked_mapping_refuses_without_touching_referent(target: conftest.Target) -> None:
    link = target.home.parent / "linked-workspace"
    link.symlink_to(target.discovery.target.host_root, target_is_directory=True)
    discovery = dataclasses.replace(
        target.discovery,
        target=dataclasses.replace(target.discovery.target, command_root=link),
    )
    with pytest.raises(container.ContainerError, match=r"mapping|symlink|directory"):
        container.prepare(discovery)
    assert not list(target.discovery.target.host_root.iterdir())


@pytest.mark.parametrize("damage", ["architecture", "dynamic", "not-elf", "truncated"])
def test_incompatible_launcher_refuses_before_bootstrap(
    target: conftest.Target,
    static_launcher: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    binary = target.home / "bad-launcher"
    data = bytearray(static_launcher.read_bytes())
    if damage == "architecture":
        data[18:20] = (183 if os.uname().machine == "x86_64" else 62).to_bytes(2, "little")
    elif damage == "dynamic":
        data = bytearray(pathlib.Path("/bin/sh").resolve().read_bytes())
    elif damage == "not-elf":
        data[:4] = b"oops"
    else:
        data = data[:60]
    binary.write_bytes(data)
    monkeypatch.setenv("AGENT6_JAIL_BIN", str(binary))
    discovery = dataclasses.replace(target.discovery, exec=("/must-not-run-bootstrap",))
    with pytest.raises(container.ContainerError, match=r"ELF|static|architecture"):
        container.prepare(discovery)


def test_busybox_head_cannot_consume_the_policy_frame(
    target: conftest.Target, static_launcher: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Buffered head must finish before policy bytes enter the shared stream."""
    busybox = shutil.which("busybox")
    if busybox is None:
        pytest.skip("BusyBox is required for the buffered bootstrap regression")
    (target.tool_dir / "head").symlink_to(busybox)
    binary = target.home / "unaligned-launcher"
    data = static_launcher.read_bytes()
    binary.write_bytes(data + b"\0" * ((1 - len(data)) % 4096))
    assert binary.stat().st_size % 4096 == 1
    monkeypatch.setenv("AGENT6_JAIL_BIN", str(binary))
    with conftest.opened(target) as session:
        result = session.run(("/bin/echo", "policy-intact"))
        assert isinstance(result, kinds.CommandResult)
        assert result.returncode == 0 and result.stdout == "policy-intact\n"


def test_missing_bootstrap_utilities_name_the_requirements(target: conftest.Target) -> None:
    discovery = dataclasses.replace(target.discovery, exec=("/usr/bin/env", "-i", "PATH=/absent"))
    with pytest.raises(jail.JailUnavailableError, match=r"bootstrap.*(uname|mktemp|head|chmod)"):
        container.prepare(discovery)


def test_target_architecture_mismatch_refuses(target: conftest.Target) -> None:
    uname = target.tool_dir / "uname"
    uname.write_text("#!/bin/sh\necho incompatible-cpu\n")
    uname.chmod(0o755)
    with pytest.raises(jail.JailUnavailableError, match="architecture"):
        container.prepare(target.discovery)


def test_runtime_stderr_flood_does_not_block_channel(target: conftest.Target) -> None:
    prefix = (
        sys.executable,
        "-c",
        "import os,sys; os.write(2,b'x'*2000000); os.execv(sys.argv[1],sys.argv[1:])",
        *target.discovery.exec,
    )
    with conftest.opened(
        dataclasses.replace(target, discovery=dataclasses.replace(target.discovery, exec=prefix))
    ) as session:
        assert "truncated" in session.startup_stderr
        result = session.run(("/bin/true",))
        assert isinstance(result, kinds.CommandResult) and result.returncode == 0


def test_replacement_launcher_cannot_claim_old_jobs(target: conftest.Target) -> None:
    """A fresh launcher has no authority over a lost launcher's job handles."""
    with conftest.opened(target) as first:
        handle = first.start_background(("/bin/sleep", "30"))
        assert first.target is not None
        instance = first.target.launcher_instance
        proc = first._proc  # pyright: ignore[reportPrivateUsage]
        proc.kill()
        proc.wait(timeout=5)
        with conftest.opened(target) as replacement:
            assert replacement.target is not None
            assert replacement.target.launcher_instance != instance
            status = replacement.status_background(handle)
            assert status.returncode is None and status.error == "unknown job handle"


@pytest.mark.parametrize("entry", ["host", "target"])
def test_probe_entries_refuse_symlinks(target: conftest.Target, entry: str) -> None:
    """Neither direction of the mount proof may read or overwrite a symlink target."""
    with conftest.opened(target) as session:
        name = ".agent6-probe-" + "a" * 32
        directory = target.discovery.target.host_root / name
        directory.mkdir()
        victim = target.home / "victim"
        nonce = "a" * 64
        contents = nonce if entry == "host" else "do-not-touch"
        victim.write_text(contents)
        (directory / "host").write_text(nonce)
        (directory / entry).unlink(missing_ok=True)
        (directory / entry).symlink_to(victim)
        try:
            with pytest.raises(jail.JailUnavailableError, match="mapping"):
                session.prove_mapping(name, nonce)
            assert victim.read_text() == contents
        finally:
            for path in directory.iterdir():
                path.unlink()
            directory.rmdir()


@pytest.mark.parametrize("option", ["--tty", "--tty=true", "-i", "-it", "-t", "-ti"])
def test_tty_prefix_refuses_before_bootstrap(target: conftest.Target, option: str) -> None:
    """TTY options refuse before spawning; -i still opens the JSON transport."""
    marker = target.home / "bootstrap-started"
    prefix = (
        sys.executable,
        "-c",
        "import os,pathlib,sys; pathlib.Path(sys.argv[1]).write_text(sys.argv[2]); "
        "os.execv(sys.argv[3],sys.argv[3:])",
        str(marker),
        option,
        *target.discovery.exec,
    )
    discovery = container.discover(
        target.discovery.target.host_root, container.ContainerOptions(exec=prefix)
    )
    if option == "-i":
        with conftest.opened(dataclasses.replace(target, discovery=discovery)) as session:
            result = session.run(("/bin/echo", "pipe-intact"))
            assert isinstance(result, kinds.CommandResult) and result.returncode == 0, result
            assert result.stdout == "pipe-intact\n"
        assert marker.read_text() == option
    else:
        with pytest.raises(container.ContainerError, match=r"container\.exec.*tty.*JSON channel"):
            session = container.prepare(discovery)
            session.close()
        assert not marker.exists(), "TTY prefix ran before preparation refused"


def test_bootstrap_runs_once_and_never_receives_model_argv(target: conftest.Target) -> None:
    log = target.home / "exec-argv.jsonl"
    prefix = (
        sys.executable,
        "-c",
        "import json,os,sys; "
        f"open({str(log)!r},'a').write(json.dumps(sys.argv[1:])+'\\n'); "
        "os.execv(sys.argv[1],sys.argv[1:])",
        *target.discovery.exec,
    )
    with conftest.opened(
        dataclasses.replace(target, discovery=dataclasses.replace(target.discovery, exec=prefix))
    ) as session:
        probe = session.probe()
        assert probe.uid == os.getuid() and probe.gid == os.getgid()
        assert probe.mounts and probe.architecture == os.uname().machine
        for _ in range(2):
            result = session.run(("/bin/echo", "MODEL-ARGV-;$(not-a-command)"))
            assert isinstance(result, kinds.CommandResult)
            assert result.stdout == "MODEL-ARGV-;$(not-a-command)\n"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 1
    assert "MODEL-ARGV" not in str(rows)


@pytest.mark.parametrize("protocol,changed", [(999, False), (4, True)])
def test_protocol_and_instance_changes_refuse(
    tmp_path: pathlib.Path, protocol: int, changed: bool
) -> None:
    target = container.discover(tmp_path, container.ContainerOptions(exec=(sys.executable,))).target
    script = (
        "import json,sys\n"
        "sys.stdin.buffer.readline()\n"
        f"print(json.dumps({{'ready':True,'protocol':{protocol},'instance':'a'*32}}),flush=True)\n"
        "request=json.loads(sys.stdin.buffer.readline())\n"
        "print(json.dumps({'id':request['id'],'instance':'b'*32,'returncode':0}),flush=True)\n"
    )
    proc = subprocess.Popen(
        (sys.executable, "-c", script),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        if changed:
            session = jail_session.JailSession.connect(
                proc,
                kinds.JailPolicy(cwd=tmp_path, argv=(), isolation="none"),
                target=target,
            )
            with pytest.raises(jail.JailUnavailableError, match="instance"):
                session.run(("/bin/true",))
        else:
            with pytest.raises(jail.JailUnavailableError, match="protocol"):
                jail_session.JailSession.connect(
                    proc,
                    kinds.JailPolicy(cwd=tmp_path, argv=(), isolation="none"),
                    target=target,
                )
    finally:
        if proc.poll() is None:
            os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream:
                stream.close()


def test_non_utf8_probe_environment_never_discloses_its_value(target: conftest.Target) -> None:
    """A real launcher must refuse non-UTF-8 environment bytes without a panic disclosure."""
    discovery = dataclasses.replace(
        target.discovery,
        exec=(*target.discovery.exec, "SECRET=" + os.fsdecode(b"private-canary\xff")),
    )
    with pytest.raises(container.ContainerError) as error:
        container.prepare(discovery)
    diagnostic = "".join(traceback.format_exception(error.value))
    assert "private-canary" not in diagnostic
    assert "panicked" not in diagnostic
    assert "environment" in diagnostic and "UTF-8" in diagnostic


@pytest.mark.parametrize("envelope", [False, True])
def test_invalid_probe_does_not_expose_environment_values(
    tmp_path: pathlib.Path, envelope: bool
) -> None:
    target = container.discover(tmp_path, container.ContainerOptions(exec=(sys.executable,))).target
    script = (
        "import json,sys\n"
        "sys.stdin.readline()\n"
        "print(json.dumps({'ready':True,'protocol':4,'instance':'a'*32}),flush=True)\n"
        "request=json.loads(sys.stdin.readline())\n"
        f"print(json.dumps({{'{'missing_id' if envelope else 'id'}':request['id'],"
        "'instance':'a'*32,'probe':{"
        "'uid':1,'gid':1,'architecture':'x86_64','mounts':'',"
        "'environment':{'SECRET':['probe-secret-canary']}}}),flush=True)\n"
        "sys.stdin.read()\n"
    )
    proc = subprocess.Popen(
        (sys.executable, "-c", script),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    session = jail_session.JailSession.connect(
        proc,
        kinds.JailPolicy(cwd=tmp_path, argv=(), isolation="none"),
        target=target,
    )
    try:
        with pytest.raises(jail.JailUnavailableError, match="invalid") as error:
            session.probe()
        assert "probe-secret-canary" not in "".join(traceback.format_exception(error.value))
    finally:
        session.close()
