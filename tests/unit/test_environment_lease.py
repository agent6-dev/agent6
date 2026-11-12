# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""An environment lease: the run that started a container owns it; attaching grants nothing."""

from __future__ import annotations

import json
import os
import pathlib
import sys
from unittest import mock

import pytest

from agent6 import kinds, paths
from agent6.app import environment_lease, session_runtime
from agent6.app import reporter as app_reporter
from agent6.config import Config
from agent6.sandbox import container
from agent6.sessions import journal, layout, process

_ID = "a" * 64
_OTHER = "b" * 64
_CONFIG: dict[str, object] = {"User": "1001:1001", "Labels": {}, "Env": []}


class Fake:
    """A runtime whose `ps` answers from marker files, and start and stop scripts that set them."""

    def __init__(self, tmp_path: pathlib.Path, workspace: pathlib.Path) -> None:
        self.dir = tmp_path / "fake"
        self.dir.mkdir()
        self.log = self.dir / "calls.jsonl"
        self.markers = self.dir / "running"
        self.markers.mkdir()
        self.runtime = self.dir / "docker"
        row: dict[str, object] = {
            "Id": _ID,
            "State": {"Running": True, "Status": "running"},
            "Config": {**_CONFIG, "Labels": {"devcontainer.local_folder": str(workspace)}},
            "Mounts": [
                {"Type": "bind", "Source": str(workspace), "Destination": "/work", "RW": True}
            ],
            "Name": "/devcontainer",
        }
        self.runtime.write_text(
            f"#!{sys.executable}\n"
            "import json, pathlib, sys\n"
            f"log = pathlib.Path({str(self.log)!r})\n"
            f"markers = pathlib.Path({str(self.markers)!r})\n"
            "with log.open('a') as fh:\n"
            "    fh.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "if sys.argv[1] == 'ps':\n"
            "    for m in sorted(markers.iterdir()):\n"
            "        print(m.name)\n"
            "elif sys.argv[1] == 'inspect':\n"
            f"    row = {row!r}\n"
            "    row['Id'] = sys.argv[-1]\n"
            "    print(json.dumps([row]))\n",
            encoding="utf-8",
        )
        self.runtime.chmod(0o755)

    def script(self, name: str, body: str) -> tuple[str, ...]:
        path = self.dir / name
        path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
        path.chmod(0o755)
        return (str(path),)

    def running(self, *ids: str) -> None:
        for child in self.markers.iterdir():
            child.unlink()
        for id_ in ids:
            (self.markers / id_).touch()

    def calls(self) -> list[list[str]]:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def options(self, **over: object) -> container.ContainerOptions:
        return container.ContainerOptions(runtime=(str(self.runtime),), **over)  # pyright: ignore[reportArgumentType]


@pytest.fixture
def workspace(tmp_path: pathlib.Path) -> pathlib.Path:
    root = tmp_path / "workspace"
    root.mkdir()
    return root


@pytest.fixture
def fake(tmp_path: pathlib.Path, workspace: pathlib.Path) -> Fake:
    return Fake(tmp_path, workspace)


@pytest.fixture
def state_dir(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, workspace: pathlib.Path
) -> pathlib.Path:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    return paths.state_dir(workspace)


def _live_session(state_dir: pathlib.Path, session_id: str, *, alive: bool) -> None:
    saved = layout.SessionLayout(state_dir, session_id)
    saved.ensure()
    # A dead worker: a pid no process holds; the kernel's pid max bounds the live ones.
    process.write_worker_pid(saved.session_dir, os.getpid() if alive else 2**22 - 1)


def _start(fake: Fake) -> tuple[str, ...]:
    return fake.script("start", f"touch {fake.markers}/{_ID}")


def _stop(fake: Fake) -> tuple[str, ...]:
    return fake.script("stop", f"rm -f {fake.markers}/{_ID}")


def test_attaching_to_a_running_container_never_stops_it(
    fake: Fake, workspace: pathlib.Path, state_dir: pathlib.Path
) -> None:
    """A container that ran before the run is borrowed; release leaves it and runs no stop."""
    fake.running(_ID)
    options = fake.options(start=_start(fake), stop=_stop(fake))
    held = environment_lease.acquire(
        state_dir, workspace, options, session_id="r1", execution_id="e1"
    )
    assert held.role == "borrower" and "already running" in held.reason
    assert held.discovery.target.container_id == _ID
    outcome = environment_lease.release(
        state_dir, held.ref, session_id="r1", host_root=workspace, options=options
    )
    assert outcome == kinds.EnvironmentOutcome("left_running", "attached, not owned")
    assert all(call[0] in ("ps", "inspect") for call in fake.calls())
    assert (fake.markers / _ID).exists()
    lease = environment_lease.read_lease(state_dir, held.ref)
    assert lease is not None and lease.owner is None and lease.borrowers == ()


def test_an_empty_start_starts_nothing(
    fake: Fake, workspace: pathlib.Path, state_dir: pathlib.Path
) -> None:
    """No running container and no start command refuses before any lease exists."""
    with pytest.raises(container.ContainerNotFoundError):
        environment_lease.acquire(
            state_dir, workspace, fake.options(), session_id="r1", execution_id="e1"
        )
    assert not (state_dir / environment_lease.LEASES_DIR).exists()


def test_an_owned_start_with_no_stop_leaves_the_container_running(
    fake: Fake, workspace: pathlib.Path, state_dir: pathlib.Path
) -> None:
    """The run that started the container owns it; without a stop command it is left running."""
    options = fake.options(start=_start(fake))
    held = environment_lease.acquire(
        state_dir, workspace, options, session_id="r1", execution_id="e1"
    )
    assert held.role == "owner" and held.reason == "started by this run; owned"
    lease = environment_lease.read_lease(state_dir, held.ref)
    assert lease is not None and lease.owner is not None
    assert lease.owner.observed_before == "absent" and lease.owner.container_id == _ID
    outcome = environment_lease.release(
        state_dir, held.ref, session_id="r1", host_root=workspace, options=options
    )
    assert outcome == kinds.EnvironmentOutcome("left_running", "owned; container.stop unset")
    assert (fake.markers / _ID).exists()
    after = environment_lease.read_lease(state_dir, held.ref)
    assert after is not None and after.owner is not None  # a resume may still stop it


def test_the_start_is_announced_before_it_runs_and_timed(
    fake: Fake, workspace: pathlib.Path, state_dir: pathlib.Path
) -> None:
    """`on_start` sees the command while no container exists; the owner keeps the start's length."""
    seen: list[tuple[tuple[str, ...], bool]] = []
    options = fake.options(start=_start(fake))

    def on_start(argv: tuple[str, ...]) -> None:
        seen.append((argv, (fake.markers / _ID).exists()))

    held = environment_lease.acquire(
        state_dir, workspace, options, session_id="r1", execution_id="e1", on_start=on_start
    )
    assert seen == [(options.start, False)]
    assert held.started and held.start_s is not None and held.start_s >= 0.0
    borrowed = environment_lease.acquire(
        state_dir, workspace, options, session_id="r2", execution_id="e1", on_start=on_start
    )
    assert len(seen) == 1
    assert borrowed.role == "borrower" and not borrowed.started and borrowed.start_s is None


def test_a_prepared_run_journals_its_start_and_its_lease(
    fake: Fake, workspace: pathlib.Path, state_dir: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`environment.starting` precedes the start and `environment.acquired` names the lease held."""
    options = fake.options(start=_start(fake))
    rows: list[journal.Event] = []
    notes: list[str] = []
    for name in ("exposures", "check_checkout", "check_hidden_paths"):
        monkeypatch.setattr(container, name, mock.Mock(return_value=()))
    monkeypatch.setattr(container, "prepare", mock.Mock())
    session_runtime.prepare_container_execution(
        Config(),
        workspace,
        options,
        state_dir=state_dir,
        session_id="r1",
        execution_id="e1",
        reporter=app_reporter.Reporter(out=notes.append, err=notes.append),
        emit=rows.append,
    )
    starting, acquired = rows
    assert isinstance(starting, journal.EnvironmentStarting)
    assert tuple(starting.command) == options.start and starting.execution_id == "e1"
    assert isinstance(acquired, journal.EnvironmentAcquired)
    assert acquired.role == "owner" and acquired.started and acquired.start_s is not None
    assert acquired.lifecycle_ref == environment_lease.lease_key(
        environment_lease.selector(workspace, options)
    )
    assert notes[0].endswith(f"container: starting: {options.start[0]}")
    assert notes[1].endswith("container: started by this run; owned")


def test_the_owner_stops_what_it_started(
    fake: Fake, workspace: pathlib.Path, state_dir: pathlib.Path
) -> None:
    """With a stop command, the owner's release stops the container and the lease records it."""
    options = fake.options(start=_start(fake), stop=_stop(fake))
    held = environment_lease.acquire(
        state_dir, workspace, options, session_id="r1", execution_id="e1"
    )
    assert held.role == "owner"
    outcome = environment_lease.release(
        state_dir, held.ref, session_id="r1", host_root=workspace, options=options
    )
    assert outcome.outcome == "stopped"
    assert not (fake.markers / _ID).exists()
    lease = environment_lease.read_lease(state_dir, held.ref)
    assert lease is not None and lease.owner is None and lease.stop is not None
    assert lease.stop.outcome == "stopped"


def test_a_failed_start_acquires_nothing(
    fake: Fake, workspace: pathlib.Path, state_dir: pathlib.Path
) -> None:
    """A start command that fails refuses the run and leaves no owner or borrower behind."""
    options = fake.options(start=fake.script("start", "echo refused >&2; exit 7"))
    with pytest.raises(
        container.ContainerLifecycleError, match=r"container\.start failed \(7\): refused"
    ):
        environment_lease.acquire(state_dir, workspace, options, session_id="r1", execution_id="e1")
    assert not (state_dir / environment_lease.LEASES_DIR).exists()
    options = fake.options(start=fake.script("start", "exit 0"))
    with pytest.raises(
        container.ContainerLifecycleError, match="no running dev container followed"
    ):
        environment_lease.acquire(state_dir, workspace, options, session_id="r1", execution_id="e1")


def test_an_ambiguous_start_owns_nothing(
    fake: Fake, workspace: pathlib.Path, state_dir: pathlib.Path
) -> None:
    """Two containers after a start cannot say which one the run made; nothing is owned."""
    options = fake.options(
        start=fake.script("start", f"touch {fake.markers}/{_ID} {fake.markers}/{_OTHER}")
    )
    with pytest.raises(container.ContainerAmbiguousError):
        environment_lease.acquire(state_dir, workspace, options, session_id="r1", execution_id="e1")
    assert not (state_dir / environment_lease.LEASES_DIR).exists()


def test_the_first_lane_to_finish_cannot_stop_its_siblings(
    fake: Fake, workspace: pathlib.Path, state_dir: pathlib.Path
) -> None:
    """The owner's stop waits for every live borrower; a borrower's release stops nothing."""
    options = fake.options(start=_start(fake), stop=_stop(fake))
    owner = environment_lease.acquire(
        state_dir, workspace, options, session_id="owner", execution_id="e1"
    )
    _live_session(state_dir, "owner", alive=True)
    for lane in ("lane-a", "lane-b"):
        _live_session(state_dir, lane, alive=True)
        held = environment_lease.acquire(
            state_dir, workspace, options, session_id=lane, execution_id="e1"
        )
        assert held.role == "borrower" and held.reason == "attached; owned by run owner"
    first = environment_lease.release(
        state_dir, owner.ref, session_id="lane-a", host_root=workspace, options=options
    )
    assert first == kinds.EnvironmentOutcome("left_running", "attached, not owned")
    waiting = environment_lease.release(
        state_dir, owner.ref, session_id="owner", host_root=workspace, options=options
    )
    assert waiting == kinds.EnvironmentOutcome("left_running", "still attached: lane-b")
    assert (fake.markers / _ID).exists()
    environment_lease.release(
        state_dir, owner.ref, session_id="lane-b", host_root=workspace, options=options
    )
    stopped = environment_lease.release(
        state_dir, owner.ref, session_id="owner", host_root=workspace, options=options
    )
    assert stopped.outcome == "stopped"


def test_a_borrower_that_died_holds_the_stop(
    fake: Fake, workspace: pathlib.Path, state_dir: pathlib.Path
) -> None:
    """A lane that ended without releasing never settled its jobs: the owner fails closed."""
    options = fake.options(start=_start(fake), stop=_stop(fake))
    owner = environment_lease.acquire(
        state_dir, workspace, options, session_id="owner", execution_id="e1"
    )
    _live_session(state_dir, "lane", alive=False)
    environment_lease.acquire(state_dir, workspace, options, session_id="lane", execution_id="e1")
    outcome = environment_lease.release(
        state_dir, owner.ref, session_id="owner", host_root=workspace, options=options
    )
    assert outcome == kinds.EnvironmentOutcome(
        "left_running", "attached runs ended without releasing: lane"
    )
    assert (fake.markers / _ID).exists()


def test_an_orphaned_lease_grants_no_stop_authority(
    fake: Fake, workspace: pathlib.Path, state_dir: pathlib.Path
) -> None:
    """A later run finds the owner dead, says so, attaches, and leaves the container running."""
    options = fake.options(start=_start(fake), stop=_stop(fake))
    environment_lease.acquire(state_dir, workspace, options, session_id="gone", execution_id="e1")
    _live_session(state_dir, "gone", alive=False)
    held = environment_lease.acquire(
        state_dir, workspace, options, session_id="later", execution_id="e1"
    )
    assert held.role == "borrower"
    assert held.reason == "attached; the owner, run gone, is not running (orphaned lease)"
    outcome = environment_lease.release(
        state_dir, held.ref, session_id="later", host_root=workspace, options=options
    )
    assert outcome == kinds.EnvironmentOutcome("left_running", "attached, not owned")
    assert (fake.markers / _ID).exists()


def test_a_resumed_owner_keeps_ownership_of_the_same_container(
    fake: Fake, workspace: pathlib.Path, state_dir: pathlib.Path
) -> None:
    """A later execution of the owning session finds its container and may still stop it."""
    options = fake.options(start=_start(fake), stop=_stop(fake))
    first = environment_lease.acquire(
        state_dir, workspace, options, session_id="r1", execution_id="e1"
    )
    environment_lease.release(
        state_dir,
        first.ref,
        session_id="r1",
        host_root=workspace,
        options=fake.options(start=_start(fake)),
    )
    resumed = environment_lease.acquire(
        state_dir, workspace, options, session_id="r1", execution_id="e2"
    )
    assert resumed.role == "owner" and resumed.reason.startswith("owned since ")
    lease = environment_lease.read_lease(state_dir, resumed.ref)
    assert lease is not None and lease.owner is not None and lease.owner.execution_id == "e2"
    outcome = environment_lease.release(
        state_dir, resumed.ref, session_id="r1", host_root=workspace, options=options
    )
    assert outcome.outcome == "stopped"


def test_a_replacement_container_cannot_inherit_stop_authority(
    fake: Fake, workspace: pathlib.Path, state_dir: pathlib.Path
) -> None:
    """The owning session resumes onto a different container: ownership is lost, not transferred."""
    options = fake.options(start=_start(fake), stop=_stop(fake))
    first = environment_lease.acquire(
        state_dir, workspace, options, session_id="r1", execution_id="e1"
    )
    fake.running(_OTHER)
    resumed = environment_lease.acquire(
        state_dir, workspace, options, session_id="r1", execution_id="e2"
    )
    assert resumed.role == "borrower"
    assert resumed.reason == f"ownership lost: the container was replaced ({_ID} to {_OTHER})"
    assert resumed.ref == first.ref
    outcome = environment_lease.release(
        state_dir, resumed.ref, session_id="r1", host_root=workspace, options=options
    )
    assert outcome == kinds.EnvironmentOutcome("left_running", resumed.reason)
    assert (fake.markers / _OTHER).exists()
    assert all(call[0] != "stop" for call in fake.calls())


def test_a_failed_stop_reads_unknown_and_keeps_the_owner(
    fake: Fake, workspace: pathlib.Path, state_dir: pathlib.Path
) -> None:
    """A stop command that fails, or leaves the container running, is never reported as stopped."""
    failing = fake.options(start=_start(fake), stop=fake.script("stop", "exit 3"))
    held = environment_lease.acquire(
        state_dir, workspace, failing, session_id="r1", execution_id="e1"
    )
    outcome = environment_lease.release(
        state_dir, held.ref, session_id="r1", host_root=workspace, options=failing
    )
    assert outcome.outcome == "unknown" and "container.stop failed (3)" in outcome.reason
    lease = environment_lease.read_lease(state_dir, held.ref)
    assert lease is not None and lease.owner is not None and lease.stop is not None
    noop = fake.options(start=_start(fake), stop=fake.script("stop", "exit 0"))
    outcome = environment_lease.release(
        state_dir, held.ref, session_id="r1", host_root=workspace, options=noop
    )
    assert outcome == kinds.EnvironmentOutcome(
        "unknown", "container.stop ran; the container still runs"
    )


def test_an_opaque_exec_prefix_is_never_owned(
    fake: Fake, workspace: pathlib.Path, state_dir: pathlib.Path
) -> None:
    """Behind a manual exec prefix nothing can be observed, so a start proves no ownership."""
    options = container.ContainerOptions(
        exec=("/usr/bin/env", "-i"), start=_start(fake), stop=_stop(fake)
    )
    held = environment_lease.acquire(
        state_dir, workspace, options, session_id="r1", execution_id="e1"
    )
    assert held.role == "borrower" and "ownership unproved" in held.reason
    assert (fake.markers / _ID).exists()  # the start ran
    outcome = environment_lease.release(
        state_dir, held.ref, session_id="r1", host_root=workspace, options=options
    )
    assert outcome == kinds.EnvironmentOutcome("left_running", "attached, not owned")
    assert (fake.markers / _ID).exists()


def test_a_checkout_outside_the_mounted_workspace_refuses(
    fake: Fake, workspace: pathlib.Path, state_dir: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """A worktree or git dir the container cannot reach refuses before setup or model commands."""
    fake.running(_ID)
    found = container.discover(workspace, fake.options())
    inside = workspace / "wt"
    inside.mkdir()
    container.check_checkout(found, inside, workspace / ".git")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    with pytest.raises(container.ContainerMappingError, match=r"checkout .* is outside"):
        container.check_checkout(found, outside, None)
    with pytest.raises(container.ContainerMappingError, match=r"git dir .* is outside"):
        container.check_checkout(found, inside, outside / ".git")
