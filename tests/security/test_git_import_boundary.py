# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A child's import brings no object its own jailed commands could not read.

A child that can write its clone's `.git` (`git.control = "model"`, `sandbox.protect_git =
"off"`, `partial` `sandbox.isolation`) can point it at another repository on the machine.
The clone's side of the import runs under the clone's jail, where that repository is out
of reach. Each rigging is first shown to send the other repository's commit on the host.
"""

from __future__ import annotations

import ast
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator

import pytest

import agent6
from agent6.config import model as config_model
from agent6.harness.parallel import repo as child_repo
from agent6.vcs.git import ops as git_ops
from agent6.vcs.git import read as git_read

_jailed = pytest.mark.needs_namespaces
# `partial` needs no namespaces: Landlock and seccomp bind without them.
_LEVELS = [pytest.param("full", marks=_jailed), "partial"]

_SESSION = "kid1"
_BRANCH = "agent6/kid1"

# One fetch from a clone under its `full` jail, with git's timeout cut to 3 s.
_FETCH = """
import pathlib, sys
from agent6 import errors
from agent6.config import model as config_model
from agent6.tools import policy
from agent6.vcs.git import command, ops
command.GIT_TIMEOUT_S, command.GIT_TERM_GRACE_S = 3.0, 1.0
origin, child = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
full = config_model.Config.model_validate({"sandbox": {"isolation": "full"}})
sender = policy.workspace_jail(child, full, ran_under="full")
try:
    ops.fetch_branch(origin, child, sys.argv[3], sender=sender)
except errors.GitError as exc:
    print(exc)
"""


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def _repo(root: pathlib.Path, name: str, text: str) -> pathlib.Path:
    subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True)
    (root / name).write_text(text, encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", f"add {name}")
    return root


def _has(repo: pathlib.Path, sha: str) -> bool:
    found = subprocess.run(
        ["git", "--no-replace-objects", "-C", str(repo), "cat-file", "-e", sha],
        capture_output=True,
        check=False,
    )
    return found.returncode == 0


def _config(control: str, isolation: str = "full") -> config_model.Config:
    """Return a config under which the child's commands can write its `.git`."""
    own = {"git": {"control": "model"}} if control == "model" else {}
    return config_model.Config.model_validate(
        {"sandbox": {"isolation": isolation, "protect_git": "off"}, **own}
    )


@pytest.fixture
def elsewhere() -> Iterator[pathlib.Path]:
    """A directory no isolation level grants: `partial` grants the host's /tmp wholesale."""
    base = pathlib.Path("/var/tmp")
    if not (base.is_dir() and os.access(base, os.W_OK)):
        pytest.skip("/var/tmp is not writable here")
    made = pathlib.Path(tempfile.mkdtemp(prefix="agent6-import-", dir=base)).resolve()
    try:
        yield made
    finally:
        shutil.rmtree(made, ignore_errors=True)


def _child(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """Return an origin, a child's clone with one commit on its branch, and its session dir."""
    origin = _repo(tmp_path / "origin", "app.py", "print('origin')\n")
    child = tmp_path / "child"
    child_repo.clone_worktree(origin, child)
    _git(child, "switch", "-q", "-c", _BRANCH)
    (child / "child.txt").write_text("child work\n", encoding="utf-8")
    _git(child, "add", "-A")
    _git(child, "commit", "-qm", "child work")
    session = tmp_path / "state" / "sessions" / "runs" / _SESSION
    session.mkdir(parents=True)
    (session / "manifest.json").write_text("{}\n", encoding="utf-8")
    return origin, child, session


def _rig(rigging: str, child: pathlib.Path, other: pathlib.Path) -> None:
    """Point the child's `.git` at the other repository, as a command in the child can."""
    git_dir, other_git = child / ".git", other / ".git"
    if rigging in ("alternates", "symlinked-ref"):
        alternates = git_dir / "objects" / "info" / "alternates"
        alternates.write_text(f"{other_git / 'objects'}\n", encoding="utf-8")
    if rigging == "alternates":
        _git(child, "update-ref", f"refs/heads/{_BRANCH}", _git(other, "rev-parse", "main"))
    elif rigging == "symlinked-ref":
        ref = git_dir / "refs" / "heads" / _BRANCH
        ref.unlink()
        ref.symlink_to(other_git / "refs" / "heads" / "main")
    elif rigging == "gitdir":
        shutil.rmtree(git_dir)
        git_dir.write_text(f"gitdir: {other_git}\n", encoding="utf-8")
    elif rigging == "commondir":
        (git_dir / "commondir").write_text(f"{other_git}\n", encoding="utf-8")


@pytest.mark.parametrize("isolation", _LEVELS)
@pytest.mark.parametrize("control", ["agent6", "model"])
@pytest.mark.parametrize("rigging", ["alternates", "symlinked-ref", "gitdir", "commondir"])
def test_a_rigged_clone_imports_nothing_of_another_repository(
    tmp_path: pathlib.Path, elsewhere: pathlib.Path, rigging: str, control: str, isolation: str
) -> None:
    """The import ran the clone's `git upload-pack` on the host, which followed its `.git`.

    Another repository's commit then reached the origin, where the parent's jailed git reads
    it. The other repository holds a branch of the child's name, so every rigging has a ref
    to send under both controls. It lies where neither level's commands can read.
    """
    origin, child, session = _child(tmp_path)
    other = _repo(elsewhere / "other", "notes.txt", "another repository's file\n")
    _git(other, "branch", _BRANCH)
    theirs = _git(other, "rev-list", "--objects", "--all").split("\n")
    theirs = [line.split(" ", 1)[0] for line in theirs]
    _rig(rigging, child, other)
    chain = git_read.chain_ref_for(_SESSION)
    spec = f"HEAD:{chain}" if control == "model" else f"{_BRANCH}:{_BRANCH}"

    # The rigging is live: the same fetch with the clone's side on the host sends the commit.
    host = tmp_path / "host"
    subprocess.run(["git", "clone", "-q", str(origin), str(host)], check=True)
    git_ops.fetch_branch(host, child, spec, sender=None)
    assert _has(host, theirs[0]), "the rigging sends nothing on the host: the test pins nothing"

    refused = False
    try:
        child_repo.import_run(
            origin,
            child,
            _BRANCH,
            session,
            _config(control, isolation),
            model_control=control == "model",
        )
    except child_repo.ChildRepoError:
        refused = True
    arrived = [sha for sha in theirs if _has(origin, sha)]
    assert not arrived, f"the import brought another repository's objects: {arrived}"
    assert refused, "the import of a rigged clone reported success"
    assert not git_read.branch_exists(origin, _BRANCH)
    assert git_read.chain_tip(origin, chain) is None


@pytest.mark.parametrize("weaker", ["config", "environment"])
def test_an_import_refuses_a_weaker_level_than_the_child_ran_under(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, weaker: str
) -> None:
    """A later import read a clone rigged under `full` with the host's reach.

    The import took its level from its own config and environment. `sessions prune` under a
    weaker `sandbox.isolation`, or with the sandbox disabled by the environment, ran the
    clone's `git upload-pack` where the other repository is in reach.
    """
    origin, child, session = _child(tmp_path)
    (session / "manifest.json").write_text(
        json.dumps({"policy": {"isolation": "full"}}), encoding="utf-8"
    )
    other = _repo(tmp_path / "other", "notes.txt", "another repository's file\n")
    theirs = _git(other, "rev-parse", "main")
    _rig("alternates", child, other)
    config = _config("agent6")
    if weaker == "environment":
        monkeypatch.setenv("AGENT6_DISABLE_SANDBOX", "1")
    else:
        config = config_model.Config.model_validate(
            {"sandbox": {"isolation": "none", "protect_git": "off"}}
        )

    with pytest.raises(child_repo.ChildRepoError, match=r"'full'.*'none'"):
        child_repo.import_run(origin, child, _BRANCH, session, config)

    assert not _has(origin, theirs), "the import brought another repository's commit"
    assert not git_read.branch_exists(origin, _BRANCH)


@pytest.mark.parametrize("isolation", _LEVELS)
@pytest.mark.parametrize("control", ["agent6", "model"])
def test_a_clean_clone_still_imports_under_its_jail(
    tmp_path: pathlib.Path, control: str, isolation: str
) -> None:
    """A jail that hides the clone's own `.git` from its `git upload-pack` imports nothing."""
    origin, child, session = _child(tmp_path)
    tip = _git(child, "rev-parse", "HEAD")

    imported = child_repo.import_run(
        origin,
        child,
        _BRANCH,
        session,
        _config(control, isolation),
        model_control=control == "model",
    )

    ref = git_read.chain_ref_for(_SESSION) if control == "model" else _BRANCH
    assert (imported.ref, imported.tip) == (ref, tip)
    assert _git(origin, "rev-parse", ref) == tip
    assert _git(origin, "show", f"{tip}:child.txt") == "child work"


@_jailed
def test_a_sender_that_never_answers_ends_with_its_fetch(tmp_path: pathlib.Path) -> None:
    """A stuck `git upload-pack` outlived its fetch and held the pipes the fetch's caller reads.

    The import then never returned. The clone's config is a FIFO nobody writes.
    """
    origin, child, _session = _child(tmp_path)
    config = child / ".git" / "config"
    config.unlink()
    os.mkfifo(config)
    # Its own process group: a caller that never returns is stopped with all it started.
    fetch = subprocess.Popen(
        [sys.executable, "-P", "-c", _FETCH, str(origin), str(child), f"{_BRANCH}:{_BRANCH}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        out, err = fetch.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(fetch.pid, signal.SIGKILL)
        fetch.communicate()
        pytest.fail("the fetch's caller never returned after the fetch timed out")
    assert "got no answer within 3s" in out, out + err


# The fetch: it starts the sender as git does, then dies once the shell has become the module.
_DYING_FETCH = """
import os, pathlib, subprocess, sys, time
sender = subprocess.Popen(
    ["sh", "-c", sys.argv[1]], stdin=0, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
)
cmdline = pathlib.Path(f"/proc/{sender.pid}/cmdline")
while b"upload_pack" not in cmdline.read_bytes():
    time.sleep(0.001)
print(sender.pid, flush=True)
os._exit(0)
"""


def _running(pid: int) -> bool:
    """Return whether a process still runs; one that exited and waits to be reaped does not."""
    try:
        stat = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def test_a_sender_whose_fetch_died_while_it_started_does_not_serve(
    tmp_path: pathlib.Path,
) -> None:
    """A sender tied itself to whoever its parent was by the time it looked.

    A fetch killed while the module started left the sender a child of the process that
    adopted it. The sender took that one for its fetch and served, with nothing to end it:
    a `git upload-pack` that waits holds the pipes the fetch's caller reads.
    """
    from agent6.tools import policy
    from agent6.vcs.git import upload_pack

    _origin, child, _session = _child(tmp_path)
    unconfined = config_model.Config.model_validate({"sandbox": {"isolation": "none"}})
    sender = policy.workspace_jail(child, unconfined, ran_under="none")
    line = upload_pack.command(sender, child, tmp_path / "sender.json")
    fetch = subprocess.Popen(
        [sys.executable, "-c", _DYING_FETCH, f"{line} '{child}'"],
        stdin=subprocess.PIPE,  # held open: an upload-pack that started waits on it
        stdout=subprocess.PIPE,
        text=True,
    )
    assert fetch.stdout is not None and fetch.stdin is not None
    pid = int(fetch.stdout.readline())
    assert fetch.wait(timeout=10) == 0
    try:
        deadline = time.monotonic() + 10
        while _running(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _running(pid), "the sender outlived a fetch that died before it started"
    finally:
        fetch.stdin.close()
        if _running(pid):
            os.kill(pid, signal.SIGKILL)


def test_only_a_fetch_out_of_a_repository_runs_its_sender_on_the_host() -> None:
    """A fetch that passes `sender=None` runs the sending side unjailed.

    A new one is a security review, not a list to extend in passing: each of these fetches
    out of a repository, into its clone or from the remote the Git page names.
    """
    src = pathlib.Path(agent6.__file__).resolve().parent
    found = sorted(
        py.relative_to(src).as_posix()
        for py in src.rglob("*.py")
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "sender"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is None
    )
    assert found == [
        "app/machine/agent.py",  # a state's clone, seeded from the checkout
        "app/parallel/resume.py",  # a child's clone, restored from its parent's repository
        "app/resume.py",  # a session's clone, brought up to its repository
        "app/resume.py",  # and the branch a resume merges
        "vcs/git/fetch.py",  # the Git page's Fetch of a remote
    ]
