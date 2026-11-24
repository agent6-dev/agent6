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
import os
import pathlib
import shutil
import signal
import subprocess
import sys

import pytest

import agent6
from agent6.config import Config
from agent6.harness.parallel import repo as child_repo
from agent6.vcs.git import ops as git_ops
from agent6.vcs.git import read as git_read

_jailed = pytest.mark.needs_namespaces

_SESSION = "kid1"
_BRANCH = "agent6/kid1"

# One fetch from a clone under its `full` jail, with git's timeout cut to 3 s.
_FETCH = """
import pathlib, sys
from agent6 import errors
from agent6.config import Config
from agent6.tools import policy
from agent6.vcs.git import command, ops
command.GIT_TIMEOUT_S, command.GIT_TERM_GRACE_S = 3.0, 1.0
origin, child = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
sender = policy.workspace_jail(child, Config.model_validate({"sandbox": {"isolation": "full"}}))
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


def _config(control: str) -> Config:
    """Return a `full` config under which the child's commands can write its `.git`."""
    own = {"git": {"control": "model"}} if control == "model" else {}
    return Config.model_validate({"sandbox": {"isolation": "full", "protect_git": "off"}, **own})


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


@_jailed
@pytest.mark.parametrize("control", ["agent6", "model"])
@pytest.mark.parametrize("rigging", ["alternates", "symlinked-ref", "gitdir", "commondir"])
def test_a_rigged_clone_imports_nothing_of_another_repository(
    tmp_path: pathlib.Path, rigging: str, control: str
) -> None:
    """The import ran the clone's `git upload-pack` on the host, which followed its `.git`.

    Another repository's commit then reached the origin, where the parent's jailed git reads
    it. The other repository holds a branch of the child's name, so every rigging has a ref
    to send under both controls.
    """
    origin, child, session = _child(tmp_path)
    other = _repo(tmp_path / "other", "notes.txt", "another repository's file\n")
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
            origin, child, _BRANCH, session, _config(control), model_control=control == "model"
        )
    except child_repo.ChildRepoError:
        refused = True
    arrived = [sha for sha in theirs if _has(origin, sha)]
    assert not arrived, f"the import brought another repository's objects: {arrived}"
    assert refused, "the import of a rigged clone reported success"
    assert not git_read.branch_exists(origin, _BRANCH)
    assert git_read.chain_tip(origin, chain) is None


@_jailed
@pytest.mark.parametrize("control", ["agent6", "model"])
def test_a_clean_clone_still_imports_under_its_jail(tmp_path: pathlib.Path, control: str) -> None:
    """A jail that hides the clone's own `.git` from its `git upload-pack` imports nothing."""
    origin, child, session = _child(tmp_path)
    tip = _git(child, "rev-parse", "HEAD")

    imported = child_repo.import_run(
        origin, child, _BRANCH, session, _config(control), model_control=control == "model"
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
