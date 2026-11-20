# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A program the repository's own git config names never runs under agent6's own git.

A jailed command writes `.git/config` where `.git` is writable (`partial`
`sandbox.isolation`, `protect_git = "off"`, `git.control = "model"`), and a copied
repository brings one. Each test sets the keys of one route and runs the agent6 operations
that took it: signature verification on a log, maintenance after a commit, the transport of
a fetch between two local repositories, and the lazy fetch of a missing object.
"""

from __future__ import annotations

import http.server
import os
import pathlib
import subprocess
import threading

import pytest

from agent6 import errors
from agent6.vcs import commit_graph
from agent6.vcs.git import history as git_history
from agent6.vcs.git import ops as git_ops
from agent6.vcs.git import read as git_read


def _git(root: pathlib.Path, *args: str, stdin: str | None = None) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True, input=stdin
    ).stdout.strip()


def _repo(root: pathlib.Path) -> pathlib.Path:
    subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True)
    _git(root, "config", "user.email", "x@y.z")
    _git(root, "config", "user.name", "x")
    (root / "f.txt").write_text("v1\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "base")
    return root


def _program(tmp_path: pathlib.Path) -> tuple[str, pathlib.Path]:
    """Return a program that records its run and fails, with the file it records in."""
    marker = tmp_path / "ran"
    program = tmp_path / "program"
    program.write_text(f"#!/bin/sh\necho ran >> {marker}\nexit 1\n", encoding="utf-8")
    program.chmod(0o755)
    return str(program), marker


def _clone_with_work(root: pathlib.Path, dest: pathlib.Path, ref: str) -> pathlib.Path:
    """Clone the repository and commit once in the clone, on a ref the repository lacks."""
    subprocess.run(["git", "clone", "-q", str(root), str(dest)], check=True)
    _git(dest, "config", "user.email", "x@y.z")
    _git(dest, "config", "user.name", "x")
    (dest / "new.txt").write_text("new\n", encoding="utf-8")
    _git(dest, "add", "-A")
    _git(dest, "commit", "-qm", "work")
    _git(dest, "update-ref", ref, "HEAD")
    return dest


# `gc --auto` finds work at two packs, and runs in the foreground: a hook it runs has run
# before the test asserts.
_MAINTENANCE = {
    "gc.autoPackLimit": "1",
    "gc.autoDetach": "false",
    "maintenance.autoDetach": "false",
}


def _make_maintenance_due(root: pathlib.Path) -> None:
    """Leave the repository two packs and one old unreachable object, which a prune asks about."""
    _git(root, "repack", "-q")
    (root / "f.txt").write_text("v2\n", encoding="utf-8")
    _git(root, "commit", "-qam", "second")
    _git(root, "repack", "-q")
    old = _git(root, "hash-object", "-w", "--stdin", stdin="unreachable\n")
    os.utime(root / ".git" / "objects" / old[:2] / old[2:], (1_600_000_000, 1_600_000_000))


def test_a_log_never_verifies_a_signature(tmp_path: pathlib.Path) -> None:
    """`log.showSignature` ran the repository's `gpg.program` on every log, show and stash list."""
    root = _repo(tmp_path / "r")
    program, marker = _program(tmp_path)
    base = _git(root, "rev-parse", "HEAD")
    signed = _git(
        root,
        "hash-object",
        "-t",
        "commit",
        "-w",
        "--stdin",
        stdin=(
            f"tree {_git(root, 'rev-parse', 'HEAD^{tree}')}\nparent {base}\n"
            "author x <x@y.z> 1700000000 +0000\ncommitter x <x@y.z> 1700000000 +0000\n"
            "gpgsig -----BEGIN PGP SIGNATURE-----\n \n AAAA\n -----END PGP SIGNATURE-----\n"
            "\nsigned\n"
        ),
    )
    _git(root, "update-ref", "refs/heads/main", signed)
    _git(root, "config", "gpg.program", program)
    _git(root, "config", "log.showSignature", "true")

    assert git_history.commit_info(root, signed).subject == "signed"
    assert [c.sha for c in git_history.log_commits(root, ["main"], limit=5)] == [signed, base]
    assert len(git_history.search_log(root, "", limit=5)) == 2
    assert commit_graph.parents(root, signed) == (base,)
    assert "signed" in git_read.recent_log(root)
    assert [row.sha for row in git_read.list_run_commits(root, base, "main")] == [signed]
    assert git_read.chain_task_ranges(root, base, signed) == [("", signed)]
    assert not marker.exists(), "the repository's gpg.program ran on the host"


def test_a_commit_runs_no_maintenance_of_the_repository(tmp_path: pathlib.Path) -> None:
    """`git commit` ran `gc --auto`, which runs the repository's `gc.recentObjectsHook`."""
    root = _repo(tmp_path / "r")
    program, marker = _program(tmp_path)
    _make_maintenance_due(root)
    for key, value in (_MAINTENANCE | {"gc.recentObjectsHook": program}).items():
        _git(root, "config", key, value)

    (root / "f.txt").write_text("v3\n", encoding="utf-8")
    git_ops.commit_paths(root, "third", ("f.txt",))
    (root / "f.txt").write_text("v4\n", encoding="utf-8")
    git_ops.stage_paths(root, ["f.txt"])
    git_ops.commit_index(root, "fourth")
    assert not marker.exists(), "the repository's gc.recentObjectsHook ran on the host"


@pytest.mark.parametrize(
    ("route", "arrives"),
    [
        ("ext", False),
        ("ssh", False),
        ("git", False),
        ("alternate", True),
        ("remote-section", True),
        ("maintenance", True),
    ],
)
def test_a_local_fetch_runs_no_program_the_receiving_config_names(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, route: str, arrives: bool
) -> None:
    """A fetch from a session's clone ran the transport programs of the receiving config.

    A rewritten URL is refused by git; every other route fetches and runs nothing.
    """
    root = _repo(tmp_path / "r")
    ref = "refs/agent6/s1/head"
    clone = _clone_with_work(root, tmp_path / "clone", ref)
    program, marker = _program(tmp_path)
    source = clone
    config = {
        "ext": {f"url.ext::{program} .insteadOf": str(clone), "protocol.ext.allow": "always"},
        "ssh": {"url.ssh://host.invalid/.insteadOf": f"{tmp_path}/", "core.sshCommand": program},
        "git": {"url.git://host.invalid/.insteadOf": f"{tmp_path}/", "core.gitProxy": program},
        "alternate": {"core.alternateRefsCommand": program},
        "remote-section": {"remote.../clone.uploadpack": program},
        "maintenance": _MAINTENANCE | {"gc.recentObjectsHook": program},
    }[route]
    if route == "alternate":
        alternates = root / ".git" / "objects" / "info" / "alternates"
        alternates.write_text(f"{clone / '.git' / 'objects'}\n", encoding="utf-8")
    elif route == "remote-section":
        # git matches a `remote.<name>` section to a relative path, never to an absolute one.
        monkeypatch.chdir(root)
        source = pathlib.Path("../clone")
    elif route == "maintenance":
        _make_maintenance_due(root)
    for key, value in config.items():
        _git(root, "config", key, value)

    for fetch in (
        lambda: git_ops.mirror_session(root, source, ref, None),
        lambda: git_ops.fetch_branch(root, source, f"{ref}:refs/agent6/s2/head"),
    ):
        refusal = ""
        try:
            fetch()
        except errors.GitError as exc:
            refusal = str(exc)
        assert not marker.exists(), f"the {route} program of the receiving config ran on the host"
        assert ("not allowed" in refusal) is not arrives, refusal
    assert (git_read.chain_tip(root, ref) is not None) is arrives


def test_a_local_fetch_downloads_no_bundle(tmp_path: pathlib.Path) -> None:
    """`fetch.bundleURI` made a local fetch request a URL and run `credential.helper` for it."""
    requests: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append(self.path)
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="x"')
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        root = _repo(tmp_path / "r")
        ref = "refs/agent6/s1/head"
        clone = _clone_with_work(root, tmp_path / "clone", ref)
        program, marker = _program(tmp_path)
        _git(root, "config", "fetch.bundleURI", f"http://127.0.0.1:{server.server_port}/b")
        _git(root, "config", "credential.helper", program)
        git_ops.mirror_session(root, clone, ref, None)
    finally:
        server.shutdown()
        server.server_close()
    assert git_read.chain_tip(root, ref) is not None
    assert requests == [], "the fetch requested the repository's bundle URI"
    assert not marker.exists(), "the repository's credential.helper ran on the host"


def test_a_local_fetch_enters_no_submodule(tmp_path: pathlib.Path) -> None:
    """A fetched gitlink change made git fetch the submodule, through the submodule's own config."""
    program, marker = _program(tmp_path)
    sub_source = _repo(tmp_path / "sub-source")
    root = _repo(tmp_path / "r")
    _git(root, "-c", "protocol.file.allow=always", "submodule", "--quiet", "add", str(sub_source))
    _git(root, "commit", "-qm", "add the submodule")
    ref = "refs/agent6/s1/head"
    subprocess.run(["git", "clone", "-q", str(root), str(tmp_path / "clone")], check=True)
    clone = tmp_path / "clone"
    _git(clone, "config", "user.email", "x@y.z")
    _git(clone, "config", "user.name", "x")
    _git(sub_source, "commit", "-q", "--allow-empty", "-m", "newer")
    newer = _git(sub_source, "rev-parse", "HEAD")
    _git(clone, "update-index", "--cacheinfo", f"160000,{newer},sub-source")
    _git(clone, "commit", "-qm", "move the submodule")
    _git(clone, "update-ref", ref, "HEAD")
    _git(root / "sub-source", "config", "remote.origin.url", f"ext::{program} x")
    _git(root / "sub-source", "config", "protocol.ext.allow", "always")

    git_ops.mirror_session(root, clone, ref, None)
    assert git_read.chain_tip(root, ref) is not None
    assert not marker.exists(), "the submodule's remote program ran on the host"


def test_a_missing_object_is_never_fetched_from_a_promisor_remote(tmp_path: pathlib.Path) -> None:
    """A promisor remote in the config made any read of an absent object run `core.sshCommand`."""
    root = _repo(tmp_path / "r")
    program, marker = _program(tmp_path)
    _git(root, "config", "remote.far.url", "ssh://host.invalid/x")
    _git(root, "config", "remote.far.promisor", "true")
    _git(root, "config", "core.sshCommand", program)
    absent = "1234567890" * 4

    assert git_read.object_exists(root, absent) is False
    assert git_history.resolve_commit(root, absent) == ""
    assert git_read.is_ancestor(root, absent, "HEAD") is False
    assert git_read.commit_diff(root, absent) == ""
    assert not marker.exists(), "the repository's core.sshCommand ran on the host"
