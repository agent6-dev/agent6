# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the Git page's fetch: the project's own config never reaches the network step."""

from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

from agent6 import errors, paths
from agent6.vcs.git import command as git_command
from agent6.vcs.git import fetch as git_fetch
from agent6.vcs.git import ops as git_ops


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _commit(repo: pathlib.Path, name: str) -> str:
    (repo / name).write_text(f"{name}\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", name)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def pair(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """An upstream holding `main`, `dev` and pull request 7, and a clone of it."""
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git(upstream, "init", "-q", "-b", "main")
    _commit(upstream, "a.txt")
    _git(upstream, "checkout", "-q", "-b", "dev")
    _commit(upstream, "d.txt")
    _git(upstream, "checkout", "-q", "main")
    _git(upstream, "update-ref", "refs/pull/7/head", "dev")
    work = tmp_path / "work"
    subprocess.run(["git", "clone", "-q", str(upstream), str(work)], check=True)
    return upstream, work


def test_a_fetch_moves_only_the_remotes_tracking_refs(
    pair: tuple[pathlib.Path, pathlib.Path],
) -> None:
    """New branches and commits arrive under `refs/remotes/<name>/`; nothing else moves."""
    upstream, work = pair
    tip = _commit(upstream, "b.txt")
    _git(upstream, "branch", "topic")
    _git(upstream, "tag", "v2")
    local = _git(work, "rev-parse", "main")

    assert git_fetch.fetch_remote(work, "origin") == "fetched origin: main, topic"

    assert _git(work, "rev-parse", "refs/remotes/origin/main") == tip
    assert _git(work, "rev-parse", "refs/remotes/origin/topic") == tip
    assert _git(work, "rev-parse", "main") == local
    assert _git(work, "tag", "--list") == ""
    assert not (work / ".git" / "FETCH_HEAD").exists()
    assert not list((paths.state_base() / "fetch").iterdir())
    assert _git(work, "fsck", "--no-dangling") == ""
    assert _git(work, "reflog", "show", "--format=%gs", "-1", "origin/main") == (
        "agent6: fetch origin"
    )
    assert git_fetch.fetch_remote(work, "origin") == "origin is up to date"


def test_a_pull_request_lands_under_the_remotes_pr_refs(
    pair: tuple[pathlib.Path, pathlib.Path],
) -> None:
    """`refs/pull/N/head` becomes `<remote>/pr/N`, which `/review` and the graph can name."""
    upstream, work = pair
    dev = _git(upstream, "rev-parse", "dev")
    before = _git(work, "for-each-ref", "refs/remotes/origin")

    assert git_fetch.fetch_remote(work, "origin", pull=7) == f"fetched origin/pr/7 at {dev[:12]}"

    assert _git(work, "rev-parse", "origin/pr/7") == dev
    assert _git(work, "for-each-ref", "refs/remotes/origin").startswith(before)
    with pytest.raises(errors.GitError, match="couldn't find remote ref refs/pull/404/head"):
        git_fetch.fetch_remote(work, "origin", pull=404)
    with pytest.raises(errors.GitError, match="no remote 'upstream'"):
        git_fetch.fetch_remote(work, "upstream")


def test_nothing_the_repositorys_config_names_runs_or_redirects_a_fetch(
    pair: tuple[pathlib.Path, pathlib.Path], tmp_path: pathlib.Path
) -> None:
    """A jailed command can write `.git/config` below `full` isolation; the click obeys none of it.

    Git's own fetch over the same config is the control: it runs the upload-pack command.
    """
    upstream, work = pair
    marks = tmp_path / "marks"
    marks.mkdir()
    decoy = tmp_path / "decoy"
    subprocess.run(["git", "clone", "-q", str(upstream), str(decoy)], check=True)
    _commit(decoy, "decoy.txt")
    _git(work, "config", f"url.{decoy}.insteadOf", str(upstream))
    _git(work, "config", "remote.origin.uploadpack", f"touch {marks}/uploadpack; git-upload-pack")
    _git(work, "config", "core.sshCommand", f"touch {marks}/ssh; ssh")
    _git(work, "config", "credential.helper", f"!touch {marks}/credential")
    _git(work, "config", "core.fsmonitor", f"touch {marks}/fsmonitor")
    _git(work, "config", "fetch.bundleURI", f"file://{marks}/bundle")
    _git(work, "config", "remote.origin.fetch", "+refs/heads/*:refs/heads/*")
    # With an alternate object store, git asks the configured command for its refs.
    alternates = work / ".git" / "objects" / "info" / "alternates"
    alternates.write_text(f"{decoy / '.git' / 'objects'}\n", encoding="utf-8")
    _git(work, "config", "core.alternateRefsCommand", f"touch {marks}/alternate-refs;")
    hook = work / ".git" / "hooks" / "reference-transaction"
    hook.parent.mkdir(exist_ok=True)
    hook.write_text(f"#!/bin/sh\ntouch {marks}/hook\n", encoding="utf-8")
    hook.chmod(0o755)
    tip = _commit(upstream, "b.txt")
    local = _git(work, "-c", "core.fsmonitor=false", "rev-parse", "main")

    assert git_fetch.fetch_remote(work, "origin") == "fetched origin: main"

    assert [p.name for p in marks.iterdir()] == []
    assert _git(work, "-c", "core.fsmonitor=false", "rev-parse", "origin/main") == tip
    assert _git(work, "-c", "core.fsmonitor=false", "rev-parse", "main") == local
    subprocess.run(["git", "-C", str(work), "fetch", "origin"], check=False, capture_output=True)
    assert "uploadpack" in {p.name for p in marks.iterdir()}


@pytest.mark.parametrize(
    ("url", "named"),
    [
        ("ext::sh -c 'touch {mark}'", "the ext transport"),
        ("http://127.0.0.1:1/x.git", "the http transport"),
        ("git://127.0.0.1:1/x.git", "the git transport"),
        ("-oProxyCommand=touch {mark}", "an option-shaped URL"),
    ],
)
def test_a_transport_other_than_https_ssh_or_a_local_path_is_refused(
    pair: tuple[pathlib.Path, pathlib.Path], tmp_path: pathlib.Path, url: str, named: str
) -> None:
    """The refusal names the transport before git is asked to use it."""
    _, work = pair
    mark = tmp_path / "mark"
    _git(work, "remote", "remove", "origin")
    with (work / ".git" / "config").open("a", encoding="utf-8") as config:
        config.write(f'[remote "origin"]\n\turl = "{url.format(mark=mark)}"\n')

    with pytest.raises(errors.GitError, match=re.escape(f"remote origin uses {named}")):
        git_fetch.fetch_remote(work, "origin")

    assert not mark.exists()


def test_gits_own_protocol_list_refuses_a_transport_the_url_check_missed(
    pair: tuple[pathlib.Path, pathlib.Path], tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rewrite in the operator's own config can change the transport after the URL check."""
    _, work = pair
    mark = tmp_path / "mark"

    def rewritten(_path: pathlib.Path, _remote: git_fetch.Remote) -> str:
        return f"ext::sh -c 'touch {mark}'"

    monkeypatch.setattr(git_fetch, "_fetch_url", rewritten)

    with pytest.raises(errors.GitError, match="transport 'ext' not allowed"):
        git_fetch.fetch_remote(work, "origin")

    assert not mark.exists()


def test_a_failed_fetch_and_the_remote_list_name_no_http_credentials(
    pair: tuple[pathlib.Path, pathlib.Path],
) -> None:
    """A token in a remote's URL stays out of the page and out of the failure's reason."""
    _, work = pair
    _git(work, "config", "remote.origin.url", "https://bot:s3cret-token@127.0.0.1:1/x.git")

    assert [(r.name, r.shown_url) for r in git_fetch.list_remotes(work)] == [
        ("origin", "https://127.0.0.1:1/x.git")
    ]
    with pytest.raises(errors.GitError, match="fetch from origin failed: ") as caught:
        git_fetch.fetch_remote(work, "origin")
    assert "s3cret-token" not in str(caught.value)
    assert not list((paths.state_base() / "fetch").iterdir())


def test_a_fetch_works_from_a_linked_worktree_and_in_a_shallow_clone(
    pair: tuple[pathlib.Path, pathlib.Path], tmp_path: pathlib.Path
) -> None:
    """The scratch repository borrows the shared object store and the clone's shallow boundary."""
    upstream, work = pair
    linked = tmp_path / "linked"
    _git(work, "worktree", "add", "-q", str(linked), "-b", "side")
    shallow = tmp_path / "shallow"
    subprocess.run(
        ["git", "clone", "-q", "--depth", "1", f"file://{upstream}", str(shallow)], check=True
    )
    tip = _commit(upstream, "b.txt")

    assert git_fetch.fetch_remote(linked, "origin") == "fetched origin: main"
    assert _git(work, "rev-parse", "origin/main") == tip
    assert git_fetch.fetch_remote(shallow, "origin") == "fetched origin: dev, main"
    assert _git(shallow, "rev-parse", "origin/main") == tip
    assert _git(shallow, "fsck", "--no-dangling") == ""


@pytest.mark.parametrize("name", ["..", ".git"])
def test_a_fetch_refuses_a_tree_git_would_not_check_out(
    pair: tuple[pathlib.Path, pathlib.Path], name: str
) -> None:
    """The fetched pack was copied in unchecked, a tree entry named `..` or `.git` with it."""
    upstream, work = pair

    def made(*args: str, stdin: str) -> str:
        return subprocess.run(
            ["git", "-C", str(upstream), *args],
            check=True,
            capture_output=True,
            text=True,
            input=stdin,
        ).stdout.strip()

    blob = made("hash-object", "-w", "--stdin", stdin="planted\n")
    inner = made("mktree", stdin=f"100644 blob {blob}\tplanted\n")
    tree = made("mktree", stdin=f"040000 tree {inner}\t{name}\n")
    commit = made("commit-tree", tree, "-p", "main", stdin="bad\n")
    _git(upstream, "update-ref", "refs/heads/bad", commit)
    packs = sorted((work / ".git" / "objects" / "pack").iterdir())

    with pytest.raises(errors.GitError, match="fetch from origin failed"):
        git_fetch.fetch_remote(work, "origin")

    assert sorted((work / ".git" / "objects" / "pack").iterdir()) == packs
    names = _git(work, "for-each-ref", "--format=%(refname)", "refs/remotes/origin")
    assert "refs/remotes/origin/bad" not in names.split()


def test_the_scratch_repository_is_out_of_a_jailed_commands_reach(
    pair: tuple[pathlib.Path, pathlib.Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scratch repository sat under `.git`, where a jailed command could write its config."""
    upstream, work = pair
    _commit(upstream, "b.txt")
    scratches: set[pathlib.Path] = set()
    real = git_command.subprocess.Popen

    def spy(*args: object, **kwargs: object) -> object:
        env = kwargs.get("env")
        if isinstance(env, dict) and "GIT_DIR" in env:
            scratches.add(pathlib.Path(env["GIT_DIR"]))  # pyright: ignore[reportUnknownArgumentType]
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(git_command.subprocess, "Popen", spy)
    git_fetch.fetch_remote(work, "origin")
    git_ops.fetch_branch(work, upstream, "refs/heads/main:refs/agent6/s/head")

    for scratch in scratches:
        assert not scratch.is_relative_to(work)
        assert scratch.parent == paths.state_base() / "fetch"
    assert len(scratches) == 2
