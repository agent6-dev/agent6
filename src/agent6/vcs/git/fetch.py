# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Fetch a remote's branches, one pull request, or a local repository's refs.

A fetch runs on the host over a `.git/config` a jailed command can write below `full`
`sandbox.isolation`. Its transport step therefore runs in a scratch repository made for it
in agent6's state base, which no jail reaches. There git reads the operator's global and
system config alone: no key of the project's `.git/config` names a transport command, a
proxy, a credential helper or a URL rewrite. The project lends its objects (an alternate, so
only new history travels). The new pack is then copied in, and the caller writes the refs.
The sending side of a fetch from a workspace's clone runs under that workspace's jail
(`upload_pack`): a jailed command can write the clone's `.git` too.
"""

from __future__ import annotations

import dataclasses
import errno
import os
import pathlib
import re
import shutil
import tempfile

from agent6 import errors, kinds, paths, terminal
from agent6.vcs.git import command as git_command
from agent6.vcs.git import upload_pack

# The transports a fetch may use; `ext::`, `fd::`, `git://`, plain http and every remote
# helper are refused by git itself under this list, whatever a URL rewrite yields.
ALLOWED_PROTOCOLS = "file:https:ssh"

_REMOTE_URL_KEY = re.compile(r"^remote\.(?P<name>.+)\.url$")
_HTTP_USERINFO = re.compile(r"(?i)(https?://)[^/@\s]+@")
# `[user@]host:path`, git's scp-like spelling of an ssh URL.
_SCP_LIKE = re.compile(r"^[A-Za-z0-9._~-]+(@[A-Za-z0-9._-]+)?:")


@dataclasses.dataclass(frozen=True, slots=True)
class Remote:
    """One remote of the repository.

    Attributes:
        name: The remote's name.
        url: Its fetch URL as configured.
    """

    name: str
    url: str

    @property
    def shown_url(self) -> str:
        """The URL for display, an http(s) URL's user and password left out."""
        return redact(self.url)


def redact(text: str) -> str:
    """Return the text with the user and password of every http(s) URL left out."""
    return _HTTP_USERINFO.sub(r"\1", text)


def list_remotes(path: pathlib.Path) -> tuple[Remote, ...]:
    """Return the repository's remotes, each with the first URL its config names.

    Args:
        path: The repository.

    Returns:
        The remotes in config order; empty when there are none.
    """
    fields = git_command.nul_fields(
        path,
        "config",
        "--local",
        "--includes",
        "-z",
        "--get-regexp",
        r"^remote\..*\.url$",
        check=False,
    )
    remotes: dict[str, Remote] = {}
    for field in fields:
        key, _, url = field.partition("\n")
        match = _REMOTE_URL_KEY.match(key)
        if match is not None and url and match["name"] not in remotes:
            remotes[match["name"]] = Remote(match["name"], url)
    return tuple(remotes.values())


def _fetch_url(path: pathlib.Path, remote: Remote) -> str:
    """Return the URL to hand git, a relative path made absolute against the checkout.

    Args:
        path: The repository.
        remote: The remote.

    Returns:
        The URL.

    Raises:
        GitError: The URL names a transport other than https, ssh or a local path.
    """
    url = remote.url
    scheme, sep, _rest = url.partition("://")
    helper = url.split("/", 1)[0].partition("::")
    if url.startswith("-"):
        named = "an option-shaped URL"
    elif helper[1]:
        named = f"the {helper[0]} transport"
    elif sep and scheme.lower() not in ("https", "ssh", "file"):
        named = f"the {scheme} transport"
    elif sep or _SCP_LIKE.match(url):
        return url
    else:
        return str((path / url).resolve())
    raise errors.GitError(
        f"remote {remote.name} uses {named}; the page fetches over https, ssh or a local"
        " path (run `git fetch` in a terminal for another)"
    )


def fetch_remote(path: pathlib.Path, name: str, *, pull: int | None = None) -> str:
    """Fetch a remote's branches, or one pull request, into its remote-tracking refs.

    Nothing is merged and no checkout, branch or tag moves: the refs written are
    `refs/remotes/<name>/*`, or `refs/remotes/<name>/pr/<pull>` for a pull request
    (the remote's `refs/pull/<pull>/head`, as GitHub, Gitea and Forgejo name it).

    Args:
        path: The repository.
        name: The remote's name.
        pull: The pull request's number; None fetches the remote's branches.

    Returns:
        The one-line outcome.

    Raises:
        GitError: No such remote, a refused transport, a failed or unanswered fetch, or a
            state base or `.git` this process cannot write. The reason carries no http(s)
            credentials.
    """
    remote = next((r for r in list_remotes(path) if r.name == name), None)
    tracking = f"refs/remotes/{name}/"
    if (
        remote is None
        or not git_command.run(
            pathlib.Path("/"), "check-ref-format", f"{tracking}x", check=False
        ).ok
    ):
        raise errors.GitError(f"no remote {name!r}")
    # A nested remote name shares tracking refs with branch names under its parent.
    for other in _lines(path, "remote"):
        if name.startswith(other + "/") or other.startswith(name + "/"):
            raise errors.GitError(
                f"overlapping remote names {name!r} and {other!r}; rename one before fetching here"
            )
    url = _fetch_url(path, remote)
    refspec = (
        f"+refs/heads/*:{tracking}*"
        if pull is None
        else f"+refs/pull/{pull}/head:{tracking}pr/{pull}"
    )
    before = refs(path, tracking)
    # The remote's side is the remote's own, on its host or at the local path the page shows.
    after = scratch_fetch(
        path, url, [refspec], protocols=ALLOWED_PROTOCOLS, source=name, sender=None
    )
    if any(not ref.startswith(tracking) for ref in before.keys() | after.keys()):
        raise errors.GitError(f"ref outside {tracking}; no refs changed")
    if pull is not None and set(after) != {f"{tracking}pr/{pull}"}:
        raise errors.GitError(f"unexpected refs for pull request {pull}; no refs changed")
    if pull is None and any(
        ref in (tracking + "HEAD", tracking + "pr") or ref.startswith(tracking + "pr/")
        for ref in after
    ):
        raise errors.GitError(
            f"remote {name} has a branch at a reserved tracking ref; no refs changed"
        )
    moved = {ref: sha for ref, sha in after.items() if before.get(ref) != sha}
    # HEAD is the remote's default branch link. PR refs come from separate fetches.
    stale = sorted(
        ref
        for ref in before.keys() - after.keys()
        if pull is None and ref != tracking + "HEAD" and not ref.startswith(tracking + "pr/")
    )
    if moved or stale:
        git_command.run(
            path,
            "update-ref",
            "--no-deref",
            "-m",
            f"agent6: fetch {name}",
            "--stdin",
            "-z",
            stdin_text="".join(
                f"update {ref}\0{sha}\0{before.get(ref, '0' * len(sha))}\0"
                for ref, sha in moved.items()
            )
            + "".join(f"delete {ref}\0{before[ref]}\0" for ref in stale),
        )
    if pull is not None:
        return f"fetched {name}/pr/{pull} at {after[f'{tracking}pr/{pull}'][:12]}"
    if not moved and not stale:
        return f"{name} is up to date"
    names = [terminal.printable_name(ref.removeprefix(tracking)) for ref in moved]
    names.extend(f"pruned {terminal.printable_name(ref.removeprefix(tracking))}" for ref in stale)
    more = f" and {len(names) - 3} more" if len(names) > 3 else ""
    return f"fetched {name}: {', '.join(names[:3])}{more}"


def scratch_fetch(
    path: pathlib.Path,
    url: str,
    refspecs: list[str],
    *,
    protocols: str,
    source: str,
    sender: kinds.JailPolicy | None,
) -> dict[str, str]:
    """Fetch from a URL in a scratch repository and copy the new objects into the repository.

    No ref of the repository moves: the caller applies the refs returned.

    Args:
        path: The repository.
        url: The URL or absolute path fetched from.
        refspecs: What to fetch; a destination names a ref returned.
        protocols: The transports git may use, as `GIT_ALLOW_PROTOCOL` lists them.
        source: What a failure calls the other side.
        sender: The policy the commands of the workspace at `url` run under: its
            `git upload-pack` runs under it. None leaves the sending side to git.

    Returns:
        The full name and sha of each ref the fetch wrote in the scratch repository.

    Raises:
        GitError: A failed or unanswered fetch, or a state base or `.git` this process
            cannot write. The reason carries no http(s) credentials.
    """
    objects, common, hashes = git_command.run(
        path,
        "rev-parse",
        "--path-format=absolute",
        "--git-path",
        "objects",
        "--git-common-dir",
        "--show-object-format",
    ).stdout.splitlines()
    try:
        base = paths.state_base() / "fetch"
        paths.mkdir_for_real_user(base)
        scratch = pathlib.Path(tempfile.mkdtemp(prefix="agent6-fetch-", dir=base))
    except OSError as exc:
        raise errors.GitError(f"cannot make a scratch repository to fetch in: {exc}") from exc
    try:
        env = {"GIT_DIR": str(scratch), "GIT_ALLOW_PROTOCOL": protocols}
        git_command.run(
            scratch,
            "init",
            "--quiet",
            "--bare",
            "--template=",
            f"--object-format={hashes}",
            str(scratch),
        )
        (scratch / "objects" / "info" / "alternates").write_text(f"{objects}\n", encoding="utf-8")
        shallow = pathlib.Path(common, "shallow")
        if shallow.is_file():
            # A shallow clone's boundary: without it git walks into history the clone lacks.
            _copy_shallow(shallow, scratch / "shallow", hashes)
        serving = (
            ()
            if sender is None
            else (
                "--upload-pack="
                + upload_pack.command(sender, pathlib.Path(url), scratch / "sender.json"),
            )
        )
        try:
            # unpackLimit 1 keeps every fetch as one pack, the unit copied in below; git checks
            # the pack's objects first, and refuses a tree holding a `..` or `.git` entry.
            fetched = git_command.run(
                scratch,
                "-c",
                "fetch.unpackLimit=1",
                "-c",
                "fetch.fsckObjects=true",
                "fetch",
                "--quiet",
                "--no-tags",
                "--no-recurse-submodules",
                "--no-write-fetch-head",
                "--no-auto-maintenance",
                *serving,
                "--end-of-options",
                url,
                *refspecs,
                check=False,
                env_extra=env,
            )
        except errors.GitError as exc:  # with check off, only the timeout raises
            raise errors.GitError(
                f"fetch from {source} got no answer within {git_command.GIT_TIMEOUT_S:.0f}s"
            ) from exc
        if not fetched.ok:
            said = redact(fetched.stderr.strip()) or f"git fetch exited {fetched.returncode}"
            raise errors.GitError(f"fetch from {source} failed: {said}")
        written = refs(scratch, env=env)
        try:
            _adopt_packs(scratch / "objects" / "pack", pathlib.Path(objects, "pack"))
        except OSError as exc:
            raise errors.GitError(f"cannot write under the repository's .git: {exc}") from exc
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return written


def _copy_shallow(source: pathlib.Path, target: pathlib.Path, hashes: str) -> None:
    """Copy a shallow file's object ids, and refuse a link or a file that holds anything else.

    A jailed command that can write `.git` can link the file to any other. Git writes the
    file itself, never a link, and quotes a line it cannot read as an object id.

    Args:
        source: The repository's shallow file.
        target: The scratch repository's.
        hashes: The repository's object format, which sets an id's length.

    Raises:
        GitError: The file is a link, or a line is no object id; the reason quotes no line.
        OSError: The file cannot be opened for another reason.
    """
    width = 40 if hashes == "sha1" else 64
    object_id = re.compile(rb"[0-9a-f]{%d}\n?" % width)
    try:
        opened = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        if exc.errno != errno.ELOOP:
            raise
        raise errors.GitError(f"{source} is a link; nothing was fetched") from exc
    with os.fdopen(opened, "rb") as lines, target.open("wb") as copy:
        # At most one id and its newline per read: no other line is ever held whole.
        while line := lines.readline(width + 2):
            if object_id.fullmatch(line) is None:
                raise errors.GitError(f"{source} is not a list of object ids; nothing was fetched")
            copy.write(line.rstrip(b"\n") + b"\n")


def refs(path: pathlib.Path, *patterns: str, env: dict[str, str] | None = None) -> dict[str, str]:
    """Return each ref with its tip.

    Args:
        path: The repository git runs in.
        *patterns: Ref names or namespaces, as `git for-each-ref` matches them; none lists all.
        env: Environment entries naming another git directory.

    Returns:
        The refs' full names and shas.
    """
    rows = _lines(path, "for-each-ref", "--format=%(refname) %(objectname)", *patterns, env=env)
    return dict(line.split(" ", 1) for line in rows)


def _lines(path: pathlib.Path, *args: str, env: dict[str, str] | None = None) -> list[str]:
    """Return Git's LF-delimited names with their original bytes.

    Args:
        path: The repository.
        *args: A Git listing with no LF in its fields.
        env: Environment entries for a scratch repository.

    Returns:
        Each record decoded for use in a filesystem operation.

    Raises:
        GitError: The listing failed.
    """
    _, code, out, err = git_command.spawn(path, args, env, None)
    if code:
        error = git_command.failure(args, code, os.fsdecode(out), os.fsdecode(err))
        raise error
    return os.fsdecode(out).removesuffix("\n").split("\n") if out else []


def _adopt_packs(source: pathlib.Path, target: pathlib.Path) -> None:
    """Move every pack of a directory into another, each index last.

    A pack is named for its content, so an existing name is the same pack. Git reads a
    pack once its index exists, so the index lands after the pack it describes.

    Args:
        source: The scratch repository's pack directory.
        target: The repository's pack directory.
    """
    target.mkdir(parents=True, exist_ok=True)
    for pack in sorted(source.glob("pack-*.pack")):
        for suffix in (".pack", ".rev", ".idx"):
            part = pack.with_suffix(suffix)
            final = target / part.name
            if part.is_file() and not final.exists():
                staged = target / f"tmp_agent6_{part.name}"
                shutil.move(part, staged)
                staged.replace(final)
