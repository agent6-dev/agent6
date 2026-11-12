# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Fetch a remote's branches, or one pull request, without the repository's own config.

The fetch is the operator's click, and it runs on the host over a `.git/config` a jailed
command can write below `full` `sandbox.isolation`. The network step therefore runs in a
scratch repository made for it, where git reads the operator's global and system config
alone: no key of the project's `.git/config` names a transport command, a proxy, a
credential helper or a URL rewrite. The project lends its objects (an alternate, so only
new history travels) and the remote's URL. The new pack and the remote-tracking refs are
then copied in.
"""

from __future__ import annotations

import dataclasses
import pathlib
import re
import shutil
import tempfile

from agent6 import errors
from agent6.vcs.git import command as git_command
from agent6.vcs.git import read as git_read

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
            `.git` this process cannot write. The reason carries no http(s) credentials.
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
    url = _fetch_url(path, remote)
    refspec = (
        f"+refs/heads/*:{tracking}*"
        if pull is None
        else f"+refs/pull/{pull}/head:{tracking}pr/{pull}"
    )
    objects = pathlib.Path(
        git_command.run(
            path, "rev-parse", "--path-format=absolute", "--git-path", "objects"
        ).stdout.strip()
    )
    before = _tracking_refs(path, tracking)
    common = git_read.git_common_dir(path)
    try:
        scratch = pathlib.Path(tempfile.mkdtemp(prefix="agent6-fetch-", dir=common))
    except OSError as exc:
        raise errors.GitError(f"cannot write under the repository's .git: {exc}") from exc
    try:
        env = {"GIT_DIR": str(scratch), "GIT_ALLOW_PROTOCOL": ALLOWED_PROTOCOLS}
        hashes = git_command.run(path, "rev-parse", "--show-object-format").stdout.strip()
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
        if (common / "shallow").is_file():
            # A shallow clone's boundary: without it git walks into history the clone lacks.
            shutil.copyfile(common / "shallow", scratch / "shallow")
        try:
            # unpackLimit 1 keeps every fetch as one pack, the unit copied in below.
            fetched = git_command.run(
                scratch,
                "-c",
                "fetch.unpackLimit=1",
                "fetch",
                "--quiet",
                "--no-tags",
                "--no-recurse-submodules",
                "--no-write-fetch-head",
                "--no-auto-maintenance",
                "--end-of-options",
                url,
                refspec,
                check=False,
                env_extra=env,
            )
        except errors.GitError as exc:  # with check off, only the timeout raises
            raise errors.GitError(
                f"fetch from {name} got no answer within {git_command.GIT_TIMEOUT_S:.0f}s"
            ) from exc
        if not fetched.ok:
            said = redact(fetched.stderr.strip()) or f"git fetch exited {fetched.returncode}"
            raise errors.GitError(f"fetch from {name} failed: {said}")
        after = _tracking_refs(scratch, tracking, env)
        _adopt_packs(scratch / "objects" / "pack", objects / "pack")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    moved = {ref: sha for ref, sha in after.items() if before.get(ref) != sha}
    if moved:
        git_command.run(
            path,
            "update-ref",
            "-m",
            f"agent6: fetch {name}",
            "--stdin",
            stdin_text="".join(f"update {ref} {sha}\n" for ref, sha in moved.items()),
        )
    if pull is not None:
        return f"fetched {name}/pr/{pull} at {after[f'{tracking}pr/{pull}'][:12]}"
    if not moved:
        return f"{name} is up to date"
    names = [ref.removeprefix(tracking) for ref in moved]
    more = f" and {len(names) - 3} more" if len(names) > 3 else ""
    return f"fetched {name}: {', '.join(names[:3])}{more}"


def _tracking_refs(
    path: pathlib.Path, prefix: str, env: dict[str, str] | None = None
) -> dict[str, str]:
    """Return each ref under a prefix with its tip.

    Args:
        path: The repository git runs in.
        prefix: The ref namespace, ending in `/`.
        env: Environment entries naming another git directory.

    Returns:
        The refs' full names and shas.
    """
    out = git_command.run(
        path, "for-each-ref", "--format=%(refname) %(objectname)", prefix, env_extra=env
    ).stdout
    return dict(line.split(" ", 1) for line in out.splitlines() if " " in line)


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
