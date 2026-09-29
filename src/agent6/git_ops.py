# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Run git for agent6 with hard safety invariants.

No code path here spells a push, a force flag or a history rewrite, so there is
nothing to refuse at runtime (pinned by `test_git_ops_never_spells_a_destructive_verb`);
the one sanctioned exception is `force_delete_squash_merged_branch`. Config can
loosen benign options (auto-stash, branch-per-run) and never these.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
from collections.abc import Collection

from agent6 import child_env, commit_message, kinds


class GitError(Exception):
    """Generic git failure."""


# Local ops finish in well under a second; this fires only on a stuck filesystem or a held lock.
_GIT_TIMEOUT_S = 120.0

# How long a timed-out git gets to unlink its lockfiles on SIGTERM before SIGKILL.
_GIT_TERM_GRACE_S = 5.0


@dataclasses.dataclass(frozen=True, slots=True)
class GitStatus:
    """The worktree against HEAD.

    Attributes:
        branch: The checked-out branch.
        head_sha: HEAD's sha; "" on an unborn branch.
        is_clean: No tracked file is modified and no untracked file exists outside the
            caller's exclude set.
        untracked_count: Untracked files outside the exclude set.
        modified_count: Tracked files with uncommitted changes, the operator's work a
            start gate reads.
    """

    branch: str
    head_sha: str
    is_clean: bool
    untracked_count: int
    modified_count: int


@dataclasses.dataclass(frozen=True, slots=True)
class CommitIdentity:
    """The identity and provenance trailer a run's commits carry.

    Attributes:
        name: The `[git.commit]` name override; None lets git's own config decide.
        email: The `[git.commit]` email override; None lets git's own config decide.
        trailer: A rendered trailer line, appended once per commit.
    """

    name: str | None = None
    email: str | None = None
    trailer: str | None = None


def verify_git_identity(path: pathlib.Path, identity: CommitIdentity) -> tuple[str, str]:
    """Resolve the author identity future commits use.

    Per field, the `[git.commit]` override wins over the repo's `git config`.
    A silently auto-generated identity is noticed weeks later in `git log`, so
    an empty field refuses.

    Args:
        path: The repository.
        identity: The configured overrides.

    Returns:
        The name and email.

    Raises:
        GitError: Either field is empty after both sources.
    """
    name = identity.name or _run(path, "config", "user.name", check=False).stdout.strip()
    email = identity.email or _run(path, "config", "user.email", check=False).stdout.strip()
    missing: list[str] = []
    if not name:
        missing.append("user.name")
    if not email:
        missing.append("user.email")
    if missing:
        joined = " and ".join(missing)
        raise GitError(
            f"Git identity not configured: {joined} is empty. Either run\n"
            f"    git -C {path} config user.name 'Your Name'\n"
            f"    git -C {path} config user.email 'you@example.com'\n"
            f"or set [git.commit].name / [git.commit].email in your agent6 config."
        )
    return name, email


def _git() -> str:
    """Return the git executable's path.

    Raises:
        GitError: git is not on PATH.
    """
    git = shutil.which("git")
    if git is None:
        raise GitError("git executable not found on PATH")
    return git


# Repo-config keys that run a repo-controlled command on the host, blanked with `-c` (highest
# precedence): fsmonitor on every index refresh, diff.external on every diff, gpg.program on
# every commit. None costs correctness: agent6's per-step commits are unsigned by design.
# Content drivers (filter.*, merge.*.driver) have no blanket switch; `_repo_driver_overrides`
# blanks each by name. The prefix keys pin the `a/` and `b/` headers every cited path reads.
_GIT_HARDENING: tuple[str, ...] = (
    "-c",
    "core.fsmonitor=false",
    "-c",
    "diff.external=",
    "-c",
    "commit.gpgsign=false",
    "-c",
    "diff.noprefix=false",
    "-c",
    "diff.mnemonicPrefix=false",
    "-c",
    "diff.srcPrefix=a/",
    "-c",
    "diff.dstPrefix=b/",
)

# A repo hook is repo-controlled host code, so honoring one on agent6's own commit is an RCE
# vector; set once from `git.run_repo_hooks` at startup. Mutated, never rebound.
_hook_policy: dict[str, bool] = {"honor_repo_hooks": False}


def set_repo_hook_policy(honor: bool) -> None:
    """Configure whether agent6's own git ops fire the repo's `.git/hooks/*`."""
    _hook_policy["honor_repo_hooks"] = honor


# A driver in `.git/config` is a host command, an RCE vector for a cloned poisoned repo;
# honoring them is the Git-LFS opt-in. Set once from `git.run_repo_filters` at startup.
_filter_policy: dict[str, bool] = {"honor_repo_filters": False}


def set_repo_filter_policy(honor: bool) -> None:
    """Configure whether agent6's git ops honor the repo's own content drivers."""
    _filter_policy["honor_repo_filters"] = honor


# The config keys that name a driver command.
_DRIVER_KEY_RE = r"^(filter\..*\.(clean|smudge|process)|merge\..*\.driver)$"


def _repo_driver_overrides(cwd: pathlib.Path) -> tuple[str, ...]:
    """Return the `-c` flags that blank every content driver the repo's own config defines.

    An empty `filter.<n>.clean` passes through and an empty `merge.<n>.driver` reports
    a conflict instead of running. Only the repo's `.git/config` and its includes are
    read: a filter in the operator's `~/.gitconfig` is trusted. Re-read per call, so a
    driver a jailed command writes mid-run is caught too.

    Args:
        cwd: The repository.

    Returns:
        The flags; empty when the policy honors drivers or the repo defines none.
    """
    if _filter_policy["honor_repo_filters"]:
        return ()
    try:
        proc = subprocess.run(
            [
                _git(),
                "-C",
                str(cwd),
                "config",
                "--local",
                # A git op follows `[include]` to a repo-controlled file; this enumeration must too.
                "--includes",
                "--name-only",
                "--get-regexp",
                _DRIVER_KEY_RE,
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    # A filter with both clean and smudge yields two keys for one driver.
    filters: set[str] = set()
    merges: set[str] = set()
    for key in proc.stdout.split():
        if key.startswith("filter."):
            filters.add(key[len("filter.") : key.rindex(".")])
        elif key.startswith("merge."):
            merges.add(key[len("merge.") : key.rindex(".")])
    overrides: list[str] = []
    for name in sorted(filters):
        overrides += [
            "-c",
            f"filter.{name}.clean=",
            "-c",
            f"filter.{name}.smudge=",
            "-c",
            f"filter.{name}.process=",
        ]
    for name in sorted(merges):
        overrides += ["-c", f"merge.{name}.driver="]
    return tuple(overrides)


# Force git's builtin diff/show renderer: `diff.external` and per-file `diff.<d>.textconv` run
# host commands the `-c` overrides do not cover. Placed after the subcommand.
DIFF_SHOW_SAFETY_FLAGS: tuple[str, ...] = ("--no-ext-diff", "--no-textconv")


def git_hardening_flags(cwd: pathlib.Path) -> tuple[str, ...]:
    """Return the `-c` overrides every agent6 git invocation carries, placed before the subcommand.

    Public so the callers that shell out to git outside this module carry the same
    hardening; diff and show callers also add `DIFF_SHOW_SAFETY_FLAGS` after the subcommand.

    Args:
        cwd: The repository, for its content drivers.

    Returns:
        The fixed set, the hooks path unless the policy honors repo hooks, and a blank
        override per content driver the repo defines.
    """
    # /dev/null is not a directory, so git finds no hooks there.
    hooks = () if _hook_policy["honor_repo_hooks"] else ("-c", "core.hooksPath=/dev/null")
    return (*_GIT_HARDENING, *hooks, *_repo_driver_overrides(cwd))


def _run(
    cwd: pathlib.Path,
    *args: str,
    check: bool = True,
    env_extra: dict[str, str] | None = None,
    stdin_text: str | None = None,
) -> kinds.CommandResult:
    """Run one hardened git command in the repository.

    Args:
        cwd: The repository.
        *args: The git subcommand and its arguments.
        check: Raise on a non-zero exit.
        env_extra: Environment entries added for this call.
        stdin_text: Text fed to git's stdin.

    Returns:
        The command's result, its output decoded lossily.

    Raises:
        GitError: The command failed with check set, or timed out.
    """
    # GIT_TERMINAL_PROMPT=0 fails a credential prompt fast; LC_ALL=C keeps the stash
    # rescue's parse of git's prose valid.
    env = child_env.without_provider_keys(
        {**os.environ, "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C", **(env_extra or {})}
    )
    hardening = git_hardening_flags(cwd)
    # git 2.53 dies rc=128 on the empty `diff.external` override, so diff and show force the
    # builtin renderer by flag.
    argv = list(args)
    if argv and argv[0] in ("diff", "show"):
        argv[1:1] = DIFF_SHOW_SAFETY_FLAGS
    # Drivers are blanked on every op: a list of driver-running verbs would enumerate badness.
    full_argv = (_git(), *hardening, *argv)
    index_lock = cwd / ".git" / "index.lock"
    lock_preexisted = index_lock.exists()
    proc = subprocess.Popen(
        full_argv,
        cwd=cwd,
        stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
        # Bytes: diff and show emit raw file bytes, and a latin-1 file is not binary to git.
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    try:
        stdin_bytes = stdin_text.encode() if stdin_text is not None else None
        out, err = proc.communicate(input=stdin_bytes, timeout=_GIT_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        # git's TERM handler unlinks its own lockfiles, so a lock left afterwards is another git's.
        proc.terminate()
        try:
            proc.communicate(timeout=_GIT_TERM_GRACE_S)
        except subprocess.TimeoutExpired:
            # SIGKILL skips git's cleanup: clear the lock only when it appeared under this child.
            proc.kill()
            proc.communicate()
            if not lock_preexisted:
                with contextlib.suppress(OSError):
                    index_lock.unlink(missing_ok=True)
        raise GitError(
            f"git {' '.join(args)} timed out after {_GIT_TIMEOUT_S:.0f}s"
            " (a stuck filesystem or a held .git/index.lock?)"
        ) from exc
    result = kinds.CommandResult(
        argv=full_argv,
        returncode=proc.returncode,
        stdout=out.decode(errors="replace"),
        stderr=err.decode(errors="replace"),
        duration_s=0.0,
    )
    if check and not result.ok:
        # `git commit` explains most failures on stdout, not stderr.
        stderr_msg = result.stderr.strip()
        stdout_msg = result.stdout.strip()
        if stderr_msg and stdout_msg:
            detail = f"{stderr_msg} | stdout: {stdout_msg}"
        else:
            detail = stderr_msg or stdout_msg or f"exit {result.returncode}"
        raise GitError(f"git {' '.join(args)} failed: {detail}")
    return result


def is_git_repo(path: pathlib.Path) -> bool:
    """Return whether the path is inside a git work tree."""
    res = _run(path, "rev-parse", "--is-inside-work-tree", check=False)
    return res.ok and res.stdout.strip() == "true"


def toplevel(path: pathlib.Path) -> pathlib.Path | None:
    """Return the enclosing work tree's root, or None outside a repo or inside `.git`."""
    res = _run(path, "rev-parse", "--show-toplevel", check=False)
    if not res.ok:
        return None
    text = res.stdout.strip()
    return pathlib.Path(text) if text else None


def paths_dirty(path: pathlib.Path, rel_paths: tuple[str, ...]) -> bool:
    """Return whether a path-limited commit of these paths would record something.

    Args:
        path: The repository.
        rel_paths: Repo-relative paths.

    Returns:
        True when any path is untracked, modified or staged; dirt elsewhere is ignored.
    """
    if not rel_paths:
        return False
    res = _run(path, "status", "--porcelain", "--", *rel_paths, check=False)
    return bool(res.stdout.strip())


def _porcelain_entries(path: pathlib.Path) -> list[tuple[str, str]]:
    """Return the status code and repo-relative path of every changed or untracked file.

    NUL-separated, so any filename round-trips; a rename's source path is skipped.

    Args:
        path: The repository.

    Returns:
        The (code, path) pairs.
    """
    res = _run(path, "status", "--porcelain=v1", "-z", "--untracked-files=all", check=False)
    fields = res.stdout.split("\0")
    out: list[tuple[str, str]] = []
    i = 0
    while i < len(fields):
        entry = fields[i]
        i += 1
        if len(entry) < 4:
            continue
        code, rel = entry[:2], entry[3:]
        out.append((code, rel))
        if code[0] in "RC":
            i += 1
    return out


def modified_paths(path: pathlib.Path) -> list[str]:
    """Return the tracked files with uncommitted changes; untracked ones are never listed."""
    return [rel for code, rel in _porcelain_entries(path) if code != "??"]


def untracked_paths(path: pathlib.Path) -> frozenset[str]:
    """Return every untracked, non-ignored file, repo-relative.

    Taken once at run start as `untracked_at_start`: those files are the operator's,
    so chain commits and dirty checks leave them out.
    """
    return frozenset(rel for code, rel in _porcelain_entries(path) if code == "??")


def status(path: pathlib.Path, *, exclude: Collection[str] = ()) -> GitStatus:
    """Return the worktree's status against HEAD.

    Args:
        path: The repository.
        exclude: Untracked files not counted, a run's `untracked_at_start`.

    Returns:
        The status.

    Raises:
        GitError: The path is not a git repository.
    """
    if not is_git_repo(path):
        raise GitError(f"Not a git repository: {path}")
    branch_res = _run(path, "rev-parse", "--abbrev-ref", "HEAD", check=False)
    if branch_res.ok:
        branch = branch_res.stdout.strip()
    else:
        # An unborn HEAD fails rev-parse, but `branch --show-current` still names the branch.
        branch = _run(path, "branch", "--show-current", check=False).stdout.strip()
    head_res = _run(path, "rev-parse", "HEAD", check=False)
    head_sha = head_res.stdout.strip() if head_res.ok else ""
    untracked = 0
    modified = 0
    for code, rel in _porcelain_entries(path):
        if code == "??":
            if rel not in exclude:
                untracked += 1
        else:
            modified += 1
    return GitStatus(
        branch=branch,
        head_sha=head_sha,
        is_clean=(untracked == 0 and modified == 0),
        untracked_count=untracked,
        modified_count=modified,
    )


def stash_tracked_changes(path: pathlib.Path, message: str) -> None:
    """Stash the tracked files' uncommitted changes; untracked files stay in place."""
    _run(path, "stash", "push", "--message", message)


def auto_stash_message(session_id: str) -> str:
    """Return the auto-stash message the finalizer finds the stash by, never by position."""
    return f"agent6 auto-stash before run {session_id}"


@dataclasses.dataclass(frozen=True, slots=True)
class StashEntry:
    """One `git stash list` entry.

    Attributes:
        ref: The position at lookup time (`stash@{1}`), for hints only; it shifts the
            moment anyone pushes or drops a stash.
        sha: The immutable commit; anything that mutates restores by it.
    """

    ref: str
    sha: str


def find_stash(path: pathlib.Path, message: str) -> StashEntry | None:
    """Return the newest stash pushed with exactly this message, or None.

    The subject is `On <branch>: MSG`, so `: MSG` anchored at the end matches the whole
    message: lane ids are ordinal, so one message can be a prefix of another.
    """
    res = _run(path, "stash", "list", "--format=%gd%x09%H%x09%gs", check=False)
    for line in res.stdout.splitlines():
        ref, _, rest = line.partition("\t")
        sha, _, subject = rest.partition("\t")
        if subject.endswith(f": {message}"):
            return StashEntry(ref=ref, sha=sha)
    return None


# `git stash drop` prints "Dropped stash@{0} (<sha>)".
_DROPPED_SHA_RE = re.compile(r"^Dropped .*\(([0-9a-f]{7,64})\)", re.MULTILINE)


def restore_stash(path: pathlib.Path, stash: StashEntry) -> bool:
    """Apply the stash back onto the working tree by sha, dropping it on a clean apply.

    A conflicted apply leaves the markers and the stash in place: nothing undoes it.

    Args:
        path: The repository.
        stash: The entry to restore.

    Returns:
        True on a clean apply; False when the apply failed and everything stays put.

    Raises:
        GitError: A raced drop took a concurrent stash and putting it back failed; the
            apply has landed and the message carries the recovery command.
    """
    if not _run(path, "stash", "apply", stash.sha, check=False).ok:
        return False
    _drop_by_sha(path, stash.sha)
    return True


def _drop_by_sha(path: pathlib.Path, sha: str) -> None:
    """Drop the stash entry with this commit, putting back a bystander taken by mistake.

    `git stash drop` takes a position, and a stash pushed between the lookup and the
    drop shifts every position. git names the commit it dropped: a stranger's is stored
    straight back under its own subject, and agent6's own then stays listed, since a
    second attempt would race the same way.

    Raises:
        GitError: The bystander could not be stored back.
    """
    listed = _run(path, "stash", "list", "--format=%gd%x09%H", check=False)
    ref = ""
    for line in listed.stdout.splitlines():
        entry_ref, _, entry_sha = line.partition("\t")
        if entry_sha == sha:
            ref = entry_ref
            break
    if not ref:
        return
    dropped = _DROPPED_SHA_RE.search(_run(path, "stash", "drop", ref, check=False).stdout)
    if dropped is None:
        return
    taken = dropped.group(1)
    # Prefix either way: an abbreviated oid must not read as a stranger's stash.
    if sha.startswith(taken) or taken.startswith(sha):
        return
    # The stash commit's subject is the reflog's "On <branch>: <message>".
    subject = _run(path, "log", "-1", "--format=%s", taken, check=False).stdout.strip()
    subject = subject or "restored by agent6"
    store = _run(path, "stash", "store", "-m", subject, taken, check=False)
    if not store.ok:
        # The commit still exists under `taken`, so the message carries the recovery.
        detail = store.stderr.strip() or store.stdout.strip() or f"exit {store.returncode}"
        raise GitError(
            f"a stash pushed concurrently ({subject!r}) was taken by a raced drop and "
            f"putting it back failed ({detail}); restore it with:\n"
            f"    git stash store -m {subject!r} {taken}"
        )


def branch_exists(path: pathlib.Path, name: str) -> bool:
    """Return whether the local branch exists."""
    return _run(path, "rev-parse", "--verify", "--quiet", f"refs/heads/{name}", check=False).ok


def valid_branch_name(name: str) -> bool:
    """Return whether the name passes `git check-ref-format --branch`.

    A session id becomes a branch and a chain ref; a branch name is the stricter role,
    so this is the one check a session id needs. Run from "/": no repo is involved.
    """
    return _run(pathlib.Path("/"), "check-ref-format", "--branch", name, check=False).ok


def list_run_branches(path: pathlib.Path) -> tuple[str, ...]:
    """Return the local branches under `agent6/`, sorted."""
    res = _run(path, "for-each-ref", "--format=%(refname:short)", "refs/heads/agent6/", check=False)
    return tuple(b for b in res.stdout.splitlines() if b.strip())


def run_ref_tips(path: pathlib.Path) -> dict[str, str]:
    """Return the tip sha of every run branch and chain ref, in one git call.

    A per-row rev-parse would put about 50 subprocesses on the hub's poll.

    Args:
        path: The repository.

    Returns:
        Branch short names and full chain ref names, each to its sha.
    """
    res = _run(
        path,
        "for-each-ref",
        "--format=%(refname) %(objectname)",
        "refs/heads/agent6/",
        "refs/agent6/*/head",
        check=False,
    )
    out: dict[str, str] = {}
    for line in res.stdout.splitlines():
        ref, _, sha = line.partition(" ")
        if ref.startswith("refs/heads/"):
            ref = ref.removeprefix("refs/heads/")
        if ref and sha:
            out[ref] = sha
    return out


def is_ancestor(path: pathlib.Path, maybe_ancestor: str, ref: str) -> bool:
    """Return whether the first commit is reachable from the second."""
    return _run(path, "merge-base", "--is-ancestor", maybe_ancestor, ref, check=False).ok


def delete_branch_if_merged(path: pathlib.Path, branch: str) -> bool:
    """Delete the branch with the safe delete, which git refuses unless it is reachable-merged.

    Args:
        path: The repository.
        branch: The branch.

    Returns:
        True when deleted; False when git refused, a squash-merged or unmerged branch.
    """
    return _run(path, "branch", "-d", branch, check=False).ok


def branch_tip_sha(path: pathlib.Path, branch: str) -> str | None:
    """Return the sha the branch points at, or None when it does not resolve."""
    res = _run(path, "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}", check=False)
    sha = res.stdout.strip()
    return sha or None


def merge_stamp_holds(
    path: pathlib.Path, session_id: str, run_branch: str, merged_tip: str
) -> bool:
    """Return whether a run's merged stamp still describes everything it committed.

    A resumed run keeps committing under a prior execution's stamp; the claim holds
    while the stamp's tip is the chain's or the run branch's. A gone chain and branch,
    unreadable git, or a stamp with no tip keeps the claim. Every surface reads this.

    Args:
        path: The repository.
        session_id: The run's session.
        run_branch: The run's visible branch.
        merged_tip: The tip the stamp recorded; "" keeps the claim.

    Returns:
        Whether the claim holds.
    """
    if not merged_tip:
        return True
    tips: set[str] = set()
    with contextlib.suppress(GitError):
        chain = chain_tip(path, chain_ref_for(session_id)) if session_id else None
        branch = branch_tip_sha(path, run_branch) if run_branch else None
        tips = {t for t in (chain, branch) if t}
    return not tips or merged_tip in tips


def force_delete_squash_merged_branch(path: pathlib.Path, branch: str) -> bool:
    """Force-delete a squash-merged run branch, the one sanctioned force-delete.

    The safe delete refuses a squash-merged branch, since its commits are not reachable
    from the base even though its content is. Only `sessions prune --delete-squashed`
    calls this, for a branch the manifest confirms was squash-merged.

    Args:
        path: The repository.
        branch: The branch.

    Returns:
        True when deleted.
    """
    return _run(path, "branch", "-D", branch, check=False).ok


def create_branch(path: pathlib.Path, name: str, *, start_point: str | None = None) -> None:
    """Create the branch and check it out, or only check it out when it exists.

    An existing branch is never moved, so a resumed run reuses its branch.

    Args:
        path: The repository.
        name: The branch.
        start_point: Where a new branch is cut from; None is HEAD.
    """
    existing = _run(path, "branch", "--list", name, check=False)
    if existing.ok and existing.stdout.strip():
        _run(path, "checkout", name)
    elif start_point:
        _run(path, "checkout", "-b", name, start_point)
    else:
        _run(path, "checkout", "-b", name)


_CHAIN_NS = "refs/agent6"
_CHAIN_KIND = "head"


def machine_chain_ref_for(machine_id: str) -> str:
    """Return the chain ref a machine's states continue from."""
    return chain_ref_for(f"machine-{machine_id}")


# Every visible agent6 branch: a run's `agent6/<id>`, a machine's `agent6/machine-<id>`.
BRANCH_PREFIX = "agent6/"


def run_branch_for(session_id: str) -> str:
    """Return the visible branch a run's chain advances under `[git].branch_per_run`."""
    return f"{BRANCH_PREFIX}{session_id}"


def machine_branch_for(machine_id: str) -> str:
    """Return the visible branch a machine's `mode="run"` states land on."""
    return f"{BRANCH_PREFIX}machine-{machine_id}"


def chain_ref_for(session_id: str) -> str:
    """Return the ref holding a session's commit chain, `refs/agent6/<id>/head`.

    The id is a namespace, not the ref: git resolves `refs/<name>` before
    `refs/heads/<name>`, so a ref at `refs/agent6/<id>` would make the visible branch's
    short name ambiguous. The kind under the id follows `refs/pull/<n>/head`.
    """
    return f"{_CHAIN_NS}/{session_id}/{_CHAIN_KIND}"


def set_ref(path: pathlib.Path, ref: str, sha: str) -> None:
    """Point one of agent6's own refs at a sha with no checkout; branches use `create_branch_at`."""
    _run(path, "update-ref", ref, sha)


def delete_ref(path: pathlib.Path, ref: str) -> None:
    """Delete the ref; a missing one is a no-op."""
    _run(path, "update-ref", "-d", ref, check=False)


def list_chain_refs(path: pathlib.Path) -> tuple[tuple[str, str], ...]:
    """Return the session id and sha of every chain ref, sorted by id.

    Globbed on the kind, so another per-session ref beside it is not read as a chain.
    """
    pattern = f"{_CHAIN_NS}/*/{_CHAIN_KIND}"
    out = _run(path, "for-each-ref", "--format=%(refname)%00%(objectname)", pattern).stdout
    rows: list[tuple[str, str]] = []
    for line in out.splitlines():
        ref, _, sha = line.partition("\x00")
        if sha and ref.startswith(f"{_CHAIN_NS}/") and ref.endswith(f"/{_CHAIN_KIND}"):
            rows.append((ref[len(_CHAIN_NS) + 1 : -(len(_CHAIN_KIND) + 1)], sha))
    return tuple(sorted(rows))


def checkout_detached(path: pathlib.Path, rev: str) -> None:
    """Check out the rev detached, in an agent6-owned clone, never the operator's checkout."""
    _run(path, "checkout", "-q", "--detach", rev)


def add_worktree(path: pathlib.Path, dest: pathlib.Path, sha: str) -> None:
    """Add a linked worktree detached at the sha, a fork's own checkout.

    It shares the repository's refs and objects, so a chain commit made there is
    visible from every checkout; the operator's checkout, HEAD and index are untouched.

    Args:
        path: The repository.
        dest: Where the worktree goes.
        sha: The commit to check out.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(path, "worktree", "add", "--detach", str(dest.absolute()), sha)


def remove_worktree(path: pathlib.Path, dest: pathlib.Path) -> bool:
    """Delete a linked worktree of the repository and git's record of it.

    Only that entry goes: a repo-wide `worktree prune` would also drop the record of an
    operator's worktree whose directory is missing for the moment.

    Args:
        path: The repository.
        dest: The worktree.

    Returns:
        True when deleted; False when the directory is not a worktree of this repository
        or could not be removed.
    """
    pointer = dest / ".git"
    if not pointer.is_file():
        return False
    text = pointer.read_text(encoding="utf-8", errors="replace").strip()
    if not text.startswith("gitdir:"):
        return False
    admin = pathlib.Path(text[len("gitdir:") :].strip())
    if not admin.is_absolute():
        admin = dest / admin
    admin = admin.resolve()
    if admin.parent != git_common_dir(path) / "worktrees":
        return False
    try:
        shutil.rmtree(dest)
    except OSError:
        return False  # what stayed keeps its record, so git and the caller still see it
    shutil.rmtree(admin, ignore_errors=True)
    return True


def git_common_dir(path: pathlib.Path) -> pathlib.Path:
    """Return the repository's shared `.git` directory, absolute: the main checkout's."""
    out = _run(path, "rev-parse", "--path-format=absolute", "--git-common-dir").stdout.strip()
    return pathlib.Path(out).resolve()


def _worktree_of_branch(path: pathlib.Path, branch: str) -> pathlib.Path | None:
    """Return the checkout that has the branch checked out, or None when none does."""
    out = _run(path, "worktree", "list", "--porcelain").stdout
    where: pathlib.Path | None = None
    for line in out.splitlines():
        if line.startswith("worktree "):
            where = pathlib.Path(line[len("worktree ") :])
        elif line == f"branch refs/heads/{branch}" and where is not None:
            return where
    return None


def create_branch_at(path: pathlib.Path, name: str, sha: str) -> None:
    """Create the branch at the sha without checking it out.

    Additive only: HEAD and the working tree are untouched, so a fork cuts its branch
    at a historical sha while the operator's checkout stays put.

    Args:
        path: The repository.
        name: The branch.
        sha: The commit it points at.

    Raises:
        GitError: The branch exists pointing elsewhere; a branch is never moved.
    """
    existing = _run(path, "rev-parse", "--verify", "--quiet", f"refs/heads/{name}", check=False)
    if existing.ok and existing.stdout.strip():
        if existing.stdout.strip() == sha:
            return
        raise GitError(
            f"branch {name!r} already exists at {existing.stdout.strip()[:12]}, not {sha[:12]}; "
            "refusing to move it"
        )
    _run(path, "branch", name, sha)


def init_repo(path: pathlib.Path) -> None:
    """Initialize a repository at the path."""
    _run(path, "init")


def clone_repo(origin: pathlib.Path, dest: pathlib.Path) -> None:
    """Clone one local repository into a destination.

    Both are plain paths, so git hardlinks on the same filesystem and copies across
    devices. Run from "/", so a missing origin fails as a GitError from git itself.

    Args:
        origin: The repository to clone.
        dest: Where the clone goes.
    """
    _run(pathlib.Path("/"), "clone", str(origin.absolute()), str(dest.absolute()))


def unignored(path: pathlib.Path, candidates: tuple[str, ...]) -> tuple[str, ...]:
    """Return the repo-relative candidates git does not ignore.

    Args:
        path: The repository.
        candidates: Repo-relative paths.

    Returns:
        The candidates not matched by an ignore rule, in their given order.
    """
    if not candidates:
        return ()
    # check-ignore prints the ignored inputs and exits 1 when none match; only stdout is read.
    res = _run(path, "check-ignore", "--", *candidates, check=False)
    ignored = {line.strip() for line in res.stdout.splitlines() if line.strip()}
    return tuple(c for c in candidates if c not in ignored)


def commit_all(
    path: pathlib.Path,
    message: str,
    *,
    trailers: dict[str, str] | None = None,
    identity: CommitIdentity | None = None,
) -> str:
    """Stage everything and commit.

    Args:
        path: The repository.
        message: The commit message.
        trailers: Trailer lines appended to the message.
        identity: The author and committer overrides; None uses the repo's config,
            validated at startup by `verify_git_identity`.

    Returns:
        The new HEAD sha.
    """
    _run(path, "add", "-A")
    return _commit(path, message, trailers=trailers, identity=identity)


def commit_paths(
    path: pathlib.Path,
    message: str,
    paths: tuple[str, ...],
    *,
    trailers: dict[str, str] | None = None,
    identity: CommitIdentity | None = None,
) -> str:
    """Stage and commit only these repo-relative paths.

    Unrelated staged changes stay staged and uncommitted, so `agent6 init`'s scaffold
    commit never folds the operator's work in.

    Args:
        path: The repository.
        message: The commit message.
        paths: The repo-relative paths to record.
        trailers: Trailer lines appended to the message.
        identity: The author and committer overrides; None uses the repo's config.

    Returns:
        The new HEAD sha.

    Raises:
        GitError: No paths were given.
    """
    if not paths:
        raise GitError("commit_paths requires at least one path")
    _run(path, "add", "--", *paths)
    return _commit(path, message, trailers=trailers, identity=identity, only_paths=paths)


def _identity_env(identity: CommitIdentity | None) -> dict[str, str] | None:
    """Return the author and committer env for a commit, or None for the repo's own identity."""
    if identity is None:
        return None
    env: dict[str, str] = {}
    if identity.name:
        env["GIT_AUTHOR_NAME"] = env["GIT_COMMITTER_NAME"] = identity.name
    if identity.email:
        env["GIT_AUTHOR_EMAIL"] = env["GIT_COMMITTER_EMAIL"] = identity.email
    return env or None


def _full_message(
    message: str, trailers: dict[str, str] | None, identity: CommitIdentity | None
) -> str:
    """Return the message with the identity trailer and any extra trailers appended once."""
    merged = dict(trailers or {})
    if identity is not None and identity.trailer and identity.trailer not in message:
        key, _, value = identity.trailer.partition(": ")
        merged[key] = value
    if not merged:
        return message
    trailer_lines = "\n".join(f"{k}: {v}" for k, v in merged.items())
    return f"{message}\n\n{trailer_lines}"


def chain_tip(path: pathlib.Path, ref: str) -> str | None:
    """Return the commit sha the ref resolves to, or None when it does not exist."""
    res = _run(path, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", check=False)
    sha = res.stdout.strip()
    return sha if res.returncode == 0 and sha else None


def commit_is_reachable(path: pathlib.Path, sha: str) -> bool:
    """Return whether any ref reaches the commit.

    Existence is a different question: a loose commit no ref reaches is one `git gc`
    from gone, and a sweep deleting the last copy of work asks this one.
    """
    res = _run(
        path, "for-each-ref", "--contains", sha, "--count=1", "--format=%(refname)", check=False
    )
    return res.ok and bool(res.stdout.strip())


def worktree_tree(path: pathlib.Path, seed: str | None, exclude: Collection[str]) -> str:
    """Return the tree sha of the worktree's content, staged into a temp index.

    The shared index is never read or written. The seed's tree fills the index first:
    ignore rules apply only to untracked files, so from an empty index `add -A` would
    drop every tracked-but-ignored file, which a later merge turns into deletions.

    Args:
        path: The repository.
        seed: The commit the tree is diffed or parented against; None in an unborn repo.
        exclude: Repo-relative paths kept out of the tree, the run's `untracked_at_start`,
            passed as pathspecs from a file so any size and filename work.

    Returns:
        The tree sha.
    """
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="agent6-chain-"))
    env = {"GIT_INDEX_FILE": str(tmp / "index")}
    try:
        if seed is not None:
            _run(path, "read-tree", seed, env_extra=env)
        if exclude:
            spec = tmp / "pathspec"
            spec.write_bytes(
                b"\0".join(
                    [b":/", *(f":(top,exclude,literal){rel}".encode() for rel in sorted(exclude))]
                )
            )
            _run(
                path,
                "add",
                "-A",
                f"--pathspec-from-file={spec}",
                "--pathspec-file-nul",
                env_extra=env,
            )
        else:
            _run(path, "add", "-A", env_extra=env)
        return _run(path, "write-tree", env_extra=env).stdout.strip()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# git's empty tree, the diff base of a chain with no commits yet.
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


def worktree_matches(path: pathlib.Path, ref: str, paths: Collection[str]) -> bool:
    """Return whether the worktree's content of the paths equals the ref's, ignoring the index."""
    res = _run(path, "diff", "--quiet", ref, "--", *sorted(paths), check=False)
    return res.returncode == 0


def _tree_sha(path: pathlib.Path, rev: str) -> str:
    """Return the tree sha of a commit-ish."""
    return _run(path, "rev-parse", f"{rev}^{{tree}}").stdout.strip()


def chain_dirty(
    path: pathlib.Path, ref: str, fallback_parent: str | None, *, exclude: Collection[str] = ()
) -> bool:
    """Return whether the worktree's content differs from the chain tip's tree.

    Args:
        path: The repository.
        ref: The chain ref.
        fallback_parent: The base when the ref does not exist; None means the empty tree.
        exclude: Repo-relative paths left out of the comparison.

    Returns:
        Whether the trees differ.

    Raises:
        GitError: The path is not a repository; callers read that as clean.
    """
    base = chain_tip(path, ref) or fallback_parent
    base_tree = _tree_sha(path, base) if base else EMPTY_TREE
    return base_tree != worktree_tree(path, base, exclude)


def chain_dirty_paths(
    path: pathlib.Path,
    ref: str,
    fallback_parent: str | None,
    limit: int,
    *,
    exclude: Collection[str] = (),
) -> list[str]:
    """Return the paths whose worktree content differs from the chain tip's tree.

    Args:
        path: The repository.
        ref: The chain ref.
        fallback_parent: The base when the ref does not exist; None means the empty tree.
        limit: The most paths returned.
        exclude: Repo-relative paths left out of the comparison.

    Returns:
        The differing paths, at most the limit.
    """
    base = chain_tip(path, ref) or fallback_parent
    base_tree = _tree_sha(path, base) if base else EMPTY_TREE
    return tree_diff_paths(path, base_tree, worktree_tree(path, base, exclude))[:limit]


def tree_paths(path: pathlib.Path, ref: str) -> frozenset[str]:
    """Return every path in the ref's tree, spelled as `status -z` spells it.

    Args:
        path: The repository.
        ref: The tree-ish.

    Returns:
        The paths; empty for a ref that does not exist.

    Raises:
        GitError: The ref is not a tree.
    """
    if _run(path, "rev-parse", "--verify", "--quiet", ref, check=False).returncode != 0:
        return frozenset()
    out = _run(path, "ls-tree", "-r", "--name-only", "-z", ref).stdout
    return frozenset(name for name in out.split("\0") if name)


def tree_diff_paths(path: pathlib.Path, old_tree: str, new_tree: str) -> list[str]:
    """Return the paths whose content differs between two trees."""
    out = _run(path, "diff-tree", "-r", "--name-only", old_tree, new_tree).stdout
    return [line for line in out.splitlines() if line]


def chain_commit(
    path: pathlib.Path,
    message: str,
    *,
    ref: str,
    fallback_parent: str | None,
    trailers: dict[str, str] | None = None,
    identity: CommitIdentity | None = None,
    also_branch: str | None = None,
    exclude: Collection[str] = (),
) -> str | None:
    """Record the worktree's content on the run's commit chain, touching no checkout.

    The tree is staged into a temp index and committed parented on the ref's current
    value: the ref is the chain state, so resume and concurrent runs compose without
    bookkeeping.

    Args:
        path: The repository.
        message: The commit message.
        ref: The chain ref, advanced to the new commit.
        fallback_parent: The parent when the ref does not exist yet, HEAD at run start;
            None makes a root commit in an unborn repo.
        trailers: Trailer lines appended to the message.
        identity: The author and committer overrides.
        also_branch: A visible branch moved with the ref; a checked-out one has its
            index brought forward.
        exclude: Repo-relative paths kept out, the run's `untracked_at_start`.

    Returns:
        The new sha, or None when the tree equals the parent's.
    """
    parent = chain_tip(path, ref) or fallback_parent
    tree = worktree_tree(path, parent, exclude)
    parent_args: list[str] = []
    if parent is not None:
        if _tree_sha(path, parent) == tree:
            return None
        parent_args = ["-p", parent]
    sha = _run(
        path,
        "commit-tree",
        tree,
        *parent_args,
        "-m",
        _full_message(message, trailers, identity),
        env_extra=_identity_env(identity),
    ).stdout.strip()
    _run(path, "update-ref", ref, sha)
    _advance_run_branch(path, also_branch, sha, expected=parent)
    return sha


def _advance_run_branch(
    path: pathlib.Path, branch: str | None, sha: str, *, expected: str | None
) -> None:
    """Move the run's visible branch to the sha, only from the tip the commit was built on.

    A compare-and-swap: a bare `update-ref` would rewind the operator's own commit on
    the branch. A branch that moved keeps its tip; the chain ref is the record either
    way. A checked-out branch has its index and worktree brought forward.

    Args:
        path: The repository.
        branch: The branch; None does nothing.
        sha: The new tip.
        expected: The tip the branch must still be at; None only creates a missing branch.
    """
    if not branch:
        return
    ref = f"refs/heads/{branch}"
    current = chain_tip(path, ref)
    if current is None:
        _run(path, "update-ref", ref, sha)  # the run's first commit creates it
        return
    if expected is not None and current == expected:
        _run(path, "update-ref", ref, sha, expected)
        _bring_index_forward(path, branch, expected, sha)


def chain_merge(
    path: pathlib.Path,
    merge_rev: str,
    message: str,
    *,
    ref: str,
    fallback_parent: str | None = None,
    identity: CommitIdentity | None = None,
    also_branch: str | None = None,
) -> str | None:
    """Merge a rev into the chain without touching HEAD, and sync the worktree to it.

    A rev descending from the tip fast-forwards instead of stacking an empty merge.

    Args:
        path: The repository.
        merge_rev: The commit to merge in.
        message: The merge commit's message.
        ref: The chain ref, advanced to the result.
        fallback_parent: The base when the ref does not exist yet.
        identity: The author and committer overrides.
        also_branch: A visible branch moved with the ref.

    Returns:
        The new tip, the old one when it already contained the rev, or None on a
        textual conflict, which leaves the chain and worktree untouched.

    Raises:
        GitError: The ref does not exist and no fallback parent was given.
    """
    ours = chain_tip(path, ref) or fallback_parent
    if ours is None:
        raise GitError(f"chain ref {ref} does not exist and no fallback parent was given")
    theirs = _run(path, "rev-parse", "--verify", f"{merge_rev}^{{commit}}").stdout.strip()
    if _run(path, "merge-base", "--is-ancestor", theirs, ours, check=False).returncode == 0:
        return ours
    if _run(path, "merge-base", "--is-ancestor", ours, theirs, check=False).returncode == 0:
        sha = theirs
    else:
        res = _run(path, "merge-tree", "--write-tree", ours, theirs, check=False)
        if res.returncode != 0:
            return None
        tree = res.stdout.strip().splitlines()[0]
        sha = _run(
            path,
            "commit-tree",
            tree,
            "-p",
            ours,
            "-p",
            theirs,
            "-m",
            _full_message(message, None, identity),
            env_extra=_identity_env(identity),
        ).stdout.strip()
    _run(path, "update-ref", ref, sha)
    sync_worktree(path, ours, sha)
    _advance_run_branch(path, also_branch, sha, expected=ours)
    return sha


def sync_worktree(path: pathlib.Path, from_rev: str, to_rev: str) -> None:
    """Move the worktree's files from one tree to another through a temp index.

    HEAD and the shared index stay untouched. The worktree must match the first tree,
    the chain invariant after a chain commit. The temp index is refreshed before the
    merge: `read-tree` records no stat data, and the merge touches only entries it can
    prove up to date.

    Args:
        path: The repository.
        from_rev: The tree the worktree holds.
        to_rev: The tree to move it to.
    """
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="agent6-chain-"))
    env = {"GIT_INDEX_FILE": str(tmp / "index")}
    try:
        _run(path, "read-tree", from_rev, env_extra=env)
        _run(path, "update-index", "--refresh", env_extra=env, check=False)
        _run(path, "read-tree", "-m", "-u", from_rev, to_rev, env_extra=env)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _commit(
    path: pathlib.Path,
    message: str,
    *,
    trailers: dict[str, str] | None,
    identity: CommitIdentity | None,
    only_paths: tuple[str, ...] | None = None,
) -> str:
    """Commit the index, or only the given paths from the worktree.

    Returns:
        The new HEAD sha.
    """
    env_extra = _identity_env(identity)
    full_message = _full_message(message, trailers, identity)
    argv = ["commit", "-m", full_message]
    if only_paths is not None:
        argv.extend(["--", *only_paths])
    _run(path, *argv, env_extra=env_extra)
    return _run(path, "rev-parse", "HEAD").stdout.strip()


@dataclasses.dataclass(frozen=True, slots=True)
class MergeResult:
    """The outcome of landing a run branch on a target.

    Attributes:
        merged_sha: The target's new tip; "" on a conflict, where nothing moved.
        conflicted: Whether the merge conflicted.
        conflicts: The conflicted paths.
        left_behind: Paths the checkout keeps its own version of, so it does not hold
            what was merged.
    """

    merged_sha: str
    conflicted: bool
    conflicts: tuple[str, ...]
    left_behind: tuple[str, ...] = ()


def fetch_branch(path: pathlib.Path, remote_path: pathlib.Path, refspec: str) -> None:
    """Fetch a refspec from another repository's path, without adding a remote."""
    _run(path, "fetch", str(remote_path), refspec)


def _merge_tree(
    path: pathlib.Path, ours: str, theirs: str, merge_base: str | None
) -> tuple[str, tuple[str, ...]]:
    """Merge two commits into a tree without touching the worktree.

    Args:
        path: The repository.
        ours: The first parent.
        theirs: The commit merged in.
        merge_base: The base to use instead of the one git would find.

    Returns:
        The merged tree's oid and no paths, or "" and the conflicted paths.

    Raises:
        GitError: git predates `--merge-base` (2.40), or the merge failed outright.
    """
    base = [f"--merge-base={merge_base}"] if merge_base else []
    res = _run(path, "merge-tree", "--write-tree", "--name-only", *base, ours, theirs, check=False)
    lines = res.stdout.splitlines()
    if res.returncode == 1:
        # The tree oid, the conflicted paths, a blank line, then git's messages.
        paths: list[str] = []
        for line in lines[1:]:
            if not line.strip():
                break
            paths.append(line)
        return "", tuple(paths)
    if res.returncode == 129:
        # git's usage exit: `--merge-base` arrived in 2.40, the floor docs/installation.md states.
        version = _run(path, "--version", check=False).stdout.strip() or "this git"
        raise GitError(
            f"merge-tree rejected its arguments ({version}); agent6 needs git 2.40 or newer"
        )
    if res.returncode != 0 or not lines:
        raise GitError(f"merge-tree failed: {res.stderr.strip() or 'exit'}")
    return lines[0].strip(), ()


def plumb_merge(
    path: pathlib.Path,
    target: str,
    merge_rev: str,
    *,
    strategy: str,
    message: str | None = None,
    identity: CommitIdentity | None = None,
    merge_base: str | None = None,
) -> MergeResult:
    """Land a rev on a branch with plumbing only: no checkout, no clean-tree requirement.

    The operator's worktree is never the medium, so a worktree carrying the run's own
    work does not block the landing. When the target is checked out, the index entries
    the merge changed and the operator did not are brought forward.

    Args:
        path: The repository.
        target: The branch to land on.
        merge_rev: The commit to land.
        strategy: "merge" (a two-parent commit), "squash" (a single-parent commit) or
            "ff" (the ref moves to the rev).
        message: The merge commit's message.
        identity: The author and committer overrides.
        merge_base: The base instead of the one git would find: a run squash-merged
            earlier is content the target holds under a commit unrelated to the chain.

    Returns:
        The result; a rev the target already contains returns the unchanged tip, and a
        conflict moves nothing and lists the paths.

    Raises:
        GitError: The "ff" strategy cannot fast-forward, or the target moved concurrently.
    """
    ref = f"refs/heads/{target}"
    ours = _run(path, "rev-parse", "--verify", f"{ref}^{{commit}}").stdout.strip()
    theirs = _run(path, "rev-parse", "--verify", f"{merge_rev}^{{commit}}").stdout.strip()
    if _run(path, "merge-base", "--is-ancestor", theirs, ours, check=False).returncode == 0:
        return MergeResult(ours, False, ())
    ff_able = _run(path, "merge-base", "--is-ancestor", ours, theirs, check=False).returncode == 0
    if strategy == "ff":
        if not ff_able:
            raise GitError(f"{target!r} has moved; a fast-forward to {merge_rev!r} is impossible")
        sha = theirs
    else:
        if ff_able:
            tree = _tree_sha(path, theirs)
        else:
            tree, conflicts = _merge_tree(path, ours, theirs, merge_base)
            if conflicts:
                return MergeResult("", True, conflicts)
        if tree == _tree_sha(path, ours):
            return MergeResult(ours, False, ())  # content-identical: nothing to land
        text = message or f"Merge {merge_rev}"
        trailer = identity.trailer if identity else None
        if trailer and trailer not in text:
            text = f"{text}\n\n{trailer}"
        parent_args = ["-p", ours] if strategy == "squash" else ["-p", ours, "-p", theirs]
        sha = _run(
            path,
            "commit-tree",
            tree,
            *parent_args,
            "-m",
            text,
            env_extra=_identity_env(identity),
        ).stdout.strip()
    # Compare-and-swap: refuses when the target moved concurrently.
    _run(path, "update-ref", ref, sha, ours)
    return MergeResult(sha, False, (), _bring_index_forward(path, target, ours, sha))


def _bring_index_forward(
    path: pathlib.Path, target: str, old_tip: str, new_tip: str
) -> tuple[str, ...]:
    """Bring a checked-out branch's index and worktree forward after its ref moved.

    Each path the move changed is updated only where it still matches the old tip, so
    the operator's own staged or edited content stays as it is. Without the index half
    `git status` shows a phantom staged reversal; without the worktree half a merge onto
    a reverted checkout leaves the files behind.

    Args:
        path: The repository.
        target: The branch; the checkout is whichever worktree has it checked out.
        old_tip: The tip before the move.
        new_tip: The tip after it.

    Returns:
        The paths the checkout keeps a third version of, which the move did not reach;
        empty when no live checkout has the branch.
    """
    checkout = _worktree_of_branch(path, target)
    if checkout is None or not checkout.is_dir():
        return ()
    path = checkout
    changed = _run(path, "diff-tree", "-r", "--no-renames", "-z", old_tip, new_tip).stdout
    staged = {
        entry.split("\t", 1)[1]: entry.split("\t", 1)[0].split()
        for entry in _run(path, "ls-files", "--stage", "-z").stdout.split("\x00")
        if "\t" in entry
    }  # path -> [mode, sha, stage]
    updates: list[str] = []
    left: list[str] = []
    records = changed.split("\x00")
    i = 0
    while i + 1 < len(records):
        meta, rel = records[i], records[i + 1]
        i += 2
        if not meta.startswith(":"):
            continue
        old_mode, new_mode, old_sha, new_sha, _status = meta[1:].split(" ")[:5]
        entry = staged.get(rel)
        if (entry is None and old_mode == "000000") or (
            entry is not None and entry[0] == old_mode and entry[1] == old_sha
        ):
            # mode 000000 removes the entry (a merge-side deletion).
            updates.append(f"{new_mode} {new_sha}\t{rel}")
        if not _bring_worktree_file_forward(path, rel, old_mode, old_sha, new_mode, new_sha):
            left.append(rel)
    if updates:
        _run(path, "update-index", "-z", "--index-info", stdin_text="\x00".join(updates) + "\x00")
    return tuple(left)


def _bring_worktree_file_forward(
    path: pathlib.Path, rel: str, old_mode: str, old_sha: str, new_mode: str, new_sha: str
) -> bool:
    """Move one worktree file from the old tip's content to the new tip's.

    Only a file still at the old tip moves (absent matches a deletion or a not-yet-added
    path); symlinks and submodule pointers are left to the operator.

    Args:
        path: The checkout.
        rel: The file, repo-relative.
        old_mode: The mode at the old tip; "000000" when absent.
        old_sha: The blob at the old tip.
        new_mode: The mode at the new tip; "000000" when deleted.
        new_sha: The blob at the new tip.

    Returns:
        False when the checkout keeps a third version, which the caller names.
    """
    file = path / rel
    if new_mode in ("120000", "160000") or old_mode in ("120000", "160000"):
        return True  # left to the operator, and never named as left behind
    try:
        current = (
            ""
            if not file.is_file() or file.is_symlink()
            else _run(path, "hash-object", "--", rel, check=False).stdout.strip()
        )
        matches_old = not file.exists() if old_mode == "000000" else current == old_sha
        if not matches_old:
            # Already at the new content is landed, not left behind.
            return current == new_sha or (new_mode == "000000" and not file.exists())
        if new_mode == "000000":
            file.unlink(missing_ok=True)
            return True
        file.parent.mkdir(parents=True, exist_ok=True)
        # Bytes straight from git: `_run` decodes lossily, which would corrupt a binary blob.
        with file.open("wb") as out:
            subprocess.run(
                [_git(), *git_hardening_flags(path), "cat-file", "blob", new_sha],
                cwd=path,
                stdout=out,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=_GIT_TIMEOUT_S,
            )
        if new_mode == "100755":
            file.chmod(file.stat().st_mode | 0o111)
    except (GitError, OSError, subprocess.SubprocessError):
        return False  # an unwritable path leaves truthful dirt, never a crash
    return True


def list_run_commits(
    path: pathlib.Path, base_sha: str, run_branch: str
) -> tuple[commit_message.CommitRow, ...]:
    """Return the commits on the run branch since the base, oldest first."""
    # A body can hold any byte but NUL, so records split on NUL and fields at most twice.
    fmt = "%H%x1f%s%x1f%B"
    res = _run(
        path, "log", "-z", "--reverse", f"--format={fmt}", f"{base_sha}..{run_branch}", check=False
    )
    if not res.ok:
        return ()
    rows: list[commit_message.CommitRow] = []
    for rec in res.stdout.split("\x00"):
        if not rec.strip():
            continue
        fields = rec.split("\x1f", 2)
        if len(fields) >= 3:
            rows.append(
                commit_message.CommitRow(
                    sha=fields[0].strip(), subject=fields[1], message=fields[2]
                )
            )
    return tuple(rows)


def worktree_name_status(
    path: pathlib.Path, *, exclude: Collection[str] = ()
) -> tuple[tuple[str, str], ...]:
    """Return the status letter and path of every pending change, untracked reported as `A`.

    The conventional-subject deriver's input at checkpoint time.

    Args:
        path: The repository.
        exclude: Untracked paths left out, the run's `untracked_at_start`.

    Returns:
        The (status, path) pairs.
    """
    res = _run(path, "status", "--porcelain", check=False)
    pairs: list[tuple[str, str]] = []
    for line in res.stdout.splitlines():
        if len(line) < 4:
            continue
        rel = line[3:].strip()
        if rel in exclude:
            continue
        code = line[:2].strip() or "M"
        pairs.append(("A" if code in ("??", "A", "AM") else code[:1], rel))
    return tuple(pairs)


def range_name_status(path: pathlib.Path, base: str, head: str) -> tuple[tuple[str, str], ...]:
    """Return the status letter and path of every change in a range, the deriver's squash input."""
    res = _run(path, "diff", "--name-status", f"{base}..{head}", check=False)
    pairs: list[tuple[str, str]] = []
    for line in res.stdout.splitlines():
        cols = line.split("\t")
        if len(cols) >= 2:
            pairs.append((cols[0][:1], cols[-1]))
    return tuple(pairs)


def recent_log(path: pathlib.Path, n: int = 20) -> str:
    """Return the last commits as one-line log text; "" when the log cannot be read."""
    res = _run(path, "log", f"-n{n}", "--oneline", check=False)
    return res.stdout if res.ok else ""


def tracked_files(path: pathlib.Path) -> tuple[str, ...]:
    """Return the tracked files in git's own order.

    Returns:
        The paths with POSIX separators; empty outside a repo, which callers read as no
        map available rather than an empty repo.
    """
    res = _run(path, "ls-files", "-z", check=False)
    if not res.ok:
        return ()
    return tuple(p for p in res.stdout.split("\x00") if p)


def diff_since(path: pathlib.Path, base_sha: str, *, exclude: Collection[str] = ()) -> str:
    """Return the worktree's diff against a base, untracked files included as additions.

    Untracked files are registered with an intent-to-add on a temp copy of the index:
    an intent entry left in the real index turns a later plumbing merge into a staged
    deletion that reads as dirt.

    Args:
        path: The repository.
        base_sha: The diff base.
        exclude: Untracked paths left out, the run's `untracked_at_start`; a review diff
            showing them as the run's additions would have a panel order their removal.

    Returns:
        The diff text; "" when the diff fails.
    """
    specs = [f":(top,exclude,literal){rel}" for rel in sorted(exclude)]
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="agent6-review-diff-"))
    try:
        index_copy = tmp / "index"
        real_index = path / ".git" / "index"
        if real_index.is_file():
            shutil.copyfile(real_index, index_copy)
        env = {"GIT_INDEX_FILE": str(index_copy)}
        _run(path, "add", "-N", "--", ".", *specs, check=False, env_extra=env)
        res = _run(path, "diff", base_sha, "--", ".", *specs, check=False, env_extra=env)
        return res.stdout if res.ok else ""
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def diff_range(path: pathlib.Path, base_sha: str, ref: str) -> str:
    """Return the diff a committed range introduces; "" when the range does not resolve."""
    res = _run(path, "diff", f"{base_sha}..{ref}", check=False)
    return res.stdout if res.ok else ""


def commit_diff(path: pathlib.Path, sha: str, *, max_bytes: int = 16384) -> str:
    """Return the patch one commit introduced, without its message.

    Args:
        path: The repository.
        sha: The commit.
        max_bytes: Where the patch is cut, so no caller holds an unbounded diff.

    Returns:
        The patch text; "" when git fails.
    """
    res = _run(path, "show", "--format=", "--no-color", sha, "--", ".", check=False)
    if not res.ok:
        return ""
    return res.stdout[:max_bytes]
