# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Git operations with hard safety invariants.

The destructive operations (push, force, history rewrite) are not exposed as a
code path here AT ALL -- there is nothing to refuse at runtime because nothing
spells them (pinned by test_git_ops_never_spells_a_destructive_verb). The one
sanctioned exception is force_delete_squash_merged_branch. The config can
*loosen* benign options (auto-stash, branch-per-run) and never these.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import os
import re
import shutil
import subprocess
import textwrap
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from agent6.types import CommandResult


class GitError(Exception):
    """Generic git failure."""


# Upper bound on any single git invocation. Local ops finish in well under a
# second; this only fires on a pathological hang (stuck filesystem, held lock).
_GIT_TIMEOUT_S = 120.0

# How long a timed-out git gets to exit on SIGTERM before SIGKILL. Its TERM
# handler only has to unlink its lockfiles, so this is generous.
_GIT_TERM_GRACE_S = 5.0


@dataclass(frozen=True, slots=True)
class GitStatus:
    branch: str
    head_sha: str
    is_clean: bool
    untracked_count: int
    modified_count: int


@dataclass(frozen=True, slots=True)
class CommitIdentity:
    """Resolved name/email + provenance trailer used for commits this run.

    `name` and `email` are populated from `[git.commit]` overrides when set,
    otherwise left as None to mean "let git's own config resolution decide".
    `verify_git_identity` ensures that when both are None the project's git
    config has a usable identity before any commit is attempted. `trailer` is
    a rendered git trailer line (see :func:`render_commit_trailer`), appended
    once per commit.
    """

    name: str | None = None
    email: str | None = None
    trailer: str | None = None

    @property
    def has_override(self) -> bool:
        return bool(self.name or self.email or self.trailer)


def render_commit_trailer(fmt: str, *, model: str, role: str) -> str | None:
    """The `[git.commit].trailer` format string as a concrete trailer line,
    or None when unset. The config validator pins the placeholder set
    ({model}, {role}) and the "Key: value" shape."""
    if not fmt:
        return None
    return fmt.format(model=model, role=role)


def verify_git_identity(path: Path, identity: CommitIdentity) -> tuple[str, str]:
    """Resolve the effective author identity, or raise GitError.

    Returns `(name, email)` that future commits will use. Order of
    precedence per field:

      1. The `[git.commit]` override (`identity.name` / `identity.email`).
      2. `git config user.name` / `git config user.email` in this repo.

    If after both steps either field is empty, we refuse to start. This is
    deliberately strict: silently committing as a missing/auto-generated
    identity is the kind of thing a user only notices weeks later when they
    `git log --author`.
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


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _git() -> str:
    git = shutil.which("git")
    if git is None:
        raise GitError("git executable not found on PATH")
    return git


# Always-on hardening: neutralize repo-config keys that would otherwise run a
# repo-controlled command on the HOST (outside the jail) during agent6's own git
# operations. `-c` has the highest precedence, overriding `.git/config`.
# `core.fsmonitor` fires a command on every index refresh (status/add/commit);
# `diff.external` fires one on `git diff` (review/diff); `commit.gpgsign` fires
# the configured `gpg.program` (arbitrary host command) on every commit. All are
# pure overrides with no correctness cost here: fsmonitor is a perf cache, an
# empty diff.external uses git's builtin diff, and agent6 has no signing feature
# (its per-step auto-commits are unsigned by design; the operator signs at the
# end). The edit tools already refuse writes into `.git` under protect_git, but a
# repo cloned with a pre-poisoned `.git/config` would otherwise execute its
# payload the first time agent6 ran git here.
# NOTE: content-semantic drivers a commit/merge legitimately runs -- clean/smudge
# `filter.*` and `merge.*.driver` -- are deliberately NOT disabled: blanket-
# disabling them would silently corrupt commits in repos that use them (Git LFS
# is the common case). They are per-name with no blanket `-c` off switch.
_GIT_EGRESS_HARDENING: tuple[str, ...] = (
    "-c",
    "core.fsmonitor=false",
    "-c",
    "diff.external=",
    "-c",
    "commit.gpgsign=false",
)

# Whether the repo's own `.git/hooks/*` run during agent6's git ops (notably the
# per-step auto-commit). Default false -- a repo hook is repo-controlled HOST
# code, so honoring it on agent6's commit is a host-RCE vector for an adversarial
# repo. Set once from `git.run_repo_hooks` at run/review startup. A module-level
# dict (mutated, not rebound) keeps the process-wide policy without a `global`
# statement -- same shape as providers.egress._BROKER_ROUTES.
_hook_policy: dict[str, bool] = {"honor_repo_hooks": False}


def set_repo_hook_policy(honor: bool) -> None:
    """Configure whether agent6's own git ops fire the repo's `.git/hooks/*`."""
    _hook_policy["honor_repo_hooks"] = honor


# Flags that force git's builtin diff/show renderer so a poisoned `.git/config`
# cannot run a host command: `--no-ext-diff` disables the `diff.external` driver,
# `--no-textconv` the per-file `diff.<d>.textconv` driver (neither is covered by
# the `-c` overrides above). Single source of truth so no diff/show call site
# drifts. Place AFTER the subcommand, alongside `git_hardening_flags()` before it.
DIFF_SHOW_SAFETY_FLAGS: tuple[str, ...] = ("--no-ext-diff", "--no-textconv")


def git_hardening_flags() -> tuple[str, ...]:
    """The `-c` overrides every agent6 git invocation must carry (see
    _GIT_EGRESS_HARDENING). Public so the few callers that shell out to git
    outside this module (`agent6 review`/`sessions diff` collectors) apply the
    same hardening; place them BEFORE the subcommand. Diff/show callers also add
    DIFF_SHOW_SAFETY_FLAGS after the subcommand."""
    if _hook_policy["honor_repo_hooks"]:
        return _GIT_EGRESS_HARDENING
    # /dev/null is not a directory, so git finds (and runs) no hooks there.
    return (*_GIT_EGRESS_HARDENING, "-c", "core.hooksPath=/dev/null")


def _run(
    cwd: Path,
    *args: str,
    check: bool = True,
    env_extra: dict[str, str] | None = None,
) -> CommandResult:
    # GIT_TERMINAL_PROMPT=0: a git op that would otherwise block on a
    # username/password prompt (a network remote without cached creds) fails
    # fast instead of hanging with no output. Local ops are unaffected.
    # LC_ALL=C: git translates its human-readable output, and the bystander
    # rescue reads one of those sentences to learn which stash it dropped.
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C", **(env_extra or {})}
    hardening = git_hardening_flags()
    # A poisoned `.git/config` reaches a host command two ways on a diff/show:
    # `diff.external` and per-file `diff.<d>.textconv`. The `-c` overrides above
    # cover neither cleanly (and git 2.53 dies rc=128 on the empty `diff.external`
    # override), so force the builtin renderer with DIFF_SHOW_SAFETY_FLAGS.
    argv = list(args)
    if argv and argv[0] in ("diff", "show"):
        argv[1:1] = DIFF_SHOW_SAFETY_FLAGS
    full_argv = (_git(), *hardening, *argv)
    index_lock = cwd / ".git" / "index.lock"
    lock_preexisted = index_lock.exists()
    proc = subprocess.Popen(
        full_argv,
        cwd=cwd,
        # Bytes, decoded lossily below: git diff/show emit raw file bytes,
        # so a changed non-UTF-8 text file (latin-1 has no NULs, so git does
        # not classify it binary) would make a strict text=True decode raise
        # UnicodeDecodeError mid-run. Filenames are core.quotePath-escaped to
        # ASCII, and shas/porcelain status are ASCII, so replacement only
        # ever touches diff/show content.
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    try:
        # A local git op is fast; a bound turns a pathological hang (a wedged
        # NFS/filesystem, an index.lock held by a crashed process) into a
        # loud error instead of a silent freeze with no feedback.
        out, err = proc.communicate(timeout=_GIT_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        # SIGTERM first: git's TERM handler unlinks its own lockfiles
        # (index.lock included), so a terminated child cleans up after itself
        # and a lock still on disk afterwards is a concurrent git's.
        proc.terminate()
        try:
            proc.communicate(timeout=_GIT_TERM_GRACE_S)
        except subprocess.TimeoutExpired:
            # TERM ignored (wedged uninterruptible): SIGKILL skips git's
            # cleanup, so clear the lock -- ONLY when it appeared under this
            # child. One that predates the spawn belongs to a concurrent git
            # process (operator shell, another lane), and deleting it would
            # break git's index mutual exclusion.
            proc.kill()
            proc.communicate()
            if not lock_preexisted:
                with contextlib.suppress(OSError):
                    index_lock.unlink(missing_ok=True)
        raise GitError(
            f"git {' '.join(args)} timed out after {_GIT_TIMEOUT_S:.0f}s"
            " (a stuck filesystem or a held .git/index.lock?)"
        ) from exc
    result = CommandResult(
        argv=full_argv,
        returncode=proc.returncode,
        stdout=out.decode(errors="replace"),
        stderr=err.decode(errors="replace"),
        duration_s=0.0,
    )
    if check and not result.ok:
        # Surface stdout too. `git commit` writes its informational
        # output (including "nothing to commit, working tree clean", pre-
        # commit hook output, and most user-facing messages) to STDOUT,
        # not stderr. Capturing only stderr produced empty error strings
        # like "git commit -m <subject> failed: " that gave the operator
        # zero signal when triaging a failed auto-commit in the wild.
        stderr_msg = result.stderr.strip()
        stdout_msg = result.stdout.strip()
        if stderr_msg and stdout_msg:
            detail = f"{stderr_msg} | stdout: {stdout_msg}"
        else:
            detail = stderr_msg or stdout_msg or f"exit {result.returncode}"
        raise GitError(f"git {' '.join(args)} failed: {detail}")
    return result


def slugify(text: str, max_len: int = 40) -> str:
    """Lowercase ASCII slug for use in branch names."""
    slug = _SLUG_RE.sub("-", text.lower()).strip("-")
    return (slug or "run")[:max_len]


def is_git_repo(path: Path) -> bool:
    res = _run(path, "rev-parse", "--is-inside-work-tree", check=False)
    return res.ok and res.stdout.strip() == "true"


# One check-ignore per grep, however many files it walks; measured at 0.05s for
# 1600 paths. A bound so a wedged filesystem cannot hold a read-only tool.
_CHECK_IGNORE_TIMEOUT_S = 10.0


def ignored_paths(repo_root: Path, rel_paths: Sequence[str]) -> frozenset[str]:
    """Which of *rel_paths* the repo's own ignore rules exclude; empty outside a
    repo or without git.

    Git's answer, not a hand-rolled matcher and not a fixed `target`/
    `node_modules` blocklist: `.gitignore` files nest, and the repo already
    says what is not source. TRACKED files are never reported (git's default),
    so a checked-in file matching an ignore pattern stays visible.

    The paths ride on stdin, NUL-delimited -- never on argv, and a filename may
    itself contain a newline. Empty on any failure: the caller then behaves as
    it did before there was a filter.
    """
    if not rel_paths:
        return frozenset()
    payload = b"".join(p.encode() + b"\0" for p in rel_paths)
    try:
        proc = subprocess.run(
            (_git(), *git_hardening_flags(), "check-ignore", "-z", "--stdin"),
            cwd=repo_root,
            input=payload,
            capture_output=True,
            check=False,
            timeout=_CHECK_IGNORE_TIMEOUT_S,
        )
    except (GitError, OSError, subprocess.TimeoutExpired):
        return frozenset()
    # rc 1 is "none of them are ignored", not a failure; 128 is "not a repo".
    if proc.returncode not in (0, 1):
        return frozenset()
    return frozenset(p.decode(errors="replace") for p in proc.stdout.split(b"\0") if p)


def paths_dirty(path: Path, rel_paths: tuple[str, ...]) -> bool:
    """True iff any of ``rel_paths`` has uncommitted changes (untracked,
    modified, or staged) versus HEAD, i.e. a path-limited commit of just those
    paths would record something. Unlike whole-tree ``status().is_clean``, this
    ignores unrelated dirt elsewhere in the worktree."""
    if not rel_paths:
        return False
    res = _run(path, "status", "--porcelain", "--", *rel_paths, check=False)
    return bool(res.stdout.strip())


def dirty_paths(path: Path, *, limit: int = 10) -> list[str]:
    """Up to *limit* working-tree paths with uncommitted changes, each tagged
    ``M`` (modified/staged) or ``?`` (untracked), for a dirty-tree refusal that
    names what to deal with instead of just saying 'not clean'."""
    res = _run(path, "status", "--porcelain=v1", "--untracked-files=all", check=False)
    out: list[str] = []
    for line in res.stdout.splitlines():
        if not line.strip():
            continue
        tag = "?" if line.startswith("??") else "M"
        out.append(f"{tag} {line[3:]}")
        if len(out) >= limit:
            break
    return out


def status(path: Path) -> GitStatus:
    if not is_git_repo(path):
        raise GitError(f"Not a git repository: {path}")
    branch_res = _run(path, "rev-parse", "--abbrev-ref", "HEAD", check=False)
    if branch_res.ok:
        branch = branch_res.stdout.strip()
    else:
        # Unborn HEAD (freshly `git init`, no commits yet): `rev-parse HEAD`
        # fails, but `branch --show-current` still reports the checked-out branch
        # name. Without this, every agent6 entry point that loads the repo
        # summary crashes in a brand-new repo.
        branch = _run(path, "branch", "--show-current", check=False).stdout.strip()
    head_res = _run(path, "rev-parse", "HEAD", check=False)
    head_sha = head_res.stdout.strip() if head_res.ok else ""
    porcelain = _run(path, "status", "--porcelain=v1", "--untracked-files=all").stdout
    untracked = 0
    modified = 0
    for line in porcelain.splitlines():
        if line.startswith("??"):
            untracked += 1
        elif line.strip():
            modified += 1
    return GitStatus(
        branch=branch,
        head_sha=head_sha,
        is_clean=(untracked == 0 and modified == 0),
        untracked_count=untracked,
        modified_count=modified,
    )


def stash_all(path: Path, message: str) -> None:
    _run(path, "stash", "push", "--include-untracked", "--message", message)


def auto_stash_message(session_id: str) -> str:
    """The auto-stash identity: the run pushes with this message and the
    finalizer finds the stash BY it -- never by position, since stash@{0} may
    be a stash someone else pushed while the run was running."""
    return f"agent6 auto-stash before run {session_id}"


@dataclass(frozen=True, slots=True)
class StashEntry:
    """One ``git stash list`` entry: its position at lookup time (``ref``,
    e.g. ``stash@{1}``, for operator-facing hints only) and its immutable
    commit (``sha``). Anything that mutates restores by sha: the position
    shifts the moment anyone pushes or drops a stash."""

    ref: str
    sha: str


def find_stash(path: Path, message: str) -> StashEntry | None:
    """The newest stash pushed with exactly *message*, or None.
    ``git stash push -m MSG`` records ``On <branch>: MSG`` and ':' cannot
    appear in a ref name, so anchoring ``": MSG"`` at the end matches the
    whole message -- lane run ids are ordinal (``…-l1``, ``…-l10``), so one
    message can be a prefix of another."""
    res = _run(path, "stash", "list", "--format=%gd%x09%H%x09%gs", check=False)
    for line in res.stdout.splitlines():
        ref, _, rest = line.partition("\t")
        sha, _, subject = rest.partition("\t")
        if subject.endswith(f": {message}"):
            return StashEntry(ref=ref, sha=sha)
    return None


# `git stash drop` names the commit it removed: "Dropped stash@{0} (<sha>)".
_DROPPED_SHA_RE = re.compile(r"^Dropped .*\(([0-9a-f]{7,64})\)", re.MULTILINE)


def restore_stash(path: Path, stash: StashEntry) -> bool:
    """Apply *stash* back onto the working tree BY SHA -- a stash@{N} recorded
    earlier applies whatever sits at that position NOW, which is the wrong
    stash the moment another one was pushed. On a clean apply, drop the entry.
    On conflict (or any non-zero apply), leave everything in place so the
    user's work is never lost, and return False. We never `reset --hard` to
    undo a conflicted apply (refused), so a conflict leaves the markers for
    the user to resolve with their stash still intact. Raises GitError when a
    raced drop took a concurrent stash and putting it back failed; the apply
    itself has landed by then, and the message carries the recovery command."""
    if not _run(path, "stash", "apply", stash.sha, check=False).ok:
        return False
    _drop_by_sha(path, stash.sha)
    return True


def _drop_by_sha(path: Path, sha: str) -> None:
    """Drop the stash entry whose commit is *sha*, putting back a bystander we
    take by mistake.

    ``git stash drop`` addresses an entry by POSITION and refuses a sha outright
    ("is not a stash reference"), so the position has to be re-resolved from the
    list -- and a stash pushed in between shifts every position, aiming the drop
    at someone else's entry. git names the commit it dropped, so check it: one
    that is not ours is stored straight back under its own subject (position is
    not identity, so it returns at the top of the stack). Ours then stays
    listed; re-resolving to drop it again would race the same way, and leaking
    our own stash beats taking a second bystander.
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
        return  # no drop happened, or git stopped naming what it dropped
    taken = dropped.group(1)
    # Prefix either way: git prints the full oid today, but an abbreviated one
    # must not read as a stranger's stash and get stored back on top of us.
    if sha.startswith(taken) or taken.startswith(sha):
        return
    # A stash commit's own subject is the "On <branch>: <message>" the reflog
    # showed, so the restored entry reads exactly as it did before.
    subject = _run(path, "log", "-1", "--format=%s", taken, check=False).stdout.strip()
    subject = subject or "restored by agent6"
    store = _run(path, "stash", "store", "-m", subject, taken, check=False)
    if not store.ok:
        # The bystander's entry is already gone from the list; its commit
        # still exists under `taken`, so the message carries the recovery.
        detail = store.stderr.strip() or store.stdout.strip() or f"exit {store.returncode}"
        raise GitError(
            f"a stash pushed concurrently ({subject!r}) was taken by a raced drop and "
            f"putting it back failed ({detail}); restore it with:\n"
            f"    git stash store -m {subject!r} {taken}"
        )


def branch_exists(path: Path, name: str) -> bool:
    """True if a local branch *name* exists."""
    return _run(path, "rev-parse", "--verify", "--quiet", f"refs/heads/{name}", check=False).ok


def list_run_branches(path: Path) -> tuple[str, ...]:
    """Local branches under the `agent6/` namespace (run branches), sorted."""
    res = _run(path, "for-each-ref", "--format=%(refname:short)", "refs/heads/agent6/", check=False)
    return tuple(b for b in res.stdout.splitlines() if b.strip())


def is_ancestor(path: Path, maybe_ancestor: str, ref: str) -> bool:
    """True if *maybe_ancestor* is reachable from *ref* (`git merge-base
    --is-ancestor`). Used to tell a reachable-merged run branch (an ancestor of its
    base) from a squash-merged one (content in the base, but not reachable)."""
    return _run(path, "merge-base", "--is-ancestor", maybe_ancestor, ref, check=False).ok


def delete_branch_if_merged(path: Path, branch: str) -> bool:
    """Delete *branch* with `git branch -d`, the SAFE delete: git refuses unless the
    branch is reachable-merged into the current HEAD (or its upstream). Returns True
    if deleted, False if git refused -- a squash-merged or genuinely unmerged branch,
    since neither is reachable. Never `branch -D` here; see
    ``force_delete_squash_merged_branch`` for the one operator-gated exception."""
    return _run(path, "branch", "-d", branch, check=False).ok


def branch_tip_sha(path: Path, branch: str) -> str | None:
    """The commit sha a branch points at, or None if it does not resolve."""
    res = _run(path, "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}", check=False)
    sha = res.stdout.strip()
    return sha or None


def force_delete_squash_merged_branch(path: Path, branch: str) -> bool:
    """`git branch -D` a run branch, the ONE sanctioned force-delete in agent6.

    `git branch -d` refuses a squash-merged branch because its commits are not
    reachable from the base (the squash collapsed them into one commit ON the
    base), even though the branch's content is safely in that base commit. So a
    default `sessions prune` can never remove a squash-merged branch, and the whole
    run -> merge -> prune lifecycle leaves the branch behind.

    This is the operator's explicit `sessions prune --delete-squashed` opt-in, only
    for a branch the manifest CONFIRMS was squash-merged into an existing base
    (its content is in that base commit; the individual per-step commits were
    collapsed anyway and survive in the reflog until GC). Operator-initiated,
    fixed argv, content-preserving. Returns True if deleted."""
    return _run(path, "branch", "-D", branch, check=False).ok


def create_branch(path: Path, name: str, *, start_point: str | None = None) -> None:
    """Create *name* and check it out, or just check it out if it already exists.

    *start_point* (a branch/sha) is where a NEW branch is cut from; None means
    the current HEAD. Idempotent: an existing branch is only checked out, never
    moved (that would be a force/rewrite, which is refused), so re-running or
    resuming a run reuses the run's branch. ``git.branch_from`` uses start_point
    to cut a run from its base instead of stacking on the current checkout."""
    existing = _run(path, "branch", "--list", name, check=False)
    if existing.ok and existing.stdout.strip():
        _run(path, "checkout", name)
    elif start_point:
        _run(path, "checkout", "-b", name, start_point)
    else:
        _run(path, "checkout", "-b", name)


def create_branch_at(path: Path, name: str, sha: str) -> None:
    """Create branch *name* pointing at *sha* WITHOUT checking it out.

    Additive only (``git branch <name> <sha>``): it never touches HEAD or the
    working tree, so `agent6 fork` can cut the new run's branch at a historical
    sha while the operator's checkout stays put. No-op if *name* already points
    at *sha*; raises ``GitError`` if it exists pointing elsewhere (we never move
    a branch -- that would be a force/rewrite, which is refused)."""
    existing = _run(path, "rev-parse", "--verify", "--quiet", f"refs/heads/{name}", check=False)
    if existing.ok and existing.stdout.strip():
        if existing.stdout.strip() == sha:
            return
        raise GitError(
            f"branch {name!r} already exists at {existing.stdout.strip()[:12]}, not {sha[:12]}; "
            "refusing to move it"
        )
    _run(path, "branch", name, sha)


def init_repo(path: Path) -> None:
    """`git init` a new repository at *path*. Creating a repo is not a push /
    force / history-rewrite, so it is outside the refusal set."""
    _run(path, "init")


def clone_repo(origin: Path, dest: Path) -> None:
    """`git clone` *origin* into *dest*. Both are plain filesystem paths, so
    git's local-clone optimization applies automatically: hardlinks on the same
    filesystem, falling back to a copy across devices -- no flag needed.

    cwd is "/" (always exists), not *origin*: both argv paths are absolutized
    here so the anchor is irrelevant, and a missing *origin* must fail as a
    GitError from git itself, not a raw FileNotFoundError from subprocess's
    chdir before git ever runs."""
    _run(Path("/"), "clone", str(origin.absolute()), str(dest.absolute()))


def unignored(path: Path, candidates: tuple[str, ...]) -> tuple[str, ...]:
    """Return the subset of repo-relative *candidates* that git does NOT ignore.

    Used so the `init` git-setup offer commits only the trackable scaffold
    (AGENTS.md, .gitignore) and not files the just-written .gitignore covers
    (e.g. the per-repo config under the ignored agent6 dir)."""
    if not candidates:
        return ()
    # check-ignore prints the ignored inputs (one per line) and exits 1 when
    # none match, both are fine, we only read stdout. The "--" stops a path
    # that begins with "-" from being parsed as a git flag.
    res = _run(path, "check-ignore", "--", *candidates, check=False)
    ignored = {line.strip() for line in res.stdout.splitlines() if line.strip()}
    return tuple(c for c in candidates if c not in ignored)


def commit_all(
    path: Path,
    message: str,
    *,
    trailers: dict[str, str] | None = None,
    identity: CommitIdentity | None = None,
) -> str:
    """Stage everything and commit. Returns the new HEAD sha.

    `identity` lets the caller override the author/committer name+email and
    append a `Co-authored-by:` trailer. When `identity` is None the commit
    uses the project's existing git config identity, callers should have
    already validated that via `verify_git_identity` at startup.
    """
    _run(path, "add", "-A")
    return _commit(path, message, trailers=trailers, identity=identity)


def commit_paths(
    path: Path,
    message: str,
    paths: tuple[str, ...],
    *,
    trailers: dict[str, str] | None = None,
    identity: CommitIdentity | None = None,
) -> str:
    """Stage only `paths` (repo-relative) and commit JUST those paths. Returns
    the new HEAD sha.

    The commit is path-limited (``git commit -- <paths>``), so unrelated
    changes the user already STAGED stay staged and uncommitted, and unrelated
    WIP in the worktree is never swept in. Used by `agent6 init`'s scaffold
    commit and the startup `.gitignore` auto-update, which must not fold the
    user's in-progress work into their commit.
    """
    if not paths:
        raise GitError("commit_paths requires at least one path")
    _run(path, "add", "--", *paths)
    return _commit(path, message, trailers=trailers, identity=identity, only_paths=paths)


def _identity_env(identity: CommitIdentity | None) -> dict[str, str] | None:
    """Author + committer env for a commit-creating git invocation, or None to fall
    back to the repo's configured identity."""
    if identity is None:
        return None
    env: dict[str, str] = {}
    if identity.name:
        env["GIT_AUTHOR_NAME"] = env["GIT_COMMITTER_NAME"] = identity.name
    if identity.email:
        env["GIT_AUTHOR_EMAIL"] = env["GIT_COMMITTER_EMAIL"] = identity.email
    return env or None


def _commit(
    path: Path,
    message: str,
    *,
    trailers: dict[str, str] | None,
    identity: CommitIdentity | None,
    only_paths: tuple[str, ...] | None = None,
) -> str:
    merged_trailers = dict(trailers or {})
    env_extra = _identity_env(identity)
    if identity is not None and identity.trailer and identity.trailer not in message:
        key, _, value = identity.trailer.partition(": ")
        merged_trailers[key] = value
    full_message = message
    if merged_trailers:
        trailer_lines = "\n".join(f"{k}: {v}" for k, v in merged_trailers.items())
        full_message = f"{message}\n\n{trailer_lines}"
    argv = ["commit", "-m", full_message]
    if only_paths is not None:
        # Path-limited commit: record only these paths (from the worktree),
        # disregarding anything else already staged. Callers that want the
        # whole index (commit_all, record_commit) pass no only_paths.
        argv.extend(["--", *only_paths])
    _run(path, *argv, env_extra=env_extra)
    return _run(path, "rev-parse", "HEAD").stdout.strip()


@dataclass(frozen=True, slots=True)
class MergeResult:
    """Outcome of merging a run branch into a target. On conflict the merge is
    undone so the working tree is left clean (merged_sha empty, conflicts listed)."""

    merged_sha: str
    conflicted: bool
    conflicts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CommitRow:
    """One commit on a run branch (oldest-first), for listing + squash-condensing."""

    sha: str
    subject: str
    message: str  # full %B


def _conflicted_paths(path: Path) -> tuple[str, ...]:
    res = _run(path, "diff", "--name-only", "--diff-filter=U", check=False)
    return tuple(p for p in res.stdout.splitlines() if p.strip())


def _untracked_files(path: Path) -> frozenset[str]:
    res = _run(path, "ls-files", "--others", "--exclude-standard", "-z", check=False)
    return frozenset(p for p in res.stdout.split("\x00") if p)


def fetch_branch(path: Path, remote_path: Path, refspec: str) -> None:
    """`git fetch <remote_path> <refspec>` into *path*, e.g. `"b:b"` to land a
    same-named local branch from another repo without adding a remote."""
    _run(path, "fetch", str(remote_path), refspec)


def merge_branch(
    path: Path,
    branch: str,
    *,
    ff_only: bool = False,
    message: str | None = None,
    identity: CommitIdentity | None = None,
) -> MergeResult:
    """Merge *branch* into the current HEAD. A real merge commit (`--no-ff`, with
    *message* and *identity*, including identity.trailer) keeps the run's
    per-step history; *ff_only* instead fast-forwards and raises if the target
    has moved (no commit is created, so message and identity do not apply). On
    conflict, `git merge --abort` (not a history rewrite) leaves the tree clean
    and the result reports the conflicts."""
    if ff_only:
        args: tuple[str, ...] = ("merge", "--ff-only", branch)
        env_extra = None
    else:
        text = message
        trailer = identity.trailer if identity else None
        if trailer:
            base = text or f"Merge {branch}"
            text = base if trailer in base else f"{base}\n\n{trailer}"
        msg_args = ("-m", text) if text else ("--no-edit",)
        args = ("merge", "--no-ff", *msg_args, branch)
        env_extra = _identity_env(identity)
    res = _run(path, *args, check=False, env_extra=env_extra)
    if res.ok:
        return MergeResult(_run(path, "rev-parse", "HEAD").stdout.strip(), False, ())
    conflicts = _conflicted_paths(path)
    _run(path, "merge", "--abort", check=False)
    if not conflicts:
        raise GitError(f"merge failed: {res.stderr.strip() or res.stdout.strip() or 'exit'}")
    return MergeResult("", True, conflicts)


def squash_merge(
    path: Path,
    branch: str,
    message: str | None,
    *,
    identity: CommitIdentity | None,
) -> MergeResult:
    """Squash *branch* into HEAD as ONE commit. `git merge --squash` stages the
    branch's cumulative tree without committing or moving HEAD (not a rebase or
    reset, so policy-clean); we then commit once with *message* plus
    identity.trailer (once, however many per-step commits carried it). A squash
    with nothing to merge is a clean no-op (returns HEAD). On conflict, restore
    the pre-merge tree (reset --mixed + checkout, plus removing only the files
    this squash newly staged, which otherwise survive as untracked; never
    reset --hard, as a squash leaves no MERGE_HEAD to --abort) and report the
    conflicted paths."""
    head = _run(path, "rev-parse", "HEAD").stdout.strip()
    pre_untracked = _untracked_files(path)
    res = _run(path, "merge", "--squash", branch, check=False)
    if not res.ok:
        conflicts = _conflicted_paths(path)
        rollback_to_known_good(path, head)
        # A conflicted squash also stages new files from the branch; reset --mixed
        # demotes them to untracked and checkout cannot remove them. Clean only the
        # files this merge introduced, so the user's pre-existing untracked files
        # are untouched (and still never reset --hard).
        stray = tuple(sorted(_untracked_files(path) - pre_untracked))
        if stray:
            _run(path, "clean", "-fdq", "--", *stray, check=False)
        if not conflicts:
            raise GitError(
                f"squash merge failed: {res.stderr.strip() or res.stdout.strip() or 'exit'}"
            )
        return MergeResult("", True, conflicts)
    if _run(path, "diff", "--cached", "--quiet", check=False).ok:
        # Nothing staged: the branch was already up to date / an ancestor. A clean
        # no-op, matching merge_branch's "Already up to date" behavior.
        return MergeResult(head, False, ())
    if message is None:
        # The `combine` style: git's own squash message, written to SQUASH_MSG
        # by `merge --squash`.
        msg_path = Path(_run(path, "rev-parse", "--git-path", "SQUASH_MSG").stdout.strip())
        if not msg_path.is_absolute():
            msg_path = path / msg_path
        try:
            message = msg_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            message = f"Squash {branch}"
    try:
        sha = _commit(path, message, trailers=None, identity=identity)
    except GitError:
        # The squash already staged a tree; if the commit step itself fails (a
        # rejecting hook, say), restore the clean pre-merge tree rather than leave
        # the index staged. Mirrors the conflict cleanup above.
        rollback_to_known_good(path, head)
        stray = tuple(sorted(_untracked_files(path) - pre_untracked))
        if stray:
            _run(path, "clean", "-fdq", "--", *stray, check=False)
        raise
    return MergeResult(sha, False, ())


def list_run_commits(path: Path, base_sha: str, run_branch: str) -> tuple[CommitRow, ...]:
    """Commits on *run_branch* since *base_sha*, oldest first."""
    # NUL-separate commits (-z): a commit body can contain any byte except NUL, so
    # \x1f/\x1e separators in a body would corrupt records/fields. Within a record,
    # split the \x1f fields at most twice so the body (last field) keeps any \x1f.
    fmt = "%H%x1f%s%x1f%B"
    res = _run(
        path, "log", "-z", "--reverse", f"--format={fmt}", f"{base_sha}..{run_branch}", check=False
    )
    if not res.ok:
        return ()
    rows: list[CommitRow] = []
    for rec in res.stdout.split("\x00"):
        if not rec.strip():
            continue
        fields = rec.split("\x1f", 2)
        if len(fields) >= 3:
            rows.append(CommitRow(sha=fields[0].strip(), subject=fields[1], message=fields[2]))
    return tuple(rows)


_ITER_SUBJECT_RE = re.compile(r"^agent6 iter \d+:\s*", re.IGNORECASE)


def condense_commit_message(rows: tuple[CommitRow, ...], *, subject: str) -> str:
    """Fold per-step commits into one readable message, so a squash reads as a
    single authored commit, not a squashed series.

    *subject* is the run's task (the headline). The body lists the distinct,
    de-noised per-step subjects (the `agent6 iter N:` prefix and checkpoint
    noise stripped). The provenance trailer is the commit emitter's job
    (identity.trailer), not this message's."""
    bullets: list[str] = []
    seen: set[str] = set()
    for row in rows:
        s = _ITER_SUBJECT_RE.sub("", row.subject).strip()
        if not s or s.lower().startswith("checkpoint") or s.lower() in seen:
            continue
        seen.add(s.lower())
        bullets.append(s)
    task = _ITER_SUBJECT_RE.sub("", subject).strip()
    headline = _headline_subject(task) or (bullets[0] if bullets else "agent6 run")
    parts = [headline]
    # If the subject truncated the task, wrap the full task into the body so
    # nothing is lost (git never wraps the subject line itself).
    full = " ".join(task.split())
    if full and full != headline:
        parts.append("")
        parts.extend(textwrap.wrap(full, width=72))
    if bullets:
        parts.append("")
        parts.extend(f"- {b}" for b in bullets)
    return "\n".join(parts)


_SUBJECT_LIMIT = 72  # git's soft subject cap; conventional tooling truncates past it


def _is_testish(p: str) -> bool:
    parts = PurePosixPath(p).parts
    name = parts[-1] if parts else ""
    return parts[:1] == ("tests",) or name.startswith("test_") or name == "conftest.py"


def _is_docish(p: str) -> bool:
    pp = PurePosixPath(p)
    return pp.suffix.lower() in (".md", ".rst") or pp.parts[:1] == ("docs",)


def _conventional_scope(paths: Sequence[str]) -> str:
    """The one common area the change touches, or "" when there is none: the
    package dir under ``src/<pkg>/`` (the module stem for a file directly under
    the package), else a second-level dir every path shares."""
    parts = [PurePosixPath(p).parts for p in paths if p]
    if not parts:
        return ""
    src_pkgs = [pp for pp in parts if len(pp) >= 3 and pp[0] == "src"]
    if src_pkgs:
        names = {pp[2] if len(pp) > 3 else str(PurePosixPath(pp[2]).stem) for pp in src_pkgs}
        return names.pop() if len(names) == 1 else ""
    tops = {pp[0] for pp in parts}
    if len(tops) != 1:
        return ""
    seconds = {pp[1] for pp in parts if len(pp) >= 3}
    return seconds.pop() if len(seconds) == 1 else ""


def conventional_commit_subject(changes: Sequence[tuple[str, str]], *, summary: str) -> str:
    """A Conventional Commits subject from ``(status, path)`` pairs, without a
    model call: all-tests -> ``test``, all-docs -> ``docs``, any added file ->
    ``feat``, else ``fix`` (``chore`` when nothing changed). Scope is the one
    common area (:func:`_conventional_scope`); the subject is *summary* with
    its head lowercased and any trailing period stripped, capped at 72."""
    paths = [p for _, p in changes]
    if not paths:
        ctype = "chore"
    elif all(_is_testish(p) for p in paths):
        ctype = "test"
    elif all(_is_docish(p) for p in paths):
        ctype = "docs"
    elif any(status.startswith("A") for status, _ in changes):
        ctype = "feat"
    else:
        ctype = "fix"
    scope = _conventional_scope(paths)
    head = f"{ctype}({scope}): " if scope else f"{ctype}: "
    subject = " ".join(summary.split()).rstrip(".")
    subject = (subject[:1].lower() + subject[1:]) if subject else "update"
    return (head + subject)[:_SUBJECT_LIMIT]


def worktree_name_status(path: Path) -> tuple[tuple[str, str], ...]:
    """``(status, path)`` pairs for every pending change (``status
    --porcelain``), untracked reported as ``A``: the conventional-subject
    deriver's input at checkpoint time."""
    res = _run(path, "status", "--porcelain", check=False)
    pairs: list[tuple[str, str]] = []
    for line in res.stdout.splitlines():
        if len(line) < 4:
            continue
        code = line[:2].strip() or "M"
        pairs.append(("A" if code in ("??", "A", "AM") else code[:1], line[3:].strip()))
    return tuple(pairs)


def range_name_status(path: Path, base: str, head: str) -> tuple[tuple[str, str], ...]:
    """``(status, path)`` pairs for ``base..head``: the conventional-subject
    deriver's input at squash time."""
    res = _run(path, "diff", "--name-status", f"{base}..{head}", check=False)
    pairs: list[tuple[str, str]] = []
    for line in res.stdout.splitlines():
        cols = line.split("\t")
        if len(cols) >= 2:
            pairs.append((cols[0][:1], cols[-1]))
    return tuple(pairs)


def _headline_subject(task: str, *, limit: int = _SUBJECT_LIMIT) -> str:
    """A short commit subject derived from the task's first clause: its first
    line, up to the first sentence end, whitespace-collapsed, capped at *limit*
    (an ellipsis marks a truncation). A run's whole task text as the subject
    reads as one unwrapped 180-char line that every git tool clips; the full
    task is wrapped into the body by the caller when this truncates it."""
    first_line = next((ln for ln in task.splitlines() if ln.strip()), "")
    match = re.search(r"[.!?](?:\s|$)", first_line)
    clause = first_line[: match.start()] if match else first_line
    clause = " ".join(clause.split())
    if len(clause) <= limit:
        return clause
    return clause[: limit - 1].rstrip() + "…"


def recent_log(path: Path, n: int = 20) -> str:
    res = _run(path, "log", f"-n{n}", "--oneline", check=False)
    return res.stdout if res.ok else ""


def revert_head(path: Path) -> str:
    """Forward-revert HEAD via ``git revert HEAD --no-edit``.

    Backs the interactive REPL's ``/undo`` so the operator
    can roll back the last auto-commit without rewriting history. Returns
    the SHA of the new revert commit. AGENTS.md forbids ``reset --hard``
    and force operations; ``revert`` is the policy-compliant undo.
    """
    _run(path, "revert", "HEAD", "--no-edit")
    sha_res = _run(path, "rev-parse", "HEAD")
    return sha_res.stdout.strip() if sha_res.ok else ""


def tracked_files(path: Path) -> tuple[str, ...]:
    """Return the list of repo-tracked files via ``git ls-files``.

    POSIX-style separators, sorted by ``git``'s own order. Empty tuple
    outside a git repo or when ls-files fails - callers must treat this
    as "no map available" rather than "empty repo".
    """
    res = _run(path, "ls-files", "-z", check=False)
    if not res.ok:
        return ()
    return tuple(p for p in res.stdout.split("\x00") if p)


def co_change_pairs(
    path: Path,
    *,
    n_commits: int = 200,
    min_pair_count: int = 2,
    max_pairs: int = 30,
) -> list[tuple[str, str, int]]:
    """Mine git history for co-change file pairs.

    Walks the last *n_commits* commits, groups changed files per commit,
    and returns the top *max_pairs* most-frequent unordered (fileA, fileB)
    pairs that co-changed in at least *min_pair_count* commits. Each
    tuple is (file_a, file_b, count). Sorted by count descending, ties
    broken alphabetically.

    Cheap signal for the planner: "file A and file B change together
    73% of the time" is a strong prior for "if you edit A, you probably
    also need to touch B". Returns an empty list if git history is too
    shallow to find any qualifying pairs (e.g. the fresh-clone bench
    case with --depth=1).

    Skips merge commits (--no-merges) so multi-parent diffs don't
    artificially inflate co-change frequencies.
    """
    res = _run(
        path,
        "log",
        f"-n{n_commits}",
        "--no-merges",
        "--name-only",
        "--pretty=format:%x00",
        check=False,
    )
    if not res.ok:
        return []
    # Output is groups of (NUL-separator, blank line, file paths...) per
    # commit. Split on NUL to get per-commit file lists.
    pair_counter: Counter[tuple[str, str]] = Counter()
    for chunk in res.stdout.split("\x00"):
        files = [line.strip() for line in chunk.strip().splitlines() if line.strip()]
        # Skip binary-marker lines and any non-file entries (defensive).
        files = sorted(set(f for f in files if "/" in f or "." in f))
        if len(files) < 2:
            continue
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                pair_counter[(files[i], files[j])] += 1
    qualifying = [(a, b, c) for (a, b), c in pair_counter.items() if c >= min_pair_count]
    qualifying.sort(key=lambda t: (-t[2], t[0], t[1]))
    return qualifying[:max_pairs]


def diff_since(path: Path, base_sha: str) -> str:
    # `git diff <base>` only considers tracked content. Newly created files
    # from a worker edit are untracked at this point (commit_all stages and
    # commits later, after the reviewer is consulted), so a plain diff would
    # be empty and the reviewer would falsely conclude "the worker did
    # nothing". Register untracked files with `git add -N` (intent-to-add)
    # so they show up as additions in the diff. -N doesn't add content to the
    # index; commit_all's later `git add -A` overwrites the intent entries.
    _run(path, "add", "-N", "--", ".", check=False)
    res = _run(path, "diff", base_sha, "--", ".", check=False)
    return res.stdout if res.ok else ""


def diff_range(path: Path, base_sha: str, ref: str) -> str:
    """The committed diff ``base_sha..ref`` introduces, captured. "" when the
    range is unresolvable (a pruned branch, a bad sha) -- read-only, never blocks
    a caller comparing several candidates. Goes through ``_run`` so it carries the
    same host-RCE hardening + builtin-renderer flags every git diff here does."""
    res = _run(path, "diff", f"{base_sha}..{ref}", check=False)
    return res.stdout if res.ok else ""


def commit_diff(path: Path, sha: str, *, max_bytes: int = 16384) -> str:
    """The patch a single commit introduced (``git show <sha>``), or "" on error.

    Read-only, used to surface "what the worker just changed" to a live viewer.
    ``--format=`` keeps it to just the diff (no commit message). Truncated to
    ``max_bytes`` here so callers don't materialize an unbounded diff in memory."""
    res = _run(path, "show", "--format=", "--no-color", sha, "--", ".", check=False)
    if not res.ok:
        return ""
    return res.stdout[:max_bytes]


def reset_to(path: Path, sha: str, *, mode: str) -> None:
    """Move HEAD (and optionally the index) to *sha* on the current branch.

    *mode* must be ``"soft"`` (HEAD only; index + worktree unchanged) or
    ``"mixed"`` (HEAD + index; worktree unchanged). ``"hard"`` is
    intentionally not accepted here: this module never performs a
    data-destroying reset, so a caller cannot accidentally obtain one.
    The commits this reset orphans remain reachable via
    reflog, so the operation is recoverable.
    """
    if mode not in {"soft", "mixed"}:
        raise GitError(f"reset_to: mode must be 'soft' or 'mixed', got {mode!r}")
    _run(path, "reset", f"--{mode}", sha)


def rollback_to_known_good(path: Path, sha: str) -> None:
    """Restore branch tip + worktree to *sha* after a regressing commit.

    ``reset --mixed`` then ``checkout -- .``, deliberately two steps rather
    than ``reset --hard``: the no-destructive-resets invariant means callers
    never get a primitive that unconditionally clobbers uncommitted work.
    Orphaned commits stay reachable in the reflog.
    """
    if not sha:
        raise GitError("rollback_to_known_good: sha must be non-empty")
    _run(path, "reset", "--mixed", sha)
    _run(path, "checkout", "--", ".")


def make_run_branch_name(prefix: str = "agent6", task_slug: str | None = None) -> str:
    """Build a run branch name like ``agent6/20260526-120000-fix-bug``.

    ``task_slug`` is the slugified user task; when omitted the slug falls
    back to the prefix so the function remains useful for ad-hoc callers.
    """
    ts = _dt.datetime.now(tz=_dt.UTC).strftime("%Y%m%d-%H%M%S")
    slug = task_slug if task_slug else slugify(prefix)
    return f"{prefix}/{ts}-{slug}"


def show_commit(path: Path, sha: str, *, max_bytes: int = 16_384) -> str:
    """Return `git show --stat <sha>` truncated to *max_bytes* for telemetry.

    Best-effort: returns empty string on error rather than raising.
    """
    res = _run(path, "show", "--stat", "--patch", sha, check=False)
    if not res.ok:
        return ""
    out = res.stdout
    if len(out) > max_bytes:
        return out[:max_bytes] + f"\n... [truncated, full size {len(out)} bytes]"
    return out


def add_worktree(repo: Path, dest: Path, sha: str) -> None:
    """Check *sha* out into *dest* as a detached worktree.

    The exact tree of that commit and nothing else: a clone plus `reset --mixed`
    leaves every file added after *sha* on disk as untracked, which silently
    turns "the base commit" into "the base commit plus this run's new files".
    """
    _run(repo, "worktree", "add", "--detach", str(dest.absolute()), sha)


def prune_worktrees(repo: Path) -> None:
    """Drop bookkeeping for worktrees whose directories are gone.

    The caller deletes the throwaway directory itself, so this only tidies
    `.git/worktrees`. `worktree remove` would need `--force` once a gate has
    written build output there, and this module spells no destructive verb.
    """
    _run(repo, "worktree", "prune")
