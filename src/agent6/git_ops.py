# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Git operations with hard safety invariants.

Every dangerous operation (push, force, history rewrite) raises GitSafetyError
unconditionally. The config can *loosen* benign options (auto-stash, branch-per-run)
but the destructive operations are not exposed as a code path here at all.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from agent6.types import CommandResult


class GitError(Exception):
    """Generic git failure."""


class GitSafetyError(GitError):
    """Refused to perform a destructive git operation."""


@dataclass(frozen=True, slots=True)
class GitStatus:
    branch: str
    head_sha: str
    is_clean: bool
    untracked_count: int
    modified_count: int


@dataclass(frozen=True, slots=True)
class CommitIdentity:
    """Resolved name/email/coauthor used for commits this run.

    `name` and `email` are populated from `[git.commit]` overrides when set,
    otherwise left as None to mean "let git's own config resolution decide".
    `verify_git_identity` ensures that when both are None the project's git
    config has a usable identity before any commit is attempted.
    """

    name: str | None = None
    email: str | None = None
    coauthor: str | None = None

    @property
    def has_override(self) -> bool:
        return bool(self.name or self.email or self.coauthor)


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
# `diff.external` fires one on `git diff` (review/diff). Both are pure overrides
# with no correctness cost (fsmonitor is a perf cache; an empty diff.external
# uses git's builtin diff). The edit tools already refuse writes into `.git`
# under protect_git, but a repo cloned with a pre-poisoned `.git/config` would
# otherwise execute its payload the first time agent6 ran git here.
_GIT_EGRESS_HARDENING: tuple[str, ...] = ("-c", "core.fsmonitor=false", "-c", "diff.external=")

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


def _git_hardening() -> tuple[str, ...]:
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
    env = None
    if env_extra:
        env = {**os.environ, **env_extra}
    hardening = _git_hardening()
    proc = subprocess.run(
        [_git(), *hardening, *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    result = CommandResult(
        argv=(_git(), *hardening, *args),
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        duration_s=0.0,
    )
    if check and not result.ok:
        # Surface stdout too. `git commit` writes its informational
        # output (including "nothing to commit, working tree clean", pre-
        # commit hook output, and most user-facing messages) to STDOUT,
        # not stderr. Capturing only stderr produced empty error strings
        # like "git commit -m <subject> failed: " that gave the operator
        # zero signal when triaging a failed auto-commit in the wild.
        stderr_msg = proc.stderr.strip()
        stdout_msg = proc.stdout.strip()
        if stderr_msg and stdout_msg:
            detail = f"{stderr_msg} | stdout: {stdout_msg}"
        else:
            detail = stderr_msg or stdout_msg or f"exit {proc.returncode}"
        raise GitError(f"git {' '.join(args)} failed: {detail}")
    return result


def slugify(text: str, max_len: int = 40) -> str:
    """Lowercase ASCII slug for use in branch names."""
    slug = _SLUG_RE.sub("-", text.lower()).strip("-")
    return (slug or "run")[:max_len]


def is_git_repo(path: Path) -> bool:
    res = _run(path, "rev-parse", "--is-inside-work-tree", check=False)
    return res.ok and res.stdout.strip() == "true"


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


def restore_stash(path: Path) -> bool:
    """Apply the latest stash back onto the working tree. On a clean apply, drop
    the stash and return True. On conflict (or any non-zero apply), leave the
    stash in place so the user's work is never lost, and return False. We never
    `reset --hard` to undo a conflicted apply (refused), so a conflict leaves the
    markers for the user to resolve with their stash still safe at stash@{0}."""
    if _run(path, "stash", "apply", check=False).ok:
        _run(path, "stash", "drop", check=False)
        return True
    return False


def create_branch(path: Path, name: str) -> None:
    """Create *name* from HEAD and check it out, or just check it out if it
    already exists. Idempotent so re-running/resuming a run reuses the run's
    branch instead of cutting a near-duplicate (no `branch -D` needed, which we
    refuse anyway)."""
    existing = _run(path, "branch", "--list", name, check=False)
    if existing.ok and existing.stdout.strip():
        _run(path, "checkout", name)
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
    """Stage only `paths` (repo-relative) and commit. Returns the new HEAD sha.

    Useful when callers must NOT touch unrelated WIP changes in the worktree
    (e.g. the startup `.gitignore` auto-update, which runs before the
    dirty-worktree pre-flight and therefore must not sweep up the user's
    in-progress edits).
    """
    if not paths:
        raise GitError("commit_paths requires at least one path")
    _run(path, "add", "--", *paths)
    return _commit(path, message, trailers=trailers, identity=identity)


def _commit(
    path: Path,
    message: str,
    *,
    trailers: dict[str, str] | None,
    identity: CommitIdentity | None,
) -> str:
    merged_trailers = dict(trailers or {})
    env_extra: dict[str, str] = {}
    if identity is not None:
        if identity.name:
            env_extra["GIT_AUTHOR_NAME"] = identity.name
            env_extra["GIT_COMMITTER_NAME"] = identity.name
        if identity.email:
            env_extra["GIT_AUTHOR_EMAIL"] = identity.email
            env_extra["GIT_COMMITTER_EMAIL"] = identity.email
        if identity.coauthor:
            merged_trailers["Co-authored-by"] = identity.coauthor
    full_message = message
    if merged_trailers:
        trailer_lines = "\n".join(f"{k}: {v}" for k, v in merged_trailers.items())
        full_message = f"{message}\n\n{trailer_lines}"
    _run(path, "commit", "-m", full_message, env_extra=env_extra or None)
    return _run(path, "rev-parse", "HEAD").stdout.strip()


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
    intentionally not accepted here: data-destroying resets must go
    through ``refuse_history_rewrite`` so a caller cannot accidentally
    obtain one. The commits this reset orphans remain reachable via
    reflog, so the operation is recoverable.
    """
    if mode not in {"soft", "mixed"}:
        raise GitError(f"reset_to: mode must be 'soft' or 'mixed', got {mode!r}")
    _run(path, "reset", f"--{mode}", sha)


def rollback_to_known_good(path: Path, sha: str) -> None:
    """Restore branch tip + worktree to *sha* after a regressing commit.

    Used by metric-driven workflows when the latest commit measurably
    regressed past the run's starting baseline: instead of compounding
    edits on top of a known-broken state, we rewind the branch tip to
    the last-known-good commit and restore the worktree to match. The
    rewound commits remain reachable via reflog (audit trail), so this
    is recoverable.

    Implementation: ``git reset --mixed sha`` rewinds HEAD + index while
    leaving the worktree alone; the follow-up ``git checkout -- .``
    then snaps the worktree back to the index (i.e. to *sha*'s tree).
    Two steps rather than ``reset --hard`` so we stay within the
    "no destructive resets" invariant, anything orphaned is still in
    the reflog and ``git_ops`` callers never get a primitive that
    unconditionally clobbers uncommitted work.
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


# ---------------------------------------------------------------------------
# Refusals, operations agent6 will not perform under any circumstances.
# ---------------------------------------------------------------------------


def refuse_push(*_args: object, **_kwargs: object) -> None:
    raise GitSafetyError("git push is disabled by agent6")


def refuse_force(*_args: object, **_kwargs: object) -> None:
    raise GitSafetyError("git --force operations are disabled by agent6")


def refuse_history_rewrite(*_args: object, **_kwargs: object) -> None:
    raise GitSafetyError(
        "git history-rewriting operations (rebase, amend, reset --hard, gc, branch -D)"
        " are disabled by agent6"
    )
