# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Read git status, refs, trees and diffs without changing the shared index or refs."""

from __future__ import annotations

import contextlib
import dataclasses
import pathlib
import shutil
import tempfile
from collections.abc import Collection

from agent6 import errors
from agent6.vcs import _git_command, commit_message

# A name handed to git as a pathspec is the file, never a glob: `a*.txt` matches only itself.
LITERAL_PATHSPECS = {"GIT_LITERAL_PATHSPECS": "1"}

# What `git rev-parse --abbrev-ref HEAD` prints on a detached HEAD, where no branch is checked out.
DETACHED_HEAD = "HEAD"


@dataclasses.dataclass(frozen=True, slots=True)
class GitStatus:
    """The worktree against HEAD.

    Attributes:
        branch: The checked-out branch; `DETACHED_HEAD` when HEAD is detached.
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
    name = (
        identity.name or _git_command.run(path, "config", "user.name", check=False).stdout.strip()
    )
    email = (
        identity.email or _git_command.run(path, "config", "user.email", check=False).stdout.strip()
    )
    missing: list[str] = []
    if not name:
        missing.append("user.name")
    if not email:
        missing.append("user.email")
    if missing:
        joined = " and ".join(missing)
        raise errors.GitError(
            f"Git identity not configured: {joined} is empty. Either run\n"
            f"    git -C {path} config user.name 'Your Name'\n"
            f"    git -C {path} config user.email 'you@example.com'\n"
            f"or set [git.commit].name / [git.commit].email in your agent6 config."
        )
    return name, email


def is_git_repo(path: pathlib.Path) -> bool:
    """Return whether the path is inside a git work tree."""
    res = _git_command.run(path, "rev-parse", "--is-inside-work-tree", check=False)
    return res.ok and res.stdout.strip() == "true"


def toplevel(path: pathlib.Path) -> pathlib.Path | None:
    """Return the enclosing work tree's root, or None outside a repo or inside `.git`."""
    res = _git_command.run(path, "rev-parse", "--show-toplevel", check=False)
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
    res = _git_command.run(
        path, "status", "--porcelain", "--", *rel_paths, check=False, env_extra=LITERAL_PATHSPECS
    )
    return bool(res.stdout.strip())


def porcelain_entries(path: pathlib.Path) -> list[tuple[str, str]]:
    """Return the status code and repo-relative path of every changed or untracked file.

    NUL-separated, so any filename round-trips; a rename's source path is skipped.

    Args:
        path: The repository.

    Returns:
        The (code, path) pairs.
    """
    fields = _git_command.nul_fields(
        path, "status", "--porcelain=v1", "-z", "--untracked-files=all", check=False
    )
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
    return [rel for code, rel in porcelain_entries(path) if code != "??"]


def untracked_paths(path: pathlib.Path) -> frozenset[str]:
    """Return every untracked, non-ignored file, repo-relative.

    Taken once at run start as `untracked_at_start`: those files are the operator's,
    so chain commits and dirty checks leave them out.

    Args:
        path: The repository.

    Returns:
        The paths; empty outside a repo.
    """
    return frozenset(rel for code, rel in porcelain_entries(path) if code == "??")


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
        raise errors.GitError(f"Not a git repository: {path}")
    branch_res = _git_command.run(path, "rev-parse", "--abbrev-ref", "HEAD", check=False)
    if branch_res.ok:
        branch = branch_res.stdout.strip()
    else:
        # An unborn HEAD fails rev-parse, but `branch --show-current` still names the branch.
        branch = _git_command.run(path, "branch", "--show-current", check=False).stdout.strip()
    head_res = _git_command.run(path, "rev-parse", "HEAD", check=False)
    head_sha = head_res.stdout.strip() if head_res.ok else ""
    untracked = 0
    modified = 0
    for code, rel in porcelain_entries(path):
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

    Args:
        path: The repository.
        message: The stash message.

    Returns:
        The entry, or None.
    """
    res = _git_command.run(path, "stash", "list", "--format=%gd%x09%H%x09%gs", check=False)
    for line in res.stdout.splitlines():
        ref, _, rest = line.partition("\t")
        sha, _, subject = rest.partition("\t")
        if subject.endswith(f": {message}"):
            return StashEntry(ref=ref, sha=sha)
    return None


def branch_exists(path: pathlib.Path, name: str) -> bool:
    """Return whether the local branch exists."""
    return _git_command.run(
        path, "rev-parse", "--verify", "--quiet", f"refs/heads/{name}", check=False
    ).ok


def valid_branch_name(name: str) -> bool:
    """Return whether the name passes `git check-ref-format --branch`.

    A session id becomes a branch and a chain ref; a branch name is the stricter role,
    so this is the one check a session id needs. Run from "/": no repo is involved.

    Args:
        name: The bare name, without `refs/heads/`.

    Returns:
        True when git accepts it as a branch name.
    """
    return _git_command.run(pathlib.Path("/"), "check-ref-format", "--branch", name, check=False).ok


def list_run_branches(path: pathlib.Path) -> tuple[str, ...]:
    """Return the local branches under `agent6/`, sorted."""
    res = _git_command.run(
        path, "for-each-ref", "--format=%(refname:short)", "refs/heads/agent6/", check=False
    )
    return tuple(b for b in res.stdout.splitlines() if b.strip())


def run_ref_tips(path: pathlib.Path) -> dict[str, str]:
    """Return the tip sha of every run branch and chain ref, in one git call.

    A per-row rev-parse would put about 50 subprocesses on the hub's poll.

    Args:
        path: The repository.

    Returns:
        Branch short names and full chain ref names, each to its sha.
    """
    res = _git_command.run(
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
    return _git_command.run(
        path, "merge-base", "--is-ancestor", maybe_ancestor, ref, check=False
    ).ok


def branch_tip_sha(path: pathlib.Path, branch: str) -> str | None:
    """Return the sha the branch points at, or None when it does not resolve."""
    res = _git_command.run(
        path, "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}", check=False
    )
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
    with contextlib.suppress(errors.GitError):
        chain = chain_tip(path, chain_ref_for(session_id)) if session_id else None
        branch = branch_tip_sha(path, run_branch) if run_branch else None
        tips = {t for t in (chain, branch) if t}
    return not tips or merged_tip in tips


CHAIN_NS = "refs/agent6"

CHAIN_KIND = "head"


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

    Args:
        session_id: The ref's `<id>`.

    Returns:
        The full ref name.
    """
    return f"{CHAIN_NS}/{session_id}/{CHAIN_KIND}"


def list_chain_refs(path: pathlib.Path) -> tuple[tuple[str, str], ...]:
    """Return the session id and sha of every chain ref, sorted by id.

    Globbed on the kind, so another per-session ref beside it is not read as a chain.

    Args:
        path: The repository.

    Returns:
        The (session id, sha) pairs.
    """
    pattern = f"{CHAIN_NS}/*/{CHAIN_KIND}"
    out = _git_command.run(
        path, "for-each-ref", "--format=%(refname)%00%(objectname)", pattern
    ).stdout
    rows: list[tuple[str, str]] = []
    for line in out.splitlines():
        ref, _, sha = line.partition("\x00")
        if sha and ref.startswith(f"{CHAIN_NS}/") and ref.endswith(f"/{CHAIN_KIND}"):
            rows.append((ref[len(CHAIN_NS) + 1 : -(len(CHAIN_KIND) + 1)], sha))
    return tuple(sorted(rows))


def git_common_dir(path: pathlib.Path) -> pathlib.Path:
    """Return the repository's shared `.git` directory, absolute: the main checkout's."""
    out = _git_command.run(
        path, "rev-parse", "--path-format=absolute", "--git-common-dir"
    ).stdout.strip()
    return pathlib.Path(out).resolve()


def worktree_of_branch(path: pathlib.Path, branch: str) -> pathlib.Path | None:
    """Return the checkout that has the branch checked out, or None when none does."""
    out = _git_command.run(path, "worktree", "list", "--porcelain").stdout
    where: pathlib.Path | None = None
    for line in out.splitlines():
        if line.startswith("worktree "):
            where = pathlib.Path(line[len("worktree ") :])
        elif line == f"branch refs/heads/{branch}" and where is not None:
            return where
    return None


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
    # The names go in on stdin, so one a model wrote never reaches the argv.
    ignored = set(
        _git_command.nul_fields(
            path, "check-ignore", "-z", "--stdin", stdin_text="\0".join(candidates), check=False
        )
    )
    return tuple(c for c in candidates if c not in ignored)


def chain_tip(path: pathlib.Path, ref: str) -> str | None:
    """Return the commit sha the ref resolves to, or None when it does not exist."""
    res = _git_command.run(
        path, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", check=False
    )
    sha = res.stdout.strip()
    return sha if res.returncode == 0 and sha else None


def commit_is_reachable(path: pathlib.Path, sha: str) -> bool:
    """Return whether any ref reaches the commit.

    Existence is a different question: a loose commit no ref reaches is one `git gc`
    from gone, and a sweep deleting the last copy of work asks this one.

    Args:
        path: The repository.
        sha: The commit.

    Returns:
        True when a ref contains it; False also for a commit the repository lacks.
    """
    res = _git_command.run(
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
            _git_command.run(path, "read-tree", seed, env_extra=env)
        # git refuses a pathspec naming an ignored path, exclude or not, and `add -A` never
        # takes one: an excluded path the ignore rules cover by now needs no exclude.
        exclude = unignored(path, tuple(sorted(exclude)))
        if exclude:
            spec = tmp / "pathspec"
            spec.write_bytes(
                b"\0".join([b":/", *(f":(top,exclude,literal){rel}".encode() for rel in exclude)])
            )
            _git_command.run(
                path,
                "add",
                "-A",
                f"--pathspec-from-file={spec}",
                "--pathspec-file-nul",
                env_extra=env,
            )
        else:
            _git_command.run(path, "add", "-A", env_extra=env)
        return _git_command.run(path, "write-tree", env_extra=env).stdout.strip()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# git's empty tree, the diff base of a chain with no commits yet.
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


def worktree_matches(path: pathlib.Path, ref: str, paths: Collection[str]) -> bool:
    """Return whether the worktree's content of the paths equals the ref's, ignoring the index.

    Args:
        path: Any directory of the checkout.
        ref: The commit to compare against.
        paths: Repo-relative paths, whichever directory `path` is.

    Returns:
        True when the contents are equal; False also on a bad ref.
    """
    specs = (f":(top,literal){p}" for p in sorted(paths))
    res = _git_command.run(path, "diff", "--quiet", ref, "--", *specs, check=False)
    return res.returncode == 0


def tree_sha(path: pathlib.Path, rev: str) -> str:
    """Return the tree sha of a commit-ish."""
    return _git_command.run(path, "rev-parse", f"{rev}^{{tree}}").stdout.strip()


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
    base_tree = tree_sha(path, base) if base else EMPTY_TREE
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
    base_tree = tree_sha(path, base) if base else EMPTY_TREE
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
    if _git_command.run(path, "rev-parse", "--verify", "--quiet", ref, check=False).returncode != 0:
        return frozenset()
    return frozenset(
        filter(None, _git_command.nul_fields(path, "ls-tree", "-r", "--name-only", "-z", ref))
    )


def tree_diff_paths(path: pathlib.Path, old_tree: str, new_tree: str) -> list[str]:
    """Return the paths whose content differs between two trees, unquoted.

    Print one through `terminal.printable_name`: a name can hold a newline.

    Args:
        path: The repository.
        old_tree: The tree or commit diffed from.
        new_tree: The tree or commit diffed to.

    Returns:
        The repo-relative paths; empty when the trees match.
    """
    return list(
        filter(
            None,
            _git_command.nul_fields(
                path, "diff-tree", "-r", "-z", "--name-only", old_tree, new_tree
            ),
        )
    )


def list_run_commits(
    path: pathlib.Path, base_sha: str, run_branch: str
) -> tuple[commit_message.CommitRow, ...]:
    """Return the commits on the run branch since the base, oldest first."""
    # A body can hold any byte but NUL, so records split on NUL and fields at most twice.
    fmt = "%H%x1f%s%x1f%B"
    res = _git_command.run(
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
    res = _git_command.run(path, "status", "--porcelain", check=False)
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
    res = _git_command.run(path, "diff", "--name-status", f"{base}..{head}", check=False)
    pairs: list[tuple[str, str]] = []
    for line in res.stdout.splitlines():
        cols = line.split("\t")
        if len(cols) >= 2:
            pairs.append((cols[0][:1], cols[-1]))
    return tuple(pairs)


def recent_log(path: pathlib.Path, n: int = 20) -> str:
    """Return the last commits as one-line log text; "" when the log cannot be read."""
    res = _git_command.run(path, "log", f"-n{n}", "--oneline", check=False)
    return res.stdout if res.ok else ""


def read_file_at(path: pathlib.Path, ref: str, name: str) -> str:
    """Return a regular file's text at a git ref, or "" when the tree has no such file.

    Args:
        path: The repository.
        ref: The commit or tree to read.
        name: The repository-relative file name.

    Returns:
        The blob's text, with undecodable bytes replaced; symlinks and directories give "".

    Raises:
        GitError: The ref cannot be read or git fails.
    """
    metadata = tree_entry(path, ref, name)
    if metadata.startswith(("100644 blob ", "100755 blob ")):
        return _git_command.run(path, "cat-file", "blob", metadata.split()[2]).stdout
    return ""


def tree_entry(path: pathlib.Path, rev: str, name: str) -> str:
    """Return the `<mode> <type> <sha>` of a tree entry, or "" when the tree has no such entry.

    Args:
        path: The repository.
        rev: The tree or commit.
        name: The entry's path in the tree.

    Returns:
        The entry's `<mode> <type> <sha>`, or "".

    Raises:
        GitError: The rev cannot be read or git fails.
    """
    entries = _git_command.nul_fields(
        path, "ls-tree", "-z", rev, "--", name, env_extra=LITERAL_PATHSPECS
    )
    for entry in entries:
        metadata, _, found = entry.partition("\t")
        if found == name:
            return metadata
    return ""


def tracked_files(path: pathlib.Path) -> tuple[str, ...]:
    """Return the tracked files in git's own order.

    Args:
        path: The repository; a subdirectory lists only the files under it, relative to it.

    Returns:
        The paths with POSIX separators; empty outside a repo, which callers read as no
        map available rather than an empty repo.
    """
    # Outside a repo git exits nonzero and lists nothing; a timeout still raises.
    return tuple(filter(None, _git_command.nul_fields(path, "ls-files", "-z", check=False)))


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
        # The copy keeps the index's mtime: git re-hashes an entry recorded in the index
        # file's own second, and a copy with a fresh mtime would trust it unchecked.
        if real_index.is_file():
            shutil.copy2(real_index, index_copy)
        env = {"GIT_INDEX_FILE": str(index_copy)}
        _git_command.run(path, "add", "-N", "--", ".", *specs, check=False, env_extra=env)
        res = _git_command.run(
            path, "diff", base_sha, "--", ".", *specs, check=False, env_extra=env
        )
        return res.stdout if res.ok else ""
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def diff_range(path: pathlib.Path, base_sha: str, ref: str) -> str:
    """Return the diff a committed range introduces; "" when the range does not resolve."""
    res = _git_command.run(path, "diff", f"{base_sha}..{ref}", check=False)
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
    res = _git_command.run(path, "show", "--format=", "--no-color", sha, "--", ".", check=False)
    if not res.ok:
        return ""
    return res.stdout[:max_bytes]
