"""Build the comparison and first message for a session's `/review` directive."""

from __future__ import annotations

import dataclasses
import pathlib
import shlex

from agent6 import errors, review_records
from agent6.harness import repo_context
from agent6.prompts import review_directive as review_prompt
from agent6.vcs import git_read, review

PATCH_MAX_BYTES = 64 * 1024
NO_CHANGES = "no uncommitted changes to review"


class ReviewDirectiveError(Exception):
    """A review target or its base rules cannot be prepared."""


@dataclasses.dataclass(frozen=True, slots=True)
class ReviewWorld:
    """The resolved comparison, first message, checkout requirement, head branch and rules.

    Attributes:
        target: The immutable comparison.
        message: The first message: the resolved revisions, the rules, the stat and the patch.
        needs_worktree: Whether the head needs a detached worktree.
        empty: Whether there is nothing to review.
        head_branch: The local branch the reviewed head is, or "" for a commit or a tag.
        rules: The base tree's AGENTS.md text.
        diff: The whole comparison diff, which the review panel grounds its findings on.
    """

    target: review_records.ReviewTarget
    message: str
    needs_worktree: bool
    empty: bool
    head_branch: str = ""
    rules: str = ""
    diff: str = ""


def build(root: pathlib.Path, target: str, *, hide: tuple[str, ...] = ()) -> ReviewWorld:
    """Resolve a review and build its bounded first message without switching the checkout.

    Args:
        root: The session's checkout.
        target: The argument after `/review`, empty for uncommitted changes.
        hide: The session's hidden paths.

    Returns:
        The immutable comparison and whether its head needs a detached worktree.

    Raises:
        ReviewDirectiveError: Git cannot resolve the target or the base rules are incomplete.
    """
    if "..." in target:
        base_name, head_name = target.split("...", 1)
    elif ".." in target:
        base_name, head_name = target.split("..", 1)
    else:
        base_name, head_name = "HEAD", target or "working tree"
    try:
        compared = review.comparison(
            root,
            base=target if ".." in target or not target else f"HEAD...{target}",
            head="HEAD",
            paths=(),
            hide=hide,
        )
        status = git_read.status(root)
    except errors.GitError as exc:
        raise ReviewDirectiveError(
            f"/review {target}: could not resolve the review: {exc.safe_message}. "
            "Use a local commit or ref, A..B, or A...B."
        ) from exc
    resolved = compared.target
    needs_worktree = bool(
        resolved.head_commit and (status.head_sha != resolved.head_commit or not status.is_clean)
    )
    if (
        not target
        and not compared.diff
        and not compared.dirty_submodules
        and not compared.untracked_repositories
    ):
        return ReviewWorld(resolved, NO_CHANGES, False, True)
    rules = repo_context.review_rules(root, resolved.base_tree, hide)
    if not rules.complete:
        raise ReviewDirectiveError(rules.reason or "review rules incomplete")
    patch_bytes = compared.diff.encode("utf-8")
    patch = patch_bytes[:PATCH_MAX_BYTES].decode("utf-8", errors="ignore")
    parts = [
        f"Review: /review{(' ' + target) if target else ''}",
        f"Base: {base_name or 'HEAD'} -> {resolved.base_commit or 'empty tree'} "
        f"(tree {resolved.base_tree})\n"
        f"Head: {head_name or 'HEAD'} -> {resolved.head_commit or 'working tree snapshot'} "
        f"(tree {resolved.head_tree})",
        review_prompt.INSTRUCTION.strip(),
        f"BASE RULES:\n{rules.text or '(no AGENTS.md at the base)'}",
        f"DIFF STAT:\n{compared.stat.rstrip() or '(no file changes)'}",
        f"PATCH:\n```diff\n{patch.rstrip()}\n```",
    ]
    if len(patch_bytes) > PATCH_MAX_BYTES:
        parts.append(
            f"Patch cut at {PATCH_MAX_BYTES} bytes. Read the rest per path with "
            f"`git diff {resolved.base_tree} {resolved.head_tree} -- <path>`."
        )
    if compared.dirty_submodules:
        parts.append(
            "Submodule working changes not in the patch: " + ", ".join(compared.dirty_submodules)
        )
    if compared.untracked_repositories:
        parts.append(
            "Untracked repositories not in the patch: " + ", ".join(compared.untracked_repositories)
        )
    return ReviewWorld(
        resolved,
        "\n\n".join(parts),
        needs_worktree,
        False,
        head_branch=_head_branch(root, resolved, status, head_name),
        rules=rules.text,
        diff=compared.diff,
    )


def _head_branch(
    root: pathlib.Path, resolved: review_records.ReviewTarget, status: git_read.GitStatus, name: str
) -> str:
    """Return the local branch the reviewed head is, or "" for a commit or a tag."""
    if resolved.head_commit is None or resolved.head_commit == status.head_sha:
        return status.branch
    return name if git_read.branch_exists(root, name) else ""


def live(target: str, *, root: pathlib.Path, hide: tuple[str, ...] = ()) -> tuple[bool, str]:
    """Prepare a live review without moving the session to another tree.

    Args:
        target: The argument after `/review`.
        root: The session's checkout.
        hide: The session's hidden paths.

    Returns:
        Whether to inject the message, and the message or feedback for the operator.
    """
    try:
        world = build(root, target, hide=hide)
        if world.needs_worktree:
            current = review.comparison(root, base="", head="HEAD", paths=(), hide=hide)
            if (
                current.target.head_tree == world.target.head_tree
                and not current.dirty_submodules
                and not current.untracked_repositories
            ):
                return True, world.message
    except ReviewDirectiveError as exc:
        return False, str(exc)
    except errors.GitError as exc:
        return False, f"/review {target}: could not read the session's tree: {exc.safe_message}"
    if world.needs_worktree:
        command = shlex.join(["agent6", "ask", f"/review {target}"])
        return False, f"/review {target} needs a different tree; start a new session: {command}"
    return not world.empty, world.message
