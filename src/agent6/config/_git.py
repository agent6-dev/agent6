# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `[git]` model.

Worktree policy, the run's detached chain, merge and message styles.
"""

from __future__ import annotations

import re
import string
from typing import Literal

import pydantic

from agent6.config import _base


class GitCommitCheckpointConfig(pydantic.BaseModel):
    """Message style for the per-step commits a run makes on its branch."""

    model_config = _base.MODEL_CONFIG

    message: Literal["agent6", "conventional", "model"] = pydantic.Field(
        default="agent6",
        description=(
            "The message of each per-step commit: `agent6` (`agent6 iter N: <summary>`), "
            "`conventional` (a `type(scope): subject` derived from the diff, no model call), or "
            "`model` (the model writes it from the git facts, falling back to `agent6` with a "
            "warning on any failure)."
        ),
    )


class GitCommitSquashConfig(pydantic.BaseModel):
    """Message style for the one commit a squash merge produces."""

    model_config = _base.MODEL_CONFIG

    message: Literal["agent6", "conventional", "combine", "model"] = pydantic.Field(
        default="agent6",
        description=(
            "The message of the one commit a squash merge produces: `agent6` (the task's first "
            "clause as the subject, the steps as bullets), `conventional` (a `type(scope): "
            "subject` derived from the diff, no "
            "model call), `combine` (git's own squash message: the per-step log concatenated), or "
            "`model` (model-written, falling back to `agent6` with a warning on any failure)."
        ),
    )


class GitCommitConfig(pydantic.BaseModel):
    """The commit identity, the provenance trailer and the per-kind message styles.

    `name` and `email` unset mean the project's own `git config` identity; a run with
    neither an override nor a resolvable identity refuses at startup.
    """

    model_config = _base.MODEL_CONFIG

    name: str | None = pydantic.Field(
        default=None,
        description=(
            "Author and committer name on the commits agent6 makes; unset uses the repo's own `git "
            "config`. A run with no resolvable identity refuses to start."
        ),
    )
    email: str | None = pydantic.Field(
        default=None,
        description=(
            "Author and committer email on the commits agent6 makes; unset uses the repo's own "
            "`git config`. A run with no resolvable identity refuses to start."
        ),
    )
    trailer: str = pydantic.Field(
        default="",
        description=(
            "A git trailer line (`Key: value`) appended to every commit agent6 makes, e.g. "
            '`"Assisted-by: agent6:{model}"` or `"Co-authored-by: agent6:{model} '
            '<noreply@agent6.dev>"`. `{model}` is the model that wrote the code (several are '
            "joined with `, `). Empty: no trailer."
        ),
    )
    checkpoint: GitCommitCheckpointConfig = pydantic.Field(
        default_factory=GitCommitCheckpointConfig
    )
    squash: GitCommitSquashConfig = pydantic.Field(default_factory=GitCommitSquashConfig)

    @pydantic.field_validator("trailer")
    @classmethod
    def _trailer_is_a_trailer_line(cls, v: str) -> str:
        """Refuse a trailer that is not a `Key: value` line with only `{model}` as a placeholder.

        Args:
            v: The trailer template.

        Returns:
            The template unchanged.

        Raises:
            ValueError: An unknown placeholder, or the rendered line is not a trailer.
        """
        if not v:
            return v
        fields = {f for _, f, _, _ in string.Formatter().parse(v) if f is not None}
        unknown = fields - {"model"}
        if unknown:
            raise ValueError(
                f"unknown placeholder {sorted(unknown)} in git.commit.trailer (known: {{model}})"
            )
        rendered = v.format(model="m")
        if not re.fullmatch(r"[A-Za-z][A-Za-z-]*: .+", rendered, re.DOTALL):
            raise ValueError(
                'git.commit.trailer must be a git trailer line, "Key: value"'
                ' (e.g. "Assisted-by: agent6:{model}")'
            )
        return v


class GitConfig(pydantic.BaseModel):
    """The `[git]` table."""

    model_config = _base.MODEL_CONFIG

    # A run records the untracked files present at its start and never commits them.
    dirty_tree: Literal["ask", "stash", "include"] = pydantic.Field(
        default="ask",
        description=(
            "What a run does with tracked files' uncommitted changes at start. `ask`: ask over the "
            "ask_user channel (`stash` them for the run, `include` them in its commits, or "
            "`cancel`, which parks the run for a later resume); a run nobody can answer refuses "
            "to start. `stash`: stash them without asking; at the end the stash is applied back "
            "per `auto_stash_pop`, else its `git stash apply <sha>` line is printed. `include`: "
            "start without asking, the run's first commit records them. Untracked files never "
            "count and are never committed. `--parallel` fans out under `stash` or `include` and "
            "refuses under `ask`."
        ),
    )
    # A run that edited leaves its work in the tree: this fires after auto_merge or a no-edit run.
    auto_stash_pop: bool = pydantic.Field(
        default=False,
        description=(
            "Apply the pre-run stash back when the run ends and the tree is clean (a clean apply, "
            "no conflicts). On any doubt the stash stays and the apply line is printed. Never "
            '`reset --hard`. Requires `dirty_tree = "stash"`.'
        ),
    )
    # The chain is refs/agent6/<session>/head, parented on HEAD at run start.
    control: Literal["agent6", "model"] = pydantic.Field(
        default="agent6",
        description=(
            "Who manages git during a run: `agent6` records every step on the run's own commit "
            "chain and branch, never touching HEAD; `model` hands git to the model: no per-step "
            "chain, no run branch, the model's own commits and branches are the record, and "
            "`sessions diff`/`merge`, `/undo`, and `fork` refuse for such runs. Requires "
            "`sandbox.protect_git = false`."
        ),
    )
    branch_per_run: bool = pydantic.Field(
        default=True,
        description=(
            "Also advance a visible `agent6/<run-id>` branch to the run's chain tip; `false` keeps "
            "only the hidden `refs/agent6/<run-id>/head` ref. Forced on for `--parallel` lanes "
            "(their work is imported by branch)."
        ),
    )
    # Off, resume still works from snapshots; the step-history surfaces degrade.
    commit_per_step: bool = pydantic.Field(
        default=True,
        description=(
            "Commit each editing step onto the run's detached chain (a temp index; HEAD never "
            "moves, and your index and working tree are touched only when the run's own branch "
            "is the one checked out). `false`: agent6 never commits; the work stays only in the "
            "worktree, and resume-from-git, `sessions diff`/`merge`, and `/parallel` dispatch "
            "from a changed tree degrade."
        ),
    )
    merge_strategy: Literal["squash", "merge", "ff"] = pydantic.Field(
        default="squash",
        description=(
            "How `agent6 sessions merge` lands a run on its base: `squash` (one commit), `merge` "
            "(a merge commit keeping the per-step history), or `ff` (fast-forward). "
            "Consolidation only; per-step commits always land on the run's chain."
        ),
    )
    # With auto_stash_pop the merge lands first, then the stash goes back on top.
    auto_merge: bool = pydantic.Field(
        default=False,
        description=(
            "After a run that finished with nothing red, merge its work into its base branch "
            "automatically (never over a red or stale verify). With `branch_per_run` off it merges "
            "the hidden chain ref. On a conflict nothing moves and the instructions are printed."
        ),
    )
    # The hidden chain ref stays as the run's record until `sessions rm`.
    auto_prune: bool = pydantic.Field(
        default=False,
        description=(
            "After an `auto_merge`, delete the run branch when `git branch -d` can (a `merge` or "
            "`ff` merge). A squash-merged branch is reported with its `-D` line, never "
            "force-deleted. Requires `auto_merge`; nothing to do without a run branch."
        ),
    )
    # A hook runs on the host outside the jail: host RCE for an adversarial repo.
    # `core.fsmonitor` and `diff.external` fire on status/diff and have no legitimate use here.
    run_repo_hooks: bool = pydantic.Field(
        default=False,
        description=(
            "Run the repo's own `.git/hooks/*` during agent6's git operations. `false` skips "
            "them: a repo hook is repo-controlled code that would run on the host. "
            "`core.fsmonitor` and `diff.external` are always neutralized."
        ),
    )
    # A driver in a poisoned `.git/config` runs on the host at every stage or merge.
    run_repo_filters: bool = pydantic.Field(
        default=False,
        description=(
            "Honor the repo's content drivers (`filter.<name>.clean/smudge/process`, "
            "`merge.<name>.driver`) during agent6's git operations. `false` neutralizes each by "
            "name: a driver defined in `.git/config` is repo-controlled code that would run on "
            "the host at every commit. `true` is what Git LFS needs (its clean/smudge filters "
            "are these drivers)."
        ),
    )
    commit: GitCommitConfig = pydantic.Field(default_factory=GitCommitConfig)

    @pydantic.model_validator(mode="after")
    def _check_auto_merge(self) -> GitConfig:
        """Refuse the stash and prune settings without the setting each depends on.

        Returns:
            The model unchanged.

        Raises:
            ValueError: `auto_stash_pop` without `dirty_tree = "stash"`, or `auto_prune` without
                `auto_merge`.
        """
        if self.auto_stash_pop and self.dirty_tree != "stash":
            raise ValueError(
                'git.auto_stash_pop requires git.dirty_tree = "stash": with nothing stashed '
                "pre-run there is nothing to restore at run end."
            )
        if self.auto_prune and not self.auto_merge:
            raise ValueError(
                "git.auto_prune requires git.auto_merge: pruning a run branch only makes "
                "sense once it has been merged."
            )
        return self
