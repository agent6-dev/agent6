# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `[git]` model: worktree policy, the run's detached chain, merge and
message styles."""

from __future__ import annotations

import re
import string
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from agent6.config._base import MODEL_CONFIG


class GitCommitCheckpointConfig(BaseModel):
    """Message style for the per-step commits a run makes on its branch."""

    model_config = MODEL_CONFIG

    # agent6: the `agent6 iter N:` subject. conventional: a `type(scope): subject`
    # derived from the diff without a model call. model: the model writes the
    # message from git facts, degrading to agent6 with a warning on any failure.
    message: Literal["agent6", "conventional", "model"] = Field(
        default="agent6",
        description=(
            "Per-step message style: `agent6` (`agent6 iter N:`), `conventional` (derived from the "
            "diff, no model call), or `model` (model-written, degrading to `agent6` on failure)."
        ),
    )


class GitCommitSquashConfig(BaseModel):
    """Message style for the one commit a squash merge produces."""

    model_config = MODEL_CONFIG

    # As checkpoint's styles, plus combine: git's own squash message (the
    # concatenated per-step log).
    message: Literal["agent6", "conventional", "combine", "model"] = Field(
        default="agent6",
        description=(
            "Squash-commit style: checkpoint's styles plus `combine` (git's concatenated per-step "
            "log)."
        ),
    )


class GitCommitConfig(BaseModel):
    """Overrides for the author/committer identity on agent6 commits, the
    provenance trailer, and the per-kind message styles.

    `name`/`email` default to None = the project's own `git config` identity;
    `agent6 run` refuses at startup when neither an override nor a resolvable
    identity exists, rather than committing as `(no author) <(none)>`.
    """

    model_config = MODEL_CONFIG

    name: str | None = Field(
        default=None,
        description=(
            "Override the commit identity (else the project's `git config`). `agent6 run` refuses "
            "to start with no resolvable identity."
        ),
    )
    email: str | None = Field(
        default=None,
        description=(
            "Override the commit identity (else the project's `git config`). `agent6 run` refuses "
            "to start with no resolvable identity."
        ),
    )
    # Appended to every commit agent6 makes when non-empty, e.g.
    # "Assisted-by: agent6:{model}". {model} = the model(s) that wrote the
    # code, first worker first, ", "-joined when several contributed.
    trailer: str = Field(
        default="",
        description=(
            'Appended to every commit agent6 makes, e.g. `"Assisted-by: agent6:{model}"` or '
            '`"Co-authored-by: agent6:{model} <noreply@agent6.dev>"`. `{model}` = the model(s) '
            'that wrote the code, `", "`-joined when several contributed.'
        ),
    )
    checkpoint: GitCommitCheckpointConfig = GitCommitCheckpointConfig()
    squash: GitCommitSquashConfig = GitCommitSquashConfig()

    @field_validator("trailer")
    @classmethod
    def _trailer_is_a_trailer_line(cls, v: str) -> str:
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


class GitConfig(BaseModel):
    model_config = MODEL_CONFIG

    require_clean_worktree: bool = Field(
        default=True,
        description="Refuse to start on a dirty worktree.",
    )
    auto_stash: bool = Field(
        default=False,
        description=(
            "Stash uncommitted changes before the run; restored per `auto_stash_pop`, else the "
            "`git stash apply <sha>` line is printed (by sha, never silently left)."
        ),
    )
    # When auto_stash stashed pre-run changes, restore them at run end. Default
    # off (safe): the run-end reporter always prints how to pop the stash; with
    # this on, agent6 also pops it for you when it can do so cleanly (switching
    # back to the base branch first under branch_per_run), and otherwise leaves
    # the stash with a message rather than risk a conflicted auto-apply.
    auto_stash_pop: bool = Field(
        default=False,
        description=(
            "Pop the stash back at run end when safe (clean tree, conflict-free apply). On any "
            "doubt, leave it and print how to restore. Never `reset --hard`."
        ),
    )
    # Per-step commits land on the run's own detached chain
    # (refs/agent6/<session>/head), parented on HEAD at run start; HEAD, the
    # operator's index, and the checkout are never touched. branch_per_run
    # additionally advances a visible agent6/<slug> branch ref to the chain
    # tip (off = the hidden ref only). Forced on for --parallel lanes (work
    # is imported by branch).
    branch_per_run: bool = Field(
        default=True,
        description=(
            "Also advance a visible `agent6/<id>` branch to the run's chain tip (else the hidden "
            "`refs/agent6/<id>/head` ref only). Forced on for `--parallel` lanes (work is imported "
            "by branch)."
        ),
    )
    # Off = no per-step commits at all: sessions diff/commits/merge, fork
    # rollback, and the compare judge honestly degrade to "no step history";
    # resume still works from snapshots.
    commit_per_step: bool = Field(
        default=True,
        description=(
            "Per-step commits onto the run's detached chain (a temp index; HEAD, your index, and "
            "your checkout are never touched). Off: agent6 never commits; work stays only in the "
            "worktree, and resume-from-git, `sessions diff`/`merge`, and `/parallel` dispatch "
            "from a changed tree degrade."
        ),
    )
    # Default strategy for `agent6 sessions merge`: how the run branch lands on
    # your branch. `squash` (one combined commit), `merge` (a
    # --no-ff merge keeping the per-step history), or `ff` (fast-forward only).
    # The per-step commits always happen on the run branch during the run; this
    # only governs how they are consolidated when you merge.
    merge_strategy: Literal["squash", "merge", "ff"] = Field(
        default="squash",
        description=(
            "`agent6 sessions merge` default: `squash` (one commit), `merge` (--no-ff, keeps "
            "per-step history), `ff`. Governs consolidation only; per-step commits always land on "
            "the run's chain."
        ),
    )
    # After a successful run, automatically run `merge_strategy` to land the
    # run's work on its base (what `agent6 sessions merge` does, run for you).
    # Default off: the run's refs are kept until you choose to merge. Works
    # with branch_per_run off too (the hidden chain ref is merged). With
    # auto_stash_pop the merge lands first, then your stashed pre-run changes
    # go back on top.
    auto_merge: bool = Field(
        default=False,
        description=(
            "After a run with nothing red, land the run's work on its base automatically (never "
            "over a red/stale verify). With `branch_per_run` off it merges the hidden chain ref. "
            "On conflict nothing moves and instructions are printed."
        ),
    )
    # After auto_merge, delete the run branch when it is safely deletable
    # (`git branch -d`: reachable-merged, so merge/ff strategies). A squash-merged
    # branch is unreachable and is reported with the `git branch -D` to remove it by
    # hand, never force-deleted. Requires auto_merge; no-op when branch_per_run
    # is off (there is no branch, and the hidden chain ref stays as the run's
    # record until `sessions rm`). With both on, run branches stop
    # accumulating, so agent6 looks like a direct-to-branch agent while keeping
    # the per-step commits during the run. Default off.
    auto_prune: bool = Field(
        default=False,
        description=(
            "After `auto_merge`, delete the run branch when `git branch -d` can (merge/ff). A "
            "squash-merged branch is reported with the `-D` line, never force-deleted. Requires "
            "`auto_merge`; no-op without a run branch."
        ),
    )
    # Whether the repo's own git hooks (`.git/hooks/*`) run during agent6's
    # OWN git operations (notably the per-step auto-commit). Default false:
    # secure-by-default (a hook is repo-controlled code that would execute on
    # the HOST, outside the jail, when agent6 commits -- a host-RCE vector for
    # an adversarial repo) and also avoids re-running a slow pre-commit hook on
    # every micro-commit. The verify_command is agent6's real success gate, not
    # git hooks. Set true to honor the repo's hooks (trust the repo). Either
    # way `core.fsmonitor`/`diff.external` stay neutralized (those fire on
    # status/diff and have no legitimate use here).
    run_repo_hooks: bool = Field(
        default=False,
        description=(
            "Run the repo's own `.git/hooks/*` during agent6's git ops. Off: a repo hook is "
            "repo-controlled host code, an RCE vector on an untrusted repo. "
            "`core.fsmonitor`/`diff.external` are always neutralized."
        ),
    )
    # Whether the repo's own content drivers -- `filter.<n>.clean/smudge/process`
    # and `merge.<n>.driver` -- run during agent6's OWN git operations. Default
    # false: like a hook, a driver defined in `.git/config` is repo-controlled
    # code that executes on the HOST, outside the jail, when agent6 stages or
    # merges (a host-RCE vector for a repo cloned with a poisoned `.git/config`).
    # agent6 neutralizes each repo-defined driver by name. Set true to honor
    # them -- the setting a Git-LFS repo needs, since LFS's clean/smudge filters
    # are exactly these drivers.
    run_repo_filters: bool = Field(
        default=False,
        description=(
            "Honor the repo's own content drivers (`filter.<n>.clean/smudge/process`, "
            "`merge.<n>.driver`) during agent6's git ops. Off: a driver defined in `.git/config` "
            "is repo-controlled host code that runs on the auto-commit's `git add` (or a chain "
            "merge), the same RCE class as a hook; agent6 neutralizes each by name. Turn on to "
            "support **Git-LFS** (its clean/smudge filters are exactly these)."
        ),
    )
    commit: GitCommitConfig = Field(default_factory=GitCommitConfig)

    @model_validator(mode="after")
    def _check_auto_merge(self) -> GitConfig:
        if self.auto_stash_pop and not self.auto_stash:
            raise ValueError(
                "git.auto_stash_pop requires git.auto_stash: with nothing stashed "
                "pre-run there is nothing to restore at run end."
            )
        if self.auto_prune and not self.auto_merge:
            raise ValueError(
                "git.auto_prune requires git.auto_merge: pruning a run branch only makes "
                "sense once it has been merged."
            )
        return self
