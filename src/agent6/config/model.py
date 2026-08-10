# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Config loading, TOML to pydantic.

This is a trust boundary (untrusted text -> structured types), so we use
pydantic and surface field-pointing errors.

Field policy: **secure by default, fully auditable**. Every field has a
default, and security-sensitive fields default to the *safe* value
(``sandbox.network = "auto"``,
``sandbox.run_commands = "ask"``, ``sandbox.protect_git = true``; git push,
``--force``, and history rewrites are refused unconditionally by ``git_ops``,
with no config override at all). This means a config can be layered (global ``$XDG_CONFIG_HOME``
defaults, per-repo config (out of the workspace, under the state dir) overrides)
and a repo can be
zero-config when the global config supplies providers + models. Use
``agent6 config show`` to audit the *effective* value of every field and
exactly where it came from (default / global / repo / flag). The one thing a
run genuinely cannot guess, a provider+key, is checked by
:meth:`Config.require_runnable` with a friendly pointer to ``agent6 connect``
rather than a load-time failure, so ``config show`` always works. The repo's
``verify_command`` is optional: `agent6 run`/`plan` infer one per run when it
is unset (see :mod:`agent6.verify_infer`), else run gateless.
"""

from __future__ import annotations

import math
import re
import string
import tomllib
from collections.abc import Callable
from ipaddress import ip_address
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from agent6.config._base import MODEL_CONFIG
from agent6.config._providers import ProviderEntry
from agent6.config._sandbox import MCPConfig, SandboxConfig
from agent6.errors import OperatorError
from agent6.types import RoleName


class ConfigError(OperatorError):
    """Raised when the config file is missing, malformed, or fails validation.

    An OperatorError: the config is the operator's file, so ``cli_main``
    presents it as a refusal, never a crash report.
    """


ThinkingLevel = Literal["off", "low", "medium", "high"]
# The review-seat depth (`[review].tier`); ReviewSeat.tier mirrors this, so the
# vocabulary has one owner.
ReviewTier = Literal["diff", "explore"]


class RoleModel(BaseModel):
    """One role's `(provider, model)` assignment.

    `provider` is the name (TOML table key) of an entry in `[providers.*]`.

    `temperature` is the sampling temperature agent6 will pin on every
    call for this role. Defaults to ``0.0``, agent6's tool-use loop is a
    search-and-act feedback loop and high-temperature sampling causes
    observable degeneration on some open-weights models (caught
    Kimi K2.6 emitting 15997 literal ``\\n`` escapes in a single
    ``old_string`` argument before hitting the completion-tokens cap).
    Anthropic and OpenAI models are tuned to behave well at any
    temperature; OpenRouter routes to provider defaults that vary by
    model, so pinning is the only way to make benches reproducible.
    Set to ``null`` only if you specifically want the provider's default
    behaviour. TOML has no null literal and ``temperature = nan`` fails the
    0.0-2.0 bounds, so null is reachable only via the Python API; omitting the
    key leaves the ``0.0`` default, not the provider's default.
    """

    model_config = MODEL_CONFIG

    provider: str = Field(
        min_length=1,
        description="A `[providers.*]` name.",
    )
    model: str = Field(
        min_length=1,
        description="Model id at that provider.",
    )
    temperature: float | None = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Pinned per call (`0.0` to `2.0`). `0.0` keeps tool use stable.",
    )
    # Reasoning/thinking effort for this role. ``None`` leaves the
    # provider default; ``off`` disables it explicitly. Mapped per
    # provider: OpenAI-compatible reasoning models receive a
    # ``reasoning.effort`` knob, Anthropic models receive an
    # ``extended_thinking`` budget. Non-reasoning models ignore it.
    thinking: ThinkingLevel | None = Field(
        default=None,
        description=(
            "Reasoning effort: `off`/`low`/`medium`/`high`. Anthropic maps it to a thinking budget "
            "(≈ 4k/8k/16k tokens); non-reasoning models ignore it."
        ),
    )


class ModelsConfig(BaseModel):
    """Per-role provider + model routing.

    Three roles, all optional:

    - ``worker`` drives the single-loop agent (``agent6 run`` / ``agent6
      resume``); its pricing also drives the USD -> token budget
      conversion.
    - ``planner`` drives ``agent6 plan`` (the read-only planning pass).
      Unset -> falls back to ``worker`` (set it to a frontier model + high
      thinking for careful up-front planning).
    - ``reviewer`` drives the one-shot ``agent6 review`` subcommand and the
      optional in-loop critic. Unset -> falls back to ``worker``.

    Any configured provider may serve any role. Leaving every role unset is
    valid (e.g. a global config that only declares providers); a role is
    only *required* for the command that uses it, checked by
    :meth:`Config.require_runnable`.
    """

    model_config = MODEL_CONFIG

    worker: RoleModel | None = None
    reviewer: RoleModel | None = None
    planner: RoleModel | None = None

    def configured(self) -> dict[str, RoleModel]:
        """Only the roles explicitly set (used for validation/key checks)."""
        out: dict[str, RoleModel] = {}
        if self.worker is not None:
            out["worker"] = self.worker
        if self.reviewer is not None:
            out["reviewer"] = self.reviewer
        if self.planner is not None:
            out["planner"] = self.planner
        return out

    def resolve(self, role: RoleName) -> RoleModel | None:
        """The effective model for *role*, applying worker fallbacks."""
        if role == "worker":
            return self.worker
        if role == "planner":
            return self.planner or self.worker
        if role == "reviewer":
            return self.reviewer or self.worker
        return None

    def source_role(self, role: RoleName) -> RoleName:
        """The configured entry ``resolve(role)`` reads: *role* itself when
        explicitly set, else the worker fallback. Lets an error message name
        the config key the user actually wrote (mirrors ``resolve`` above)."""
        return role if role in self.configured() else "worker"


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


class MetricConfig(BaseModel):
    """Optional continuous-score metric for tasks that have a measurable goal
    (cycles, wall time, kB, bench score) distinct from binary verify pass/fail.

    When configured, ``run_metric_command`` (the metric tool) runs ``command``
    in the jail (same env as ``verify_command``) and parses ``pattern``'s
    first capture group as a number. ``goal = "minimize"`` for things like
    cycles/time; ``"maximize"`` for bench scores. ``pattern`` is a Python
    regex; the FIRST capture group must be a base-10 integer or float. If
    the pattern does not match in the command's combined stdout+stderr the
    metric is treated as missing.
    """

    model_config = MODEL_CONFIG

    command: tuple[str, ...] = Field(
        min_length=1,
        description="argv to run.",
    )
    pattern: str = Field(
        min_length=1,
        description="Regex; first capture group = the number.",
    )
    goal: Literal["minimize", "maximize"] = Field(
        description='`"minimize"` or `"maximize"`.',
    )


class WorkflowConfig(BaseModel):
    model_config = MODEL_CONFIG

    # The command agent6 runs to decide whether a step "succeeded". This is
    # inherently repo-specific, so it has no useful global default and defaults
    # to empty. Optional: `agent6 run`/`plan` infer one per run when it is unset
    # (AGENTS.md -> repo signals -> a cheap LLM call; see agent6.verify_infer),
    # falling back to a gateless run. `agent6 init` can pin one.
    verify_command: tuple[str, ...] = Field(
        default=(),
        description=(
            'argv defining "a step succeeded" (no shell; wrap a pipeline as `["sh","-c","a '
            '&& b"]`). Optional: unset infers per run (AGENTS.md `## Verify command`, then repo '
            "manifests, then a cheap model call), injected in-memory and printed. None "
            "inferable = the run starts gateless; a recognizable project created mid-run "
            "adopts the first resolvable inferred gate. Set it to pin one."
        ),
    )
    # per-call timeout for verify_command (and metric_command) in
    # seconds. Defaults to the jail's general 600s but should be cranked
    # MUCH lower for benches where the verify is a fast correctness test
    # (perf-takehome's CorrectnessTests run in ~2s; a 30s cap detects
    # infinite-loop / quadratic edits 20x faster than the 600s default).
    # Setting too low for slow legitimate tests will cause false-positive
    # failures, so leave at 600 unless the verify is reliably fast.
    verify_timeout_s: float = Field(
        gt=0.0,
        default=600.0,
        description=(
            "Per-call timeout for `verify_command` / `metric.command`. The operator's gate needs a "
            "verdict, so it is bounded; a model-chosen `run_command` is not (see "
            "`command_checkin_s`)."
        ),
    )
    # How long a run_command may run before the model is handed it back as a
    # background job. NOT a timeout: nothing is killed, the command keeps
    # running and the model decides whether to wait, poll or stop it -- a
    # judgement a number cannot make. 0 disables the hand-back (wait while it
    # lives), which is right when a human is watching and can interrupt.
    # 900 because the hand-back is non-destructive, so it can afford to be
    # patient: the cost of being early is a poll cycle of tokens, and the cost
    # of being late is nothing at all.
    command_checkin_s: float = Field(
        ge=0.0,
        default=900.0,
        description=(
            "How long a model's `run_command` may run before it is **handed back** as a "
            "background job. Not a timeout: nothing is killed, the command keeps running, and "
            "the model is told (`returncode: null`, `still_running: true`, a `background_id`) so "
            "it can wait with `read_background`, stop it, or carry on. `0` disables the "
            "hand-back, which is right when a human is watching and can interrupt."
        ),
    )
    # When true, finish_session is refused while the last verify is red (or a verify
    # command is configured but was never run): the worker must get verify green
    # or explicitly stop. Default false keeps finish_session always honorable, but
    # even then a finish over a red verify is reported honestly (session.end
    # all_passed=False -> "finished", never "passed"); this flag turns the honest
    # signal into a hard gate for operators who want it.
    require_verify_to_finish: bool = Field(
        default=False,
        description=(
            "Refuse `finish_session` while the last verify is red or never ran (bounded nudges). "
            'Regardless, a finish over red is always reported "finished", never "passed".'
        ),
    )
    # Opt-in: bounce the FIRST finish_session over a green verify once, with a
    # directive to re-check every spec requirement (the committed suite may
    # cover a subset). Targets the finish-on-green-but-incomplete failure
    # mode measured on bench/coreagent's eventflow task; costs about one
    # extra turn per run when on. See docs/config.md for the measurements.
    spec_recheck_on_finish: bool = Field(
        default=False,
        description=(
            "Bounce the first finish over a green verify once for a spec re-check. Measured "
            "(n=6/arm, 3 models): no gain beyond noise, one score drop, +38-88% cost. Kept off; "
            "candidate for removal."
        ),
    )
    # Optional. None means "no metric; ``run_metric_command`` is unavailable".
    metric: MetricConfig | None = None


class ContextConfig(BaseModel):
    """``[context]`` section: tiered context-compaction thresholds."""

    model_config = MODEL_CONFIG

    # Tiered context-compaction thresholds (approximate chars; tokens ~=
    # chars/4). When cumulative *tool_result* content grows past
    # ``drop_at_chars`` the oldest tool_results are replaced by a
    # short placeholder (the worker can re-call the tool to refetch). When the
    # *whole* context (text + tool_use inputs + surviving tool_results) grows
    # past ``summarise_at_chars`` -- which must be > drop, so tier-2
    # escalates above tier-1 -- the conversation is summarized and restarted
    # (the durable task DAG survives; the restart notice points the worker at
    # ``list_tasks`` to recover task-level state).
    # ``summary_max_tokens`` caps the summarizer's output.
    #
    # Default ``None`` == ADAPTIVE: agent6 sizes both thresholds from the worker
    # model's context window (tier-1 at ~45%, tier-2 at ~80% of it), resolving
    # the window from a bundled table of tested models + the live model cache
    # (see ``models.registry.compaction_thresholds``). Pin them by setting BOTH
    # explicitly (e.g. a self-hosted model agent6 can't size); leave BOTH unset
    # to stay adaptive. When the window is unknown the historical 256k/768k
    # fixed defaults apply.
    drop_at_chars: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Tier 1: oldest tool results become placeholders. Unset sizes from the worker's "
            "context window (~45%); set BOTH thresholds to pin."
        ),
    )
    summarise_at_chars: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Tier 2: summarise elided history and restart (the task DAG survives). Unset ≈ 80% of "
            "the window. Must exceed `drop_at_chars`."
        ),
    )
    summary_max_tokens: int = Field(
        gt=0,
        default=2048,
        description="Cap on the tier-2 summary (and gist distillation calls).",
    )
    # Tier-1 gist elision: a large read_file result about to be elided decays
    # to a placeholder carrying a model-written gist of the file first (one
    # batched reviewer-model call per drop event), then to the bare marker
    # under continued pressure. Measured on the longhorizon bench: bare
    # elision of reference docs halves a retention task's score under a small
    # window. False = straight to bare markers (no distiller calls).
    elision_gists: bool = Field(
        default=True,
        description=(
            "Tier 1 decays a large `read_file` to a model-written gist before the bare marker "
            "(demoted under continued pressure so the byte bound holds). `false` = straight to "
            "bare markers."
        ),
    )

    @model_validator(mode="after")
    def _check_compaction_thresholds(self) -> ContextConfig:
        drop, summarise = self.drop_at_chars, self.summarise_at_chars
        if (drop is None) != (summarise is None):
            raise ValueError(
                "set BOTH context.drop_at_chars and"
                " summarise_at_chars, or NEITHER (neither == adaptive,"
                " sized from the worker model's context window). Both at once:"
                " agent6 config set context"
                " '{ drop_at_chars = 200000, summarise_at_chars = 400000 }'"
            )
        if drop is not None and summarise is not None and summarise <= drop:
            raise ValueError(
                "context.summarise_at_chars"
                f" ({summarise}) must be greater than"
                f" drop_at_chars ({drop}): tier-2"
                " summarise must escalate above tier-1 elision."
            )
        return self


class PromptConfig(BaseModel):
    """``[prompt]`` section: system-prompt override, structural priors, and
    one-shot task-prompt revision."""

    model_config = MODEL_CONFIG

    # ADVANCED: replace run-mode's static base system prompt (role + edit/tool-use/
    # dag/scope rules) with the contents of this file. The dynamic blocks (verify,
    # metric, budget, repo-priors + AGENTS.md) still append, so repo context and
    # the budget cap are preserved. Empty = the built-in default. You own keeping
    # the tool contracts intact (apply_edit/apply_patch, run_verify_command,
    # finish_session); run startup warns if the override omits them. Inspect the
    # assembled result with `agent6 prompt show`.
    system_prompt_file: str = Field(
        default="",
        description=(
            "ADVANCED: replace run-mode's static base prompt with this file (dynamic blocks still "
            "append). Warned at startup if core tool names are missing."
        ),
    )
    # Include the structural-prior blocks in the run-mode <repo-priors>: hot
    # symbols (cross-file reference ranking), git co-change pairs, and the
    # tree-sitter symbol outline. Default on. Set false for a leaner/cheaper
    # prompt that relies purely on on-demand exploration (outline/find_definition)
    # -- the base repo map + AGENTS.md still ship.
    structural_priors: bool = Field(
        default=True,
        description=(
            "Include the `<repo-priors>` block (hot symbols, co-change, outline). `false` for a "
            "leaner prompt."
        ),
    )
    # one-shot task prompt revision before the worker loop starts.
    # Reuses the reviewer model, takes no tools, and is budget-tracked like
    # any other provider call. Default off: crisp prompts and frontier models
    # do not need revision.
    revise_prompt: Literal["off", "auto", "interactive"] = Field(
        default="off",
        description=(
            "One-shot task-prompt revision before the loop: `off` / `auto` / `interactive`."
        ),
    )
    # Front-load task decomposition (run mode). When on the worker's system
    # prompt swaps the "DAG is optional" guidance for a "decompose first"
    # directive: lay the task out as ordered subtasks before editing, then work
    # one focused subtask at a time (the existing surface-current-task and
    # finish-gate machinery walks the frontier). Helps small/open models that
    # lose track of multi-part tasks; a capable model decomposes implicitly and
    # only pays the 2-4x turn overhead. "auto" (default) enables it ONLY for
    # worker models with a measured win in the capability registry
    # (models.registry.decompose_default); the CLI pins auto to on/off at run
    # start via ``with_decompose``, and the engine treats any value other than
    # "on" as off. No effect on plan/ask/machine/agent modes. See
    # docs/config.md for the measured per-model effect.
    decompose: Literal["auto", "on", "off"] = Field(
        default="auto",
        description=(
            "Front-load task decomposition (run mode): `on` helps small models that under-finish "
            "multi-part tasks (measured on mistral-small; capable models just pay 2-4x overhead). "
            "`auto` resolves per worker model from the capability registry; `config show` displays "
            "the resolved value. `--decompose` forces one run."
        ),
    )

    @model_validator(mode="after")
    def _check_system_prompt_file(self) -> PromptConfig:
        # Fail loud at config time if the override path is set but missing, rather
        # than silently falling back to the default prompt at run start.
        if self.system_prompt_file:
            p = Path(self.system_prompt_file).expanduser()
            if not p.is_file():
                raise ValueError(f"prompt.system_prompt_file: not a readable file: {p}")
        return self


class SkillsConfig(BaseModel):
    """``[skills]`` section: operator-installed SKILL.md packs (agentskills.io).

    Skills live under ``<data-dir>/skills/<name>/`` (``agent6 skills install``)
    plus any ``extra_dirs``. Installed means enabled: the run-mode system
    prompt lists each enabled skill's name + description and the worker loads
    content on demand; the ``state`` map holds only the exceptions. Skills are
    trusted like config (operator-chosen prompt content); nothing in a skill
    is ever executed by the loader.
    """

    model_config = MODEL_CONFIG

    # Master switch for the whole subsystem. Off = no index block, no
    # use_skill tool, slash commands don't register.
    enabled: bool = Field(
        default=True,
        description="Master switch: off = no index, no `use_skill`, no slash commands.",
    )
    # Additional skill directories scanned BEFORE the installed dir (a local
    # checkout during skill development wins over an installed copy). Each may
    # hold skill subdirectories or be a single skill dir itself.
    extra_dirs: tuple[str, ...] = Field(
        default=(),
        description="Additional skill dirs, scanned BEFORE the installed dir.",
    )
    # Per-skill exceptions, one value per skill so contradictory states are
    # unrepresentable: "disabled" drops it from the index; "always" injects
    # the full SKILL.md text into the system prompt instead of indexing it.
    # Absent = "enabled". Layered configs merge this map key-wise, so a repo
    # config can flip one skill without restating the rest.
    state: dict[str, Literal["enabled", "disabled", "always"]] = Field(
        default_factory=dict,
        description=(
            'Per-skill: `"disabled"` drops it; `"always"` injects the full text into the '
            "system prompt. Layers merge key-wise; `agent6 skills enable/disable [--repo]` "
            "writes it."
        ),
    )


class ReviewConfig(BaseModel):
    """``[review]`` section: critic-in-loop trigger + the adversarial review panel."""

    model_config = MODEL_CONFIG

    # critic-in-loop. When != "off", Workflow runs the
    # ``reviewer`` model as a critic at the chosen trigger and injects
    # its critique as a user message the worker sees next turn.
    #   off              - never (default; behaviour unchanged).
    #   on_verify_fail   - after every verify failure.
    #   before_finish    - intercept ``finish_session``; reject if critic
    #                      is not satisfied and inject critique.
    #   periodic         - every ``period`` iterations.
    # The reviewer provider must already be configured in
    # ``[models.reviewer]`` (same one ``agent6 review`` uses).
    trigger: Literal["off", "on_verify_fail", "before_finish", "periodic"] = Field(
        default="off",
        description=(
            "In-loop review panel trigger: `off` / `on_verify_fail` / `before_finish` / `periodic`."
        ),
    )
    period: int = Field(
        ge=1,
        default=10,
        description="Iterations between reviews for `periodic`.",
    )
    # Adversarial review panel (opt-in). ``seats`` is THE roster: flat
    # "persona[@provider/model]" strings (e.g. "security" routes via
    # [models.reviewer]; "security@openrouter/moonshotai/kimi-k2" pins a
    # model). The `agent6 review --reviewers N`/`--personas` flags synthesize
    # an in-memory equivalent. ``decision`` is only a GATE in-loop; "advisory"
    # (default) just injects findings as guidance and never blocks.
    decision: Literal["advisory", "veto", "quorum", "all"] = Field(
        default="advisory",
        description="`advisory` (inject findings, never block) / `veto` / `quorum` / `all`.",
    )
    quorum: int = Field(
        ge=1,
        default=2,
        description="K for `quorum`; counts distinct MODELS, so same-model seats can't fake it.",
    )
    # Per-run cap on total panel blocks before the gate auto-downgrades to
    # advisory for the rest of the run (so a gating panel can never stall forever).
    max_total_rejections: int = Field(
        ge=1,
        default=4,
        description="Per-run blocks before the gate auto-disarms to advisory.",
    )
    # Budget floor: the in-loop review panel is SKIPPED (approve-and-proceed) once
    # the run's remaining token budget falls below this fraction -- reviewing costs
    # most exactly when budget is scarcest. Default 0.25 = skip the panel in the
    # last quarter of the budget.
    budget_fraction: float = Field(
        gt=0.0,
        le=1.0,
        default=0.25,
        description="Skip the in-loop panel once remaining budget falls below this fraction.",
    )
    seats: tuple[str, ...] = Field(
        default=(),
        description=(
            'Panel roster: `"persona"` routes via `[models.reviewer]`; '
            '`"persona@provider/model"` pins a model per seat. `agent6 review --reviewers N '
            "[--personas …]` synthesizes an equivalent."
        ),
    )
    # Seat concurrency for the in-loop panel (1 = sequential). The post-hoc
    # `agent6 review` runs all seats in parallel regardless (fast one-shot).
    concurrency: int = Field(
        ge=1,
        default=1,
        description="In-loop seat parallelism (post-hoc `agent6 review` is always parallel).",
    )
    # Reviewer tier: "diff" (one grounded call over the diff) or "explore" (a
    # read-only tool-using mini-loop that reads the broader repo first to catch
    # cross-file impact). explore is more thorough but costs several calls/seat.
    tier: ReviewTier = Field(
        default="diff",
        description=(
            "`diff` (one grounded call over the diff) or `explore` (read-only tool-using reviewer, "
            "cross-file)."
        ),
    )

    @model_validator(mode="after")
    def _check_review_seats(self) -> ReviewConfig:
        # Each seats entry is "persona", "persona@provider/model", or
        # "@provider/model"; an "@" form must name BOTH a provider and a model so
        # a typo doesn't silently degrade to the reviewer route.
        for spec in self.seats:
            if not spec.strip():
                raise ValueError("review.seats entries must be non-empty")
            _persona, sep, route = spec.partition("@")
            if sep:
                provider, slash, model = route.partition("/")
                if not (provider.strip() and slash and model.strip()):
                    raise ValueError(
                        f"review.seats: {spec!r} must be"
                        " 'persona@provider/model' (both provider and model required)"
                    )
        return self

    @model_validator(mode="after")
    def _check_review_quorum(self) -> ReviewConfig:
        if self.decision == "quorum" and self.quorum > 1:
            models = {(s.partition("@")[2].strip() if "@" in s else "") for s in self.seats}
            if len(models) < self.quorum:
                raise ValueError(
                    f"review.decision='quorum' with quorum={self.quorum}"
                    f" needs >= {self.quorum} DISTINCT models (the gate counts one block per"
                    " distinct model). Provide them via seats"
                    " ('persona@provider/model'), or use decision='veto'."
                )
        return self


class BudgetConfig(BaseModel):
    """``[budget]``: every provider call is bounded in exactly ONE currency.

    A call the runtime can meter (provider-reported cost, else price x tokens
    at the model's fetched rates, cache-aware) counts against ``max_usd``; a
    call it cannot price counts its input+output tokens against
    ``max_tokens_fallback``. Both fields share one rule: ``-1`` = unlimited,
    ``0`` = refuse calls in that ledger up front (``max_tokens_fallback = 0``
    means never run an unmeterable model), ``> 0`` = the cap. Hitting a cap
    ends the run resumably (``budget_exhausted``); each resumed leg gets a
    fresh budget. The ``--max-usd`` / ``--max-tokens-fallback`` flags override
    per run."""

    model_config = MODEL_CONFIG

    max_usd: float = Field(
        default=10.0,
        description="Cap on metered spend (cache-aware, per model).",
    )
    max_tokens_fallback: int = Field(
        ge=-1,
        default=2_000_000,
        description="Token cap for UNMETERED calls only (local models, price gaps).",
    )

    @field_validator("max_usd")
    @classmethod
    def _usd_unlimited_is_exactly_minus_one(cls, v: float) -> float:
        # Non-finite never binds (nan fails every comparison; inf exceeds any
        # spend), which would silently disable the hard budget.
        if not math.isfinite(v) or (v < 0 and v != -1):
            raise ValueError("max_usd is a finite cap >= 0, or exactly -1 for unlimited")
        return v


class MachineNotifyConfig(BaseModel):
    """Optional out-of-band notify hook for a running machine.

    When ``on_event`` is set, `agent6 machine run` runs the argv tuple on each
    `machine.notify` (a state's ``notify`` message) and on the terminal
    `machine.end`, on the host OUTSIDE the jail (mirror of
    ``[notify].on_complete``). The argv is operator-controlled and never
    includes LLM output. Env vars passed:

    - ``AGENT6_MACHINE_ID``      , the machine id
    - ``AGENT6_MACHINE_DIR``     , absolute path to the instance dir
    - ``AGENT6_MACHINE_EVENT``   , ``notify`` or ``end``
    - ``AGENT6_MACHINE_STATE``   , the state that emitted it
    - ``AGENT6_MACHINE_MESSAGE`` , the notify message (or the end reason)
    - ``AGENT6_MACHINE_LEVEL``   , ``info``/``warn``/``error`` for notify, or the
                                   ``ok``/``failed`` status for end

    Use it to fan out to a phone (ntfy/Pushover/Telegram/email); agent6 owns no
    push infra. A failed hook is logged and does not change the exit code.
    """

    model_config = MODEL_CONFIG

    on_event: tuple[str, ...] = Field(
        default=(),
        description="argv per notify/end (empty = disabled).",
    )
    timeout_s: float = Field(
        gt=0.0,
        default=30.0,
        description="Hook timeout.",
    )


class MachineConfig(BaseModel):
    """State-machine runtime knobs (`agent6 machine run`)."""

    model_config = MODEL_CONFIG

    # How many recent blackboard snapshots to keep per machine instance.
    # Recovery only reads the latest and `machine replay` rebuilds from the
    # journal, so old snapshots are an audit convenience, not state. 0 keeps
    # every snapshot (one file per transition; budget disk accordingly for
    # long-running machines).
    snapshot_keep: int = Field(
        ge=0,
        default=5,
        description=(
            "Blackboard snapshots kept per instance (recovery reads only the latest; `machine "
            "replay` rebuilds from the journal). `0` keeps all."
        ),
    )
    notify: MachineNotifyConfig = Field(default_factory=MachineNotifyConfig)


def is_loopback_host(host: str) -> bool:
    """True iff *host* is a loopback bind (the one source of truth for the web
    UI's secure-by-default gate; a wildcard like 0.0.0.0/:: is NOT loopback)."""
    normalized = host.strip()
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    if normalized.lower() == "localhost":
        return True
    try:
        return ip_address(normalized).is_loopback
    except ValueError:
        return False


class WebConfig(BaseModel):
    """`agent6 web` server bind. Secure by default: loopback only.

    Remote access is expected behind `tailscale serve` (HTTPS + WireGuard) in
    front of the loopback bind; the tailnet identity is the access control, so
    there is no app-level auth. Binding a non-loopback address exposes the write
    surface (spawn runs, answer prompts) to anyone who can reach the port, so it
    is gated behind `allow_non_loopback = true` and carries no default.
    """

    model_config = MODEL_CONFIG

    host: str = Field(
        default="127.0.0.1",
        description="Bind address; non-loopback requires `allow_non_loopback = true`.",
    )
    port: int = Field(
        ge=1,
        le=65535,
        default=7658,
        description="Listen port.",
    )
    # Opt-in required to bind a non-loopback host. Off by default so a typo or a
    # copied config can never silently expose the agent to the local network.
    allow_non_loopback: bool = Field(
        default=False,
        description=(
            "Opt-in for a non-loopback bind, so a typo can never silently expose the write surface."
        ),
    )

    @model_validator(mode="after")
    def _guard_non_loopback(self) -> WebConfig:
        if not is_loopback_host(self.host) and not self.allow_non_loopback:
            raise ValueError(
                f"[web].host = {self.host!r} is not loopback. Binding a non-loopback"
                " address exposes the web UI's write surface; set [web]"
                " allow_non_loopback = true to opt in (and prefer `tailscale serve`"
                " in front of a 127.0.0.1 bind instead)."
            )
        return self


class Agent6Section(BaseModel):
    model_config = MODEL_CONFIG

    config_version: int = Field(
        ge=1,
        le=1,
        default=1,
        description="Config schema version (must be `1`).",
    )
    # Absolute base directory for per-repo agent6 state (this per-repo config +
    # all run state), which lives OUT of the workspace under ``<base>/<repo-id>/``
    # (default ``$XDG_STATE_HOME/agent6``; see ``agent6.paths.state_base``). Can
    # ONLY be set in the GLOBAL config: it locates the per-repo config, so a
    # per-repo/flag value would be chicken-and-egg. Must be absolute. Point it
    # at a persisted, out-of-cwd path (e.g. a mounted volume) to keep run state
    # across devcontainer rebuilds.
    state_dir: str | None = Field(
        default=None,
        description=(
            "Absolute base for all per-repo state (`<state_dir>/<repo-id>/`), out of the "
            "workspace. Global-config only; `AGENT6_STATE_HOME` overrides. In a devcontainer the "
            "default is ephemeral: point it at a persisted volume to keep runs across rebuilds."
        ),
    )

    @field_validator("state_dir")
    @classmethod
    def _check_state_dir(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not Path(v).expanduser().is_absolute():
            raise ValueError(f"[agent6].state_dir must be an absolute path, got {v!r}")
        return v


class NotifyConfig(BaseModel):
    """Optional post-run notification hook.

    When ``on_complete`` is set, agent6 runs the argv tuple after the
    workflow returns (``agent6 run`` or ``agent6 resume``). The argv is
    operator-controlled, it never includes LLM output, and runs in the
    user's shell environment, NOT in the jail, with these env vars:

    - ``AGENT6_SESSION_ID``      , session id under the per-repo state dir
    - ``AGENT6_SESSION_OK``      , ``1`` if the workflow finished cleanly, ``0`` otherwise
    - ``AGENT6_SESSION_REASON``  , workflow termination reason (e.g. ``finish_session``,
                                 ``budget_exhausted``, ``provider_error``)
    - ``AGENT6_SESSION_DIR``     , absolute path to the session dir

    Use cases: desktop notification (``notify-send``), shell-bell, ssh
    push notification, mailx, etc. A failure of the notify command is
    logged but does not change the agent6 exit code.
    """

    model_config = MODEL_CONFIG

    on_complete: tuple[str, ...] = Field(
        default=(),
        description="argv to run (empty = disabled).",
    )
    timeout_s: float = Field(
        gt=0.0,
        default=30.0,
        description="Hook timeout.",
    )


class ParallelConfig(BaseModel):
    """``[parallel]`` section: fan-out defaults for `agent6 run --parallel`.

    ``--parallel N`` (or a comma-separated model list) runs N isolated lanes,
    each a disposable clone of the repo, and auto-compares the results. These
    knobs bound and place that fan-out; nothing here mutates the origin repo.
    """

    model_config = MODEL_CONFIG

    # Hard cap on lanes per fan-out. `--parallel` over this refuses up front so a
    # typo (or a long model list) can't spawn an unbounded pile of clones+runs.
    max_lanes: int = Field(
        ge=1,
        default=4,
        description="Hard cap per fan-out; more refuses up front.",
    )
    # Base directory for lane workspaces (each fan-out gets `<workdir>/<fanout-id>/
    # lane-<i>`). "" resolves to `<cache_dir>/parallel`, a regenerable cache the
    # orchestrator cleans up after importing each lane. Point it at a fast disk
    # for large repos.
    workdir: str = Field(
        default="",
        description=(
            'Base dir for lane clones. `""` = `<cache_dir>/parallel`, cleaned up after import.'
        ),
    )


class Config(BaseModel):
    model_config = MODEL_CONFIG

    agent6: Agent6Section = Field(default_factory=Agent6Section)
    providers: dict[str, ProviderEntry] = Field(default_factory=dict)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    git: GitConfig = Field(default_factory=GitConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    machine: MachineConfig = Field(default_factory=MachineConfig)
    notify: NotifyConfig = Field(default_factory=NotifyConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    parallel: ParallelConfig = Field(default_factory=ParallelConfig)
    # Named strategy PRESET: fills in many settings at once (BUILTIN_PRESETS +
    # user `[presets.<name>]`). "" / "standard" = plain defaults; injection
    # order and stacking rules: `config.layer._apply_preset`.
    preset: str = Field(
        default="",
        description=(
            "Named strategy preset: fills many settings at once (see the Presets section). "
            "Top-level because it overrides every section. `agent6 config set preset <name>` "
            "(`--repo`); `--preset` overrides per run."
        ),
    )

    @model_validator(mode="after")
    def _cross_validate_provider_routing(self) -> Config:
        # Only configured roles are checked here, and only when their
        # provider is actually present; an empty/partial config is valid
        # at load time (require_runnable enforces completeness per command).
        for role, rm in self.models.configured().items():
            if self.providers and rm.provider not in self.providers:
                known = ", ".join(sorted(self.providers)) or "(none)"
                raise ValueError(
                    f"models.{role}.provider = {rm.provider!r} but"
                    f" [providers.{rm.provider}] is not configured."
                    f" Known providers: {known}."
                )
        return self

    def with_budget_overrides(
        self,
        *,
        max_usd: float | None = None,
        max_tokens_fallback: int | None = None,
    ) -> Config:
        """Return a copy with budget fields overridden (the per-run CLI flags,
        each writing the config field of the same name). ``None`` keeps the
        existing value."""
        if max_usd is None and max_tokens_fallback is None:
            return self
        data = self.model_dump(mode="python")
        budget = data.setdefault("budget", {})
        if max_usd is not None:
            budget["max_usd"] = max_usd
        if max_tokens_fallback is not None:
            budget["max_tokens_fallback"] = max_tokens_fallback
        return Config.model_validate(data)

    def with_sandbox_overrides(
        self,
        *,
        disable_sandbox: bool = False,
        auto_approve: bool = False,
        no_commands: bool = False,
    ) -> Config:
        """Return a copy with per-invocation sandbox overrides from CLI flags.

        ``disable_sandbox`` forces ``sandbox.isolation = "none"`` (unconfined).
        ``auto_approve`` upgrades ``run_commands`` ``"ask" -> "yes"`` but never
        resurrects a withheld ``"no"`` (a per-invocation flag must not grant a
        capability the standing policy denied); it covers every MCP server's
        ``approve`` too, because "do not prompt me this run" that still prompted
        would not be that. ``no_commands`` pins ``run_commands`` to ``"no"`` and
        always may: tightening needs no permission. All are operator-supplied
        (flag/env); the LLM can reach none of them.
        """
        if not disable_sandbox and not auto_approve and not no_commands:
            return self
        data = self.model_dump(mode="python")
        sandbox = data.setdefault("sandbox", {})
        if disable_sandbox:
            sandbox["isolation"] = "none"
        if auto_approve and self.sandbox.run_commands != "no":
            sandbox["run_commands"] = "yes"
        if auto_approve:
            for server in data.get("mcp", {}).get("servers", {}).values():
                server["approve"] = "yes"
        if no_commands:
            sandbox["run_commands"] = "no"
        return Config.model_validate(data)

    def with_machine_agent_overrides(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        thinking: str | None = None,
        temperature: float | None = None,
        max_usd: float | None = None,
        max_tokens_fallback: int | None = None,
    ) -> Config:
        """Return a copy with a machine ``agent`` state's per-state knobs applied.

        Overrides the ``worker`` role (the role machine agent loops run as)
        and the budget ledgers. ``None`` means "inherit the effective config".
        Re-validates so the provider-name checks run against the merged result."""
        data = self.model_dump(mode="python")
        worker = data.setdefault("models", {}).get("worker")
        if worker is None:
            worker = {}
            data["models"]["worker"] = worker
        if provider is not None:
            worker["provider"] = provider
        if model is not None:
            worker["model"] = model
        if thinking is not None:
            worker["thinking"] = thinking
        if temperature is not None:
            worker["temperature"] = temperature
        budget = data.setdefault("budget", {})
        if max_usd is not None:
            budget["max_usd"] = max_usd
        if max_tokens_fallback is not None:
            budget["max_tokens_fallback"] = max_tokens_fallback
        return Config.model_validate(data)

    def with_verify_command(self, argv: tuple[str, ...]) -> Config:
        """Return a copy whose ``workflow.verify_command`` is *argv*, `()` for
        a gateless run.

        How `agent6 run`/`plan` inject a verify command inferred at run start,
        and how a run whose policy withholds command tools drops the gate it
        could never execute. IN-MEMORY only -- runs never write config; the
        operator is shown what was picked and can pin it explicitly.
        """
        data = self.model_dump(mode="python")
        data.setdefault("workflow", {})["verify_command"] = list(argv)
        return Config.model_validate(data)

    def clamped_for_ask(self) -> Config:
        """Return a copy with ``sandbox.run_commands`` clamped for `agent6 ask`.

        An ask is a question with the operator sitting there, often in a
        directory that is not even a repo, so it must never execute anything
        unwatched: ``"yes"`` becomes ``"ask"``. Only ever tightens -- ``"no"``
        stays refused, because a run can never loosen a boundary the operator
        set. IN-MEMORY only, like ``with_verify_command``: `config show` keeps
        reporting what the operator actually configured.
        """
        if self.sandbox.run_commands != "yes":
            return self
        data = self.model_dump(mode="python")
        data.setdefault("sandbox", {})["run_commands"] = "ask"
        return Config.model_validate(data)

    def with_decompose(self, value: Literal["on", "off"]) -> Config:
        """Return a copy with ``prompt.decompose`` pinned to *value*.

        Used by the CLI to resolve ``"auto"`` (from the model-capability
        registry) before the workflow starts, so the engine only ever sees
        on/off. IN-MEMORY only, like ``with_verify_command``.
        """
        data = self.model_dump(mode="python")
        data.setdefault("prompt", {})["decompose"] = value
        return Config.model_validate(data)

    def require_runnable(self, role: RoleName = "worker") -> None:
        """Raise ConfigError unless *role* can actually run.

        Checks (in order) that a provider is configured and the role resolves
        to a model whose provider exists. Messages point at the command that
        fixes the gap so a fresh user is never stuck. ``verify_command`` is NOT
        required: `agent6 run`/`plan` infer one when unset (and fall back to a
        gateless run if even that fails) -- see ``agent6.verify_infer``.
        """
        if not self.providers:
            raise ConfigError(
                "No providers configured. Run `agent6 connect` to add one"
                " (stored in your global config), or add a [providers.*]"
                " block to the per-repo config."
            )
        rm = self.models.resolve(role)
        if rm is None:
            raise ConfigError(
                f"No model configured for the {role!r} role. Run `agent6 model`"
                " to set it, or add a [models.worker] block to your config."
            )
        if rm.provider not in self.providers:
            known = ", ".join(sorted(self.providers)) or "(none)"
            raise ConfigError(
                f"models.{role}.provider = {rm.provider!r} but [providers.{rm.provider}]"
                f" is not configured. Known providers: {known}."
            )


def _format_validation_error(
    err: ValidationError, source: str, locate: Callable[[str], str | None] | None = None
) -> str:
    lines = [f"Config validation failed: {source}"]
    for issue in err.errors():
        loc = ".".join(str(part) for part in issue["loc"]) or "<root>"
        lines.append(f"  - {loc}: {issue['msg']} (type={issue['type']})")
        if locate is not None and (where := locate(loc)):
            lines.append(where)
    return "\n".join(lines)


def validate_config(
    raw: dict[str, object],
    *,
    source: str = "<config>",
    locate: Callable[[str], str | None] | None = None,
) -> Config:
    """Validate an already-parsed (and possibly layer-merged) config dict.

    Shared by :func:`load_config` and the layered loader
    (``agent6.config.layer``) so both surface identical field-pointing errors.
    ``locate`` maps a dotted leaf to a "which file, how to fix" hint appended to
    its error line, so a stale value in a layered config names its own source.
    """
    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc, source, locate)) from exc


def load_config(path: Path) -> Config:
    """Load and strictly validate the TOML config at *path*.

    Raises ConfigError on any problem; never returns a partially valid config.
    """
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Config file is not valid TOML ({path}): {exc}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"Config file cannot be read ({path}): {exc}") from exc
    return validate_config(raw, source=str(path))
