# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The loop-behaviour models: `[workflow]` (+ its metric), `[review]`,
`[context]`, `[prompt]`, and `[budget]`."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from agent6.config._base import MODEL_CONFIG, Argv, StrTuple

# The review-seat depth (`[review].tier`); ReviewSeat.tier mirrors this, so the
# vocabulary has one owner.
ReviewTier = Literal["diff", "explore"]


class MetricConfig(BaseModel):
    """Optional continuous-score metric for tasks that have a measurable goal
    (cycles, wall time, kB, bench score) distinct from binary verify pass/fail.

    When configured, `run_metric_command` (the metric tool) runs `command`
    in the jail (same env as `verify_command`) and parses `pattern`'s
    first capture group as a number. `goal = "minimize"` for things like
    cycles/time; `"maximize"` for bench scores. `pattern` is a Python
    regex; the FIRST capture group must be a base-10 integer or float. If
    the pattern does not match in the command's combined stdout+stderr the
    metric is treated as missing.
    """

    model_config = MODEL_CONFIG

    command: Argv = Field(
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
    verify_command: Argv = Field(
        default=(),
        description=(
            'argv defining "a step succeeded" (no shell; wrap a pipeline as `["sh","-c","a '
            '&& b"]`). Optional: unset infers per run (AGENTS.md `## Verify command`, then repo '
            "manifests, then a model call over the manifests, skipped when there are none), "
            "injected in-memory and printed. None inferable = the run starts gateless; a "
            "recognizable project created mid-run adopts the first resolvable inferred gate. "
            "Set it to pin one."
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
    metric: MetricConfig | None = Field(
        default=None,
        description="Optional continuous-score metric; unset = no `run_metric_command` tool.",
    )


class ContextConfig(BaseModel):
    """`[context]` section: tiered context-compaction thresholds."""

    model_config = MODEL_CONFIG

    # Tiered context-compaction thresholds (approximate chars; tokens ~=
    # chars/4). When cumulative *tool_result* content grows past
    # `drop_at_chars` the oldest tool_results are replaced by a
    # short placeholder (the worker can re-call the tool to refetch). When the
    # *whole* context (text + tool_use inputs + surviving tool_results) grows
    # past `summarise_at_chars` -- which must be > drop, so tier-2
    # escalates above tier-1 -- the conversation is summarized and restarted
    # (the durable task DAG survives; the restart notice points the worker at
    # `list_tasks` to recover task-level state).
    # `summary_max_tokens` caps the summarizer's output.
    #
    # Default `None` == ADAPTIVE: agent6 sizes both thresholds from the worker
    # model's context window (tier-1 at ~45% of it, tier-2 at the window
    # minus a 16k-token reserve), resolving
    # the window from a bundled table of tested models + the live model cache
    # (see `models.registry.compaction_thresholds`). Pin them by setting BOTH
    # explicitly (e.g. a self-hosted model agent6 can't size); leave BOTH unset
    # to stay adaptive. When the window is unknown, fixed 256k/768k
    # defaults apply.
    drop_at_chars: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Tier 1: oldest tool results become placeholders. Unset sizes from the worker's "
            "context window (~45%); set both thresholds to pin."
        ),
    )
    summarise_at_chars: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Tier 2: summarise elided history and restart (the task DAG survives). Unset = the "
            "window minus a 16k-token reserve. Must exceed `drop_at_chars`."
        ),
    )
    keep_recent_chars: int = Field(
        ge=0,
        default=80_000,
        description=(
            "Verbatim recent-history tail kept through a tier-2 restart (chars; 0 keeps none)."
        ),
    )
    keep_thinking_turns: int = Field(
        ge=0,
        default=0,
        description=(
            "Drop thinking from assistant turns older than N assistant turns, at tier-1 "
            "moments. `0` (default) keeps all thinking, matching pi; Claude Code clears old "
            "thinking. Anthropic-format providers only: the openai wire never re-sends thinking."
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
                "set both context.drop_at_chars and"
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
    """`[prompt]` section: system-prompt override, structural priors, and
    one-shot task-prompt revision."""

    model_config = MODEL_CONFIG

    # Advanced: replace run-mode's static base system prompt (role + edit/tool-use/
    # dag/scope rules) with the contents of this file. The dynamic blocks (verify,
    # metric, budget, repo-priors + AGENTS.md) still append, so repo context and
    # the budget cap are preserved. Empty = the built-in default. You own keeping
    # the tool contracts intact (apply_edit/apply_patch, run_verify_command,
    # finish_session); run startup warns if the override omits them. Inspect the
    # assembled result with `agent6 prompt show`.
    system_prompt_file: str = Field(
        default="",
        description=(
            "Advanced: replace run-mode's static base prompt with this file (dynamic blocks still "
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
    # start via `with_decompose`, and the engine treats any value other than
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


class ReviewConfig(BaseModel):
    """`[review]` section: the in-loop review panel and its trigger."""

    model_config = MODEL_CONFIG

    # When != "off", Workflow runs the review panel at the chosen trigger and
    # injects its findings as a user message the worker sees next turn. With no
    # `seats`, the panel is one seat on `[models.reviewer]` (same route
    # `agent6 review` uses).
    #   off              - never (default).
    #   on_verify_fail   - after every verify failure.
    #   before_finish    - intercept `finish_session`; a gating `decision`
    #                      rejects the finish while the panel is unsatisfied.
    #   periodic         - every `period` iterations.
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
    # `seats` is THE roster: flat
    # "persona[@provider/model]" strings (e.g. "security" routes via
    # [models.reviewer]; "security@openrouter/moonshotai/kimi-k2" pins a
    # model). The `agent6 review --reviewers N`/`--personas` flags synthesize
    # an in-memory equivalent. `decision` is only a GATE in-loop; "advisory"
    # (default) just injects findings as guidance and never blocks.
    decision: Literal["advisory", "veto", "quorum", "all"] = Field(
        default="advisory",
        description="`advisory` (inject findings, never block) / `veto` / `quorum` / `all`.",
    )
    quorum: int = Field(
        ge=1,
        default=2,
        description="K for `quorum`; counts distinct models, so same-model seats can't fake it.",
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
    seats: StrTuple = Field(
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
    """`[budget]`: every provider call is bounded in exactly ONE currency.

    A call the runtime can meter (provider-reported cost, else price x tokens
    at the model's fetched rates, cache-aware) counts against `max_usd`; a
    call it cannot price counts its input+output tokens against
    `max_tokens_fallback`. Both fields share one rule: `-1` = unlimited,
    `0` = refuse calls in that ledger up front (`max_tokens_fallback = 0`
    means never run an unmeterable model), `> 0` = the cap. Hitting a cap
    ends the run resumably (`budget_exhausted`); each resumed leg gets a
    fresh budget. The `--max-usd` / `--max-tokens-fallback` flags override
    per run."""

    model_config = MODEL_CONFIG

    max_usd: float = Field(
        default=10.0,
        description="Cap on metered spend (cache-aware, per model).",
    )
    max_tokens_fallback: int = Field(
        ge=-1,
        default=2_000_000,
        description="Token cap for unmetered calls only (local models, price gaps).",
    )

    @field_validator("max_usd")
    @classmethod
    def _usd_unlimited_is_exactly_minus_one(cls, v: float) -> float:
        # Non-finite never binds (nan fails every comparison; inf exceeds any
        # spend), which would silently disable the hard budget.
        if not math.isfinite(v) or (v < 0 and v != -1):
            raise ValueError("max_usd is a finite cap >= 0, or exactly -1 for unlimited")
        return v
