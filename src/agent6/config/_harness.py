# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The loop-behaviour models.

`[harness]` with its metric, `[review]`, `[context]`, `[prompt]` and `[budget]`.
"""

from __future__ import annotations

import math
import pathlib
from typing import Literal

import pydantic

from agent6.config import _base


def parse_seat_spec(spec: str) -> tuple[str, str, str]:
    """Split a review seat spec, `persona[@provider/model]`, into its three parts.

    Only the first `/` after `@` splits provider from model, so the model may contain `/`.
    `"security"` gives `("security", "", "")`, routed via the reviewer role;
    `"@anthropic/claude-opus-4-8"` gives `("", "anthropic", "claude-opus-4-8")`.

    Args:
        spec: The seat spec.

    Returns:
        `(persona, provider, model)`, each stripped, the route parts empty without an `@`.

    Raises:
        ValueError: An `@` form names no provider or no model (a typo must not degrade to the
            reviewer route in silence).
    """
    persona, sep, route = spec.partition("@")
    if not sep:
        return (spec.strip(), "", "")
    provider, slash, model = route.partition("/")
    if not (provider.strip() and slash and model.strip()):
        raise ValueError(
            f"{spec!r} must be 'persona@provider/model' (both provider and model required)"
        )
    return (persona.strip(), provider.strip(), model.strip())


# The review-seat depth; ReviewSeat.tier mirrors this, so the vocabulary has one owner.
ReviewTier = Literal["diff", "explore"]


class MetricConfig(pydantic.BaseModel):
    """The `[harness.metric]` table: a continuous score beside the pass/fail gate.

    `run_metric_command` runs `command` in the jail with `verify_command`'s environment and
    parses `pattern`'s first capture group as a base-10 number; no match in the combined
    stdout and stderr reads as a missing metric.
    """

    model_config = _base.MODEL_CONFIG

    command: _base.Argv = pydantic.Field(
        min_length=1,
        description=(
            "The command that prints the score, as argv (no shell). Runs after every "
            "verify-passing edit, and on the model's `run_metric_command` call."
        ),
    )
    pattern: str = pydantic.Field(
        min_length=1,
        description=(
            "A regular expression over the command's output; its first capture group is the "
            'number, e.g. `"score: ([0-9.]+)"`.'
        ),
    )
    goal: Literal["minimize", "maximize"] = pydantic.Field(
        description=(
            "Which way is better: `minimize` (a smaller number wins) or `maximize`. The run "
            "reports the trajectory and can finish once a verified edit only ties the best."
        ),
    )


class HarnessConfig(pydantic.BaseModel):
    """The `[harness]` table."""

    model_config = _base.MODEL_CONFIG

    # Repo-specific, so no global default; the inference lives in agent6.verify_infer.
    verify_command: _base.Argv = pydantic.Field(
        default=(),
        description=(
            "The command that decides whether a step succeeded, as argv (no shell; wrap a pipeline "
            'as `["sh", "-c", "a && b"]`). Set it to pin the gate. Unset: each run infers one and '
            "prints it (an AGENTS.md `## Verify command` block first, then a root `verify.sh`, "
            "the repo's manifest files, and loose `test_*.py` files, then a model call over "
            "those manifests); a run that can infer none starts gateless and adopts the first "
            "gate a recognizable project created mid-run yields."
        ),
    )
    verify_infer: bool = pydantic.Field(
        default=True,
        description=(
            "Infer a verify command when `verify_command` is unset (AGENTS.md fence, repo "
            "signals, a model call), and adopt one mid-run when a gateless run materializes a "
            "recognizable project; an adopted gate that cannot run (exit 127, or the module "
            "its `-m` names is missing) is dropped again, never re-adopted. false: such a run "
            "stays gateless, no inference and no adoption; a set `verify_command` is unaffected."
        ),
    )
    # Matches the jail's general 600s; a bench with a 2s gate detects a runaway edit sooner at 30.
    verify_timeout_s: float = pydantic.Field(
        gt=0.0,
        default=600.0,
        description=(
            "Seconds one `verify_command` or `metric.command` call may take before it is killed "
            "and counted as failed. A pytest gate naming no paths that overruns this budget "
            "(the harness's run or the model's own `run_verify_command`) re-runs scoped to the "
            "test files nearest the run's diff, and harness gates run scoped until a full run "
            "of the gate passes; a scoped green ends the run `passed · scoped gate`. A "
            "model-chosen `run_command` is not bounded (see `command_checkin_s`)."
        ),
    )
    # Per execution: a standing run is not capped by the sum of its executions.
    max_iterations: int = pydantic.Field(
        default=200,
        description=(
            "Assistant turns one execution may take before the run stops with reason "
            "`max_iterations`; -1 is unlimited. A resumed execution gets a fresh allowance."
        ),
    )

    @pydantic.field_validator("max_iterations")
    @classmethod
    def _iterations_unlimited_is_exactly_minus_one(cls, v: int) -> int:
        """Refuse a cap that is neither positive nor exactly -1.

        Args:
            v: The configured cap.

        Returns:
            The cap unchanged.

        Raises:
            ValueError: The cap is 0 or below -1.
        """
        if v == 0 or v < -1:
            raise ValueError("max_iterations is >= 1, or exactly -1 for unlimited")
        return v

    # The loop's guards: the empty turn, the repeated call, the long silence.
    went_quiet_max_nudges: int = pydantic.Field(
        default=4,
        ge=0,
        description=(
            "Empty turns (no text, no tool call) re-asked per streak, reasoning-starvation "
            "bursts included; 0 ends the run on the first."
        ),
    )
    loop_guard_kill_threshold: int = pydantic.Field(
        default=10,
        ge=0,
        description=(
            "The same (tool, args) call this many times in a row ends the run as "
            "`loop_guard_killed` (the notice fires from three, every other turn); 0 leaves "
            "the notice alone."
        ),
    )
    stagnation_notice_after_s: float = pydantic.Field(
        default=300.0,
        ge=0.0,
        description=(
            "Seconds of wall clock with no edit and no verify before one notice (a recall "
            "spiral makes few calls with long reasoning between them); 0 disables."
        ),
    )

    # 900 because the hand-back is non-destructive: handing back early costs one poll cycle.
    command_checkin_s: float = pydantic.Field(
        ge=0.0,
        default=900.0,
        description=(
            "Seconds a model's `run_command` may run before it is handed back as a background job. "
            "Not a timeout: nothing is killed, the command keeps running, and the model is told "
            "(`returncode: null`, `still_running: true`, a `background_id`) so it can wait with "
            "`read_background`, stop it, or carry on. `0` disables the hand-back."
        ),
    )
    standing_patience: int = pydantic.Field(
        ge=-1,
        default=-1,
        description=(
            "Consecutive fruitless standing-goal re-entries (rounds with no executed tool call) "
            "the run absorbs before soft ends are honoured. `-1`: never on its own (the run ends "
            "on its budget, iteration cap, or an operator stop); `0`: the first fruitless round "
            "ends it; `N`: N fruitless re-entries get an escalating nudge, then ends are "
            "honoured. A round that lands work resets the streak."
        ),
    )
    verify_when: Literal["finish", "step", "never"] = pydantic.Field(
        default="finish",
        description=(
            "When the harness runs `verify_command` itself: `finish` (when the model calls "
            "`finish_session` and the tree changed since the last green run), `step` (also after "
            "every turn that edits the tree), `never` (only the model's own `run_verify_command` "
            "calls run it). The tool stays available in every mode; a run with no verify command "
            "has no gate to run."
        ),
    )
    verify_retries: int = pydantic.Field(
        ge=0,
        default=2,
        description=(
            "How many times a red finish certification returns to the model with the gate's "
            "output before the finish stands and the run reads `finished · gate red`, never "
            "passed. `0`: the first red ends the run. A gate that was red before the run touched "
            "anything is not returned unless this run has since made it green."
        ),
    )
    metric: MetricConfig | None = pydantic.Field(
        default=None,
        description=(
            "An optional score to iterate on beside the pass/fail gate (a benchmark, a size, a "
            "count): the run calls it after every verify-passing edit and shows the model the "
            "trend. Unset: `run_metric_command` stays on the tool list and refuses, naming this "
            "key."
        ),
    )


class ContextConfig(pydantic.BaseModel):
    """`[context]` section: tiered context-compaction thresholds."""

    model_config = _base.MODEL_CONFIG

    # Unset thresholds are sized in models.registry.compaction_thresholds.
    drop_at_chars: int | None = pydantic.Field(
        default=None,
        gt=0,
        description=(
            "Tier-1 compaction threshold: once the accumulated tool results exceed this many "
            "characters (about 4 per token), the oldest results are replaced by short placeholders "
            "the model can re-fetch. Unset: sized from the model's context window (about 45% of "
            "it); set both thresholds to pin them."
        ),
    )
    summarise_at_chars: int | None = pydantic.Field(
        default=None,
        gt=0,
        description=(
            "Tier-2 compaction threshold: once the whole context exceeds this many characters, the "
            "elided history is summarized and the conversation restarts on the summary (the task "
            "DAG survives). Unset: the model's window minus a 16k-token reserve. Must exceed "
            "`drop_at_chars`."
        ),
    )
    keep_recent_chars: int = pydantic.Field(
        ge=0,
        default=80_000,
        description=(
            "How many characters of the most recent history a tier-2 restart keeps verbatim after "
            "the summary. `0` keeps none."
        ),
    )
    keep_thinking_turns: int = pydantic.Field(
        ge=0,
        default=0,
        description=(
            "At tier-1 moments, drop the model's thinking from assistant turns older than this "
            "many turns. `0` keeps all thinking. Wires that re-send thinking (Anthropic's signed "
            "blocks, ChatGPT's reasoning items) replay less; the OpenAI wire never re-sends it."
        ),
    )
    summary_max_tokens: int = pydantic.Field(
        gt=0,
        default=2048,
        description=(
            "Cap on the tokens a tier-2 summary (and a gist distillation) may produce. A "
            "reasoning model's per-call floor (room for its reasoning tokens) overrides a "
            "smaller cap, and the chatgpt backend takes no cap."
        ),
    )
    # One batched reviewer-model call per drop event.
    # Measured on the longhorizon bench: bare elision halves a retention score under a small window.
    elision_gists: bool = pydantic.Field(
        default=True,
        description=(
            "At tier 1, replace a large `read_file` result with a model-written gist before the "
            "bare placeholder (the gist is dropped too under continued pressure, so the byte bound "
            "holds). `false`: straight to bare placeholders."
        ),
    )

    @pydantic.model_validator(mode="after")
    def _check_compaction_thresholds(self) -> ContextConfig:
        """Refuse thresholds that are half set or do not escalate.

        Returns:
            The model unchanged.

        Raises:
            ValueError: One threshold set without the other, tier 2 at or below tier 1, or tier 2
                at or below the verbatim tail.
        """
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
        if summarise is not None and summarise <= self.keep_recent_chars:
            raise ValueError(
                f"context.summarise_at_chars ({summarise}) must be greater than"
                f" keep_recent_chars ({self.keep_recent_chars}): the verbatim tail"
                " alone would re-trigger tier 2 after every restart."
            )
        return self


class PromptConfig(pydantic.BaseModel):
    """The `[prompt]` table: the system-prompt override, task revision and decomposition."""

    model_config = _base.MODEL_CONFIG

    system_prompt_file: str = pydantic.Field(
        default="",
        description=(
            "Path of a file that replaces run mode's built-in base system prompt (the dynamic "
            "blocks still append). The tool contracts become yours to state; a file missing the "
            "core tool names is warned about at startup. Empty: the built-in base. `agent6 prompt "
            "show` prints the assembled prompt, the tool definitions, and the first message."
        ),
    )
    # The revision call takes no tools and counts against the budget like any provider call.
    revise_prompt: Literal["off", "auto", "interactive"] = pydantic.Field(
        default="off",
        description=(
            "Rewrite the task prompt once with the reviewer model before the loop starts: `off`, "
            "`auto` (the revision is used as written), or `interactive` (you accept, keep the "
            "original, edit, or quit, which stops the run; needs a terminal to answer at, so a "
            "run under the TUI, an ACP client or a spawned lane skips it). A task queued into a "
            "live run (`/task`) gets the same pass, revised as `auto` does."
        ),
    )
    # `auto` resolves per model in models.registry.decompose_default; the engine reads only `on`.
    decompose: Literal["auto", "on", "off"] = pydantic.Field(
        default="auto",
        description=(
            "Front-load task decomposition in run mode: the model lays the task out as ordered DAG "
            "subtasks before editing and works them one at a time. `on` helps small models that "
            "under-finish multi-part tasks (measured on mistral-small; capable models just pay "
            "2-4x overhead), `off` never, `auto` decides per worker model from the capability "
            "registry (`config show` prints the resolved value). `--decompose` forces it for one "
            "run."
        ),
    )

    @pydantic.model_validator(mode="after")
    def _check_system_prompt_file(self) -> PromptConfig:
        """Refuse an override path that is not a file, at config time rather than at run start.

        Returns:
            The model unchanged.

        Raises:
            ValueError: `system_prompt_file` is set and is not a readable file.
        """
        if self.system_prompt_file:
            p = pathlib.Path(self.system_prompt_file).expanduser()
            if not p.is_file():
                raise ValueError(f"prompt.system_prompt_file: not a readable file: {p}")
        return self


class ReviewConfig(pydantic.BaseModel):
    """The `[review]` table: the in-loop review panel and its trigger."""

    model_config = _base.MODEL_CONFIG

    # The findings reach the model as a user message on its next turn.
    trigger: Literal["off", "on_verify_fail", "before_finish", "periodic"] = pydantic.Field(
        default="off",
        description=(
            "When the in-loop review panel runs on the diff so far and its findings reach the "
            "model as a message: `off` (never), `on_verify_fail` (after each failed verify), "
            "`before_finish` (when the model calls `finish_session`; a gating `decision` can "
            "reject the finish), or `periodic` (every `period` iterations). With no `seats` the "
            "panel is one reviewer seat on `[models.reviewer]`, the model `agent6 review` uses."
        ),
    )
    period: int = pydantic.Field(
        ge=1,
        default=10,
        description='Iterations between panels when `trigger = "periodic"`.',
    )
    decision: Literal["advisory", "veto", "quorum", "all"] = pydantic.Field(
        default="advisory",
        description=(
            "What a panel's `block` verdicts do: `advisory` (the findings are injected as "
            "guidance, nothing is blocked), `veto` (one blocking seat rejects the finish), "
            "`quorum` (`quorum` distinct models must block), or `all` (every seat must block). A "
            "gate applies to `before_finish` only; the other triggers always advise."
        ),
    )
    quorum: int = pydantic.Field(
        ge=1,
        default=2,
        description=(
            'How many seats must block for `decision = "quorum"`, counted per distinct model (two '
            "seats on one model count once, so a same-model panel cannot reach it)."
        ),
    )
    max_total_rejections: int = pydantic.Field(
        ge=1,
        default=4,
        description=(
            "How many finishes a gating panel may reject per run before it disarms to `advisory` "
            "for the rest of the run, so a panel can never stall a run forever."
        ),
    )
    budget_fraction: float = pydantic.Field(
        gt=0.0,
        le=1.0,
        default=0.25,
        description=(
            "Skip the panel (the finish is accepted) once the run's remaining budget falls below "
            "this fraction of the whole. `0.25`: no panel in the last quarter."
        ),
    )
    seats: _base.StrTuple = pydantic.Field(
        default=(),
        description=(
            'The panel roster, one entry per seat: a persona name (`"security"`), routed via '
            '`[models.reviewer]`, or `"<persona>@<provider>/<model>"` to pin a model per seat '
            '(`"correctness@openrouter/moonshotai/kimi-k2"`). A persona is any short stance the '
            "reviewer's prompt adopts; the built-in set cycled when none is named is `security`, "
            "`correctness`, `tests`, `over-engineering`, `edge-cases`. Empty: one reviewer seat "
            "when `trigger` is on. `agent6 review --reviewers N --personas ...` builds the same "
            "roster for a one-off review."
        ),
    )
    concurrency: int = pydantic.Field(
        ge=1,
        default=1,
        description=(
            "How many seats the in-loop panel runs at once (`1` = one after another; the panel's "
            "latency is its slowest seat). `agent6 review` always runs every seat in parallel."
        ),
    )
    tier: ReviewTier = pydantic.Field(
        default="diff",
        description=(
            "How much a seat reads: `diff` (one call over the diff, the task, and the verify "
            "result) or `explore` (a read-only tool-using reviewer that also reads the repo around "
            "the diff to catch cross-file impact; several calls per seat, and it reads the "
            "checkout, so the reviewed head must be checked out with a clean tree)."
        ),
    )

    @pydantic.model_validator(mode="after")
    def _check_review_seats(self) -> ReviewConfig:
        """Refuse a seat entry that is empty or does not parse.

        Returns:
            The model unchanged.

        Raises:
            ValueError: A seat is blank, or its `@` form names no provider or no model.
        """
        for spec in self.seats:
            if not spec.strip():
                raise ValueError("review.seats entries must be non-empty")
            try:
                parse_seat_spec(spec)
            except ValueError as exc:
                raise ValueError(f"review.seats: {exc}") from exc
        return self

    @pydantic.model_validator(mode="after")
    def _check_review_quorum(self) -> ReviewConfig:
        """Refuse a quorum the roster's distinct models can never reach.

        Returns:
            The model unchanged.

        Raises:
            ValueError: `decision = "quorum"` with fewer distinct seat models than `quorum`.
        """
        if self.decision == "quorum" and self.quorum > 1:
            models = {f"{p}/{m}" if p else "" for _, p, m in map(parse_seat_spec, self.seats)}
            if len(models) < self.quorum:
                raise ValueError(
                    f"review.decision='quorum' with quorum={self.quorum}"
                    f" needs >= {self.quorum} distinct models (the gate counts one block per"
                    " distinct model). Provide them via seats"
                    " ('persona@provider/model'), or use decision='veto'."
                )
        return self


class BudgetConfig(pydantic.BaseModel):
    """The `[budget]` table: every provider call is bounded in exactly one currency.

    A meterable call counts against `max_usd`, a plan-metered call against `max_percent`, and
    a call with neither against `max_tokens_fallback`. Each cap reads `-1` as unlimited and
    `0` as refuse up front.
    """

    model_config = _base.MODEL_CONFIG

    max_usd: float = pydantic.Field(
        default=10.0,
        description=(
            "Cap on the metered spend of one run (provider-reported cost, else price times tokens "
            "at the model's fetched rates, cache-aware). Hitting it ends the run resumably "
            "(`budget_exhausted`); each resumed execution gets a fresh budget. `-1`: unlimited; "
            "`0`: refuse every metered call. `--max-usd` overrides per run."
        ),
    )
    max_tokens_fallback: int = pydantic.Field(
        ge=-1,
        default=2_000_000,
        description=(
            "Token cap (input plus output) for the calls the run cannot price: local models, a "
            "model with no price data. `-1`: unlimited; `0`: never run an unmeterable model. "
            "`--max-tokens-fallback` overrides per run."
        ),
    )

    max_percent: float = pydantic.Field(
        default=-1.0,  # the float the loader validates it to, so `config fill` is idempotent
        description=(
            "Cap on the plan percentage points one run may consume on a subscription provider: "
            "the rise in the account's reported used-percent across the run, added up across "
            "window resets (so a value above 100 is meaningful; with several windows, the one "
            "that moved most). The account reports whole percents and every tick counts as a "
            "full point, so the cap ends a run early, never late; the call in flight still "
            "finishes. The reading is account-global: a concurrent run's spend counts toward "
            "whichever run observes it next. `-1`: unlimited; `0`: refuse plan-metered calls. "
            "`--max-percent` overrides per run."
        ),
    )

    # Purchased credits and extra usage are real money after the included window.
    allow_paid_credits: bool = pydantic.Field(
        default=False,
        description=(
            "Allow plan-metered calls (`chatgpt`, `claude_code`) to spend purchased credits or "
            "extra usage once the included plan window is exhausted (auto top-up can buy more "
            "with the saved payment method). `false` is a circuit breaker, not a guarantee: "
            "the backend's usage readings (a chatgpt preflight and every response's headers, "
            "every claude_code round's rate-limit event) report the account's windows and "
            "credit state, and once a window is exhausted with credits present the run stops "
            "at its next boundary; a call already in flight completes. `true`: a chatgpt credit "
            "balance's drop across the run is read as dollars and meters against `max_usd`; a "
            "claude_code run reads no credit balance, so the extra usage it spends is not "
            "metered by `max_usd`. Included-plan usage is unaffected."
        ),
    )

    @pydantic.field_validator("max_usd", "max_percent")
    @classmethod
    def _usd_unlimited_is_exactly_minus_one(cls, v: float) -> float:
        """Refuse a cap that is non-finite or negative other than exactly -1.

        A non-finite cap never binds (nan fails every comparison), which would silently disable
        the hard budget.

        Args:
            v: The configured cap.

        Returns:
            The cap unchanged.

        Raises:
            ValueError: The cap is nan, infinite, or negative and not -1.
        """
        if not math.isfinite(v) or (v < 0 and v != -1):
            raise ValueError("a budget cap is finite and >= 0, or exactly -1 for unlimited")
        return v
