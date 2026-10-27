# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The loop-behaviour models.

`[harness]` with its metric, `[review]`, `[context]`, `[prompt]` and `[budget]`.
"""

from __future__ import annotations

import pathlib
from typing import Literal

import pydantic

from agent6.config import _base

EffortLevel = Literal["off", "low", "medium", "high", "xhigh", "max"]


class RoleModel(pydantic.BaseModel):
    """One role's provider and model.

    `temperature` defaults to 0.0: high-temperature sampling degenerates on some open-weights
    models (an observed 15997 literal newline escapes in one `old_string`), and OpenRouter's
    per-model defaults vary, so pinning is what makes a bench reproducible.
    """

    model_config = _base.MODEL_CONFIG

    provider: str = pydantic.Field(
        min_length=1,
        description="A `[providers.<name>]` entry, by name.",
    )
    model: str = pydantic.Field(
        min_length=1,
        description="Model id as that provider names it (`agent6 model` lists them).",
    )
    temperature: float | None = pydantic.Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description=(
            "sampling temperature pinned on every call, `temperature = 0.0` to `2.0`; stable "
            "tool use at `temperature = 0.0`; TOML omission: `temperature = 0.0`; provider "
            "default only through `temperature=None` in the Python API"
        ),
    )
    # Per wire: `reasoning.effort` on OpenAI-family models, `output_config.effort` or a thinking
    # budget on Anthropic ones.
    effort: EffortLevel | None = pydantic.Field(
        default=None,
        description=(
            "reasoning effort via `effort`: `off`, `low`, `medium`, `high`, `xhigh` or `max`; "
            "top tiers where offered by the model, collapsed to Anthropic's highest; unset: "
            '`effort = "low"` on openai-compatible reasoning models, no thinking on Anthropic, '
            'as printed by `agent6 config show`; no effort sent with `api_format = "chatgpt"` '
            'or `"claude_code"`, leaving the model\'s own default'
        ),
    )


class ReviewerConfig(pydantic.BaseModel):
    """One reviewer's focus and optional model settings."""

    model_config = _base.MODEL_CONFIG

    focus: str = pydantic.Field(min_length=1, pattern=r"\S", description="A short review stance.")
    provider: str | None = pydantic.Field(
        default=None, min_length=1, description="A provider name; requires `model` too."
    )
    model: str | None = pydantic.Field(
        default=None, min_length=1, description="A model id; requires `provider` too."
    )
    temperature: float | None = pydantic.Field(
        default=None,
        ge=0.0,
        le=2.0,
        description=(
            "sampling temperature; route's value when unset; `temperature = 0.0` an explicit "
            "override"
        ),
    )
    effort: EffortLevel | None = pydantic.Field(
        default=None,
        description=(
            'reasoning effort; route\'s value when unset; `effort = "off"` an explicit override'
        ),
    )

    @pydantic.model_validator(mode="after")
    def _check_route(self) -> ReviewerConfig:
        """Refuse a partial provider/model route.

        Returns:
            The validated entry.

        Raises:
            ValueError: Only one routing field is set.
        """
        if (self.provider is None) != (self.model is None):
            raise ValueError("reviewer provider and model must be set together or both omitted")
        return self

    def resolve(self, reviewer: RoleModel | None) -> RoleModel | None:
        """Resolve the route and apply explicit sampling settings.

        Args:
            reviewer: The resolved reviewer role, or None when unconfigured.

        Returns:
            A pinned route uses RoleModel defaults; an unpinned route inherits the role.
            None when neither supplies a route.
        """
        route = (
            RoleModel(provider=self.provider, model=self.model)
            if self.provider is not None and self.model is not None
            else reviewer
        )
        if route is None:
            return None
        return RoleModel(
            provider=route.provider,
            model=route.model,
            temperature=route.temperature if self.temperature is None else self.temperature,
            effort=route.effort if self.effort is None else self.effort,
        )


ReviewScope = Literal["diff", "repo"]


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
            "score-printing command as argv, no shell; run after every committed edit and on "
            "the model's `run_metric_command` call"
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
            'score direction; `harness.metric.goal = "minimize"`: smaller wins; '
            '`harness.metric.goal = "maximize"`: larger wins; trajectory reported; finish '
            "allowed once a verified edit only ties the best"
        ),
    )


class HarnessConfig(pydantic.BaseModel):
    """The `[harness]` table."""

    model_config = _base.MODEL_CONFIG

    # Repo-specific, so no global default; the inference lives in agent6.verify_infer.
    verify_command: _base.Argv = pydantic.Field(
        default=(),
        description=(
            'step-success command as argv, no shell (pipeline form: `["sh", "-c", "a && b"]`); '
            "explicit value: pinned gate; unset: inferred and printed per run; inference order: "
            "AGENTS.md `## Verify command` block or inline `Verify:` line, root `verify.sh`, "
            "repo manifests and loose `test_*.py` files, then a model call over those "
            "manifests; AGENTS.md commands joined with `&&`, one per line, no multiline `for` "
            "or `if`; commands with a first word absent from sandbox PATH skipped and named; "
            "files under `sandbox.hide_paths` treated as absent; a plain-command package.json "
            "test script run directly when its runner is on sandbox PATH but `npm` is not; no "
            "inferred gate: gateless start, adoption of the first gate from a recognizable "
            "project created mid-run; `$ ` and `> ` prompts stripped; in a fence containing a "
            "`$ ` prompt or opening with a `> ` prompt: only prompted lines and their "
            "continuations treated as commands"
        ),
    )
    verify_infer: bool = pydantic.Field(
        default=True,
        description=(
            "verify-command inference when `harness.verify_command` is unset (AGENTS.md fence, "
            "repo signals, model call); mid-run adoption for a gateless run creating a "
            "recognizable project; adopted gate dropped and never re-adopted if unable to run "
            "(exit 127 or missing module named by `-m`); `harness.verify_infer = false`: no "
            "inference or adoption, run stays gateless; explicit `harness.verify_command` "
            "unaffected"
        ),
    )
    # Matches the jail's general 600s; a bench with a 2s gate detects a runaway edit sooner at 30.
    verify_timeout_s: float = pydantic.Field(
        gt=0.0,
        default=600.0,
        description=(
            "seconds per `harness.verify_command` or `harness.metric.command` call before kill "
            "and failure; run ending over a gate killed here: `finished · gate timed out`; "
            "pathless pytest gate over budget (harness call or model's `run_verify_command`): "
            "retry on test files nearest the run's diff, harness gates scoped until a full gate "
            "passes; `finish_session` over scoped green: `passed · scoped gate`; model's "
            "`run_command` unbounded (see `harness.command_checkin_s`); cap on each gate call, "
            "not cached work; `go test` may pass from a cache filled by the model's "
            "`run_command`; Go test cache disabled by `go test -count=1`"
        ),
    )
    # Per execution: a standing run is not capped by the sum of its executions.
    max_turns: int = pydantic.Field(
        default=200,
        description=(
            "loop turns per execution before stop reason `max_turns`; `harness.max_turns = -1`: "
            "unlimited; fresh allowance on resume"
        ),
    )

    @pydantic.field_validator("max_turns")
    @classmethod
    def _turns_unlimited_is_exactly_minus_one(cls, v: int) -> int:
        """Refuse a cap that is neither positive nor exactly -1.

        Args:
            v: The configured cap.

        Returns:
            The cap unchanged.

        Raises:
            ValueError: The cap is 0 or below -1.
        """
        if v == 0 or v < -1:
            raise ValueError("max_turns is >= 1, or exactly -1 for unlimited")
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
    # Measured: six idle turns cut steered runs mid-work, twelve none (bench/coreagent/FINDINGS.md).
    verify_settled_stop_after: int = pydantic.Field(
        default=12,
        ge=0,
        description=(
            "consecutive turns with no edit, commit or memory write and an unchanged tree "
            "before a plain run ends after green verify or a gateless editing step; verify runs "
            "neutral; streak reset by progress or operator steer; "
            "`harness.verify_settled_stop_after = 0`: stop disabled, nudge at three idle turns "
            "retained"
        ),
    )
    stagnation_notice_after_s: float = pydantic.Field(
        default=300.0,
        ge=0.0,
        description=(
            "wall-clock seconds without the run's own edit or verify before one notice (recall "
            "spirals: few calls, long reasoning); `harness.stagnation_notice_after_s = 0`: "
            "disabled; clock from the first turn after the start gate, excluding operator wait "
            "time"
        ),
    )
    # Measured in bench/longhorizon/FINDINGS.md: most memory writes follow this notice.
    memory_advisory: bool = pydantic.Field(
        default=True,
        description=(
            "one notice per run at the first red-to-green verify with no repo-memory write, "
            "inviting a durable repo fact; cost about two turns on a short fix; "
            "`harness.memory_advisory = false`: no notice, memory still readable and writable "
            "by the model"
        ),
    )

    # 900 because the hand-back is non-destructive: handing back early costs one poll cycle.
    command_checkin_s: float = pydantic.Field(
        ge=0.0,
        default=900.0,
        description=(
            "seconds before the model's `run_command` returns as a background job; no timeout "
            "or kill, command still running; response: `returncode: null`, `still_running: "
            "true`, `background_id`; model options: wait with `read_background`, stop the job "
            "or carry on; `harness.command_checkin_s = 0`: hand-back disabled"
        ),
    )
    standing_patience: int = pydantic.Field(
        ge=-1,
        default=-1,
        description=(
            "consecutive fruitless standing-goal re-entries (no executed tool call) before soft "
            "ends are honoured; `harness.standing_patience = -1`: no end from fruitless rounds "
            "alone, only budget, turn cap or operator stop; `harness.standing_patience = 0`: "
            "end on the first fruitless round; `harness.standing_patience = N`: escalating "
            "nudges for N fruitless re-entries, then soft ends honoured; streak reset by a "
            "round landing work"
        ),
    )
    verify_when: Literal["finish", "turn", "never"] = pydantic.Field(
        default="finish",
        description=(
            'automatic `harness.verify_command` timing; `harness.verify_when = "finish"`: '
            "`finish_session` after a tree change since the last green run; "
            '`harness.verify_when = "turn"`: also after each editing turn; `harness.verify_when '
            '= "never"`: only the model\'s own `run_verify_command` calls; tool available in '
            "every mode; no gate without a verify command"
        ),
    )
    verify_at_start: bool = pydantic.Field(
        default=True,
        description=(
            "one `harness.verify_command` call before a fresh run's first turn to record "
            "pre-existing gate failure; red output in the first request; finish leaving that "
            "same red: `finished · gate was already red`; cost: one gate call per fresh run; "
            "`harness.verify_at_start = false`: skip for a gate too slow to run twice; no start "
            'gate with `harness.verify_when = "never"` or `sandbox.run_commands = "deny"`; '
            "recorded baseline kept on resume"
        ),
    )
    verify_retries: int = pydantic.Field(
        ge=0,
        default=2,
        description=(
            "red finish certifications returned to the model with gate output before the finish "
            "stands as `finished · gate red`, never passed; `harness.verify_retries = 0`: end "
            "on the first red; pre-existing red not returned unless made green during this run"
        ),
    )
    metric: MetricConfig | None = pydantic.Field(
        default=None,
        description=(
            "optional score beside the pass/fail gate (benchmark, size or count); run after "
            "every committed edit, trend shown to the model; unset: `run_metric_command` still "
            "listed, calls refused with `harness.metric` named"
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
            "tier-1 threshold in accumulated tool-result characters (about 4 per token); above "
            "it: oldest results replaced by short, re-fetchable placeholders; unset: about 45% "
            "of the model's context window; explicit thresholds: both `context.drop_at_chars` "
            "and `context.summarise_at_chars` required"
        ),
    )
    summarise_at_chars: int | None = pydantic.Field(
        default=None,
        gt=0,
        description=(
            "tier-2 threshold in whole-context characters; above it: elided history summarized, "
            "conversation restarted on the summary, task DAG preserved; unset: model's window "
            "minus a 16k-token reserve; required above `context.drop_at_chars`"
        ),
    )
    keep_recent_chars: int = pydantic.Field(
        ge=0,
        default=80_000,
        description=(
            "characters of newest history kept verbatim after the summary on a tier-2 restart; "
            "`context.keep_recent_chars = 0`: none"
        ),
    )
    keep_thinking_turns: int = pydantic.Field(
        default=-1,
        description=(
            "thinking retained at tier 1 for the newest N assistant turns with "
            "`context.keep_thinking_turns = N` (`N >= 1`); `context.keep_thinking_turns = -1`: "
            "all thinking; `context.keep_thinking_turns = 0`: invalid, newest signed thinking "
            "block needed by an in-flight tool call; less replay for Anthropic's signed blocks "
            "and ChatGPT's reasoning items; no thinking re-sent on the OpenAI wire"
        ),
    )
    summary_max_tokens: int = pydantic.Field(
        gt=0,
        default=2048,
        description=(
            "output-token cap for a tier-2 summary or gist distillation; smaller caps "
            "overridden by a reasoning model's per-call floor for reasoning tokens; no cap "
            "accepted by the chatgpt backend"
        ),
    )
    # One batched reviewer-model call per drop event.
    # Measured on the longhorizon bench: bare elision halves a retention score under a small window.
    elision_gists: bool = pydantic.Field(
        default=True,
        description=(
            "large `read_file` results replaced at tier 1 by model-written gists before bare "
            "placeholders; gists also dropped under continued pressure to keep the byte bound; "
            "`context.elision_gists = false`: bare placeholders directly"
        ),
    )

    @pydantic.field_validator("keep_thinking_turns")
    @classmethod
    def _thinking_unlimited_is_exactly_minus_one(cls, v: int) -> int:
        """Refuse a count that is neither positive nor exactly -1.

        Args:
            v: The configured count.

        Returns:
            The count unchanged.

        Raises:
            ValueError: The count is 0 or below -1.
        """
        if v == 0 or v < -1:
            raise ValueError("keep_thinking_turns is >= 1, or exactly -1 to keep all thinking")
        return v

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
            "file path replacing run mode's built-in base system prompt; dynamic blocks still "
            "appended; tool contracts supplied by the operator; startup warning for missing "
            "core tool names; empty: built-in base; assembled prompt, tool definitions and "
            "first message shown by `agent6 prompt show`"
        ),
    )
    # The revision call takes no tools and counts against the budget like any provider call.
    revise_prompt: Literal["off", "auto", "interactive"] = pydantic.Field(
        default="off",
        description=(
            "one task rewrite by the reviewer model before the loop; `prompt.revise_prompt = "
            '"off"`: no rewrite; `prompt.revise_prompt = "auto"`: revision used as written; '
            '`prompt.revise_prompt = "interactive"`: accept, keep original, edit or quit (stop '
            "the run), terminal required, skipped under TUI, ACP or a spawned lane; same pass "
            'for `/task` queued into a live run, handled as `prompt.revise_prompt = "auto"`'
        ),
    )
    # `auto` resolves per model in models.registry.decompose_default; the engine reads only `on`.
    decompose: Literal["auto", "on", "off"] = pydantic.Field(
        default="auto",
        description=(
            "task decomposition before editing in run mode, ordered DAG subtasks worked one at "
            'a time; `prompt.decompose = "on"`: help for small models under-finishing '
            "multi-part tasks (measured on mistral-small), 2-4x overhead for capable models; "
            '`prompt.decompose = "off"`: disabled; `prompt.decompose = "auto"`: '
            "per-worker-model capability registry, resolved value in `config show`; forced for "
            "one run by `--decompose`"
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
            "in-loop panel over the diff so far, findings sent to the model as a message; "
            '`review.trigger = "off"`: never; `review.trigger = "on_verify_fail"`: after each '
            'failed verify; `review.trigger = "before_finish"`: on `finish_session`, finish '
            'rejectable by a gating `review.block_on`; `review.trigger = "periodic"`: every '
            "`review.interval_turns` turns; no `review.reviewers`: one seat on "
            "`[models.reviewer]`, also used by `agent6 review`"
        ),
    )
    interval_turns: int = pydantic.Field(
        ge=1,
        default=10,
        description='Turns between panels when `trigger = "periodic"`.',
    )
    block_on: Literal["none", "any", "model_quorum", "all"] = pydantic.Field(
        default="none",
        description=(
            'effect of panel `block` verdicts; `review.block_on = "none"`: guidance only; '
            '`review.block_on = "any"`: finish rejected by one blocking seat; `review.block_on '
            '= "model_quorum"`: `review.min_blocking_models` distinct provider/model routes '
            'required to block; `review.block_on = "all"`: every seat required to block, '
            "abstentions counted as no block; gating only with `review.trigger = "
            '"before_finish"`, other triggers advisory'
        ),
    )
    min_blocking_models: int = pydantic.Field(
        ge=1,
        default=2,
        description=(
            "distinct provider/model routes required to block with `review.block_on = "
            '"model_quorum"`; two seats on one route counted once, even with different focus or '
            "effort"
        ),
    )
    max_review_rejections: int = pydantic.Field(
        ge=1,
        default=4,
        description=(
            "ends rejectable by a gating panel per run (`finish_session`, silent finish, "
            "settled or plateau stop); count preserved across resumes, never reset; at the cap: "
            "panels and findings retained, every end accepted, no indefinite panel stall"
        ),
    )
    min_remaining_budget_fraction: float = pydantic.Field(
        gt=0.0,
        le=1.0,
        default=0.25,
        description=(
            "remaining-budget fraction below which the panel is skipped and the finish "
            "accepted; `review.min_remaining_budget_fraction = 0.25`: no panel in the last "
            "quarter"
        ),
    )
    reviewers: tuple[ReviewerConfig, ...] = pydantic.Field(
        default=(),
        strict=False,
        json_schema_extra={"display_safe": True},
        description=(
            "ordered panel roster of tables with `focus` and optional `provider`, `model`, "
            "`temperature` and `effort`; routing omitted: inherited `[models.reviewer]`; pinned "
            "route: role-model defaults, `provider` and `model` required together; empty "
            "roster: built-in focuses cycled through `security`, `correctness`, `tests`, "
            "`over-engineering`, `edge-cases`, one seat when `review.trigger` is on; one-off "
            "roster via `agent6 review --reviewers N --focus ...`"
        ),
    )
    concurrency: int = pydantic.Field(
        ge=1,
        default=1,
        description=(
            "simultaneous in-loop panel seats; `review.concurrency = 1`: one after another; "
            "`agent6 review`: all seats always parallel"
        ),
    )
    scope: ReviewScope = pydantic.Field(
        default="diff",
        description=(
            "How much a seat reads: `diff` (one call over the diff, the task, and the verify "
            "result) or `repo` (a read-only tool-using reviewer that also reads the repo around "
            "the diff to catch cross-file impact; several calls per seat, and it reads the "
            "checkout, so the reviewed head must be checked out with a clean tree)."
        ),
    )
    repo_max_turns: int = pydantic.Field(
        ge=1,
        default=20,
        description=(
            'provider-call cap for a `review.scope = "repo"` seat, tool calls included; no '
            "verdict within the cap: abstention, reported by the panel"
        ),
    )


class BudgetConfig(pydantic.BaseModel):
    """The `[budget]` table: every provider call is bounded in exactly one currency.

    A meterable call counts against `max_usd`, a plan-metered call against `max_plan_points`, and
    a call with neither against `max_tokens_fallback`. Each cap reads `-1` as unlimited and
    `0` as refuse up front.
    """

    model_config = _base.MODEL_CONFIG

    max_usd: float = pydantic.Field(
        default=10.0,
        description=(
            "cap on the metered spend of one run (provider-reported cost, else price times "
            "tokens at the model's fetched rates, cache-aware); at the cap: resumable end "
            "(`budget_exhausted`); fresh budget per resumed execution; `budget.max_usd = -1`: "
            "unlimited; `budget.max_usd = 0`: every metered call refused; per-run override: "
            "`--max-usd`"
        ),
    )
    max_tokens_fallback: int = pydantic.Field(
        ge=-1,
        default=2_000_000,
        description=(
            "input-plus-output token cap for unpriceable calls (local models, models without "
            "price data); `budget.max_tokens_fallback = -1`: unlimited; "
            "`budget.max_tokens_fallback = 0`: unmeterable models refused; per-run override: "
            "`--max-tokens-fallback`"
        ),
    )

    max_plan_points: float = pydantic.Field(
        default=-1.0,  # the float the loader validates it to, so `config fill` is idempotent
        description=(
            "subscription-plan percentage-point cap per run; rise in the account's reported "
            "used-percent, summed across window resets (values above 100 meaningful), largest "
            "movement across multiple windows; a window counts as reset when its reported reset "
            "time moves later by more than a minute; a lower reading whose reset time has not "
            "moved counts nothing; one more reading at the end of the run; whole-percent "
            "readings, each tick counted as a full point; early stop at the cap, never late; "
            "in-flight call still completed; account-global readings include other runs' "
            "spend; `budget.max_plan_points = -1`: unlimited; "
            "`budget.max_plan_points = 0`: plan-metered calls refused; per-run override: "
            "`--max-plan-points`"
        ),
    )

    # Purchased credits and extra usage are real money after the included window.
    allow_plan_overage: bool = pydantic.Field(
        default=False,
        description=(
            'plan overage for `api_format = "chatgpt"` or `"claude_code"`: purchased credits or '
            "extra usage after the included window, possible auto top-up through the saved "
            "payment method; `budget.allow_plan_overage = false`: circuit breaker, not a "
            "guarantee; account windows and credit state read at chatgpt preflight and from "
            "every response's headers, or from every claude_code round's rate-limit event; "
            "with `budget.allow_plan_overage = false`, exhausted window and credits present: "
            "stop at the next boundary, in-flight call "
            "still completed; `budget.allow_plan_overage = true`: chatgpt credit-balance drop "
            "read as dollars against `budget.max_usd`; no claude_code credit-balance reading, "
            "extra usage not metered by `budget.max_usd`; included-plan usage unaffected"
        ),
    )

    @pydantic.field_validator("max_usd", "max_plan_points")
    @classmethod
    def _usd_unlimited_is_exactly_minus_one(cls, v: float) -> float:
        """Refuse a negative cap other than exactly -1.

        Args:
            v: The configured cap.

        Returns:
            The cap unchanged.

        Raises:
            ValueError: The cap is negative and not -1.
        """
        if v < 0 and v != -1:
            raise ValueError("a budget cap is finite and >= 0, or exactly -1 for unlimited")
        return v
