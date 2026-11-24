# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The loop-behaviour models.

`[harness]` with its parallel, prompt and verify tables, `[review]`, `[compaction]`
and `[budget]`.
"""

from __future__ import annotations

import pathlib
from typing import Any, Literal

import pydantic

from agent6 import directive, kinds
from agent6.config import validation

_EFFORT_MEANINGS: dict[str, pydantic.JsonValue] = {
    "off": "Disable reasoning effort",
    "low": "Use low reasoning effort",
    "medium": "Use medium reasoning effort",
    "high": "Use high reasoning effort",
    "xhigh": "Use extra-high reasoning effort where supported",
    "max": "Use maximum reasoning effort where supported",
}


class RoleModel(pydantic.BaseModel):
    """One role's provider and model.

    `temperature` defaults to 0.0: high-temperature sampling degenerates on some open-weights
    models (an observed 15997 literal newline escapes in one `old_string`), and OpenRouter's
    per-model defaults vary, so pinning is what makes a bench reproducible.
    """

    model_config = validation.MODEL_CONFIG

    provider: str = pydantic.Field(
        default="",
        min_length=1,
        exclude=True,
        description="A `[providers.<name>]` entry, by name.",
    )
    model: str = pydantic.Field(
        default="",
        min_length=1,
        description=(
            "Fully qualified provider/model route (`agent6 model` lists them); the first "
            "segment names a `[providers.<name>]` entry, the rest is the model id verbatim; "
            "required in `models.default` for a model call, inherited from it in other roles."
        ),
    )
    temperature: float | None = pydantic.Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description=(
            "sampling temperature, `temperature = 0.0` to `2.0`, sent where the wire accepts "
            "one (not with Anthropic thinking or on `chatgpt` or `claude_code`, dropped for "
            "the run after a 400 naming it); `extra_body.temperature` wins; TOML omission: "
            "inherits `models.default.temperature` in a role, `temperature = 0.0` in the default"
        ),
    )
    # Per wire: `reasoning.effort` on OpenAI-family models, `output_config.effort` or a thinking
    # budget on Anthropic ones.
    effort: kinds.EffortLevel | None = pydantic.Field(
        json_schema_extra={"choice_meanings": _EFFORT_MEANINGS},
        default=None,
        description=(
            "Reasoning effort; an unsupported explicit level refuses the call; "
            "unset in a role: inherits `models.default.effort`; unset "
            'in the default: `effort = "low"` on openai-compatible reasoning models, no '
            'thinking on Anthropic, none sent with `backend = "chatgpt"`, `"responses"` or '
            '`"claude_code"` (the model\'s default), as printed by `agent6 config show`'
        ),
    )

    @pydantic.field_serializer("model")
    def _serialize_route(self, model: str) -> str:
        """Return the qualified settings leaf."""
        return f"{self.provider}/{model}" if model else ""


class ReviewerConfig(pydantic.BaseModel):
    """One reviewer's focus and optional model settings."""

    model_config = validation.MODEL_CONFIG

    focus: str = pydantic.Field(min_length=1, pattern=r"\S", description="A short review stance.")
    model: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="A provider/model route; inherits the reviewer role when unset.",
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
    effort: kinds.EffortLevel | None = pydantic.Field(
        json_schema_extra={"choice_meanings": _EFFORT_MEANINGS},
        default=None,
        description=(
            'reasoning effort; route\'s value when unset; `effort = "off"` an explicit override'
        ),
    )

    @pydantic.model_validator(mode="before")
    @classmethod
    def _check_route(cls, data: Any) -> Any:
        """Refuse the split route and validate a qualified model setting.

        Args:
            data: The raw reviewer entry.

        Returns:
            The entry unchanged.

        Raises:
            ValueError: A split or unqualified route.
        """
        validation.refuse_split_model(data, "review.reviewers")
        if isinstance(data, dict) and isinstance(data.get("model"), str):
            validation.model_route(data["model"])
        return data

    def resolve(self, reviewer: RoleModel | None) -> RoleModel | None:
        """Resolve the route and apply explicit sampling settings.

        Args:
            reviewer: The resolved reviewer role, or None when unconfigured.

        Returns:
            The pinned route or the role's, with the role's sampling settings under the
            entry's own; None when neither supplies a route.
        """
        route = reviewer
        if self.model is not None:
            pinned = validation.model_route(self.model)
            # The one rule: every omitted field inherits the role, model pinned or not.
            route = RoleModel(
                provider=pinned.provider,
                model=pinned.model,
                temperature=reviewer.temperature if reviewer is not None else 0.0,
                effort=reviewer.effort if reviewer is not None else None,
            )
        if route is None:
            return None
        return RoleModel(
            provider=route.provider,
            model=route.model,
            temperature=route.temperature if self.temperature is None else self.temperature,
            effort=route.effort if self.effort is None else self.effort,
        )


class ParallelConfig(pydantic.BaseModel):
    """The `[harness.parallel]` table: the child limit and the model's `parallel` tool."""

    model_config = validation.MODEL_CONFIG

    tool: bool = pydantic.Field(
        default=False,
        description=(
            "offer a run's model the `parallel` tool, which starts one task in isolated "
            "children (fresh clones of the committed tree, each an ordinary run that may edit) "
            "and completes their tasks at turn boundaries (`git.control` decides who merges), "
            "and `steer_child`, which instructs a running child or continues a stopped one; "
            "a child runs under the run's "
            "own config and its remaining budget, which the children share; "
            "`harness.parallel.tool = false`: no run is offered either; neither value changes "
            "`run --parallel`, the `/parallel` steer or the review panel"
        ),
    )

    # `le` bounds the cap itself, or a huge max_children re-opens the allocation parse_spec refuses.
    # Static, not CPU-derived: children are I/O-bound and the same config must load on every box.
    max_children: int = pydantic.Field(
        ge=1,
        le=1024,
        default=4,
        description=(
            "The most children a run has running at once, `1` to `1024`; a dispatch asking "
            "for more is refused before anything is cloned; `review.reviewers` runs in batches "
            "limited by both this setting and `review.concurrency`, with no reviewer dropped."
        ),
    )

    judges: validation.StrTuple = pydantic.Field(
        default=("reviewer",),
        description=(
            "who ranks the attempts at one task and the candidates of `sessions compare`: "
            'one role, `provider/model`, or positive reviewer count (such as `"2"`) per entry; '
            "the judges start together, so their total must fit `harness.parallel.max_children`; "
            "each reviewer child reads and runs the attempts under `review.max_turns` and answers "
            "with an order; the lowest sum of ranks wins, a tie going to the "
            "mechanical order (a passed gate first, then the lower cost); a role with no "
            "model route judges nothing; `[]`, or no usable order: the mechanical order alone"
        ),
    )

    @pydantic.field_validator("judges")
    @classmethod
    def _each_judge_is_named(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        """Refuse a blank judge, which the child grammar would read as the worker's own model.

        Args:
            v: The configured judges.

        Returns:
            The judges unchanged.

        Raises:
            ValueError: An entry is blank.
        """
        if any(not judge.strip() for judge in v):
            raise ValueError(
                "a judge is a role, a `provider/model` or a positive count; one entry is blank"
            )
        return v

    def judge_counts(self) -> list[tuple[str, int]]:
        """Count all requested judges before validation or scheduling resolves their routes.

        Returns:
            Each entry's selector and judge count, in order.

        Raises:
            ConfigError: A count is invalid or the roster exceeds `max_children`.
        """
        try:
            return directive.counted_selectors(
                self.judges, default="reviewer", limit=self.max_children
            )
        except directive.DirectiveError as exc:
            raise validation.ConfigError(str(exc)) from exc


class PromptConfig(pydantic.BaseModel):
    """The `[harness.prompt]` table: the system-prompt override and task decomposition."""

    model_config = validation.MODEL_CONFIG

    system_prompt_file: str = pydantic.Field(
        default="",
        description=(
            "file path replacing run mode's built-in base system prompt; dynamic blocks still "
            "appended; tool contracts supplied by the operator; startup warning for missing "
            "core tool names; empty: built-in base; assembled prompt, tool definitions and "
            "first message shown by `agent6 prompt show`"
        ),
    )
    # `auto` resolves per model in models.registry.decompose_default; the engine reads only `on`.
    decompose: Literal["auto", "on", "off"] = pydantic.Field(
        json_schema_extra={
            "choice_meanings": {
                "auto": "Choose from worker model capability registry",
                "on": "Require ordered task decomposition before editing",
                "off": "Do not require task decomposition",
            }
        },
        default="auto",
        description=(
            "Ordered DAG subtasks worked one at a time before editing in run mode; required "
            "decomposition helps small models under-finishing multi-part tasks (measured on "
            "mistral-small), with 2-4x overhead for capable models; `config show` prints the "
            "resolved value; `--decompose` forces it for one run."
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
                raise ValueError(f"harness.prompt.system_prompt_file: not a readable file: {p}")
        return self


class VerifyConfig(pydantic.BaseModel):
    """The `[harness.verify]` table: the gate command and when the harness runs it."""

    model_config = validation.MODEL_CONFIG

    # Repo-specific, so no global default; the inference lives in agent6.tools.verify_infer.
    command: validation.Argv = pydantic.Field(
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
    infer: bool = pydantic.Field(
        default=True,
        description=(
            "verify-command inference when `harness.verify.command` is unset (AGENTS.md fence, "
            "repo signals, model call); mid-run adoption for a gateless run creating a "
            "recognizable project; adopted gate dropped and never re-adopted if unable to run "
            "(exit 127 or missing module named by `-m`); `harness.verify.infer = false`: no "
            "inference or adoption, run stays gateless; explicit `harness.verify.command` "
            "unaffected"
        ),
    )
    # Matches the jail's general 600s; a bench with a 2s gate detects a runaway edit sooner at 30.
    timeout_s: float = pydantic.Field(
        gt=0.0,
        default=600.0,
        description=(
            "seconds per `harness.verify.command` or the session metric call before kill "
            "and failure; run ending over a gate killed here: `finished · gate timed out`; "
            "pathless pytest gate over budget (harness call or model's `run_verify_command`): "
            "retry on test files nearest the run's diff, harness gates scoped until a full gate "
            "passes; `finish_session` over scoped green: `passed · scoped gate`; model's "
            "`run_command` handed back by `harness.command_checkin_s`; cap on each gate call, "
            "not cached work; `go test` may pass from a cache filled by the model's "
            "`run_command`; Go test cache disabled by `go test -count=1`"
        ),
    )
    when: Literal["finish", "turn", "never"] = pydantic.Field(
        json_schema_extra={
            "choice_meanings": {
                "finish": "Verify uncovered final tree at finish",
                "turn": "Also verify after each editing turn",
                "never": "Verify only when model calls tool",
            }
        },
        default="never",
        description=(
            "When the harness runs `harness.verify.command` itself; the tool remains available "
            "in every mode; no gate runs without a command; an uncovered final tree reads "
            "unverified; finish-time failures return to the model up to "
            "`harness.verify.retries` times."
        ),
    )
    at_start: bool = pydantic.Field(
        default=False,
        description=(
            "one `harness.verify.command` call before a fresh run's first turn to record "
            "pre-existing gate failure; red output in the first request; a red finish over a "
            "red base: `finished · gate red at base` (red at the start and at the finish, the "
            "failures not compared); cost: one gate call per fresh run, worth it for a fast gate "
            'beside a suite that is already red; needs `harness.verify.when = "finish"` or '
            '`"turn"` and `sandbox.run_commands` other than `"deny"`; recorded baseline kept '
            "on resume"
        ),
    )
    retries: int = pydantic.Field(
        ge=0,
        default=2,
        description=(
            "red finish certifications returned to the model with gate output before the finish "
            "stands as `finished · gate red`, never passed; `harness.verify.retries = 0`: end "
            "on the first red; pre-existing red not returned unless made green during this run"
        ),
    )


class HarnessConfig(pydantic.BaseModel):
    """The `[harness]` table."""

    model_config = validation.MODEL_CONFIG

    # Per execution: a standing run is not capped by the sum of its executions.
    max_turns: int = pydantic.Field(
        default=-1,
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
    loop_guard_kill_threshold: validation.NoLimitOrPositive = pydantic.Field(
        default=10,
        description=(
            "The same (tool, args) call this many times in a row ends the run as "
            "`loop_guard_killed` (the notice fires from three, every other turn); "
            "`harness.loop_guard_kill_threshold = -1`: the notice alone, no end."
        ),
    )
    # Measured: six idle turns cut steered runs mid-work, twelve none (bench/coreagent/FINDINGS.md).
    settled_stop_after: validation.NoLimitOrPositive = pydantic.Field(
        default=12,
        description=(
            "consecutive turns with no edit, commit, memory write or task list change and an "
            "unchanged tree "
            "before a plain run ends after a green verify or an editing step; verify runs and "
            "waits on a live background command neutral; streak reset by progress or "
            "operator steer; default 12, measured; "
            "`harness.settled_stop_after = -1`: stop disabled, nudge at three idle turns "
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
            "or carry on; also bounds each `read_background` wait, including children; "
            "`harness.command_checkin_s = 0`: background wait cap and hand-back disabled, "
            "foreground command killed at 600 s"
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
    parallel: ParallelConfig = pydantic.Field(default_factory=ParallelConfig)
    prompt: PromptConfig = pydantic.Field(default_factory=PromptConfig)
    verify: VerifyConfig = pydantic.Field(default_factory=VerifyConfig)


class CompactionConfig(pydantic.BaseModel):
    """`[compaction]` section: tiered context-compaction thresholds."""

    model_config = validation.MODEL_CONFIG

    cold_start: kinds.ColdStart = pydantic.Field(
        default="drop",
        description=(
            "local history on resume when the route changes or the provider's idle limit "
            "is reached (not used when the endpoint writes summaries)"
        ),
        json_schema_extra={
            "choice_meanings": {
                "resend": "Send history without a cold-start rewrite",
                "drop": "Replace all eligible old tool results",
                "summarize": "Restart with a helper summary and the selected history",
                "truncate": "Restart with the selected history and no summary",
            }
        },
    )

    summarizer: Literal["worker", "helper"] = pydantic.Field(
        json_schema_extra={
            "choice_meanings": {
                "worker": "Worker continues its own request",
                "helper": "Replaced history with operator answers, without other tool output",
            }
        },
        default="worker",
        description=(
            "Local compaction only: "
            "tier-2 writer: worker continues its own request with a summary instruction; "
            "helper reads replaced history including operator answers but not other tool output, "
            "bounded by its known window; "
            "worker falls back once to helper on rejection or unusable output; claude_code "
            "always uses helper because its live session cannot be rolled back"
        ),
    )

    # Unset thresholds are sized in models.registry.compaction_thresholds.
    drop_at_chars: int | None = pydantic.Field(
        default=None,
        gt=0,
        description=(
            "Accumulated tool-result characters before the endpoint clears old results or "
            "agent6 replaces them with short, re-fetchable placeholders down to half "
            "the threshold where result protections allow; unset: about 45% "
            "of the model's context window; explicit thresholds: both `compaction.drop_at_chars` "
            "and `compaction.summarize_at_chars` required"
        ),
    )
    summarize_at_chars: int | None = pydantic.Field(
        default=None,
        gt=0,
        description=(
            "Whole-context character threshold for a summary by the endpoint or agent6; "
            "an accepted summary restarts the conversation and preserves the task DAG; "
            "unset: model's window "
            "minus a 16k-token reserve; explicit value must exceed `compaction.drop_at_chars` "
            "and `compaction.keep_recent_chars`"
        ),
    )
    keep_operator_messages: int = pydantic.Field(
        ge=-1,
        default=-1,
        description=(
            "Local restart: reserve the newest N messages given to the run, word for word, "
            "before selecting turns: operator steers and answers, and the whole task as the "
            "oldest; -1: no count limit; 0: reserve none outside kept turns"
        ),
    )
    keep_recent_turns: int = pydantic.Field(
        ge=-1,
        default=-1,
        description=(
            "Local restart: keep up to N consecutive replies with their inputs and paired "
            "results, newest first; -1: no count limit; "
            "0: keep no optional turns"
        ),
    )
    keep_recent_chars: int = pydantic.Field(
        ge=0,
        default=80_000,
        description=(
            "Local restart: shared character cap on reserved messages and recent "
            "turns; the newest optional turn stays whole even if it exceeds the space left; "
            "0: keep nothing optional; adaptive thresholds limit the cap to half of "
            "`compaction.summarize_at_chars`"
        ),
    )
    keep_thinking_turns: validation.NoLimitOrPositive = pydantic.Field(
        default=-1,
        description=(
            "Local compaction: keep thinking on the newest N assistant turns when rewriting "
            "results, at a cold drop or restarting; -1: keep all; minimum otherwise: 1, to "
            "keep signed thinking with an in-flight tool call"
        ),
    )
    summary_max_tokens: int = pydantic.Field(
        gt=0,
        default=2048,
        description=(
            "Output-token cap for either summary writer and for gist distillation; smaller caps "
            "overridden by a reasoning model's per-call floor for reasoning tokens; no cap "
            "accepted by the chatgpt backend"
        ),
    )
    # One batched helper-model call per drop event.
    # Measured on the longhorizon bench: bare elision halves a retention score under a small window.
    elision_gists: bool = pydantic.Field(
        default=True,
        description=(
            "Local compaction only: "
            "large `read_file` results replaced at tier 1 by model-written gists before bare "
            "placeholders; gists also dropped under continued pressure to keep the byte bound; "
            "`compaction.elision_gists = false`: bare placeholders directly"
        ),
    )

    @pydantic.model_validator(mode="after")
    def _check_compaction_thresholds(self) -> CompactionConfig:
        """Refuse thresholds that are half set or do not escalate.

        Returns:
            The model unchanged.

        Raises:
            ValueError: One threshold set without the other, tier 2 at or below tier 1, or tier 2
                at or below the verbatim tail.
        """
        drop, summarize = self.drop_at_chars, self.summarize_at_chars
        if (drop is None) != (summarize is None):
            raise ValueError(
                "set both compaction.drop_at_chars and"
                " summarize_at_chars, or NEITHER (neither == adaptive,"
                " sized from the worker model's context window). Both at once:"
                " agent6 config set compaction"
                " '{ drop_at_chars = 200000, summarize_at_chars = 400000 }'"
            )
        if drop is not None and summarize is not None and summarize <= drop:
            raise ValueError(
                "compaction.summarize_at_chars"
                f" ({summarize}) must be greater than"
                f" drop_at_chars ({drop}): tier-2"
                " summarize must escalate above tier-1 elision."
            )
        if summarize is not None and summarize <= self.keep_recent_chars:
            raise ValueError(
                f"compaction.summarize_at_chars ({summarize}) must be greater than"
                f" keep_recent_chars ({self.keep_recent_chars}): the verbatim tail"
                " alone would re-trigger tier 2 after every restart."
            )
        return self


class ReviewConfig(pydantic.BaseModel):
    """The `[review]` table: the in-loop review panel and when it sits."""

    model_config = validation.MODEL_CONFIG

    # The findings reach the model as a user message on its next turn.
    when: Literal["never", "verify_fail", "finish", "periodic"] = pydantic.Field(
        json_schema_extra={
            "choice_meanings": {
                "never": "Do not run in-loop review",
                "verify_fail": "Review after each failed verification",
                "finish": "Review when model requests finish",
                "periodic": "Review every configured interval of turns",
            }
        },
        default="never",
        description=(
            "In-loop panel over a run's diff so far, with findings sent to the model; only "
            "finish-time review can gate finish; periodic review uses "
            "`review.interval_turns`; without `review.reviewers`, one reviewer uses "
            "`[models.reviewer]`; each reviewer runs as a child with its own session."
        ),
    )
    interval_turns: int = pydantic.Field(
        ge=1,
        default=10,
        description='Turns between panels when `when = "periodic"`.',
    )
    block_on: Literal["none", "any", "model_quorum", "all"] = pydantic.Field(
        json_schema_extra={
            "choice_meanings": {
                "none": "Keep block findings advisory at finish and merge",
                "any": "One blocking reviewer rejects finish",
                "model_quorum": "Require configured number of distinct blocking model routes",
                "all": "Require every reviewer to block",
            }
        },
        default="none",
        description=(
            "Rule for new grounded block findings at finish. Quorum counts "
            "`review.min_blocking_models` distinct provider/model routes. "
            "Open blocks prevent a merge unless this is `none`."
        ),
    )
    min_blocking_models: int = pydantic.Field(
        ge=1,
        default=2,
        description=(
            "distinct provider/model routes required to block with `review.block_on = "
            '"model_quorum"`; two reviewers on one route counted once, even with different focus '
            "or effort"
        ),
    )
    max_rejections: int = pydantic.Field(
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
            "ordered panel roster of tables with `focus` and optional `model`, "
            "`temperature` and `effort`; every omitted field inherits the resolved "
            "`[models.reviewer]`, `model` pinned or not; `model` is a provider/model route; "
            'empty roster: one `security` reviewer unless `review.when = "never"`'
        ),
    )
    concurrency: int = pydantic.Field(
        ge=1,
        default=1,
        description=(
            "simultaneous reviewers, at most `harness.parallel.max_children`; a longer roster "
            "runs in batches, in roster order; `review.concurrency = 1`: one after another"
        ),
    )
    max_turns: int = pydantic.Field(
        ge=1,
        default=20,
        description=(
            "turns a reviewer may take, turns that ask for tools included; each turn tells "
            "the reviewer which reply it is, and a tool call in the last is not run; no "
            "verdict within the limit: abstention, reported by the panel"
        ),
    )


class BudgetConfig(pydantic.BaseModel):
    """The `[budget]` table: every provider call is bounded in exactly one currency.

    A meterable call counts against `max_usd`, a plan-metered call against `max_plan_points`, and
    a call with neither against `max_tokens_fallback`. Each cap reads `-1` as unlimited and
    `0` as refuse up front.
    """

    model_config = validation.MODEL_CONFIG

    max_usd: float = pydantic.Field(
        default=10.0,
        description=(
            "cap on the metered spend of one run (provider-reported cost, else price times "
            "tokens at the route's rates, including cache tokens), plus observed account credit "
            "declines; at the cap: resumable end "
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
            "in-flight call still completed; `budget.max_plan_points = -1`: unlimited; "
            "`budget.max_plan_points = 0`: plan-metered calls refused; per-run override: "
            "`--max-plan-points`; each concurrent run counts the whole account-window increase "
            "toward its own cap, not just its own share of the spend."
        ),
    )

    # Purchased credits and extra usage are real money after the included window.
    allow_plan_overage: bool = pydantic.Field(
        default=False,
        description=(
            'plan overage for `backend = "chatgpt"` or `"claude_code"`: purchased credits or '
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
