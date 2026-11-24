# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Parse the steer directives the harness and the composers share.

    /parallel [SPEC --] TASK [/parallel [SPEC --] TASK]...
    /pin <instruction that must survive context compaction>
    /compact [focus text for the summary]
    /review [REF | A..B | A...B]

A message is a directive only when it starts with the exact token.
A whitespace-delimited `/parallel` token starts each segment.
A segment's spec is a comma-separated list of counts, roles or `provider/model` entries.
A whitespace-delimited `--` token ends it.
Without `--`, the whole segment is task text for one default child.
Task text keeps its internal whitespace and newlines, including later `--` tokens.
Pure stdlib parsing with no agent6 imports: a leaf both layers sit above.
"""

from __future__ import annotations

import dataclasses
import math
import re
import shlex
from collections.abc import Sequence
from typing import Literal

# A whitespace-delimited `/parallel` token; not MULTILINE, so the anchors are the whole string's.
_SEPARATOR = re.compile(r"(?:\A|(?<=\s))/parallel(?=\s|\Z)", re.IGNORECASE)
_SPEC_SEPARATOR = re.compile(r"(?:\A|(?<=\s))--(?=\s|\Z)")


class DirectiveError(ValueError):
    """A directive or spec is malformed: a bare token, a missing task, a bad child count."""


METRIC_SYNTAX = 'maximize|minimize ["capture pattern"] [target NUMBER] -- command'
METRIC_POLICY = (
    "The metric can refuse a finish up to three times while more than a quarter of the budget "
    "remains. A plateau can end the run. Once the best score reaches or passes the target, "
    "the metric allows a finish or a plateau stop at any budget."
)
_METRIC_TOKEN = re.compile(r"\A\s*/metric(?=\s|\Z)", re.IGNORECASE)


@dataclasses.dataclass(frozen=True, slots=True)
class MetricSpec:
    """The operator's score direction, capture pattern, jailed argv and optional target."""

    goal: Literal["maximize", "minimize"]
    pattern: str | None
    command: tuple[str, ...]
    target: float | None = None

    def text(self) -> str:
        """Return the spec as input the operator can reuse."""
        pattern = (self.pattern,) if self.pattern is not None else ()
        target = ("target", str(self.target).removesuffix(".0")) if self.target is not None else ()
        return shlex.join((self.goal, *pattern, *target, "--", *self.command))


def parse_metric(text: str) -> str | None:
    """Return the text after a leading `/metric`, empty to show the current metric."""
    match = _METRIC_TOKEN.match(text)
    return text[match.end() :].strip() if match else None


def parse_metric_spec(text: str) -> MetricSpec:
    """Parse a metric flag or directive as argv, without shell evaluation.

    Args:
        text: The spec after `--metric` or `/metric`.

    Returns:
        The operator's metric.

    Raises:
        DirectiveError: The spec, capture pattern or argv is invalid.
    """
    if "\r" in text:
        raise DirectiveError("metric commands cannot contain a carriage return")
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise DirectiveError("the metric must be UTF-8 text") from exc
    refusal = f"metric expects: {METRIC_SYNTAX}"
    parts = text.strip().split(maxsplit=1)
    if len(parts) != 2 or parts[0] not in ("maximize", "minimize"):
        raise DirectiveError(refusal)
    goal, rest = parts
    try:
        args = shlex.split(rest)
    except ValueError as exc:
        raise DirectiveError(f"{refusal} ({exc})") from exc
    pattern = None
    if rest.startswith(("'", '"')):
        pattern = args.pop(0)
        try:
            groups = re.compile(pattern).groups
        except (re.error, OverflowError, RecursionError, ValueError) as exc:
            raise DirectiveError(f"invalid metric pattern: {exc}") from exc
        if not groups:
            raise DirectiveError("metric pattern needs a capture group for the score")
    target = None
    if args[:1] == ["target"]:
        try:
            target = float(args[1])
        except (IndexError, ValueError) as exc:
            raise DirectiveError("metric target must be a finite number") from exc
        if not math.isfinite(target):
            raise DirectiveError("metric target must be a finite number")
        args = args[2:]
    if len(args) < 2 or args[0] != "--" or any(not a.strip() or "\x00" in a for a in args[1:]):
        raise DirectiveError(refusal)
    return MetricSpec(
        "maximize" if goal == "maximize" else "minimize", pattern, tuple(args[1:]), target
    )


@dataclasses.dataclass(frozen=True, slots=True)
class Segment:
    """One parsed `/parallel` task.

    Attributes:
        spec: The child spec; "" is one default child.
        task: The task text, internal whitespace and newlines preserved.
        node_id: The listed task the model handed off, None for a text brief.
    """

    spec: str
    task: str
    node_id: str | None = None


def counted_selectors(entries: Sequence[str], *, default: str, limit: int) -> list[tuple[str, int]]:
    """Count one roster without building a list of its children.

    Args:
        entries: The individual counts or selectors, in roster order.
        default: The selector a count repeats.
        limit: The most children this roster may request.

    Returns:
        One (selector, count) per nonblank entry.

    Raises:
        DirectiveError: A count is invalid or the roster exceeds the limit.
    """
    counted: list[tuple[str, int]] = []
    total = 0
    for index, entry in enumerate(entries, 1):
        selector = entry.strip()
        if not selector:
            continue
        count = 1
        if selector.isdecimal():
            try:
                count = int(selector)
            except ValueError as exc:
                raise DirectiveError(
                    f"parallel child count is too long; request 1 to {limit}"
                ) from exc
            if not count:
                raise DirectiveError("parallel child count must be >= 1")
            selector = default
        total += count
        if total > limit:
            qualifier = "at least " if index < len(entries) else ""
            raise DirectiveError(
                f"parallel spec requests {qualifier}{total} children but"
                f" [harness.parallel].max_children = {limit}."
                " Request fewer, or raise [harness.parallel].max_children."
            )
        counted.append((selector, count))
    return counted


def parse_spec(spec: str, *, limit: int) -> list[str | None]:
    """Parse a child spec into one entry per child, the grammar the directive and the CLI share.

    Args:
        spec: A comma-separated list of positive counts, role names or `provider/model` entries.
        limit: The caller's `[harness.parallel].max_children`; an over-limit count refuses before
            the list is built.

    Returns:
        One entry per child: None for the worker's own settings, else the child's selector
        text for the caller to resolve. "" is one default child.

    Raises:
        DirectiveError: The count is not positive or over the limit, or the list
            names no models.
    """
    if not spec.strip():
        return [None]
    counted = counted_selectors(spec.split(","), default="", limit=limit)
    if not counted:
        raise DirectiveError(f"parallel spec {spec!r} names no models")
    return [selector or None for selector, count in counted for _ in range(count)]


# A leading `/pin` token; everything after it is one pinned instruction.
_PIN_TOKEN = re.compile(r"\A\s*/pin(?=\s|\Z)", re.IGNORECASE)


def parse_pin(text: str) -> str | None:
    """Parse a `/pin` steer.

    Args:
        text: The steer text.

    Returns:
        The instruction, or None when the text is not a pin directive.

    Raises:
        DirectiveError: The `/pin` is bare.
    """
    m = _PIN_TOKEN.match(text)
    if m is None:
        return None
    instruction = text[m.end() :].strip()
    if not instruction:
        raise DirectiveError("pin needs an instruction: /pin <text that must survive compaction>")
    return instruction


# Parsed by the composers and the pause menu, never by the loop: a compact request is a marker.
_COMPACT_TOKEN = re.compile(r"\A\s*/compact(?=\s|\Z)", re.IGNORECASE)


def parse_compact(text: str) -> str | None:
    """Return the focus a `/compact` message carries ("" when bare), or None when it is not one."""
    m = _COMPACT_TOKEN.match(text)
    if m is None:
        return None
    return text[m.end() :].strip()


# A question asked beside the run, never steer text.
_BTW_TOKEN = re.compile(r"\A\s*/btw(?=\s|\Z)", re.IGNORECASE)


def parse_btw(text: str) -> str | None:
    """Return the question a `/btw` message carries ("" when bare), or None when it is not one."""
    m = _BTW_TOKEN.match(text)
    if m is None:
        return None
    return text[m.end() :].strip()


# A task queued into the run's graph, never steer text.
_TASK_TOKEN = re.compile(r"\A\s*/task(?=\s|\Z)", re.IGNORECASE)


def parse_task(text: str) -> str | None:
    """Return the task a `/task` message carries ("" when bare), or None when it is not one."""
    m = _TASK_TOKEN.match(text)
    if m is None:
        return None
    return text[m.end() :].strip()


_REVIEW_TOKEN = re.compile(r"\A\s*/review(?=\s|\Z)", re.IGNORECASE)


def parse_review(text: str) -> str | None:
    """Return a review's target, empty for the working tree, or None for ordinary text.

    Args:
        text: The operator's message.

    Returns:
        The target as typed, without surrounding whitespace.

    Raises:
        DirectiveError: The message has flags, several targets or a malformed range.
    """
    match = _REVIEW_TOKEN.match(text)
    if match is None:
        return None
    target = text[match.end() :].strip()
    if (
        any(char.isspace() for char in target)
        or any(part.startswith("-") for part in re.split(r"\.\.\.?", target))
        or "...." in target
        or target.count("..") > 1
    ):
        raise DirectiveError("/review accepts one ref or A..B or A...B, with no flags")
    return target


# The run's standing goal, set by the operator alone; never steer text.
_STANDING_TOKEN = re.compile(r"\A\s*/standing(?=\s|\Z)", re.IGNORECASE)


def parse_standing(text: str) -> str | None:
    """Return the goal a `/standing` message carries ("" when bare), or None when it is not one."""
    m = _STANDING_TOKEN.match(text)
    if m is None:
        return None
    return text[m.end() :].strip()


# A task the operator retires, named by the id every task tree leads with.
_RETIRE_TOKEN = re.compile(r"\A\s*/retire(?=\s|\Z)", re.IGNORECASE)


def parse_retire(text: str) -> str | None:
    """Return the task id a `/retire` message names ("" when bare), or None when it is not one."""
    m = _RETIRE_TOKEN.match(text)
    if m is None:
        return None
    return text[m.end() :].strip()


# The urgency the CLI spells `steer --now`; parsed by the composers, carried by the request marker.
_NOW_TOKEN = re.compile(r"\A\s*/now(?=\s|\Z)", re.IGNORECASE)


def parse_now(text: str) -> str | None:
    """Return the steer a `/now` message carries ("" when bare), or None when it is not one."""
    m = _NOW_TOKEN.match(text)
    if m is None:
        return None
    return text[m.end() :].strip()


def spec_fragment(text: str) -> str | None:
    """Return the model fragment being typed at the end of a `/parallel` spec, a completion key.

    Args:
        text: The composer text.

    Returns:
        The fragment after the last comma, or None when the text is not a directive,
        the segment contains `--`, or the spec is a child count.
    """
    body = text.lstrip()
    matches = list(_SEPARATOR.finditer(body))
    if not matches or matches[0].start() != 0:
        return None
    raw = body[matches[-1].end() :]
    if not raw or _SPEC_SEPARATOR.search(raw) is not None:
        return None
    spec = raw.strip()
    if spec.isdecimal():
        return None
    return spec.rsplit(",", 1)[-1].strip()


def stray_directive(text: str) -> str | None:
    """Return the first directive token sitting somewhere other than the start, or None.

    A token further in travels to the model as ordinary text; naming it tells a
    mistyped command from a sentence that mentions one.

    Args:
        text: The steer text.

    Returns:
        The token, such as `/pin`, or None.
    """
    m = _STRAY.search(text)
    return m.group(1) if m is not None else None


def steer_problem(text: str) -> str | None:
    """Explain why the text cannot start an execution as its steer.

    A malformed directive or a front-end command refuses: an execution spent on a
    directive the loop can only decline reads as a silent finish.

    Args:
        text: The task or steer.

    Returns:
        The refusal, or None for ordinary text and a well-formed directive.
    """
    if (m := _FRONT_END_TOKEN.match(text)) is not None:
        return f"{m.group(1)} is a composer command, not an instruction"
    try:
        parse_pin(text)
        parse_directive(text)
        parse_review(text)
    except DirectiveError as exc:
        return str(exc)
    return None


def parse_directive(text: str) -> list[Segment] | None:
    """Split a `/parallel` message into its task segments.

    Args:
        text: The steer text.

    Returns:
        One segment per `/parallel` token, or None when the text is not a directive.

    Raises:
        DirectiveError: A segment has no task or a bare separator; the parse is all or nothing.
    """
    body = text.lstrip()
    matches = list(_SEPARATOR.finditer(body))
    if not matches or matches[0].start() != 0:
        return None
    segments: list[Segment] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        segments.append(_parse_segment(body[m.end() : end]))
    return segments


def _parse_segment(raw: str) -> Segment:
    """Parse one segment's spec and task.

    Args:
        raw: The segment's text after its `/parallel` token.

    Returns:
        The segment.

    Raises:
        DirectiveError: The segment has no task or a bare separator.
    """
    body = raw.strip()
    if not body:
        raise DirectiveError(
            "/parallel needs a task, e.g. `/parallel fix the bug` or `/parallel 2 -- fix the bug`"
        )
    separator = _SPEC_SEPARATOR.search(body)
    if separator is None:
        return Segment(spec="", task=body)
    spec = body[: separator.start()].strip()
    task = body[separator.end() :].strip()
    if not spec:
        raise DirectiveError("/parallel needs a spec before --, e.g. `/parallel 2 -- fix the bug`")
    if not task:
        raise DirectiveError(
            f"/parallel {spec} needs a task after --, e.g. `/parallel {spec} -- fix the bug`"
        )
    return Segment(spec=spec, task=task)


CommandKind = Literal[
    "starts a session", "starts children", "sets state", "controls the run", "opens a view"
]
Surface = Literal[
    "start",
    "plan/ask",
    "pause",
    "steer",
    "resume",
    "script",
    "acp start",
    "acp",
    "parent",
    "child",
    "run -i",
    "ask -i",
    "next",
]


@dataclasses.dataclass(frozen=True, slots=True)
class SlashCommand:
    """Describe one slash command without importing its handler.

    Attributes:
        usage: The command word and its arguments.
        help: One line for the operator.
        kind: What the command does.
        accepts: Where a well-formed command acts rather than travelling as text.
        handler: The dotted path of a function that carries out the command.
        refuses: Where a task or steer explicitly refuses this command word.
    """

    usage: str
    help: str
    kind: CommandKind
    accepts: frozenset[Surface]
    handler: str
    refuses: frozenset[Surface] = frozenset()

    @property
    def name(self) -> str:
        """Return the word a menu completes."""
        return self.usage.partition(" ")[0]


def _at(*surfaces: Surface) -> frozenset[Surface]:
    return frozenset(surfaces)


_LIVE = _at("pause", "steer", "script")
_STEER = _LIVE | _at("acp", "resume", "next")
_VIEW = _at("pause", "steer", "resume")
_TASK_REFUSAL = _at("start", "plan/ask", "acp start", "acp", "child", "next")

_LIVE_ONLY = _TASK_REFUSAL | _at("resume")
_VIEW_ONLY = _TASK_REFUSAL | _at("script")

# Registry order is the completion order for a bare slash.
COMMANDS: tuple[SlashCommand, ...] = (
    SlashCommand(
        "/status",
        "run status: tasks, tools, cost, context, preset",
        "opens a view",
        _at("pause"),
        "agent6.ui.cli.steer_menu._print_status",
    ),
    SlashCommand(
        "/tasks",
        "the task graph with statuses",
        "opens a view",
        _at("pause"),
        "agent6.ui.cli.steer_menu._print_tasks",
    ),
    SlashCommand(
        "/pin <text>",
        "pin an instruction that survives compaction: /pin <text>",
        "sets state",
        _STEER,
        "agent6.harness.operator_bridge.Steering.pin",
    ),
    SlashCommand(
        "/answer <question-id> <text>...",
        "answer a question (pause menu: /answer lists questions, /answer ID asks one)",
        "sets state",
        _at("pause", "acp"),
        "agent6.ui.acp.review._answer_reply",
        _at("acp start"),
    ),
    SlashCommand(
        "/compact [focus]",
        "compact the context now; /compact <focus> steers the summary",
        "controls the run",
        _LIVE,
        "agent6.ui.directive_actions._compact",
        _LIVE_ONLY,
    ),
    SlashCommand(
        "/parallel [SPEC --] TASK",
        "dispatch children: /parallel [SPEC --] TASK (SPEC: count or models; repeat to queue more)",
        "starts children",
        _STEER | _at("start", "acp start", "run -i"),
        "agent6.harness.operator_bridge.Steering.directive",
    ),
    SlashCommand(
        "/restate",
        "restate the conversation since your last message (local, no model call)",
        "opens a view",
        _VIEW,
        "agent6.viewmodel.transcript.restate",
        _VIEW_ONLY,
    ),
    SlashCommand(
        "/review [TARGET]",
        "review changes in this session: /review [TARGET] (a ref or A..B or A...B)",
        "starts a session",
        _STEER | _at("start", "plan/ask", "acp start", "child", "ask -i"),
        "agent6.app.review.directive.build",
    ),
    SlashCommand(
        "/undo",
        "fork back to before your last message (the text returns to edit and resend)",
        "starts a session",
        _VIEW | _at("run -i", "next"),
        "agent6.app.undo.undo_fork",
        _at("script"),
    ),
    SlashCommand(
        "/btw <question>",
        "ask a question beside the run: /btw <question> (answers inline, later)",
        "starts a session",
        _LIVE,
        "agent6.ui.btw.open_btw",
        _LIVE_ONLY,
    ),
    SlashCommand(
        "/task <text>",
        "queue work into the task graph: /task <text> (worked when the queue drains)",
        "sets state",
        _LIVE,
        "agent6.ui.directive_actions._task",
        _LIVE_ONLY,
    ),
    SlashCommand(
        "/metric [off | SPEC]",
        "Show, set, or turn off the run metric.",
        "sets state",
        _LIVE,
        "agent6.ui.directive_actions._metric",
        _LIVE_ONLY,
    ),
    SlashCommand(
        "/standing <text>",
        "set the goal the run returns to when the queue drains: /standing <text>",
        "sets state",
        _LIVE,
        "agent6.ui.directive_actions._standing",
        _LIVE_ONLY,
    ),
    SlashCommand(
        "/retire <task id>",
        "drop a task from the graph: /retire <task id>, the number the task tree shows",
        "sets state",
        _LIVE,
        "agent6.ui.directive_actions._retire",
        _LIVE_ONLY,
    ),
    SlashCommand(
        "/now <text>",
        "Interrupt the current call",
        "controls the run",
        _LIVE,
        "agent6.ui.directive_actions.submit_composer_line",
        _LIVE_ONLY,
    ),
    SlashCommand(
        "/continue",
        "resume the run unchanged (same as Enter)",
        "controls the run",
        _at("pause", "run -i"),
        "agent6.ui.cli.steer_menu._answer_line",
    ),
    SlashCommand(
        "/stop",
        "stop the run now, as `agent6 stop` does (resumable)",
        "controls the run",
        _LIVE | _at("run -i"),
        "agent6.app.stop.stop_session",
        _LIVE_ONLY,
    ),
    SlashCommand(
        "/shells",
        "background commands this run started, and how they ended",
        "opens a view",
        _VIEW,
        "agent6.ui.cli.steer_menu._print_shells",
        _VIEW_ONLY,
    ),
    SlashCommand(
        "/<skill> [args]",
        "use an enabled skill as the instruction",
        "sets state",
        _STEER,
        "agent6.harness.operator_bridge.Steering.skill",
    ),
    SlashCommand(
        "/exit",
        "stop the session and leave the prompt",
        "controls the run",
        _at("pause", "run -i", "ask -i", "next"),
        "agent6.ui.cli.session_prompt.end_of_session_prompt",
    ),
    SlashCommand(
        "/detach",
        "keep the run going in the background",
        "controls the run",
        _at("pause"),
        "agent6.ui.cli.steer_menu._answer_line",
    ),
    SlashCommand(
        "/help",
        "list the commands at this prompt",
        "opens a view",
        _at("pause", "run -i"),
        "agent6.ui.cli.steer_menu._print_help",
    ),
    SlashCommand(
        "/cost",
        "print the token and cost summary",
        "opens a view",
        _at("run -i", "ask -i"),
        "agent6.budget.BudgetTracker.format_summary",
    ),
    SlashCommand(
        "/diff",
        "show the diff from the base commit to the run branch tip",
        "opens a view",
        _at("run -i"),
        "agent6.ui.cli.repl.repl_run_diff",
    ),
    SlashCommand(
        "/watch",
        "show the last 20 audit events from this run",
        "opens a view",
        _at("run -i"),
        "agent6.ui.cli.repl.repl_show_recent_events",
    ),
    SlashCommand(
        "/mcp",
        "list the MCP servers and tools in this run",
        "opens a view",
        _at("run -i"),
        "agent6.ui.cli.repl.repl_list_mcp",
    ),
    SlashCommand(
        "/init",
        "run the setup wizard in the current directory",
        "sets state",
        _at("run -i"),
        "agent6.ui.cli.repl.repl_run_init",
    ),
    SlashCommand(
        "/quit",
        "stop at this prompt",
        "controls the run",
        _at("run -i", "ask -i"),
        "agent6.ui.cli.repl.build_repl_hook",
    ),
    SlashCommand(
        "/reset",
        "clear the prior questions and answers",
        "sets state",
        _at("ask -i"),
        "agent6.ui.cli.ask.run_ask_repl",
    ),
    SlashCommand(
        "/receipt",
        "Inspect recorded testing, review and reversal evidence",
        "opens a view",
        _at("acp"),
        "agent6.ui.acp.review._read_only_reply",
        _at("acp start"),
    ),
    SlashCommand(
        "/findings [--all]",
        "List this run's open review findings",
        "opens a view",
        _at("acp"),
        "agent6.ui.acp.review._read_only_reply",
        _at("acp start"),
    ),
    SlashCommand(
        "/fix <finding-id>... --resume | --all --resume",
        "Resume an idle session to reproduce stored findings; leaves them open",
        "starts a session",
        _at("acp"),
        "agent6.ui.acp.review.handle_command",
        _at("acp start"),
    ),
    SlashCommand(
        "/resolve <finding-id> --as <disposition> --reason <text> [--duplicate-of <id>]",
        "Resolve a finding with the operator's disposition and reason",
        "sets state",
        _at("acp"),
        "agent6.ui.acp.review.handle_command",
        _at("acp start"),
    ),
    SlashCommand(
        "/reopen <finding-id> --reason <text>",
        "Reopen a finding with a reason",
        "sets state",
        _at("acp"),
        "agent6.ui.acp.review.handle_command",
        _at("acp start"),
    ),
)


def commands_for(surface: Surface) -> tuple[SlashCommand, ...]:
    """Return the built-ins for a surface. Installed skills supply their own names and help."""
    return tuple(c for c in COMMANDS if surface in c.accepts and c.name != "/<skill>")


VIEW_COMMANDS = frozenset(c.name for c in COMMANDS if "script" in c.refuses)
_FRONT_END_TOKEN = re.compile(
    r"\A\s*("
    + "|".join(re.escape(c.name) for c in COMMANDS if "start" in c.refuses)
    + r")(?=\s|\Z)"
)

# A directive token inside a line; `/parallel` is excluded, since a later one separates tasks.
_STRAY = re.compile(
    r"(?<=\s)("
    + "|".join(re.escape(c.name) for c in commands_for("steer") if c.name != "/parallel")
    + r")(?=\s|\Z)"
)
