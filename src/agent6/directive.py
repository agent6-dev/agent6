# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Parse the steer directives the harness and the composers share.

    /parallel [spec] <task text> [/parallel [spec] <task text>]...
    /pin <instruction that must survive context compaction>
    /compact [focus text for the summary]

A message is a directive only when it starts with the exact token; a token
glued to a word or sitting mid-text is ordinary text, and newlines are task
characters. A `/parallel` spec is a positive lane count or a comma-separated
list of `[provider/]model` entries; a segment's first token counts as a spec
when it holds a comma or a slash, so a task starting with a path parses as a
model spec (start with a verb). Pure stdlib parsing with no agent6 imports: a
leaf both layers sit above.
"""

from __future__ import annotations

import dataclasses
import re

# A whitespace-delimited `/parallel` token; not MULTILINE, so the anchors are the whole string's.
_SEPARATOR = re.compile(r"(?:\A|(?<=\s))/parallel(?=\s|\Z)", re.IGNORECASE)


class DirectiveError(ValueError):
    """A directive or spec is malformed: a bare token, a missing task, a bad lane count."""


@dataclasses.dataclass(frozen=True, slots=True)
class Segment:
    """One parsed `/parallel` task.

    Attributes:
        spec: The lane spec; "" is one default lane.
        task: The task text, internal whitespace and newlines preserved.
    """

    spec: str
    task: str


def parse_spec(spec: str, *, limit: int) -> list[str | None]:
    """Parse a lane spec into one entry per lane, the grammar the directive and the CLI share.

    Args:
        spec: A positive lane count, a comma-separated `[provider/]model` list, or "".
        limit: The caller's `[parallel].max_lanes`; an over-limit count refuses before
            the list is built.

    Returns:
        One entry per lane: None for the configured worker model, else the lane's
        model text for the caller to resolve. "" is one default lane.

    Raises:
        DirectiveError: The count is not positive or over the limit, or the list
            names no models.
    """
    s = spec.strip()
    if not s:
        return [None]
    # isdecimal is exactly the set int() parses; isdigit accepts superscripts int() rejects.
    if s.isdecimal():
        n = int(s)
        if n < 1:
            raise DirectiveError("parallel lane count must be >= 1")
        if n > limit:
            raise DirectiveError(_over_limit(n, limit))
        return [None] * n
    models = [m.strip() for m in s.split(",") if m.strip()]
    if not models:
        raise DirectiveError(f"parallel spec {spec!r} names no models")
    if len(models) > limit:
        raise DirectiveError(_over_limit(len(models), limit))
    return list(models)


def _over_limit(requested: int, limit: int) -> str:
    """Return the refusal for a lane count over the limit."""
    return (
        f"parallel spec requests {requested} lanes but [parallel].max_lanes = {limit}."
        " Request fewer, or raise [parallel].max_lanes."
    )


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


# The spec token of the last `/parallel` segment still being typed; a following space ends it.
_SPEC_TAIL = re.compile(r"[^\S\n]+(\S*)\Z")


def spec_fragment(text: str) -> str | None:
    """Return the model fragment being typed at the end of a `/parallel` spec, a completion key.

    Args:
        text: The composer text.

    Returns:
        The fragment after the last comma, or None when the text is not a directive,
        the caret has left the spec, or the token is a lane count.
    """
    matches = list(_SEPARATOR.finditer(text))
    if not matches or matches[0].start() != 0:
        return None
    m = _SPEC_TAIL.match(text, matches[-1].end())
    if m is None:
        return None
    token = m.group(1)
    if token.isdigit():
        return None
    return token.rsplit(",", 1)[-1]


# Directives a front-end acts on itself; the loop parses none, so none can start an execution.
LIVE_RUN_COMMANDS: frozenset[str] = frozenset(
    {"/compact", "/btw", "/now", "/retire", "/standing", "/stop", "/task"}
)
_FRONT_END_COMMANDS: frozenset[str] = LIVE_RUN_COMMANDS | {"/restate", "/shells"}
_FRONT_END_TOKEN = re.compile(
    r"\A\s*(" + "|".join(map(re.escape, sorted(_FRONT_END_COMMANDS))) + r")(?=\s|\Z)"
)


def stray_directive(text: str) -> str | None:
    """Return the first directive token sitting somewhere other than the start, or None.

    A token further in travels to the model as ordinary text; naming it tells a
    mistyped command from a sentence that mentions one.
    """
    m = _STRAY.search(text)
    return m.group(1) if m is not None else None


def steer_problem(text: str) -> str | None:
    """Explain why the text cannot start an execution as its steer.

    A malformed directive or a front-end command refuses: an execution spent on a
    directive the loop can only decline reads as a silent finish.

    Returns:
        The refusal, or None for ordinary text and a well-formed directive.
    """
    if (m := _FRONT_END_TOKEN.match(text)) is not None:
        return (
            f"{m.group(1)} is a composer command, not an instruction;"
            " start this execution, then type it in the composer"
        )
    try:
        parse_pin(text)
        parse_directive(text)
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
        DirectiveError: A segment has no task; the parse is all or nothing.
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


def _is_spec_token(token: str) -> bool:
    """Return whether a segment's first token is a spec: a count, a comma list or a slash."""
    return token.isdecimal() or "," in token or "/" in token


def _parse_segment(raw: str) -> Segment:
    """Parse one segment's spec and task.

    Returns:
        The segment.

    Raises:
        DirectiveError: The segment has no task.
    """
    body = raw.strip()
    if not body:
        raise DirectiveError(
            "/parallel needs a task, e.g. `/parallel fix the bug` or `/parallel 2 fix the bug`"
        )
    parts = body.split(None, 1)
    if _is_spec_token(parts[0]):
        spec, task = parts[0], (parts[1] if len(parts) > 1 else "")
    else:
        spec, task = "", body
    if not task:
        raise DirectiveError(
            f"/parallel {parts[0]} needs a task, e.g. `/parallel {parts[0]} fix the bug`"
        )
    return Segment(spec=spec, task=task)


# The words a view acts on itself; a scripted steer has no view, so `agent6 steer` refuses them.
VIEW_COMMANDS: frozenset[str] = frozenset({"/restate", "/shells", "/undo"})

# The directives a composer completes, with their help; the web client mirrors the strings verbatim.
STEER_COMMANDS: dict[str, str] = {
    "/pin": "pin an instruction that survives compaction: /pin <text>",
    "/compact": "compact the context now; /compact <focus> steers the summary",
    "/parallel": "fan out lanes: /parallel [N|models] <task> (repeat to queue more)",
    "/restate": "restate the conversation since your last message (local, no model call)",
    "/undo": "fork back to before your last message (the text returns to edit and resend)",
    "/btw": "ask a question beside the run: /btw <question> (answers inline, later)",
    "/task": "queue work into the task graph: /task <text> (worked when the queue drains)",
    "/standing": "set the goal the run returns to when the queue drains: /standing <text>",
    "/retire": "drop a task from the graph: /retire <task id>, the number the task tree shows",
    "/now": "steer at once, aborting the call in flight: /now <text> (Ctrl+Enter on the web)",
    "/stop": "stop the run now, as `agent6 stop` does (resumable)",
    "/shells": "background commands this run started, and how they ended",
}


# A directive token inside a line; `/parallel` is excluded, since a later one separates tasks.
_STRAY = re.compile(
    r"(?<=\s)("
    + "|".join(re.escape(c) for c in sorted(STEER_COMMANDS) if c != "/parallel")
    + r")(?=\s|\Z)"
)
