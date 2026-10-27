# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Hold every key agent6 answers to, in one table.

The same key often means the same thing on several surfaces; spelling them
once keeps them from drifting and makes a collision visible in one file. A
leaf with no agent6 or textual imports, so every front-end can read it.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True, slots=True)
class Answer:
    """One answer to an approval.

    Attributes:
        key: The letter.
        answer: The value written to the answer bridge.
        label: The word every surface shows for it.
        aliases: The words the CLI prompt also takes.
        standing: Offered only when the operator may answer for the whole session.
        grants: Allows the call; the bridge's rule, pinned to this table.
    """

    key: str
    answer: str
    label: str
    aliases: tuple[str, ...] = ()
    standing: bool = False
    grants: bool = False


# The order they are offered in, everywhere.
APPROVAL_ANSWERS: tuple[Answer, ...] = (
    Answer("y", "yes", "allow", ("yes",), grants=True),
    Answer("a", "session", "allow all", ("all", "always", "session"), standing=True, grants=True),
    Answer("n", "no", "deny", ("no",)),
    Answer("d", "session-deny", "deny all", ("deny", "never"), standing=True),
)


def answer_entry(answer: str) -> Answer:
    """Return the table's row for a bridge value; one outside the table is a programming error."""
    return next(entry for entry in APPROVAL_ANSWERS if entry.answer == answer)


def answer_for(typed: str, *, standing: bool) -> str:
    """Return the answer a typed line means: the letter or an alias, else "no".

    Args:
        typed: The operator's reply; case and surrounding whitespace are ignored.
        standing: The operator may answer for the whole session; when False, a session
            answer denies, as `[y/N]` says.

    Returns:
        The bridge value: "yes", "session", "no" or "session-deny".
    """
    word = typed.strip().lower()
    for entry in APPROVAL_ANSWERS:
        if word in (entry.key, *entry.aliases) and (standing or not entry.standing):
            return entry.answer
    return "no"


def approval_prompt_suffix(*, standing: bool) -> str:
    """Return the `[y/N/a/d]` suffix the CLI prompt ends with, built from the table."""
    if not standing:
        return "[y/N]: "
    # The two plain answers first, then the scoped pair the line goes on to explain.
    ordered = [e for e in APPROVAL_ANSWERS if not e.standing] + [
        e for e in APPROVAL_ANSWERS if e.standing
    ]
    letters = "/".join(e.key.upper() if e.answer == "no" else e.key for e in ordered)
    scoped = ", ".join(f"{e.key} = {e.label}" for e in APPROVAL_ANSWERS if e.standing)
    return f"[{letters}]  ({scoped}, this session): "


FINDINGS = "Findings"
REVIEW_TEXT = {
    "back": "Back",
    "fix": "Fix",
    "resolve": "Resolve",
    "reopen": "Reopen",
    "refresh": "Refresh",
    "receipt": "Receipt",
    "revert": "Ask a run to revert",
    "help": "Help",
    "keys": "Keys & actions",
    "palette": "Command palette",
    "reason": "Reason (required, at most 2,000 characters)",
    "duplicate": "Duplicate finding ID (for duplicate only)",
    "submit": "Submit",
    "cancel": "Cancel",
    "resume": "Resume to fix",
    "confirm": "Resume this run to reproduce finding {finding_id}? The finding stays open.",
    "empty": "No recorded findings.",
    "summary": "Open: {block} block, {warn} warn, {nit} nit. Review: {completion}.",
    "detail": (
        "{finding_id} · {status} · {severity}\n{location}\n{title}\n{detail}\nFix: {fix_state}"
    ),
    "pre_existing": "  Pre-existing line: {base_location}. {reason}\n  Blame: {commits}",
    "history": "{action}: {reason}",
    "result": "{message} (request {request_id})",
    "preview": "Reviewed diff: {location}",
    "missing": "No recorded review target.",
    "no_hunk": "No cited hunk is available in this preview.",
    "truncated": "\n… (truncated)",
    "panel": "Panel {panel_id}: block_on={block_on}, rejected={blocked}, disarmed={disarmed}",
}
# The review queue's groups as the hub names them.
QUEUE_TEXT = {
    "blocked": "blocked",
    "unverified": "unverified",
    "awaiting_merge": "awaiting merge",
    "other": "other",
}
REVIEW_COLUMNS = ("ID", "Status", "Severity", "Location", "Title")
REVIEW_HINTS = ("↑↓ selects a finding; Tab focuses its detail or read-only diff.",)
REVIEW_DISPOSITIONS = (
    ("Fixed", "fixed"),
    ("False positive", "false_positive"),
    ("Intentional", "intentional"),
    ("Won't fix", "wont_fix"),
    ("Duplicate", "duplicate"),
    ("Pre-existing line", "pre_existing"),
)


# Every key a TUI screen binds to a menu action, by screen; an action absent here is reachable
# from the menu and the palette only. Commas join one action's aliases; the first carries the
# footer entry. A destructive letter is confirmed before it acts.
SCREEN_KEYS: dict[str, dict[str, str]] = {
    "hub": {
        "new_work": "n",
        "open_selected": "enter",
        "merge_selected": "m",
        "delete_selected": "d",
        "refresh": "r",
        "quit": "q",
        "open_config": "c",
        "open_machines": "M",
        "view_logs": "l",
        "view_findings": "v",
        "toggle_lanes": "space",
        "toggle_sort": "s",
        "help": "question_mark",
        "command_palette": "ctrl+p",
    },
    # The run views carry no bare letters: the composer has the keyboard.
    "conversation": {
        "close": "escape",
        "quit_hub": "ctrl+q",
        "history_search": "ctrl+r",
        "cycle_detail": "ctrl+t",
        "page_up": "pageup",
        "page_down": "pagedown",
        "scroll_top": "ctrl+home",
        "scroll_bottom": "ctrl+end",
        "copy": "ctrl+c",
        "toggle_dashboard": "ctrl+d",
        "help": "question_mark",
        "command_palette": "ctrl+p",
    },
    "dashboard": {
        "to_hub": "escape",
        "quit_hub": "ctrl+q",
        "history_search": "ctrl+r",
        "view_findings": "ctrl+f",
        "focus_next_pane": "tab",
        "focus_prev_pane": "shift+tab",
        "page_up": "pageup",
        "page_down": "pagedown",
        "scroll_top": "ctrl+home",
        "scroll_bottom": "ctrl+end",
        "copy": "ctrl+c",
        "toggle_dashboard": "ctrl+d",
        "help": "question_mark",
        "command_palette": "ctrl+p",
    },
    "findings": {
        "close": "escape,q",
        "fix": "f",
        "resolve": "s",
        "reopen": "o",
        "receipt": "e",
        "refresh": "r",
        "help": "question_mark",
        "command_palette": "ctrl+p",
    },
    "finding disposition": {"submit": "ctrl+s", "close": "escape"},
    "receipt": {"close": "escape,q", "revert": "r"},
    "new work": {
        "close": "escape",
        "quit_hub": "ctrl+q",
        "help": "question_mark",
        "command_palette": "ctrl+p",
    },
    # `l` closes the log too: the key that opened it toggles it shut.
    "event log": {
        "close": "escape,q,l",
        "page_up": "pageup",
        "page_down": "pagedown",
        "scroll_top": "ctrl+home",
        "scroll_bottom": "ctrl+end",
        "reload": "r",
        "help": "question_mark",
        "command_palette": "ctrl+p",
    },
    "config": {
        "reload": "r",
        "close": "escape,q",
        "quit": "ctrl+q",
        "edit": "e",
        "add_provider": "a",
        "add_mcp_server": "c",
        "add_skill": "s",
        "add_reviewer": "v",
        "remove_entry": "x",
        "remove_reviewer": "X",
        "reset": "d",
        "search": "/",
        "toggle_modified": "m",
        "help": "question_mark",
        "command_palette": "ctrl+p",
    },
    # `r` refreshes on every screen that refreshes, so running a machine takes the shifted letter.
    "machines": {
        "close": "escape,q",
        "quit": "ctrl+q",
        "view": "v",
        "run": "R",
        "watch": "w",
        "create": "c",
        "refresh": "r",
        "help": "question_mark",
        "command_palette": "ctrl+p",
    },
    "machine watch": {
        "close": "escape,q",
        "steer": "s",
        "poke": "m",
        "stop": "x",
        "help": "question_mark",
        "command_palette": "ctrl+p",
    },
}
