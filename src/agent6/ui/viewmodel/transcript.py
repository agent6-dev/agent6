# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Fold a run's event stream into an ordered conversation of `TranscriptItem`s.

The medium-agnostic half of live conversation rendering. `TranscriptFold` walks
`logs.jsonl` events in emission order and yields the things worth showing -- a
reasoning block, an assistant message, a tool call coalesced with its result, a
commit, the final verdict -- as plain data. Each front-end (the CLI ANSI stream,
the TUI RichLog, the web SPA) maps these items to its own styling; the glyphs and
content helpers here are shared so the three never drift.

`fold_transcript` is the batch form (whole stream at once); the CLI/TUI live
tailers feed the same `TranscriptFold` one event at a time.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import Any, Literal

# ANSI/CSI escape sequences (colored tool output). Stripped from fold previews:
# the fold is plain data consumed by non-terminal surfaces (web, saved
# transcripts, TUI widgets) that render escapes as literal garbage; the live CLI
# stream styles its own output separately.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


# Shared glyph vocabulary (text characters, not graphics, so every terminal font
# renders them). One place so cli/tui/web agree.
CALL = "→"  # a tool call
RESULT = "└"  # its result, on the line below (U+2514: base box-drawing, renders in every mono font)
COMMIT = "✎"  # an auto-commit
THINK = "·"  # a reasoning block
DONE = "●"  # run start / final verdict
OPERATOR = "❯"  # noqa: RUF001 -- deliberate prompt glyph, not a mistyped >

# Tool names the loop treats as terminal; their call is folded into the final
# verdict rather than shown as an ordinary step. Kept as literals so viewmodel
# stays free of a tools import (layering).
_FINISH_TOOLS = frozenset({"finish_run", "finish_planning"})

# Friendly word for a run.end reason on the terminal/TUI "done" line, so a stop
# reads as "stopped" (not the raw "steer_abort") and an error names itself.
_END_REASON_LABEL = {
    "steer_abort": "stopped",
    "finish_run": "finished",
    "answered": "answered",
    "provider_error": "provider error",
    "budget_exhausted": "budget exhausted",
    "went_quiet": "went quiet",
    "max_iterations": "hit iteration cap",
}

ItemKind = Literal["thinking", "text", "tool", "commit", "marker", "done", "operator"]


@dataclass(frozen=True, slots=True)
class TranscriptItem:
    """One rendered conversation step. Only the fields its `kind` needs are set."""

    kind: ItemKind
    body: str = ""  # thinking / text / marker prose; the final summary for `done`
    name: str = ""  # tool name
    arg: str = ""  # the tool's salient argument (path, pattern, command, ...)
    ok: bool | None = None  # tool or run outcome (None = not applicable / in flight)
    detail: str = ""  # tool result summary, verify badge, or commit/done metadata
    tail: str = ""  # a failed tool's captured output tail


_PRIMARY_ARGS = ("path", "file", "pattern", "query", "command", "cmd", "url", "title", "summary")


def _clip(text: str, n: int = 60) -> str:
    return text if len(text) <= n else text[: n - 3] + "…"


def salient_arg(args: Any) -> str:
    """The one argument worth showing beside a tool name (best effort). Takes
    untrusted event data, so a non-dict `args` is tolerated, not assumed away."""
    if not isinstance(args, dict) or not args:
        return ""
    # argv (run_command): a shell-style line, not a Python list repr.
    argv = args.get("argv")
    if isinstance(argv, (list, tuple)) and argv:
        return _clip(shlex.join(str(a) for a in argv))
    # ask_user: the question text, not the nested {questions:[{...}]} repr.
    questions = args.get("questions")
    if isinstance(questions, (list, tuple)) and questions:
        first = questions[0]
        q = first.get("question", "") if isinstance(first, dict) else str(first)
        more = f" (+{len(questions) - 1})" if len(questions) > 1 else ""
        return _clip(str(q)) + more
    for key in _PRIMARY_ARGS:
        value = args.get(key)
        if isinstance(value, (str, int)):
            return _clip(str(value))
    key, value = next(iter(args.items()))
    return _clip(f"{key}={value}")


class TranscriptFold:
    """Incremental event -> `TranscriptItem` fold. Feed events in order; each
    `feed` returns the items that event completed (usually zero or one)."""

    def __init__(self) -> None:
        self._thinking: list[str] = []
        self._text: list[str] = []
        # name -> salient arg for calls awaiting their result. A dict (not one
        # slot) so interleaved calls -- a concurrent explore-tier review panel
        # shares one dispatcher across threads -- pair by name, not by position.
        self._pending: dict[str, str] = {}
        self._verify: tuple[bool, str] | None = None  # (ok, badge) for run_verify_command
        self._finish = ""  # summary from the terminal finish tool
        self._tools = 0
        self._commits = 0

    def feed(self, event: dict[str, Any]) -> list[TranscriptItem]:  # noqa: PLR0911
        etype = event.get("type", "")
        if etype == "role.call":
            self._thinking.clear()
            self._text.clear()
            return []
        if etype == "role.thinking_delta":
            self._thinking.append(str(event.get("text", "")))
            return []
        if etype == "role.text_delta":
            self._text.append(str(event.get("text", "")))
            return []
        if etype == "role.result":
            return self._flush_message()
        if etype == "tool.call":
            out = self._flush_message()  # a turn's prose precedes its calls
            name = str(event.get("name", ""))
            if name in _FINISH_TOOLS:
                self._finish = str((event.get("args") or {}).get("summary", "")).strip()
                return out
            self._tools += 1
            self._pending[name] = salient_arg(event.get("args") or {})
            self._verify = None
            return out
        if etype == "verify.end":
            code = event.get("exit_code")
            dur = float(event.get("duration_s", 0) or 0)
            badge = "✓ pass" if code == 0 else f"✗ exit {code}"
            self._verify = (code == 0, f"{badge} · {dur:.1f}s")
            return []
        if etype == "tool.result":
            return self._complete_tool(event)
        if etype == "diff.updated":
            self._commits += 1
            n = len(str(event.get("patch", "")).splitlines())
            return [TranscriptItem("commit", detail=f"{n} lines")]
        if etype == "loop.steer.injected":
            # The operator's typed instruction (a steer, or the follow-up a
            # resume was started with), shown in the conversation like any
            # other turn. Older logs carry only a char count -- no item then.
            out = self._flush_message()
            text = str(event.get("text", "")).strip()
            if text:
                out.append(TranscriptItem("operator", body=text))
            return out
        if etype == "run.end":
            out = self._flush_message()
            tools = f"{self._tools} tool{'' if self._tools == 1 else 's'}"
            commits = f"{self._commits} commit{'' if self._commits == 1 else 's'}"
            counts = f"{tools} · {commits}"
            reason = str(event.get("reason", ""))
            # Pair the finish summary with the done line ONLY on a clean finish.
            # On a failure/stop the summary is from an EARLIER finish_run call and
            # pairing it (e.g. "provider error  Plan seeded.") misreads as success.
            body = self._finish if reason in ("", "finish_run") else ""
            out.append(
                TranscriptItem(
                    "done",
                    body=body,
                    ok=bool(event.get("all_passed")),
                    detail=counts,
                    name=_END_REASON_LABEL.get(reason, reason),
                )
            )
            return out
        return []

    def _flush_message(self) -> list[TranscriptItem]:
        out: list[TranscriptItem] = []
        thinking = "".join(self._thinking).strip()
        self._thinking.clear()
        if thinking:
            out.append(TranscriptItem("thinking", body=thinking))
        text = "".join(self._text).strip()
        self._text.clear()
        if text:  # only when non-empty: no more blank response blocks
            out.append(TranscriptItem("text", body=text))
        return out

    def _complete_tool(self, event: dict[str, Any]) -> list[TranscriptItem]:
        name = str(event.get("name", ""))
        if name not in self._pending:  # a finish tool's result, or an unmatched one
            return []
        arg = self._pending.pop(name)
        if name == "run_verify_command" and self._verify is not None:
            ok, detail = self._verify
            self._verify = None
        else:
            ok = event.get("ok") in (True, "True")
            detail = str(event.get("summary", "")).strip()
        # A failed tool shows why (stderr, else stdout). run_command succeeds
        # silently otherwise, so show a short stdout tail on success too: the
        # operator asked to run it to SEE the output (git status, a version).
        if not ok:
            tail = str(event.get("stderr_tail") or event.get("stdout_tail") or "").strip()
        elif name in ("run_command", "run_metric_command"):
            tail = str(event.get("stdout_tail") or "").strip()
        else:
            tail = ""
        return [
            TranscriptItem(
                "tool",
                name=name,
                arg=arg,
                ok=ok,
                detail=_strip_ansi(detail),
                tail=_strip_ansi(tail),
            )
        ]


def fold_transcript(events: list[dict[str, Any]]) -> list[TranscriptItem]:
    """Fold a whole event stream into its ordered conversation items."""
    fold = TranscriptFold()
    out: list[TranscriptItem] = []
    for event in events:
        out.extend(fold.feed(event))
    return out
