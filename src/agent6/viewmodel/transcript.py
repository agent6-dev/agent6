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

from dataclasses import dataclass
from typing import Any, Literal

# Shared glyph vocabulary (text characters, not graphics, so every terminal font
# renders them). One place so cli/tui/web agree.
CALL = "→"  # a tool call
RESULT = "⎿"  # its result, on the line below
COMMIT = "✎"  # an auto-commit
THINK = "·"  # a reasoning block
DONE = "●"  # run start / final verdict

# Tool names the loop treats as terminal; their call is folded into the final
# verdict rather than shown as an ordinary step. Kept as literals so viewmodel
# stays free of a tools import (layering).
_FINISH_TOOLS = frozenset({"finish_run", "finish_planning"})

ItemKind = Literal["thinking", "text", "tool", "commit", "marker", "done"]


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


def salient_arg(args: Any) -> str:
    """The one argument worth showing beside a tool name (best effort). Takes
    untrusted event data, so a non-dict `args` is tolerated, not assumed away."""
    if not isinstance(args, dict) or not args:
        return ""
    for key in _PRIMARY_ARGS:
        value = args.get(key)
        if isinstance(value, (str, int)):
            text = str(value)
            return text if len(text) <= 60 else text[:57] + "…"
    key, value = next(iter(args.items()))
    text = f"{key}={value}"
    return text if len(text) <= 60 else text[:57] + "…"


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
        if etype == "run.end":
            out = self._flush_message()
            counts = f"{self._tools} tools · {self._commits} commit(s)"
            out.append(
                TranscriptItem(
                    "done",
                    body=self._finish,
                    ok=bool(event.get("all_passed")),
                    detail=counts,
                    name=str(event.get("reason", "")),
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
        tail = "" if ok else str(event.get("stderr_tail") or event.get("stdout_tail") or "").strip()
        return [TranscriptItem("tool", name=name, arg=arg, ok=ok, detail=detail, tail=tail)]


def fold_transcript(events: list[dict[str, Any]]) -> list[TranscriptItem]:
    """Fold a whole event stream into its ordered conversation items."""
    fold = TranscriptFold()
    out: list[TranscriptItem] = []
    for event in events:
        out.extend(fold.feed(event))
    return out
