# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A small fish-style line reader for the pause menu.

Tab previews the matching slash commands with their descriptions below the line
and cycles them; Up and Down recall history; Ctrl-R searches it with the line as
the query; Esc restores what was typed. Enter accepts, Ctrl-C stops the run,
Ctrl-D on an empty line continues it. Hand-rolled on termios: GNU readline's
menu-complete cycles blind and libedit has only plain completion. Unix only;
callers gate on `menu_capable`. The rendering uses CR, erase-below, cursor
movement, reverse and dim only, and never wraps a row, which would break the
cursor-up arithmetic.
"""

from __future__ import annotations

import os
import select
import sys
from collections.abc import Callable

from agent6.ui.cli._terminal_guard import raw_stream
from agent6.viewmodel.transcript import scrub_terminal_controls

try:
    import termios
    import tty
except ImportError:  # pragma: no cover  # Windows callers gate on menu_capable()
    termios = None  # type: ignore[assignment]
    tty = None  # type: ignore[assignment]

_CSI_FINAL = {
    b"A": "up",
    b"B": "down",
    b"C": "right",
    b"D": "left",
    b"Z": "backtab",
    b"H": "home",
    b"F": "end",
}
_TILDE_SEQ = {b"1~": "home", b"7~": "home", b"4~": "end", b"8~": "end", b"3~": "delete"}
_CTRL = {
    b"\r": "enter",
    b"\n": "enter",
    b"\t": "tab",
    b"\x7f": "backspace",
    b"\x08": "backspace",
    b"\x03": "interrupt",
    b"\x04": "eof",
    b"\x01": "home",  # Ctrl-A
    b"\x05": "end",  # Ctrl-E
    b"\x15": "kill-line",  # Ctrl-U
    b"\x17": "kill-word",  # Ctrl-W
    b"\x0b": "kill-to-end",  # Ctrl-K
    b"\x0c": "redraw",  # Ctrl-L
    b"\x12": "history-search",  # Ctrl-R
}

# History search renders only the newest matches; typing narrows to the rest.
_SEARCH_ROWS = 8
_SEARCH_PROMPT = "search: "


def menu_capable() -> bool:
    """Return whether the reader can own the line: termios exists and both std streams are a tty."""
    return termios is not None and sys.stdin.isatty() and sys.stdout.isatty()


class LineSuperseded(Exception):  # noqa: N818  # a signal, not an error
    """The line being read was answered by another route: `until` held while it waited."""


def read_line_until(fd: int, until: Callable[[], bool] | None) -> str | None:
    """Read one line, a byte at a time, until a newline, EOF or `until` holds.

    A paste's later lines stay in the descriptor for the next prompt; a partial line
    is dropped once `until` holds, since it was aimed at a prompt that is over.

    Args:
        fd: The descriptor to read.
        until: Polled every 0.2 s while nothing is typed.

    Returns:
        The line, or None at EOF or once `until` held.
    """
    line = b""
    while True:
        if select.select([fd], [], [], 0)[0]:
            byte = os.read(fd, 1)
            if not byte:
                return line.decode("utf-8", errors="replace") if line else None
            if byte == b"\n":
                return line.decode("utf-8", errors="replace").rstrip("\r")
            line += byte
            continue
        if until is not None and until():
            return None
        select.select([fd], [], [], 0.2 if until is not None else None)


def _read_key_until(fd: int, until: Callable[[], bool]) -> str:
    """Return the next key, polling `until` every 0.2 s while nothing is typed.

    Raises:
        LineSuperseded: `until` held first.
    """
    while not select.select([fd], [], [], 0.2)[0]:
        if until():
            raise LineSuperseded
    return _read_key(fd)


def _read_escape(fd: int) -> str:
    """Return the rest of an ESC-initiated key: a name, "esc" for a bare Escape, "" if ignored."""
    # Distinguish a bare Esc from an escape sequence by a short poll.
    ready, _, _ = select.select([fd], [], [], 0.03)
    if not ready:
        return "esc"
    lead = os.read(fd, 1)
    if lead not in (b"[", b"O"):
        if lead and lead[0] >= 0xC0:  # an Alt-chord on a multibyte character
            _read_exactly(fd, _continuation_bytes(lead[0]))
        return "esc"
    seq = b""
    while True:
        ch = os.read(fd, 1)
        if not ch:
            break
        seq += ch
        if 0x40 <= ch[0] <= 0x7E:  # a CSI final byte
            break
    if seq[-1:] == b"~":
        return _TILDE_SEQ.get(seq, "")
    return _CSI_FINAL.get(seq[-1:], "")


def _read_key(fd: int) -> str:
    """Return one logical key: a name from the tables, `char:<c>` for text, "" if ignored."""
    data = os.read(fd, 1)
    if not data:
        return "eof"
    if data in _CTRL:
        return _CTRL[data]
    b = data[0]
    if b == 0x1B:
        return _read_escape(fd)
    if b < 0x20:
        return ""
    if b >= 0xC0:  # a UTF-8 lead byte
        data += _read_exactly(fd, _continuation_bytes(b))
    return "char:" + data.decode("utf-8", errors="replace")


def _continuation_bytes(lead: int) -> int:
    """Return how many bytes follow a UTF-8 lead byte."""
    return 1 if lead < 0xE0 else 2 if lead < 0xF0 else 3


def _read_exactly(fd: int, n: int) -> bytes:
    """Return exactly n bytes, so a split character leaves no byte behind; short only at EOF."""
    data = b""
    while len(data) < n:
        chunk = os.read(fd, n - len(data))
        if not chunk:
            break
        data += chunk
    return data


def _width() -> int:
    """Return the terminal width, 80 without a terminal."""
    try:
        cols = os.get_terminal_size(sys.stdout.fileno()).columns
    except OSError:
        return 80
    # A pty with no winsize set reports 0.
    return cols if cols > 0 else 80


class _Reader:
    """One `menu_input` call's state, with one small method per key."""

    def __init__(self, prompt: str, commands: dict[str, str], history: list[str]) -> None:
        self.prompt = prompt
        self.commands = commands
        self.history = history
        self.line = ""
        self.cur = 0  # the cursor's index into the line
        self.menu: list[str] | None = None
        self.sel = 0
        self.stem = ""  # what was typed before the menu opened; Esc restores it
        self.hist_idx = len(history)
        self.draft = ""  # the unsubmitted line saved when history recall starts
        self.searching = False  # Ctrl-R mode: the line is the query
        self.hits: list[str] = []
        self.more = 0  # matches beyond the rendered cap
        self.saved = ""  # the line before search began; Esc restores it

    def render(self, write: Callable[[str], None]) -> None:
        """Redraw the input row and the rows under it, never wider than the terminal."""
        width = _width()
        # The prompt is clamped first, then the line windowed into the remainder.
        prompt = (_SEARCH_PROMPT if self.searching else self.prompt)[: width - 1]
        avail = max(0, width - 1 - len(prompt))
        start = 0 if self.cur < avail else self.cur - avail + 1
        visible = self.line[start : start + avail]
        out = ["\r\x1b[J", prompt, visible]
        rows, highlight = self._rows()
        # A row's text is data, scrubbed here: this writer bypasses the stream's scrubber.
        rows = [
            (scrub_terminal_controls(label), scrub_terminal_controls(dim)) for label, dim in rows
        ]
        if rows:
            pad = max(len(label) for label, _dim in rows)
            for i, (label, dim) in enumerate(rows):
                # The SGR codes take no columns and are never sliced through.
                cell = "  " if not label else f"  {label:<{pad}}  "[: width - 1]
                tail = dim[: max(0, width - 1 - len(cell))]
                row = f"{cell}\x1b[2m{tail}\x1b[22m"
                if i == highlight:
                    row = f"\x1b[7m{row}\x1b[27m"
                out.append("\r\n" + row)
            out.append(f"\x1b[{len(rows)}A")
        col = len(prompt) + (self.cur - start)
        out.append("\r" + (f"\x1b[{col}C" if col else ""))
        write("".join(out))

    def _rows(self) -> tuple[list[tuple[str, str]], int]:
        """Return the rows under the input line as (label, dim tail) pairs and the highlight.

        The highlight is -1 for none; a search marker is all dim, with an empty label.
        """
        if self.searching:
            if not self.hits:
                return [("", "(no match)")], -1
            rows: list[tuple[str, str]] = [(hit, "") for hit in self.hits]
            if self.more:
                rows.append(("", f"… {self.more} more (type to narrow)"))
            return rows, self.sel
        if self.menu is not None:
            return [(cmd, self.commands[cmd]) for cmd in self.menu], self.sel
        return [], -1

    def close_rows(self, write: Callable[[str], None]) -> None:
        """Leave the accepted or abandoned line in scrollback with the menu erased."""
        write(f"\r\x1b[J{self.prompt}{self.line}\r\n")

    def open_menu(self, write: Callable[[str], None]) -> None:
        """Open the completion menu for a lone command word; inside steer text Tab is inert."""
        if " " in self.line or (self.line and not self.line.startswith("/")):
            write("\a")
            return
        matches = [c for c in self.commands if c.startswith(self.line)]
        if not matches:
            write("\a")
            return
        if len(matches) == 1:
            self.line = matches[0]
            self.cur = len(self.line)
            return
        self.stem = self.line
        self.menu = matches
        self.select(0)

    def select(self, i: int) -> None:
        """Select the i-th menu entry, wrapping, and put it on the line."""
        assert self.menu is not None
        self.sel = i % len(self.menu)
        self.line = self.menu[self.sel]
        self.cur = len(self.line)

    def dismiss_menu(self, *, restore: bool) -> None:
        """Close the menu, restoring the typed stem when asked."""
        if restore:
            self.line = self.stem
            self.cur = len(self.line)
        self.menu = None

    def open_search(self, write: Callable[[str], None]) -> None:
        """Start a history search with the line as the live query."""
        if not self.history:
            write("\a")
            return
        if self.menu is not None:
            self.dismiss_menu(restore=False)
        self.saved = self.line
        self.line = ""
        self.cur = 0
        self.searching = True
        self.refilter()

    def refilter(self) -> None:
        """Refresh the hits: newest-first case-insensitive substring matches, repeats collapsed."""
        q = self.line.lower()
        matches = list(dict.fromkeys(h for h in reversed(self.history) if q in h.lower()))
        self.hits = matches[:_SEARCH_ROWS]
        self.more = len(matches) - len(self.hits)
        self.sel = 0

    def close_search(self, line: str) -> None:
        """End the search with the given line."""
        self.line = line
        self.cur = len(line)
        self.searching = False
        self.hits = []
        self.more = 0

    def _search_key(self, key: str) -> None:
        """Apply a key while searching; Enter and Tab keep the highlighted match, Esc restores."""
        if key in ("enter", "tab"):
            self.close_search(self.hits[self.sel] if self.hits else self.line)
        elif key in ("esc", "eof"):
            self.close_search(self.saved)
        elif key in ("history-search", "down"):
            if self.hits:
                self.sel = (self.sel + 1) % len(self.hits)
        elif key in ("up", "backtab"):
            if self.hits:
                self.sel = (self.sel - 1) % len(self.hits)
        elif key.startswith("char:"):
            self.insert(key[5:])
            self.refilter()
        elif key in self._EDIT_KEYS:
            self.edit(key)
            self.refilter()

    def insert(self, text: str) -> None:
        """Insert text at the cursor."""
        self.line = self.line[: self.cur] + text + self.line[self.cur :]
        self.cur += len(text)

    def edit(self, key: str) -> None:
        """Apply an editing key to the line."""
        if key == "backspace" and self.cur:
            self.line = self.line[: self.cur - 1] + self.line[self.cur :]
            self.cur -= 1
        elif key == "delete":
            self.line = self.line[: self.cur] + self.line[self.cur + 1 :]
        elif key == "left":
            self.cur = max(0, self.cur - 1)
        elif key == "right":
            self.cur = min(len(self.line), self.cur + 1)
        elif key == "home":
            self.cur = 0
        elif key == "end":
            self.cur = len(self.line)
        elif key == "kill-line":
            self.line = self.line[self.cur :]
            self.cur = 0
        elif key == "kill-to-end":
            self.line = self.line[: self.cur]
        elif key == "kill-word":
            head = self.line[: self.cur].rstrip()
            cut = head.rfind(" ") + 1
            self.line = self.line[:cut] + self.line[self.cur :]
            self.cur = cut

    def recall(self, step: int) -> None:
        """Move through history by step, the draft saved at the newest end."""
        if not self.history:
            return
        if self.hist_idx == len(self.history):
            self.draft = self.line
        self.hist_idx = max(0, min(len(self.history), self.hist_idx + step))
        self.line = (
            self.draft if self.hist_idx == len(self.history) else self.history[self.hist_idx]
        )
        self.cur = len(self.line)

    _EDIT_KEYS = (
        "backspace",
        "delete",
        "left",
        "right",
        "home",
        "end",
        "kill-line",
        "kill-to-end",
        "kill-word",
    )

    def handle_key(self, key: str, write: Callable[[str], None]) -> bool:
        """Apply one key.

        Args:
            key: The key, as `_read_key` names it.
            write: The terminal writer.

        Returns:
            Whether Enter accepted the line, now in `self.line`.

        Raises:
            KeyboardInterrupt: Ctrl-C.
            EOFError: Ctrl-D on an empty line.
        """
        if key == "interrupt":
            raise KeyboardInterrupt
        if self.searching:
            self._search_key(key)
            return False
        if key == "eof":
            if self.menu is not None or self.line:
                write("\a")
                return False
            self.close_rows(write)
            raise EOFError
        if key == "enter":
            self.menu = None
            self.close_rows(write)
            if self.line.strip() and (not self.history or self.history[-1] != self.line):
                self.history.append(self.line)
            return True
        self._apply(key, write)
        return False

    def _apply(self, key: str, write: Callable[[str], None]) -> None:
        """Apply a non-terminal key: menu navigation, history recall, or an edit."""
        if key == "history-search":
            self.open_search(write)
        elif key == "tab" or (key == "backtab" and self.menu is None):
            if self.menu is None:
                self.open_menu(write)
            else:
                self.select(self.sel + 1)
        elif self.menu is not None and key in ("backtab", "up", "down", "esc"):
            if key == "esc":
                self.dismiss_menu(restore=True)
            else:
                self.select(self.sel + (1 if key == "down" else -1))
        elif key in ("up", "down"):
            self.recall(-1 if key == "up" else 1)
        elif key.startswith("char:") or key in self._EDIT_KEYS:
            # Typing keeps the selected candidate and edits from there.
            self.dismiss_menu(restore=False)
            if key.startswith("char:"):
                self.insert(key[5:])
            else:
                self.edit(key)


def menu_input(
    prompt: str,
    commands: dict[str, str],
    history: list[str],
    *,
    read_key: Callable[[], str] | None = None,
    write: Callable[[str], None] | None = None,
    until: Callable[[], bool] | None = None,
) -> str:
    """Read one line with a fish-style command preview, on `input()`'s contract.

    Args:
        prompt: The prompt.
        commands: The slash commands and their descriptions.
        history: The recallable lines; an accepted non-empty line is appended, deduped
            against the last entry.
        read_key: Reads one key; the real terminal, put in cbreak mode, when None.
        write: The terminal writer; stdout when None.
        until: Polled while nothing is typed.

    Returns:
        The line without its newline.

    Raises:
        EOFError: Ctrl-D on an empty line.
        KeyboardInterrupt: Ctrl-C, by SIGINT in cbreak mode or the byte where signals are off.
        LineSuperseded: `until` held while nothing was typed.
    """

    def restore() -> None:
        """Return None: nothing to put back until cbreak mode is set."""
        return None

    if read_key is None:
        assert termios is not None and tty is not None, "menu_input needs a Unix terminal"
        tio, drain = termios, termios.TCSADRAIN
        fd = sys.stdin.fileno()
        old_attrs = tio.tcgetattr(fd)
        tty.setcbreak(fd, drain)

        def restore() -> None:
            """Put the terminal's attributes back."""
            tio.tcsetattr(fd, drain, old_attrs)

        def terminal_key() -> str:
            """Return one key from the terminal."""
            return _read_key(fd) if until is None else _read_key_until(fd, until)

        read_key = terminal_key

    if write is None:

        def _stdout_write(text: str) -> None:
            """Write and flush past stdout's scrubber."""
            out = raw_stream(sys.stdout)
            out.write(text)
            out.flush()

        write = _stdout_write

    r = _Reader(prompt, commands, history)
    try:
        r.render(write)
        while True:
            if r.handle_key(read_key(), write):
                return r.line
            r.render(write)
    except (KeyboardInterrupt, LineSuperseded):
        # Erase the menu rows once, so they do not linger under whatever prints next.
        write("\r\n\x1b[J")
        raise
    finally:
        restore()
