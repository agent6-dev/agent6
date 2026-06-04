# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Strict unified-diff parser and applier (single file per patch).

Accepts standard `diff -u` output:

    --- a/path/to/file
    +++ b/path/to/file
    @@ -OLD_START,OLD_COUNT +NEW_START,NEW_COUNT @@
     context
    -removed
    +added
     context

Design choices (pre-1.0, opinionated):

- One file per patch. Multi-file patches are rejected. Callers loop.
- Zero fuzz. Context lines must match the on-disk file exactly. If any
  hunk fails to apply, no change is written (all-or-nothing).
- `--- /dev/null` is allowed and means "create a new file"; the target
  file must not already exist.
- `+++ /dev/null` (file delete) is rejected. Use a different tool.
- The `\\ No newline at end of file` marker is honoured: when present
  on the `-` side, the original file must lack a trailing newline; on
  the `+` side, the result is written without one.
- Hunk-header line counts are validated; an inconsistent header is a
  hard error, not silently fixed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HUNK_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
)


class PatchError(ValueError):
    """The patch could not be parsed or could not be applied cleanly."""


@dataclass(frozen=True, slots=True)
class _Hunk:
    old_start: int  # 1-based line in original file
    old_count: int
    new_start: int  # 1-based line in resulting file (informational)
    new_count: int
    # Each entry is (prefix, text). prefix is one of " ", "-", "+".
    # text has no trailing newline.
    body: tuple[tuple[str, str], ...]
    # Whether the last "-"-or-" " line lacks a trailing newline in the original.
    old_no_newline: bool
    # Whether the last "+"-or-" " line lacks a trailing newline in the result.
    new_no_newline: bool


@dataclass(frozen=True, slots=True)
class ParsedPatch:
    """A successfully-parsed single-file unified diff."""

    # Path from the `+++` header with the leading `b/` (if any) stripped.
    # For file creation (`--- /dev/null`), this is the new file's path.
    target_path: str
    # True if the patch creates a new file (i.e. `--- /dev/null`).
    is_create: bool
    hunks: tuple[_Hunk, ...]


# ---------- parsing ----------


def _strip_ab_prefix(header_path: str) -> str:
    """Strip the conventional `a/` or `b/` prefix from a diff header path.

    `--- a/foo.py` and `+++ b/foo.py` are the format `git diff` emits.
    Some models also emit bare `--- foo.py`. Accept both.
    """
    if header_path.startswith(("a/", "b/")):
        return header_path[2:]
    return header_path


def parse_patch(text: str) -> ParsedPatch:  # noqa: PLR0912, PLR0915
    """Parse a single-file unified diff. Raises PatchError on malformed input."""
    if not text.strip():
        raise PatchError("Empty patch")

    lines = text.splitlines()
    # Locate the `---` and `+++` headers. Skip leading commentary lines (e.g.
    # `diff --git a/foo b/foo`, `index abc..def 100644`).
    i = 0
    while i < len(lines) and not lines[i].startswith("--- "):
        i += 1
    if i >= len(lines):
        raise PatchError("Missing `--- ` header line")
    minus_header = lines[i][4:].strip()
    i += 1
    if i >= len(lines) or not lines[i].startswith("+++ "):
        raise PatchError("Missing `+++ ` header line after `--- ` header")
    plus_header = lines[i][4:].strip()
    i += 1

    is_create = minus_header == "/dev/null"
    if plus_header == "/dev/null":
        raise PatchError("File deletion (`+++ /dev/null`) is not supported")

    target_path = _strip_ab_prefix(plus_header)
    if not target_path or target_path == "/dev/null":
        raise PatchError(f"Invalid target path in `+++` header: {plus_header!r}")

    # Reject multi-file patches: a second `--- ` header anywhere below means
    # the caller stuffed multiple files into one payload.
    for j in range(i, len(lines)):
        if lines[j].startswith("--- "):
            raise PatchError("Multi-file patches are not supported; submit one file at a time")

    hunks: list[_Hunk] = []
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        # A trailing `\ No newline at end of file` marker may live between the
        # last `+`/`-` line of the previous hunk and the next hunk header
        # (or end of input). Attribute it to the most recent hunk.
        if line.startswith("\\ "):
            if not hunks:
                raise PatchError("`\\ No newline` marker has no preceding hunk")
            prev = hunks[-1]
            last_prefix = prev.body[-1][0] if prev.body else " "
            new_old = prev.old_no_newline or last_prefix in ("-", " ")
            new_new = prev.new_no_newline or last_prefix in ("+", " ")
            hunks[-1] = _Hunk(
                old_start=prev.old_start,
                old_count=prev.old_count,
                new_start=prev.new_start,
                new_count=prev.new_count,
                body=prev.body,
                old_no_newline=new_old,
                new_no_newline=new_new,
            )
            i += 1
            continue
        m = _HUNK_RE.match(line)
        if not m:
            raise PatchError(f"Expected hunk header `@@ -L,N +L,N @@`, got: {line!r}")
        old_start = int(m.group("old_start"))
        old_count = int(m.group("old_count")) if m.group("old_count") is not None else 1
        new_start = int(m.group("new_start"))
        new_count = int(m.group("new_count")) if m.group("new_count") is not None else 1
        i += 1
        body: list[tuple[str, str]] = []
        old_no_newline = False
        new_no_newline = False
        seen_old = 0
        seen_new = 0
        while i < len(lines) and (seen_old < old_count or seen_new < new_count):
            ln = lines[i]
            if ln.startswith("\\ "):
                # "\ No newline at end of file" — applies to the immediately
                # preceding line. Determine which side based on its prefix.
                if not body:
                    raise PatchError("`\\ No newline` marker has no preceding line")
                prev_prefix, _ = body[-1]
                if prev_prefix == "-":
                    old_no_newline = True
                elif prev_prefix == "+":
                    new_no_newline = True
                else:  # " " context line — applies to both sides
                    old_no_newline = True
                    new_no_newline = True
                i += 1
                continue
            if not ln:
                # Empty line is a legitimate context line (encoded as " " + "").
                # Some patch producers strip the leading space on otherwise-empty
                # lines; accept both shapes.
                body.append((" ", ""))
                seen_old += 1
                seen_new += 1
                i += 1
                continue
            prefix = ln[0]
            text_part = ln[1:]
            if prefix == " ":
                body.append((" ", text_part))
                seen_old += 1
                seen_new += 1
            elif prefix == "-":
                body.append(("-", text_part))
                seen_old += 1
            elif prefix == "+":
                body.append(("+", text_part))
                seen_new += 1
            else:
                raise PatchError(
                    f"Unexpected line in hunk body (expected ` `, `-`, `+`, `\\ `): {ln!r}"
                )
            i += 1
        if seen_old != old_count or seen_new != new_count:
            raise PatchError(
                f"Hunk header @@ -{old_start},{old_count} +{new_start},{new_count} @@ "
                f"declares {old_count}/{new_count} lines but body has {seen_old}/{seen_new}"
            )
        hunks.append(
            _Hunk(
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
                body=tuple(body),
                old_no_newline=old_no_newline,
                new_no_newline=new_no_newline,
            )
        )

    if not hunks:
        raise PatchError("Patch contains no hunks")
    return ParsedPatch(target_path=target_path, is_create=is_create, hunks=tuple(hunks))


# ---------- application ----------


def _split_lines_keepends(text: str) -> tuple[list[str], bool]:
    """Split *text* into lines without trailing newlines; track final-newline state."""
    if text == "":
        return [], False
    has_trailing = text.endswith("\n")
    lines = text.split("\n")
    if has_trailing:
        # Final element after split is "" — drop it.
        lines.pop()
    return lines, has_trailing


def apply_parsed_patch(patch: ParsedPatch, original: str | None) -> str:  # noqa: PLR0912
    """Apply *patch* to *original* file contents (None means file does not exist).

    Returns the new file content. Raises PatchError on any context mismatch or
    impossible-to-apply hunk. All-or-nothing: caller writes the returned string.
    """
    if patch.is_create:
        if original is not None:
            raise PatchError(
                f"Patch declares file creation (`--- /dev/null`) but "
                f"{patch.target_path!r} already exists"
            )
        base_lines: list[str] = []
        base_had_trailing = False
    else:
        if original is None:
            raise PatchError(
                f"Patch targets {patch.target_path!r} but the file does not exist; "
                f"use `--- /dev/null` to create it"
            )
        base_lines, base_had_trailing = _split_lines_keepends(original)

    # Work on a mutable copy. Apply hunks in original order; track the cumulative
    # offset between original-file line numbers and current-buffer line numbers.
    buf = list(base_lines)
    offset = 0  # buf_index = original_index + offset
    # Track whether the final newline should be present after all hunks have been
    # applied. Starts at the file's current state; a hunk that touches the last
    # line can flip it.
    result_has_trailing = base_had_trailing

    for hunk in patch.hunks:
        # Map the hunk's 1-based original line to a 0-based buffer index.
        # Special case: a pure-insertion hunk has `old_count == 0` and its
        # `old_start` is the line number *after which* to insert (0 meaning
        # "at the very beginning"). For `old_count > 0`, `old_start` is the
        # 1-based first line of the replaced range.
        buf_start = hunk.old_start + offset if hunk.old_count == 0 else hunk.old_start - 1 + offset
        if buf_start < 0 or buf_start + hunk.old_count > len(buf):
            raise PatchError(
                f"Hunk @@ -{hunk.old_start},{hunk.old_count} @@ for "
                f"{patch.target_path!r} reaches outside the file "
                f"(file has {len(base_lines)} lines)"
            )

        expected_old: list[str] = []
        replacement_new: list[str] = []
        for prefix, txt in hunk.body:
            if prefix in (" ", "-"):
                expected_old.append(txt)
            if prefix in (" ", "+"):
                replacement_new.append(txt)

        actual_old = buf[buf_start : buf_start + hunk.old_count]
        if actual_old != expected_old:
            raise PatchError(
                f"Context mismatch in {patch.target_path!r} at "
                f"hunk @@ -{hunk.old_start},{hunk.old_count} @@.\n"
                f"Expected lines:\n{_render_lines(expected_old)}\n"
                f"On-disk lines:\n{_render_lines(actual_old)}"
            )

        # Determine whether this hunk touches the file's tail. For a pure-
        # insertion the anchor is `old_start` itself; for a replacement it is
        # the 1-based last line of the replaced range.
        touches_tail = (
            hunk.old_start == len(base_lines)
            if hunk.old_count == 0
            else (hunk.old_start - 1 + hunk.old_count) == len(base_lines)
        )
        buf[buf_start : buf_start + hunk.old_count] = replacement_new
        offset += hunk.new_count - hunk.old_count
        if touches_tail:
            # The hunk's `new_no_newline` flag is authoritative for the result.
            # If the hunk didn't declare a no-newline marker on the new side,
            # the result has a trailing newline (standard diff convention).
            result_has_trailing = not hunk.new_no_newline

    if not buf:
        # Empty file — write empty string regardless of trailing-newline state.
        return ""
    out = "\n".join(buf)
    if result_has_trailing:
        out += "\n"
    return out


def _render_lines(lines: list[str]) -> str:
    if not lines:
        return "  (empty)"
    return "\n".join(f"  {i + 1}| {ln}" for i, ln in enumerate(lines))


def apply_patch_text(patch_text: str, original: str | None) -> tuple[str, str]:
    """Convenience: parse + apply. Returns (target_path, new_content)."""
    patch = parse_patch(patch_text)
    new_content = apply_parsed_patch(patch, original)
    return patch.target_path, new_content
