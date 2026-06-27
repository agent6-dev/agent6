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

    # NOTE: multi-file patches are rejected structurally, where a hunk header
    # is expected (see the `_HUNK_RE` miss below). We must NOT pre-scan raw
    # lines for `--- ` here: a removal line whose *content* begins with `-- `
    # (a SQL/Lua/Haskell/Ada comment, say) is encoded as `-` + `-- foo` =
    # `--- foo` inside a hunk body, and a raw scan would wrongly reject the
    # legitimate single-file patch as multi-file.
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
            # A `--- ` line where a hunk header is expected is a real second
            # file's header (vs a `-`-removal of `-- ...` content, which is
            # consumed inside a hunk body above and never reaches here).
            if line.startswith("--- "):
                raise PatchError("Multi-file patches are not supported; submit one file at a time")
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
                # "\ No newline at end of file", applies to the immediately
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
        # Final element after split is "", drop it.
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
        # Empty file, write empty string regardless of trailing-newline state.
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


# ---------- OpenAI "*** Begin Patch" (V4A) format ----------
#
# GPT / gpt-oss models emit patches in OpenAI's apply_patch format, NOT unified
# diff: `*** Begin Patch` / `*** End Patch` wrap one or more file directives
# (`*** Add File:` / `*** Update File:` / `*** Delete File:`); inside an Update,
# hunks use ` `/`-`/`+` line prefixes with optional `@@ <hint>` section markers
# and NO `@@ -L,N +L,N @@` line numbers (matching is by context, not position).
# Without this, every apply_patch from a GPT-family model fails ("got: '@@'")
# and the model death-spirals on re-reads. We map each context hunk onto the
# same safe unique-substring replacement apply_edit uses: zero fuzz, all-or-
# nothing, and a clear error when context is missing or ambiguous.


def is_v4a_patch(text: str) -> bool:
    """True if *text* looks like an OpenAI `*** Begin Patch` envelope."""
    return text.lstrip().startswith("*** Begin Patch")


def patch_target_path(text: str) -> str:
    """Extract the single target path from a patch (either format) without
    applying it. Raises ``PatchError`` if no path header is present."""
    if is_v4a_patch(text):
        for ln in text.splitlines():
            d = _v4a_file_directive(ln)
            if d is not None:
                return d[1]
        raise PatchError("V4A patch has no `*** Add/Update/Delete File:` directive")
    for ln in text.splitlines():
        if ln.startswith("+++ "):
            header = ln[4:].strip()
            if header == "/dev/null":
                raise PatchError("File deletion (`+++ /dev/null`) is not supported")
            return _strip_ab_prefix(header)
    raise PatchError("patch has no `+++ ` header to take a path from")


def _v4a_file_directive(line: str) -> tuple[str, str] | None:
    """Parse a `*** <Verb> File: <path>` directive into (verb, path), else None."""
    for verb in ("Add", "Update", "Delete"):
        prefix = f"*** {verb} File:"
        if line.startswith(prefix):
            return verb, line[len(prefix) :].strip()
    return None


def apply_v4a_text(patch_text: str, original: str | None) -> tuple[str, str]:  # noqa: PLR0912
    """Parse and apply a single-file OpenAI V4A patch.

    Returns ``(target_path, new_content)``. Raises ``PatchError`` on a malformed
    envelope, a multi-file patch, a missing/ambiguous context, or a file
    create/update mismatch. All-or-nothing: the caller writes the returned text.
    """
    raw = patch_text.strip().splitlines()
    if not raw or raw[0].strip() != "*** Begin Patch":
        raise PatchError("V4A patch must start with `*** Begin Patch`")
    if raw[-1].strip() != "*** End Patch":
        raise PatchError("V4A patch must end with `*** End Patch`")
    body = raw[1:-1]

    # Locate the single file directive. Multiple are rejected (one file per call,
    # same as the unified-diff applier).
    directives = [(i, _v4a_file_directive(ln)) for i, ln in enumerate(body)]
    file_starts = [(i, d) for i, d in directives if d is not None]
    if not file_starts:
        raise PatchError("V4A patch has no `*** Add/Update/Delete File:` directive")
    if len(file_starts) > 1:
        raise PatchError("Multi-file V4A patches are not supported; submit one file at a time")
    start_idx, (verb, path) = file_starts[0]
    if not path:
        raise PatchError("V4A file directive is missing a path")
    if verb == "Delete":
        raise PatchError("V4A file deletion (`*** Delete File:`) is not supported")
    section = body[start_idx + 1 :]
    # Drop a `*** Move to:` line (rename); we only honour the content change at
    # the original path and leave any rename to the caller's explicit tools.
    section = [ln for ln in section if not ln.startswith("*** Move to:")]

    if verb == "Add":
        if original is not None:
            raise PatchError(f"V4A `*** Add File: {path}` but the file already exists")
        added: list[str] = []
        for ln in section:
            if ln.startswith("+"):
                added.append(ln[1:])
            elif ln.strip() == "":
                added.append("")
            else:
                raise PatchError(f"V4A Add File body must be all `+` lines, got: {ln!r}")
        return path, ("\n".join(added) + "\n" if added else "")

    # Update File.
    if original is None:
        raise PatchError(f"V4A `*** Update File: {path}` but the file does not exist")
    hunks = _v4a_split_hunks(section)
    if not hunks:
        raise PatchError(f"V4A `*** Update File: {path}` has no hunks")
    content = original
    for hints, old_block, new_block in hunks:
        if old_block == "":
            raise PatchError(
                f"V4A hunk for {path!r} has no context/removed lines to anchor on; "
                "include the surrounding lines so the change can be located"
            )
        count = content.count(old_block)
        if count == 0:
            raise PatchError(
                f"V4A hunk context not found in {path!r}. The ` `/`-` lines must match "
                f"the file byte-for-byte. Closest-anchor failed; re-read and retry.\n"
                f"Expected block:\n{_render_lines(old_block.split(chr(10)))}"
            )
        if count == 1:
            content = content.replace(old_block, new_block, 1)
            continue
        # The block itself repeats; the `@@ <section>` hints disambiguate it. We
        # only apply when the hints pin a SINGLE occurrence -- otherwise the hunk
        # stays ambiguous and we refuse rather than edit the wrong copy.
        idx = _v4a_locate_with_hints(content, hints, old_block)
        if idx is None:
            raise PatchError(
                f"V4A hunk context is ambiguous in {path!r} ({count} matches); include "
                "more surrounding context lines, or a `@@ <section>` marker naming the "
                "enclosing def/class, so the location is unique"
            )
        content = content[:idx] + new_block + content[idx + len(old_block) :]
    return path, content


def _v4a_split_hunks(section: list[str]) -> list[tuple[tuple[str, ...], str, str]]:
    """Split a V4A Update body into ``(hints, old_block, new_block)`` tuples.

    A ``@@ <text>`` line is a section LOCATOR HINT for the hunk that follows: its
    text (typically a ``def``/``class`` line) names the enclosing region, used to
    disambiguate when the hunk's own context lines repeat elsewhere in the file.
    One or more ``@@`` lines may precede a hunk; an empty ``@@`` is a bare hunk
    separator with no hint. Within a hunk, ` `/`-` lines build the old block and
    ` `/`+` lines build the new block.
    """
    hunks: list[tuple[list[str], list[str], list[str]]] = []
    cur_hints: list[str] = []
    cur_old: list[str] = []
    cur_new: list[str] = []

    def flush() -> None:
        nonlocal cur_hints, cur_old, cur_new
        if cur_old or cur_new:
            hunks.append((cur_hints, cur_old, cur_new))
        cur_hints, cur_old, cur_new = [], [], []

    for ln in section:
        if ln.startswith("@@"):
            # A `@@` after hunk content starts a NEW hunk; flush the current one
            # first (which clears its hints). Then record this `@@`'s text as a
            # locator hint for the hunk now beginning (empty `@@` = no hint).
            if cur_old or cur_new:
                flush()
            hint = ln[2:].strip()
            if hint:
                cur_hints.append(hint)
            continue
        if ln == "" or ln.startswith(" "):
            text = ln[1:] if ln.startswith(" ") else ""
            cur_old.append(text)
            cur_new.append(text)
        elif ln.startswith("-"):
            cur_old.append(ln[1:])
        elif ln.startswith("+"):
            cur_new.append(ln[1:])
        else:
            raise PatchError(f"Unexpected V4A hunk line (expected ` `, `-`, `+`, `@@`): {ln!r}")
    flush()
    return [(tuple(h), "\n".join(o), "\n".join(n)) for h, o, n in hunks]


def _v4a_locate_with_hints(content: str, hints: tuple[str, ...], old_block: str) -> int | None:
    """Index at which to apply *old_block* when it occurs more than once, using
    the ``@@`` *hints* to disambiguate. Returns None when the hints do not
    resolve it to a SINGLE location (the caller then reports the hunk as
    ambiguous -- we never guess which occurrence to edit). Each hint must appear
    in order; ``old_block`` must then occur exactly once at or after the last
    hint's position."""
    search_from = 0
    last_hint_pos = 0
    for hint in hints:
        pos = content.find(hint, search_from)
        if pos == -1:
            return None
        last_hint_pos = pos
        search_from = pos + len(hint)
    region = content[last_hint_pos:]
    if region.count(old_block) != 1:
        return None
    return last_hint_pos + region.index(old_block)
