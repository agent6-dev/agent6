# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The unified `/parallel` grammar, shared by the coordinator steer parser
(``workflows/loop.py``) and the web + TUI new-work composers.

    /parallel [spec] <task text> [/parallel [spec] <task text>]...

- ``spec`` is a positive int (lane count) or a comma-separated model list, and
  is OPTIONAL: omitted means one lane on the configured worker model. A
  segment's first token counts as a spec when it contains a comma OR a slash
  (model ids are provider/model shaped, e.g. ``moonshotai/kimi-k2.6``); a bare
  comma-less slash-less model name (``opus``) intentionally stays task text --
  it is indistinguishable from a task word. The flip side: a task whose FIRST
  word is a path (``src/foo.py``) parses as a bogus model spec, refused
  pre-spawn with a did-you-mean (``models.validate``) when a model cache exists
  to check against, else run and failed at the provider call; start with a verb.
- The exact token ``/parallel``, whitespace-delimited, separates tasks. A
  message is a directive only when it STARTS with the exact ``/parallel`` token;
  ``/parallelfoo ...`` stays ordinary text, byte-for-byte. A mid-task
  ``/parallel`` inside a word or path (not whitespace-delimited) is ordinary text
  too.
- Newlines are ordinary task characters, so a task can span multiple lines.

One parser, imported by ``workflows`` (the coordinator) and ``ui`` (the
composers, and the CLI ``--parallel`` value via :func:`parse_spec`). Pure stdlib
string parsing, no agent6 imports -- a leaf both layers sit above."""

from __future__ import annotations

import re
from dataclasses import dataclass

# A `/parallel` token that is whitespace-delimited: preceded by string start or
# whitespace, followed by whitespace or string end. NOT re.MULTILINE -- a newline
# is task text; `\s` already covers it, so a bare whitespace-delimited /parallel
# IS a separator while `foo/parallel/bar` (in a path) is not. `\A`/`\Z` anchor to
# the whole string, never to line boundaries.
_SEPARATOR = re.compile(r"(?:\A|(?<=\s))/parallel(?=\s|\Z)")


class DirectiveError(ValueError):
    """A `/parallel` directive or spec was malformed: a bare token, a segment
    with no task, a non-positive lane count, or an empty model list."""


@dataclass(frozen=True, slots=True)
class Segment:
    """One parsed `/parallel` task: its optional ``spec`` (``""`` = one default
    lane) and the ``task`` text (internal whitespace and newlines preserved)."""

    spec: str
    task: str


def parse_spec(spec: str) -> list[str | None]:
    """A spec string -> one entry per lane: ``None`` = the configured worker
    model, else a per-lane model override. ``""`` (omitted) is one default lane.

    A positive integer ``N`` is N default lanes; a comma-separated list is one
    lane per named model (a single model id, e.g. ``provider/model``, is a
    one-lane list). Raises DirectiveError on a non-positive count or a list that
    names no models. Single source for the directive spec AND the CLI
    ``run --parallel <spec>`` value grammar."""
    s = spec.strip()
    if not s:
        return [None]
    # isdecimal, not isdigit: isdigit() is True for superscripts/circled
    # digits ('\u00b2') that int() rejects, so the guard raised a bare
    # ValueError past every DirectiveError-catching caller (the coordinator's
    # never-end-the-run contract included). isdecimal() is exactly the set
    # int() parses for a stripped, sign-less string.
    if s.isdecimal():
        n = int(s)
        if n < 1:
            raise DirectiveError("parallel lane count must be >= 1")
        return [None] * n
    models = [m.strip() for m in s.split(",") if m.strip()]
    if not models:
        raise DirectiveError(f"parallel spec {spec!r} names no models")
    return list(models)


def parse_directive(text: str) -> list[Segment] | None:
    """Split a `/parallel` message into its task segments, or ``None`` when *text*
    is not a directive (does not start with the exact ``/parallel`` token).

    Each whitespace-delimited ``/parallel`` token starts a new segment. Within a
    segment, the first whitespace-delimited token is the spec when it is a
    positive int or contains a comma or slash (a model list / model id), else
    the whole segment is the task. Raises DirectiveError on a segment with no
    task (a bare ``/parallel``, or a spec with nothing after it) --
    all-or-nothing, so a later empty segment fails the whole parse."""
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
    """A leading token is a spec iff it is a positive integer, a comma list, or
    contains a slash (a provider/model id; no natural task starts with a
    slash-containing word -- see the module docstring for the path caveat). A
    bare word (``fix``, a single model name with no comma or slash) is task
    text."""
    return token.isdecimal() or "," in token or "/" in token


def _parse_segment(raw: str) -> Segment:
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
