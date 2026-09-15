# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The operator's own words inside a composed task.

`agent6 run` may prepend context to what the operator typed: another
session's digest (`--from`, a `<prior-run ...>` block), file seeds (`<file
...>` blocks), and installed skills (`<skill ...>` blocks behind a preamble
and a `---` rule). The model reads the whole composition; a headline (the
session.start event, every listing) shows the operator's part alone."""

from __future__ import annotations

import re

_BLOCK_RE = re.compile(r"<(prior-run|file|skill|plan)\b[^>]*>.*?</\1>\s*", re.DOTALL)
_OPEN_BLOCK_RE = re.compile(r"<(prior-run|file|skill|plan)\b[^>]*>.*\Z", re.DOTALL)
SKILLS_PREAMBLE = "Apply the operator-installed skill(s) below to this task."
# An ask transcript's headers: the title (`# agent6 ask`, `# agent6 ask
# (interactive)`), then `## Question` / `## Answer` for a one-shot ask or
# `## Q1` / `## A1` (numbered) for an interactive one.
_ASK_QUESTION_HEADER = re.compile(r"^## (Question|Q\d+)$")
_ASK_ANSWER_HEADER = re.compile(r"^## (Answer|A\d+)$")
_HEADING_MARK = re.compile(r"^#{1,6}\s+")


def operator_task_text(text: str) -> str:
    """*text* with every context block, the skills preamble, and the `---`
    rule removed; the text itself when nothing was composed. An unclosed
    block (a clipped copy) drops from its opener to the end."""
    out = _BLOCK_RE.sub("", text)
    out = _OPEN_BLOCK_RE.sub("", out)
    lines = [line for line in out.splitlines() if line.strip() not in (SKILLS_PREAMBLE, "---")]
    return "\n".join(lines).strip()


def task_headline(text: str) -> str:
    """The line that names a task or ask transcript: the first line in the
    operator's words (context blocks skipped, an ask's headers skipped), a
    markdown heading's marks dropped. "" when nothing stands out."""
    for line in operator_task_text(text).splitlines():
        s = line.strip()
        if s.startswith("# agent6 ask") or _ASK_QUESTION_HEADER.match(s):
            continue
        if _ASK_ANSWER_HEADER.match(s):
            break
        if s and not s.startswith("<"):
            return _HEADING_MARK.sub("", s)
    return ""
