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

_BLOCK_RE = re.compile(r"<(prior-run|file|skill)\b[^>]*>.*?</\1>\s*", re.DOTALL)
_OPEN_BLOCK_RE = re.compile(r"<(prior-run|file|skill)\b[^>]*>.*\Z", re.DOTALL)
SKILLS_PREAMBLE = "Apply the operator-installed skill(s) below to this task."


def operator_task_text(text: str) -> str:
    """*text* with every context block, the skills preamble, and the `---`
    rule removed; the text itself when nothing was composed. An unclosed
    block (a clipped copy) drops from its opener to the end."""
    out = _BLOCK_RE.sub("", text)
    out = _OPEN_BLOCK_RE.sub("", out)
    lines = [line for line in out.splitlines() if line.strip() not in (SKILLS_PREAMBLE, "---")]
    return "\n".join(lines).strip()
