# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""ReDoS containment for the grep tool: reject a model regex that could
cause catastrophic backtracking. Lifted verbatim from dispatch.py so the dense
security screener can be read and tested in isolation."""

from __future__ import annotations

from agent6.tools.errors import ToolError

# --- grep regex safety (ReDoS containment) -----------------------------------
# `grep` compiles a model-supplied regex and runs it in agent6's own process
# (not the jail). CPython's `re` engine holds the GIL and is not interruptible
# mid-match, so a catastrophic-backtracking pattern (e.g. ``(a+)+$``) on one
# unlucky line could hang the run. We can't time-bound a single C-level match,
# so we defend in cheap layers: cap the pattern length; statically reject the
# classic nested-unbounded-quantifier shape AND the common single-quantifier
# catastrophic shapes (overlapping alternation under a repeat like '(a|a)*' /
# '(a|ab)*', and adjacent unbounded quantifiers over the same atom like 'a*a*');
# and bound total grep wall-clock across files AND lines. The static screen is
# conservative (it may reject a safe pattern; the caller can rephrase) and not
# exhaustive. RESIDUAL: a catastrophic pattern not recognised by the screen can
# still spend a long time inside one in-process re.search, because the
# wall-clock check between lines cannot interrupt a match already in progress
# under the GIL; a true interrupt would need a subprocess/ripgrep. The screens
# close the shapes that actually blow up.
_MAX_GREP_PATTERN_LEN = 1000


def _quantifier_is_unbounded(pattern: str, k: int) -> bool:
    """At index *k* (just past a token), is there an unbounded quantifier
    (``*``, ``+``, or ``{n,}`` with no upper bound)?"""
    if k >= len(pattern):
        return False
    if pattern[k] in "*+":
        return True
    if pattern[k] == "{":
        close = pattern.find("}", k)
        if close != -1:
            body = pattern[k + 1 : close]
            return body.endswith(",") or (body.count(",") == 1 and body.split(",")[1] == "")
    return False


def _has_nested_unbounded_quantifier(pattern: str) -> bool:
    """True for the classic catastrophic shape: an unbounded quantifier applied
    to a group whose body itself contains an unbounded quantifier — ``(a+)+``,
    ``(.*)*``, ``(a+)*``, ``((ab)+)+`` … Cheap single pass that skips escapes
    and character classes and propagates unbounded-ness up to parent groups."""
    seen_unbounded: list[bool] = [False]  # stack: one flag per open group body
    j, n = 0, len(pattern)
    while j < n:
        c = pattern[j]
        if c == "\\":
            j += 2
            continue
        if c == "[":  # skip a character class wholesale
            j += 1
            if j < n and pattern[j] == "^":
                j += 1
            if j < n and pattern[j] == "]":  # a literal ] as the first class member
                j += 1
            while j < n and pattern[j] != "]":
                j += 2 if pattern[j] == "\\" else 1
            j += 1
            continue
        if c == "(":
            seen_unbounded.append(False)
            j += 1
            continue
        if c == ")":
            inner = seen_unbounded.pop() if len(seen_unbounded) > 1 else False
            quant = _quantifier_is_unbounded(pattern, j + 1)
            if inner and quant:
                return True
            if inner or quant:  # an unbounded element of the parent's body
                seen_unbounded[-1] = True
            j += 1
            continue
        if _quantifier_is_unbounded(pattern, j):
            seen_unbounded[-1] = True
        j += 1
    return False


def _split_top_level_alternation(body: str) -> list[str]:
    """Split a group body on top-level ``|`` (ignoring ``|`` inside nested
    groups, character classes, and escapes)."""
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    j, n = 0, len(body)
    while j < n:
        c = body[j]
        if c == "\\":
            cur.append(body[j : j + 2])
            j += 2
            continue
        if c == "[":  # character class: copy verbatim to its closing ]
            k = j + 1
            if k < n and body[k] == "^":
                k += 1
            if k < n and body[k] == "]":
                k += 1
            while k < n and body[k] != "]":
                k += 2 if body[k] == "\\" else 1
            cur.append(body[j : k + 1])
            j = k + 1
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "|" and depth == 0:
            parts.append("".join(cur))
            cur = []
            j += 1
            continue
        cur.append(c)
        j += 1
    parts.append("".join(cur))
    return parts


def _skip_char_class(pattern: str, j: int) -> int:
    """Given *j* at a ``[``, return the index just past the closing ``]``."""
    n = len(pattern)
    j += 1
    if j < n and pattern[j] == "^":
        j += 1
    if j < n and pattern[j] == "]":  # a literal ] as the first class member
        j += 1
    while j < n and pattern[j] != "]":
        j += 2 if pattern[j] == "\\" else 1
    return j + 1


def _group_body_end(pattern: str, open_idx: int) -> int:
    """Given *open_idx* at a ``(``, return the index of its matching ``)`` plus
    one, or -1 if unbalanced. Skips escapes and character classes."""
    n = len(pattern)
    depth = 1
    k = open_idx + 1
    while k < n and depth:
        ck = pattern[k]
        if ck == "\\":
            k += 2
        elif ck == "[":
            k = _skip_char_class(pattern, k)
        else:
            if ck == "(":
                depth += 1
            elif ck == ")":
                depth -= 1
            k += 1
    return k if depth == 0 else -1


def _alternation_branches_overlap(branches: list[str]) -> bool:
    """True if any two branches are identical or one is a prefix of another."""
    for a in range(len(branches)):
        for b in range(a + 1, len(branches)):
            x, y = branches[a], branches[b]
            if x == y or x.startswith(y) or y.startswith(x):
                return True
    return False


def _has_overlapping_alternation_under_quantifier(pattern: str) -> bool:
    """True for single-quantifier catastrophic forms whose group is an
    alternation with duplicate or prefix-overlapping branches followed by an
    unbounded quantifier — ``(a|a)*``, ``(a|ab)*``, ``(a|a|b)+`` … These blow up
    on a non-matching suffix even though no quantifier is *nested*. Distinct,
    non-prefix branches like ``(ab|cd)+`` are NOT flagged."""
    j, n = 0, len(pattern)
    while j < n:
        c = pattern[j]
        if c == "\\":
            j += 2
            continue
        if c == "[":
            j = _skip_char_class(pattern, j)
            continue
        if c == "(":
            k = _group_body_end(pattern, j)
            if k != -1 and _quantifier_is_unbounded(pattern, k):
                body = pattern[j + 1 : k - 1]
                # Strip a leading non-capturing/group prefix like (?:...).
                if body.startswith("?:"):
                    body = body[2:]
                branches = [b for b in _split_top_level_alternation(body) if b]
                if len(branches) >= 2 and _alternation_branches_overlap(branches):
                    return True
            j += 1
            continue
        j += 1
    return False


def _has_adjacent_unbounded_quantifiers(pattern: str) -> bool:
    """True for runs of adjacent unbounded quantifiers over the SAME single-char
    atom — ``a*a*``, ``a+a+a+``, ``.*.*`` … which backtrack catastrophically on a
    non-matching suffix. Conservative: only flags when consecutive quantified
    atoms are identical single characters (incl. ``.``), so distinct atoms like
    ``a*b*`` are left alone."""
    prev_atom: str | None = None
    j, n = 0, len(pattern)
    while j < n:
        c = pattern[j]
        if c == "\\":
            atom = pattern[j : j + 2]
            j += 2
        elif c in "[(":
            # Groups/classes reset the run; let the other screens handle them.
            prev_atom = None
            j += 1
            continue
        else:
            atom = c
            j += 1
        if _quantifier_is_unbounded(pattern, j):
            # consume the quantifier char(s)
            if pattern[j] == "{":
                close = pattern.find("}", j)
                j = close + 1 if close != -1 else j + 1
            else:
                j += 1
            if prev_atom is not None and prev_atom == atom:
                return True
            prev_atom = atom
        else:
            prev_atom = None
    return False


def reject_pathological_regex(pattern: str) -> None:
    """Raise ToolError if *pattern* is over-long or matches a catastrophic
    shape; otherwise return (it may still be compiled)."""
    if len(pattern) > _MAX_GREP_PATTERN_LEN:
        raise ToolError(
            f"grep pattern too long ({len(pattern)} > {_MAX_GREP_PATTERN_LEN} chars); "
            "narrow the search."
        )
    if _has_nested_unbounded_quantifier(pattern):
        raise ToolError(
            "grep pattern has a nested unbounded quantifier (e.g. '(a+)+') that can "
            "cause catastrophic backtracking; rewrite it without the nested repeat."
        )
    if _has_overlapping_alternation_under_quantifier(pattern):
        raise ToolError(
            "grep pattern repeats an alternation with overlapping branches "
            "(e.g. '(a|a)*' or '(a|ab)*') that can cause catastrophic backtracking; "
            "rewrite it so the branches are disjoint."
        )
    if _has_adjacent_unbounded_quantifiers(pattern):
        raise ToolError(
            "grep pattern has adjacent unbounded quantifiers over the same atom "
            "(e.g. 'a*a*') that can cause catastrophic backtracking; collapse them "
            "into a single repeat."
        )
