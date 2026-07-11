# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tier-1 gist elision: a large read_file result decays to a distilled-gist
placeholder (batched distiller call per pass) before the bare marker, hot and
small reads are never gisted, and continued pressure demotes gists so the
byte bound still holds (bench/longhorizon FINDINGS #1)."""

from __future__ import annotations

import json
from typing import Any

from agent6.workflows._compaction import (
    ELISION_GIST_PREFIX,
    ELISION_PREFIX,
    GistRequest,
    compact_old_tool_results,
    elision_gist_placeholder,
    parse_gist_lines,
    read_file_text_from_result,
)


def _assistant_read(tid: str, path: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": tid, "name": "read_file", "input": {"path": path}}],
    }


def _result(tid: str, content: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": tid, "content": content}],
    }


def _read_result(tid: str, text: str) -> dict[str, Any]:
    return _result(tid, json.dumps({"content": text, "size": len(text)}))


class _SpyGister:
    def __init__(self, replies: dict[str, str]) -> None:
        self.replies = replies
        self.calls: list[tuple[GistRequest, ...]] = []

    def __call__(self, requests: tuple[GistRequest, ...]) -> dict[str, str]:
        self.calls.append(requests)
        return {r.path: self.replies[r.path] for r in requests if r.path in self.replies}


def test_large_read_decays_to_gist_placeholder() -> None:
    doc = "R01 requires headers under 80 chars. END: lines are exempt (R10). " * 60
    msgs = [
        _assistant_read("r0", "rules/r01.md"),
        _read_result("r0", doc),
        _assistant_read("r1", "b.py"),
        _read_result("r1", "x" * 500),
        _assistant_read("r2", "c.py"),
        _read_result("r2", "y" * 500),
    ]
    gister = _SpyGister({"rules/r01.md": "R01: headers <80 chars; END: lines exempt (R10)"})
    stats = compact_old_tool_results(msgs, max_total_bytes=1500, keep_recent=2, gister=gister)
    assert stats.elided == 1
    assert stats.gisted == 1
    got = msgs[1]["content"][0]["content"]
    assert got.startswith(ELISION_GIST_PREFIX)
    assert "rules/r01.md" in got
    assert "gist: R01: headers <80 chars; END: lines exempt (R10)" in got
    # The distiller saw the unwrapped file text, not the JSON envelope.
    assert gister.calls[0][0].content.startswith("R01 requires headers")


def test_no_gister_and_failed_gister_fall_back_to_bare() -> None:
    doc = "spec " * 1000
    for gister in (None, _SpyGister({})):
        msgs = [
            _assistant_read("r0", "docs/spec.md"),
            _read_result("r0", doc),
            _assistant_read("r1", "b.py"),
            _read_result("r1", "x" * 500),
            _assistant_read("r2", "c.py"),
            _read_result("r2", "y" * 500),
        ]
        stats = compact_old_tool_results(msgs, max_total_bytes=1500, keep_recent=2, gister=gister)
        assert stats.elided == 1
        assert stats.gisted == 0
        got = msgs[1]["content"][0]["content"]
        assert got.startswith(ELISION_PREFIX)
        assert not got.startswith(ELISION_GIST_PREFIX)


def test_small_protected_and_non_read_results_are_never_gisted() -> None:
    msgs = [
        _assistant_read("r0", "hot.py"),
        _read_result("r0", "h" * 5000),  # protected: actively edited
        _assistant_read("r1", "tiny.md"),
        _read_result("r1", "t" * 300),  # below GIST_MIN_SOURCE_CHARS
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "r2", "name": "grep", "input": {"pattern": "q"}}
            ],
        },
        _result("r2", "g" * 5000),  # not a read_file
        _assistant_read("r3", "b.py"),
        _read_result("r3", "x" * 500),
        _assistant_read("r4", "c.py"),
        _read_result("r4", "y" * 500),
    ]
    gister = _SpyGister({"hot.py": "nope", "tiny.md": "nope", "grep": "nope"})
    stats = compact_old_tool_results(
        msgs,
        max_total_bytes=1000,
        keep_recent=2,
        protect_paths=frozenset({"hot.py"}),
        gister=gister,
    )
    assert stats.elided == 3
    assert stats.gisted == 0
    assert gister.calls == []  # nothing eligible: the distiller is never dialled


def test_newest_read_per_path_wins_the_gist() -> None:
    doc_v1 = "OLD spec text. " * 300
    doc_v2 = "NEW spec text. " * 300
    msgs = [
        _assistant_read("r0", "docs/spec.md"),
        _read_result("r0", doc_v1),
        _assistant_read("r1", "docs/spec.md"),
        _read_result("r1", doc_v2),
        _assistant_read("r2", "b.py"),
        _read_result("r2", "x" * 500),
        _assistant_read("r3", "c.py"),
        _read_result("r3", "y" * 500),
    ]
    gister = _SpyGister({"docs/spec.md": "the spec facts"})
    stats = compact_old_tool_results(msgs, max_total_bytes=1200, keep_recent=2, gister=gister)
    assert stats.elided == 2
    assert stats.gisted == 1
    assert len(gister.calls) == 1 and len(gister.calls[0]) == 1
    assert gister.calls[0][0].content.startswith("NEW spec text")
    # The newer read keeps the gist; the older one gets the bare marker.
    assert msgs[1]["content"][0]["content"].startswith(ELISION_PREFIX)
    assert not msgs[1]["content"][0]["content"].startswith(ELISION_GIST_PREFIX)
    assert msgs[3]["content"][0]["content"].startswith(ELISION_GIST_PREFIX)


def test_continued_pressure_demotes_gists_oldest_first() -> None:
    doc = "authoritative spec. " * 300
    msgs = [
        _assistant_read("r0", "docs/spec.md"),
        _read_result("r0", doc),
        _assistant_read("r1", "b.py"),
        _read_result("r1", "x" * 500),
        _assistant_read("r2", "c.py"),
        _read_result("r2", "y" * 500),
    ]
    gister = _SpyGister({"docs/spec.md": "spec facts " * 30})
    first = compact_old_tool_results(msgs, max_total_bytes=1800, keep_recent=2, gister=gister)
    assert (first.gisted, first.demoted) == (1, 0)
    gist_ph = msgs[1]["content"][0]["content"]
    assert gist_ph.startswith(ELISION_GIST_PREFIX)
    # Re-running under the same budget is a no-op (idempotent).
    again = compact_old_tool_results(msgs, max_total_bytes=1800, keep_recent=2, gister=gister)
    assert (again.elided, again.gisted, again.demoted) == (0, 0, 0)
    assert msgs[1]["content"][0]["content"] == gist_ph
    # A tighter budget demotes the gist to the bare marker; the bound holds.
    tighter = compact_old_tool_results(msgs, max_total_bytes=1100, keep_recent=2, gister=gister)
    assert tighter.demoted == 1
    got = msgs[1]["content"][0]["content"]
    assert got.startswith(ELISION_PREFIX)
    assert not got.startswith(ELISION_GIST_PREFIX)
    assert "docs/spec.md" in got


def test_gist_longer_than_content_stays_bare() -> None:
    # A gist placeholder that would not shrink the block is pointless.
    text = "z" * 2100  # just over GIST_MIN_SOURCE_CHARS
    msgs = [
        _assistant_read("r0", "a.md"),
        _result("r0", text),  # raw (non-JSON) read result: falls back verbatim
        _assistant_read("r1", "b.py"),
        _read_result("r1", "x" * 500),
        _assistant_read("r2", "c.py"),
        _read_result("r2", "y" * 500),
    ]
    gister = _SpyGister({"a.md": "g" * 3000})  # clipped to GIST_MAX_CHARS, still fits
    stats = compact_old_tool_results(msgs, max_total_bytes=1500, keep_recent=2, gister=gister)
    assert stats.gisted == 1
    assert len(msgs[1]["content"][0]["content"]) < 2100


def test_elision_gist_placeholder_shares_the_elision_prefix() -> None:
    ph = elision_gist_placeholder("docs/a.md", "facts")
    assert ph.startswith(ELISION_PREFIX)
    assert ph.startswith(ELISION_GIST_PREFIX)
    assert "docs/a.md" in ph and "gist: facts" in ph


def test_parse_gist_lines_is_tolerant() -> None:
    text = (
        "- `docs/a.md`: A requires X; threshold 80\n"
        "docs/b.md: B forbids Y\n"
        "unknown.md: ignored\n"
        "no separator line\n"
        "docs/c.md:\n"  # empty gist: skipped
    )
    got = parse_gist_lines(text, paths=["docs/a.md", "docs/b.md", "docs/c.md"])
    assert got == {"docs/a.md": "A requires X; threshold 80", "docs/b.md": "B forbids Y"}


def test_read_file_text_from_result_unwraps_shapes() -> None:
    assert read_file_text_from_result(json.dumps({"content": "text", "size": 4})) == "text"
    truncated = json.dumps(
        {"_tool_result_truncated": True, "head": "the head", "tool": "read_file"}
    )
    assert read_file_text_from_result(truncated) == "the head"
    assert read_file_text_from_result(json.dumps({"error": "Not a file: x"})) == ""
    assert read_file_text_from_result("plain non-json payload") == "plain non-json payload"
