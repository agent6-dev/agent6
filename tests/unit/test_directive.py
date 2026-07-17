# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The unified `/parallel` grammar shared by the coordinator + the composers."""

from __future__ import annotations

import pytest

from agent6.directive import DirectiveError, Segment, parse_directive, parse_spec


def _segs(text: str) -> list[tuple[str, str]]:
    parsed = parse_directive(text)
    assert parsed is not None
    return [(s.spec, s.task) for s in parsed]


# --- spec grammar (single source, shared with `run --parallel <spec>`) --------


def test_parse_spec_omitted_is_one_default_lane() -> None:
    assert parse_spec("") == [None]
    assert parse_spec("   ") == [None]


def test_parse_spec_int_is_n_default_lanes() -> None:
    assert parse_spec("3") == [None, None, None]
    assert parse_spec("1") == [None]


def test_parse_spec_model_list_is_one_lane_per_model() -> None:
    assert parse_spec("gpt-5,opus") == ["gpt-5", "opus"]
    assert parse_spec("kimi, glm ") == ["kimi", "glm"]


def test_parse_spec_slash_model_id_is_one_lane() -> None:
    # provider/model ids pass through whole; the CLI --parallel value shares this
    assert parse_spec("moonshotai/kimi-k2.6") == ["moonshotai/kimi-k2.6"]
    assert parse_spec("a/b,c/d") == ["a/b", "c/d"]


def test_parse_spec_zero_and_empty_models_raise() -> None:
    with pytest.raises(DirectiveError):
        parse_spec("0")
    with pytest.raises(DirectiveError):
        parse_spec(",")


# --- directive gate: only a leading exact /parallel token -----------------


def test_non_directive_returns_none() -> None:
    assert parse_directive("just fix the bug") is None
    # a /parallel that is not at the start is ordinary text, not a directive
    assert parse_directive("do this\n/parallel nope") is None


def test_prefix_lookalike_is_not_a_directive() -> None:
    # byte-for-byte: /parallelfoo is not the exact token
    assert parse_directive("/parallelfoo do x") is None
    assert parse_directive("/parallelize the loop") is None


def test_leading_whitespace_before_the_token_is_allowed() -> None:
    assert _segs("  /parallel 2 do it") == [("2", "do it")]


# --- spec is optional; omitted = one lane ---------------------------------


def test_omitted_spec_is_one_lane_task_is_whole_segment() -> None:
    assert _segs("/parallel refactor the parser") == [("", "refactor the parser")]
    # a bare word first token is task text, never a single-model spec
    assert _segs("/parallel do it now") == [("", "do it now")]


def test_int_spec() -> None:
    assert _segs("/parallel 3 add a greeting") == [("3", "add a greeting")]


def test_model_list_spec() -> None:
    assert _segs("/parallel gpt-5,opus refactor the parser") == [
        ("gpt-5,opus", "refactor the parser")
    ]


def test_slash_model_id_spec() -> None:
    # a slash-containing first token is a model spec (provider/model shaped)
    assert _segs("/parallel moonshotai/kimi-k2.6 fix the flaky test") == [
        ("moonshotai/kimi-k2.6", "fix the flaky test")
    ]
    assert _segs("/parallel a/b,c/d task B") == [("a/b,c/d", "task B")]


def test_bare_model_name_stays_task_text() -> None:
    # a comma-less slash-less name is indistinguishable from a task word: task text
    assert _segs("/parallel opus fix the bug") == [("", "opus fix the bug")]


def test_slash_first_task_word_parses_as_spec_documented_ambiguity() -> None:
    # A task whose FIRST word is a path parses as a (bogus) model spec; the lane
    # then fails loudly at the provider (documented: start the task with a verb).
    assert _segs("/parallel src/foo.py needs a docstring") == [("src/foo.py", "needs a docstring")]


def test_slash_spec_without_task_raises() -> None:
    with pytest.raises(DirectiveError):
        parse_directive("/parallel moonshotai/kimi-k2.6")


def test_task_keeps_internal_spacing() -> None:
    assert _segs("/parallel 2 fix  the   bug") == [("2", "fix  the   bug")]


def test_multiline_task_body() -> None:
    # newlines are ordinary task characters now (no per-line splitting)
    assert _segs("/parallel 2 line one\nline two\nline three") == [
        ("2", "line one\nline two\nline three")
    ]


# --- the /parallel token separates tasks within one message ---------------


def test_separator_splits_segments() -> None:
    assert _segs("/parallel 2 task A /parallel gpt-5,opus task B") == [
        ("2", "task A"),
        ("gpt-5,opus", "task B"),
    ]


def test_separator_splits_multiline_segments() -> None:
    parsed = _segs("/parallel implement X\nwith care\n/parallel 2 implement Y\nkeep it small")
    assert parsed == [("", "implement X\nwith care"), ("2", "implement Y\nkeep it small")]


def test_mid_task_slash_parallel_in_a_path_is_not_a_separator() -> None:
    # /parallel inside a word or path (not whitespace-delimited) stays task text
    assert _segs("/parallel 2 edit src/parallel/loop.py now") == [
        ("2", "edit src/parallel/loop.py now")
    ]
    assert _segs("/parallel touch a/parallel.txt") == [("", "touch a/parallel.txt")]


def test_whitespace_delimited_mid_message_slash_parallel_is_a_separator() -> None:
    # a bare, whitespace-delimited /parallel DOES separate, by design
    assert _segs("/parallel a /parallel b") == [("", "a"), ("", "b")]


# --- malformed: an empty task after removing the spec -----------------


def test_bare_directive_raises() -> None:
    for bad in ("/parallel", "/parallel   ", "  /parallel  "):
        with pytest.raises(DirectiveError):
            parse_directive(bad)


def test_spec_without_task_raises() -> None:
    for bad in ("/parallel 2", "/parallel gpt-5,opus", "/parallel 2   "):
        with pytest.raises(DirectiveError):
            parse_directive(bad)


def test_empty_segment_mid_message_raises() -> None:
    # all-or-nothing: a later empty segment fails the whole parse
    with pytest.raises(DirectiveError):
        parse_directive("/parallel 2 good task /parallel")


def test_segment_is_a_frozen_dataclass() -> None:
    (seg,) = parse_directive("/parallel 2 x")  # type: ignore[misc]
    assert isinstance(seg, Segment)
    with pytest.raises(AttributeError):
        seg.spec = "3"  # type: ignore[misc]
