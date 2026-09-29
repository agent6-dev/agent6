# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The agent6-style checkpoint subject, and the commit identity."""

from __future__ import annotations

from pathlib import Path

from agent6.commit_message import agent6_subject
from agent6.config import Config
from agent6.workflows._chain import commit_identity


def test_empty_text_falls_back() -> None:
    assert agent6_subject("", 3) == "agent6 iter 3: verify passed"


def test_whitespace_only_text_falls_back() -> None:
    assert agent6_subject("\n  \n\t\n", 1) == "agent6 iter 1: verify passed"


def test_takes_first_non_empty_line() -> None:
    text = "\n\nAdded a failing test for the parser bug.\nThen fixed the parser.\n"
    out = agent6_subject(text, 7)
    assert out == "agent6 iter 7: Added a failing test for the parser bug."


def test_strips_markdown_heading_and_bullets() -> None:
    text = "# Plan\n- step one\n- step two"
    out = agent6_subject(text, 2)
    assert out == "agent6 iter 2: Plan"


def test_strips_leading_thinking_block() -> None:
    text = "<thinking>internal monologue here</thinking>\nFix the off-by-one in foo()."
    out = agent6_subject(text, 4)
    assert out == "agent6 iter 4: Fix the off-by-one in foo()."


def test_truncates_long_subject() -> None:
    long = "x" * 200
    out = agent6_subject(long, 5)
    # "agent6 iter 5: " prefix + 72 chars of body.
    assert out.startswith("agent6 iter 5: ")
    assert len(out) - len("agent6 iter 5: ") == 72


def test_unclosed_thinking_block_falls_back() -> None:
    out = agent6_subject("<thinking>oops never closed", 9)
    assert out == "agent6 iter 9: verify passed"


def test_the_configured_identity_reaches_the_commit(tmp_path: Path) -> None:
    """`[git.commit].name`/`.email` are the only identity on a machine whose
    git has none. Preflight accepts them, and the loop dropped them: every
    chain commit died with "Author identity unknown", the run reported
    "finished", and no branch existed."""
    cfg = Config.model_validate(
        {"git": {"commit": {"name": "Agent Six", "email": "agent6@example.com"}}}
    )
    identity = commit_identity(cfg.git.commit, "Assisted-by: agent6:m1")
    assert identity is not None
    assert (identity.name, identity.email) == ("Agent Six", "agent6@example.com")
    assert identity.trailer == "Assisted-by: agent6:m1"
    assert commit_identity(Config().git.commit, None) is None
