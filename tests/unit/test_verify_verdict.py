# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""VerifyVerdict transitions: the streak/reset rules every consumer relies on."""

from __future__ import annotations

from agent6.harness import verify_verdict


def test_pass_resets_streak_and_covers_earlier_edits() -> None:
    v = verify_verdict.VerifyVerdict()
    v.note_edit()
    v.note_fail("sig-a")
    v.note_pass()
    assert v.last_ok is True and v.ever_passed and v.ever_failed
    assert v.edited_since is False
    assert v.fail_streak == 0 and v.fail_signature == ""
    assert v.green_and_untouched


def test_same_signature_extends_the_streak_a_new_one_restarts_it() -> None:
    v = verify_verdict.VerifyVerdict()
    v.note_fail("sig-a")
    v.note_fail("sig-a")
    assert v.fail_streak == 2
    v.note_fail("sig-b")  # a NEW stuck point
    assert v.fail_streak == 1 and v.fail_signature == "sig-b"


def test_a_red_verdict_covers_its_tree_like_a_green_one() -> None:
    """A red clears edited_since like a green: an untouched red tree is not re-run or recounted."""
    v = verify_verdict.VerifyVerdict()
    v.note_edit()
    v.note_fail("sig")
    assert v.judged_and_untouched  # the red judged the tree as it stands
    assert not v.green_and_untouched  # but it is not green
    v.note_edit()
    assert not v.judged_and_untouched  # an edit moves the tree past the verdict


def test_an_edit_after_green_withdraws_the_green() -> None:
    v = verify_verdict.VerifyVerdict()
    v.note_pass()
    assert v.green_and_untouched
    v.note_edit()
    assert not v.green_and_untouched
    assert v.last_ok is True  # the observation stands; only its coverage lapsed


def test_a_green_baseline_ends_a_failure_streak() -> None:
    """A green judged on the base commit mid-run resets the streak as any green does."""
    v = verify_verdict.VerifyVerdict()
    v.note_fail("sig-a")
    v.note_fail("sig-a")
    v.note_baseline(ok=True)
    assert v.fail_streak == 0 and v.fail_signature == ""
    v.note_baseline(ok=False)
    assert v.fail_streak == 0 and v.fail_signature == ""


def test_the_fact_names_the_command_the_verdict_and_whether_the_tree_moved() -> None:
    """The worker's one-line verify fact follows the verdict through its states."""
    v = verify_verdict.VerifyVerdict()
    assert v.fact() == "No verification verdict covers the tree as it stands."
    v.command, v.verify_tree = ("make", "check"), "a" * 40
    v.note_pass()
    assert v.fact() == (
        f"Last full verification: `make check` passed on tree {'a' * 12}; the tree is"
        " unchanged since."
    )
    v.note_edit()
    assert v.fact().endswith("passed on tree aaaaaaaaaaaa; the tree has changed since.")
    v.note_fail("sig")
    assert "`make check` failed on tree aaaaaaaaaaaa; the tree is unchanged since." in v.fact()
    # A resume onto another tree keeps the verified tree as a fact and carries no verdict.
    moved = verify_verdict.VerifyVerdict(
        command=("make", "check"), verify_tree="b" * 40, edited_since=True
    )
    assert moved.fact() == (
        f"The last full verification ran `make check` on tree {'b' * 12}; the tree as it"
        " stands is not that tree, so no verdict covers it."
    )


def test_the_verified_tree_restored_reads_untouched() -> None:
    """Edits made and undone leave the verified tree standing: the verdict follows the tree."""
    v = verify_verdict.VerifyVerdict(verify_tree="t" * 40)
    v.note_pass()
    v.note_edit()
    assert not v.green_and_untouched
    v.note_tree("u" * 40)
    assert not v.green_and_untouched
    v.note_tree("t" * 40)
    assert v.green_and_untouched
    v.note_tree("")  # git could not say: the flag stands
    assert v.green_and_untouched
