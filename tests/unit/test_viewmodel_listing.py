# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The shared run-listing helpers (session_mtime, task_snippet, summarize_session_dir)."""

from __future__ import annotations

import json
import os
import pathlib
import time
from typing import Any

import pytest

from agent6.sessions import manifest as sessions_manifest
from agent6.viewmodel import (
    format,
    is_session_husk,
    is_winner,
    listing,
    manifest_branches,
    session_compare,
    session_is_live,
    session_mtime,
    summarize_session_dir,
    task_snippet,
)


def test_run_mtime_prefers_log_over_dir(tmp_path: pathlib.Path) -> None:
    d = tmp_path / "run"
    d.mkdir()
    log = d / "logs.jsonl"
    log.write_text("{}\n", encoding="utf-8")
    os.utime(log, (1000.0, 1000.0))
    os.utime(d, (5000.0, 5000.0))  # dir bumped later (a viewer wrote frontend.pid)
    assert session_mtime(d) == 1000.0  # keyed off the log, not the dir


def test_run_mtime_of_a_log_less_session_is_its_manifest(tmp_path: pathlib.Path) -> None:
    """Opening a parked run or a `fork --no-run` in a viewer never floats it to the top of a list.

    Both have no log, and the `frontends/` claim the viewer writes bumped the dir mtime.
    """
    d = tmp_path / "run"
    d.mkdir()
    manifest = d / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    os.utime(manifest, (2000.0, 2000.0))
    os.utime(d, (9000.0, 9000.0))  # a viewer's claim bumped the dir

    assert session_mtime(d) == 2000.0

    bare = tmp_path / "husk"
    bare.mkdir()
    os.utime(bare, (2000.0, 2000.0))
    assert session_mtime(bare) == 2000.0  # nothing but the dir -> the dir


def test_task_snippet_skips_seeded_file_block() -> None:
    task = (
        "# agent6 ask\n\n## Question\n\n"
        '<file path="a.py">\ndef f(): pass\nSHOULD NOT SHOW\n</file>\n\n'
        "why is the broker slow?\n\n## Answer\n"
    )
    assert task_snippet(task) == "why is the broker slow?"


def test_task_snippet_is_the_operators_words_under_a_from_seed_and_skills() -> None:
    """The headline everywhere is what the operator typed, not the prepended skill or digest block.

    A clipped copy that cuts inside the block drops the open block instead of showing its opener.
    """
    from agent6 import task_text

    seeded = (
        '<prior-run id="agile-echo-H2EWX5">\nThis question is about a PRIOR agent6 run.\n'
        "## Run task\nhow many functions?\n</prior-run>\n\n"
        "add a module docstring to calc.py"
    )
    assert task_snippet(seeded) == "add a module docstring to calc.py"
    skilled = (
        "Apply the operator-installed skill(s) below to this task.\n\n"
        '<skill name="tdd">\nwrite the test first\n</skill>\n\n---\n\n'
        "add a --json flag\nmore detail"
    )
    assert task_snippet(skilled) == "add a --json flag"
    assert task_text.operator_task_text(skilled) == "add a --json flag\nmore detail"
    assert (
        task_text.operator_task_text(seeded[:60]) == ""
    )  # clipped inside the block: nothing invented
    assert task_snippet("plain words") == "plain words"


def test_task_snippet_plain_task() -> None:
    assert task_snippet("add a --json flag\nmore detail") == "add a --json flag"


def test_task_snippet_drops_a_markdown_heading_mark() -> None:
    """A task pasted from a TASK.md opens with `# Title`; every listing and card showed the marks.

    A `#` with no space after it (`#include`) is not a heading.
    """
    assert task_snippet("# Implement `parse_url` per RFC 3986\n\nbody") == (
        "Implement `parse_url` per RFC 3986"
    )
    assert task_snippet("### deep heading") == "deep heading"
    assert task_snippet("#include <stdio.h> fails, why?") == "#include <stdio.h> fails, why?"


def test_task_snippet_falls_back_to_stripped_text() -> None:
    assert task_snippet("   ") == ""


def test_task_snippet_of_a_task_that_is_only_a_file_block_is_its_first_line() -> None:
    """With no words of the operator's, the headline is the block's opener naming the file."""
    task = '<file path="question.md">\nWhy is the broker slow?\n</file>'
    assert task_snippet(task) == '<file path="question.md">'


def _stamp(session_dir: pathlib.Path, compare: object) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "manifest.json").write_text(json.dumps({"compare": compare}), encoding="utf-8")


def test_run_compare_and_is_winner_read_the_manifest_block(tmp_path: pathlib.Path) -> None:
    # The fixture writes the legacy `group` key; the model ignores it (old-shape
    # compat), so a fan-out lane recorded before the dedup still reads its stamp.
    win = tmp_path / "win"
    _stamp(win, {"group": "fan", "rank": 1, "of": 2, "winner": True, "ranked_by": "judge"})
    assert is_winner(win) is True
    assert isinstance(session_compare(win), sessions_manifest.CompareStamp)
    loser = tmp_path / "loser"
    _stamp(loser, {"group": "fan", "rank": 2, "of": 2, "winner": False, "ranked_by": "judge"})
    assert is_winner(loser) is False
    # A run outside any fan-out (no manifest / no compare block) reads as None.
    plain = tmp_path / "plain"
    plain.mkdir()
    assert session_compare(plain) is None and is_winner(plain) is False


def test_format_compare_headline_and_rationale() -> None:
    won = format.format_compare(
        sessions_manifest.CompareStamp(
            rank=1, of=3, winner=True, ranked_by="judge", rationale="cleanest diff"
        )
    )
    assert won == ("rank 1/3 · winner · judge", "cleanest diff")
    # A loser, mechanical, no rationale.
    lost = format.format_compare(
        sessions_manifest.CompareStamp(
            rank=2, of=3, winner=False, ranked_by="mechanical", rationale=""
        )
    )
    assert lost == ("rank 2/3 · mechanical", "")
    # No stamp -> None.
    assert format.format_compare(None) is None


def test_format_branch_is_the_one_wording_and_manifest_branches_carries_it(
    tmp_path: pathlib.Path,
) -> None:
    """`branch_line` is the header's branch line: merged into its base, else the base it lands on.

    "" without a run branch; `manifest_branches` hands it to every header.
    """
    import json

    assert format.format_branch("agent6/x", "main", "") == "agent6/x → merges into main"
    assert format.format_branch("agent6/x", "main", "main") == "agent6/x (merged into main)"
    assert format.format_branch("agent6/x", "", "") == "agent6/x"
    assert format.format_branch("", "main", "") == ""
    d = tmp_path / "run-x"
    d.mkdir()
    (d / "manifest.json").write_text(
        json.dumps({"mode": "run", "run_branch": "agent6/x", "base_branch": "main"})
    )
    assert manifest_branches(d) == {
        "run_branch": "agent6/x",
        "base_branch": "main",
        "branch_line": "agent6/x → merges into main",
    }
    (d / "manifest.json").write_text(json.dumps({"mode": "ask"}))
    assert manifest_branches(d) == {}


def test_manifest_header_carries_the_fork_lineage_in_one_wording(tmp_path: pathlib.Path) -> None:
    """`forked_from` is `<parent>@turn <n> (<sha12>)` on every header of a fork, else absent."""
    import json

    from agent6.viewmodel import manifest_header

    assert (
        format.format_lineage("orig-run-AAAAAA", 3, "a" * 40)
        == "orig-run-AAAAAA@turn 3 (aaaaaaaaaaaa)"
    )
    assert format.format_lineage("orig-run-AAAAAA", 3, None) == "orig-run-AAAAAA@turn 3"
    assert format.format_lineage(None, None, None) == ""
    d = tmp_path / "fork-x"
    d.mkdir()
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "mode": "run",
                "parent_session_id": "orig-run-AAAAAA",
                "forked_from_turn": 3,
                "forked_from_sha": "b" * 40,
            }
        )
    )
    assert manifest_header(d)["forked_from"] == "orig-run-AAAAAA@turn 3 (bbbbbbbbbbbb)"
    (d / "manifest.json").write_text(json.dumps({"mode": "run"}))
    assert "forked_from" not in manifest_header(d)


def test_manifest_branches_claims_merged_only_while_the_stamp_holds(tmp_path: pathlib.Path) -> None:
    """A run resumed after its merge reads as awaiting a merge again, with the repo at hand.

    The web Merge button read the raw stamp and stayed disabled over unmerged commits.
    """
    import json
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t"}
    env["GIT_COMMITTER_EMAIL"] = "t@t"
    git = ["git", "-C", str(repo)]
    subprocess.run([*git, "init", "-q", "-b", "main"], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "base"], check=True, env=env)
    subprocess.run([*git, "branch", "agent6/x"], check=True)
    tip = subprocess.run(
        [*git, "rev-parse", "agent6/x"], check=True, capture_output=True, text=True
    ).stdout.strip()
    d = tmp_path / "run-x"
    d.mkdir()
    stamped = {
        "mode": "run",
        "run_branch": "agent6/x",
        "base_branch": "main",
        "merged": {"into": "main", "sha": tip, "tip": tip},
    }
    (d / "manifest.json").write_text(json.dumps(stamped))
    assert manifest_branches(d, repo=repo)["branch_line"] == "agent6/x (merged into main)"
    # A later commit on the run branch: the stamp no longer describes it.
    subprocess.run([*git, "checkout", "-q", "agent6/x"], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "more"], check=True, env=env)
    got = manifest_branches(d, repo=repo)
    assert "merged_into" not in got
    assert got["branch_line"] == "agent6/x → merges into main"
    # Without the repo the manifest fact stands as recorded.
    assert manifest_branches(d)["merged_into"] == "main"


def test_manifest_branches_names_the_ref_holding_the_commits(tmp_path: pathlib.Path) -> None:
    """`commits_ref` is the ref a merge or diff reads: run branch, else chain ref, else none.

    The web Merge button gated on `run_branch`, so a `branch_per_run = false` run read "no branch
    to merge" while `sessions merge` landed it.
    """
    import subprocess

    from agent6 import git_ops

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    (repo / "a.txt").write_text("x", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "a.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "i"],
        check=True,
    )
    d = tmp_path / "s"
    d.mkdir()
    (d / "manifest.json").write_text(
        json.dumps({"session_id": "x", "run_branch": None, "base_branch": "main"}), encoding="utf-8"
    )
    assert "commits_ref" not in manifest_branches(d, repo=repo)
    chain = git_ops.chain_ref_for("x")
    subprocess.run(["git", "-C", str(repo), "update-ref", chain, "HEAD"], check=True)
    assert manifest_branches(d, repo=repo)["commits_ref"] == chain
    (d / "manifest.json").write_text(
        json.dumps({"session_id": "x", "run_branch": "agent6/x", "base_branch": "main"}),
        encoding="utf-8",
    )
    assert manifest_branches(d, repo=repo)["commits_ref"] == chain  # the branch does not exist
    subprocess.run(["git", "-C", str(repo), "branch", "agent6/x"], check=True)
    assert manifest_branches(d, repo=repo)["commits_ref"] == "agent6/x"


def test_manifest_branches_names_a_branch_only_once_it_exists(tmp_path: pathlib.Path) -> None:
    """The manifest names the run branch at run start; git creates it at the first commit.

    A run stopped before one (or parked before starting) had a header reading `agent6/x → merges
    into main` and an enabled Merge that the CLI then refused with "no branch to merge"; `sessions
    show --json` already reported `run_branch` null. One rule, `existing_run_branch`, for both.
    """
    import json
    import subprocess

    from agent6.viewmodel import existing_run_branch, session_snapshot

    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-C", str(repo)]
    subprocess.run([*git, "init", "-q", "-b", "main"], check=True)
    d = tmp_path / "run-x"
    d.mkdir()
    (d / "manifest.json").write_text(
        json.dumps({"mode": "run", "run_branch": "agent6/x", "base_branch": "main"})
    )
    (d / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "user_task": "t"})
        + "\n"
        + json.dumps({"type": "session.end", "reason": "steer_abort"})
        + "\n"
    )
    assert existing_run_branch(sessions_manifest.read_manifest(d), repo) == ""
    assert manifest_branches(d, repo=repo) == {"base_branch": "main"}
    snap = session_snapshot(d, repo=repo)
    assert "run_branch" not in snap and "branch_line" not in snap
    # The first commit creates it: from here the header names it and Merge lands.
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t"}
    env["GIT_COMMITTER_EMAIL"] = "t@t"
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "base"], check=True, env=env)
    subprocess.run([*git, "branch", "agent6/x"], check=True)
    assert existing_run_branch(sessions_manifest.read_manifest(d), repo) == "agent6/x"
    assert session_snapshot(d, repo=repo)["branch_line"] == "agent6/x → merges into main"


# --- summarize_session_dir / status_word (shared by TUI hub, web hub, runs list) --


def _write_run(
    base: pathlib.Path, sub: str, session_id: str, events: list[dict[str, object]]
) -> pathlib.Path:
    """A session dir as one looks on disk: a started session has a live worker.pid.

    Tests that model a death overwrite or unlink it.
    """
    import json
    import os

    rd = base / "sessions" / sub / session_id
    rd.mkdir(parents=True)
    (rd / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    if any(e.get("type") in ("session.start", "loop.resume.start") for e in events):
        (rd / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    return rd


def test_summary_reads_mode_task_and_passed(tmp_path: pathlib.Path) -> None:
    rd = _write_run(
        tmp_path,
        "runs",
        "r1",
        [
            {"type": "session.start", "mode": "run", "user_task": "fix [the] bug"},
            {"type": "tool.call", "name": "read_file"},
            {"type": "budget.update", "usd_total": 0.12},
            {"type": "session.end", "all_passed": True, "reason": "finish_session"},
        ],
    )
    s = summarize_session_dir(rd)
    assert (s.mode, s.task, s.status, s.reason) == ("run", "fix [the] bug", "passed", "")
    assert s.cost_usd == 0.12


def test_verify_verdict_reads_the_gate_facts_not_the_status_word(tmp_path: pathlib.Path) -> None:
    """The judge's verify tri-state reads the gate facts, not the folded status word.

    finish_session over a red gate folds to "finished", so an all-red fan-out crowned a rank 1
    and exited 0.
    """
    red_finish: list[dict[str, object]] = [
        {"type": "session.start", "mode": "run", "user_task": "t"},
        {"type": "verify.end", "cmd": ["pytest"], "exit_code": 1},
        {"type": "session.end", "all_passed": False, "reason": "finish_session"},
    ]
    rd = _write_run(tmp_path, "runs", "r-red", red_finish)
    assert summarize_session_dir(rd).verify_ok is False, "a red gate read as no-verify"

    green: list[dict[str, object]] = [
        {"type": "session.start", "mode": "run", "user_task": "t"},
        {"type": "verify.end", "cmd": ["pytest"], "exit_code": 1},
        {"type": "verify.end", "cmd": ["pytest"], "exit_code": 0},
        {"type": "session.end", "all_passed": True, "reason": "finish_session"},
    ]
    assert summarize_session_dir(_write_run(tmp_path, "runs", "r-green", green)).verify_ok is True

    gateless: list[dict[str, object]] = [
        {"type": "session.start", "mode": "run", "user_task": "t"},
        {"type": "session.end", "all_passed": False, "reason": "settled"},
    ]
    assert summarize_session_dir(_write_run(tmp_path, "runs", "r-none", gateless)).verify_ok is None

    # A plan never runs its (inferred) gate: no verdict to claim.
    plan: list[dict[str, object]] = [
        {"type": "session.start", "mode": "plan", "user_task": "t"},
        {"type": "session.end", "all_passed": True, "reason": "finish_planning"},
    ]
    assert summarize_session_dir(_write_run(tmp_path, "plans", "p1", plan)).verify_ok is None

    # A prior execution's red is not this execution's: the observation is execution-scoped, like
    # the token counters (the resumed execution may never run the gate at all).
    resumed: list[dict[str, object]] = [
        {"type": "session.start", "mode": "run", "user_task": "t"},
        {"type": "verify.end", "cmd": ["pytest"], "exit_code": 1},
        {"type": "loop.resume.start"},
        {"type": "session.end", "all_passed": False, "reason": "finish_session"},
    ]
    rd = _write_run(tmp_path, "runs", "r-executions", resumed)
    assert summarize_session_dir(rd).verify_ok is None


def test_a_finish_over_a_red_gate_resumes_plainly(tmp_path: pathlib.Path) -> None:
    """A finish_session over an observed red gate takes a plain resume.

    `finished_needs_new_work` read only the end reason, against its own rule that a red verify is
    what resume is for.
    """
    red: list[dict[str, object]] = [
        {"type": "session.start", "mode": "run", "user_task": "t"},
        {"type": "verify.end", "cmd": ["pytest"], "exit_code": 1},
        {"type": "session.end", "all_passed": False, "reason": "finish_session"},
    ]
    assert listing.finished_needs_new_work(_write_run(tmp_path, "runs", "r-red", red)) is False
    green: list[dict[str, object]] = [
        {"type": "session.start", "mode": "run", "user_task": "t"},
        {"type": "session.end", "all_passed": True, "reason": "finish_session"},
    ]
    assert listing.finished_needs_new_work(_write_run(tmp_path, "runs", "r-green", green)) is True
    gateless: list[dict[str, object]] = [
        {"type": "session.start", "mode": "run", "user_task": "t"},
        {"type": "session.end", "all_passed": None, "reason": "finish_session"},
    ]
    assert listing.finished_needs_new_work(_write_run(tmp_path, "runs", "r-none", gateless)) is True


def test_needs_new_work_is_one_predicate_for_every_surface() -> None:
    """One predicate decides "needs new work"; resume's refusal and the wire both call it.

    The web composer re-derived it without the all_passed clause.
    """
    fin = "finish_session"
    assert listing.needs_new_work(finished=True, end_reason=fin, all_passed=True) is True
    assert (
        listing.needs_new_work(finished=True, end_reason=fin, all_passed=None) is True
    )  # gateless
    assert (
        listing.needs_new_work(finished=True, end_reason=fin, all_passed=False) is False
    )  # red gate
    assert (
        listing.needs_new_work(finished=True, end_reason="budget_exhausted", all_passed=True)
        is False
    )
    assert listing.needs_new_work(finished=False, end_reason=fin, all_passed=True) is False


def test_a_finish_over_a_red_gate_reads_gate_red(tmp_path: pathlib.Path) -> None:
    """A finish_session over an observed red gate reads "finished · gate red"."""
    red: list[dict[str, object]] = [
        {"type": "session.start", "mode": "run", "user_task": "t"},
        {"type": "verify.end", "cmd": ["pytest"], "exit_code": 1},
        {"type": "session.end", "all_passed": False, "reason": "finish_session"},
    ]
    s = summarize_session_dir(_write_run(tmp_path, "runs", "r-red", red))
    assert (s.status, s.reason) == ("finished", "gate red")
    gateless: list[dict[str, object]] = [
        {"type": "session.start", "mode": "run", "user_task": "t"},
        {"type": "session.end", "all_passed": None, "reason": "finish_session"},
    ]
    s = summarize_session_dir(_write_run(tmp_path, "runs", "r-none", gateless))
    assert (s.status, s.reason) == ("finished", "")


@pytest.mark.parametrize(
    ("reason", "verify_exit", "detail"),
    [
        ("silent_finish", 1, "gate red"),
        ("metric_plateau", 0, "unverified"),
    ],
)
def test_implicit_clean_ends_do_not_read_as_failures(
    tmp_path: pathlib.Path, reason: str, verify_exit: int, detail: str
) -> None:
    """Silent and metric-driven completion are clean ends; a not-green tree qualifies them."""
    events: list[dict[str, object]] = [
        {"type": "session.start", "mode": "run", "user_task": "t"},
        {"type": "verify.end", "cmd": ["pytest"], "exit_code": verify_exit},
        {"type": "session.end", "all_passed": False, "reason": reason},
    ]
    s = summarize_session_dir(_write_run(tmp_path, "runs", reason, events))
    assert (s.status, s.reason) == ("finished", detail)


def test_summary_ask_reads_answered_not_passed(tmp_path: pathlib.Path) -> None:
    # An ask verifies nothing; "passed" for a Q&A is a category error. The ask
    # flow's own banner already says "answered", so listings must agree.
    rd = _write_run(
        tmp_path,
        "asks",
        "a1",
        [
            {"type": "session.start", "mode": "ask", "user_task": "what does x do?"},
            {"type": "session.end", "all_passed": True, "reason": "answered"},
        ],
    )
    s = summarize_session_dir(rd)
    assert (s.mode, s.status, s.reason) == ("ask", "answered", "")


def test_summary_failure_carries_its_reason(tmp_path: pathlib.Path) -> None:
    """A provider_error death reads 'failed · provider_error', never a neutral 'done'."""
    rd = _write_run(
        tmp_path,
        "runs",
        "r1",
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {"type": "session.end", "all_passed": False, "reason": "provider_error"},
        ],
    )
    s = summarize_session_dir(rd)
    assert (s.status, s.reason) == ("failed", "provider_error")


def test_summary_stop_is_not_a_failure(tmp_path: pathlib.Path) -> None:
    rd = _write_run(
        tmp_path,
        "runs",
        "r1",
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {"type": "session.end", "all_passed": False, "reason": "steer_abort"},
        ],
    )
    assert summarize_session_dir(rd).status == "stopped"


def test_summary_interrupt_reads_as_stopped(tmp_path: pathlib.Path) -> None:
    # A Ctrl-C interrupt is the operator's own act, like steer_abort -- not a
    # failure the listing should flag red.
    rd = _write_run(
        tmp_path,
        "runs",
        "r1",
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {"type": "session.end", "all_passed": False, "reason": "interrupted"},
        ],
    )
    assert summarize_session_dir(rd).status == "stopped"


def test_summary_undone_reads_undone_and_never_unmerged(tmp_path: pathlib.Path) -> None:
    """/undo ends a run with reason "undone", its own word everywhere, never the unmerged mark."""
    rd = _write_run(
        tmp_path,
        "runs",
        "r-undone",
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {"type": "session.end", "reason": "undone", "all_passed": False},
        ],
    )
    (rd / "manifest.json").write_text(
        json.dumps(
            {"mode": "run", "user_task": "t", "base_sha": "b" * 40, "run_branch": "agent6/r"}
        ),
        encoding="utf-8",
    )
    s = summarize_session_dir(rd, branch_tips={"agent6/r": "c" * 40})
    assert (s.status, s.reason, s.unmerged) == ("undone", "", False)
    assert format.listing_status_label(s.mode, s.status, s.reason, unmerged=s.unmerged) == "undone"


def test_summary_task_is_the_manifests_operator_words(tmp_path: pathlib.Path) -> None:
    """The manifest owns the listing's task: the operator's words, never a clipped or composed copy.

    session.start clips user_task to 200 chars; `run --skill` and `--from` prepend blocks.
    """
    from agent6 import task_text
    from agent6.app import manifest as app_manifest
    from agent6.config import Config
    from agent6.sessions import layout

    words = "fix the bug in the parser " * 12
    composed = f'{task_text.SKILLS_PREAMBLE}\n<skill name="tidy">be tidy</skill>\n---\n{words}'
    rd = _write_run(
        tmp_path,
        "runs",
        "r-long",
        [
            {
                "type": "session.start",
                "mode": "run",
                "user_task": task_text.operator_task_text(composed)[:200],
            },
            {"type": "session.end", "reason": "finish_session", "all_passed": True},
        ],
    )
    app_manifest.write_session_manifest(
        layout.layout_of(rd),
        session_id="r-long",
        user_task=composed,
        base_sha="",
        base_branch="main",
        run_branch=None,
        cfg=Config(),
        mode="run",
    )
    s = summarize_session_dir(rd)
    assert (s.mode, s.task, s.status) == ("run", words.strip(), "passed")


def test_summary_resume_unfinishes(tmp_path: pathlib.Path) -> None:
    """A detached resume appending past the first session.end reads running again."""
    rd = _write_run(
        tmp_path,
        "runs",
        "r1",
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {"type": "session.end", "all_passed": False, "reason": "steer_abort"},
            {"type": "loop.resume.start", "iteration": 2},
        ],
    )
    assert summarize_session_dir(rd).status == "running"


def test_summary_running_and_stale(tmp_path: pathlib.Path) -> None:
    """Liveness is the worker, not log silence: a live pid file separates running from stale."""
    rd = _write_run(tmp_path, "runs", "r2", [{"type": "session.start", "mode": "plan"}])
    assert summarize_session_dir(rd).status == "running"
    (rd / "worker.pid").unlink()  # the worker's finally cleared it on the way out
    assert summarize_session_dir(rd).status == "stale"


def test_summary_unanswered_approval_reads_waiting(tmp_path: pathlib.Path) -> None:
    # A live run whose LAST event is an unanswered approval (or ask_user
    # question) is blocked on the operator; "running" read as busy, and an
    # approval-parked lane sat invisible in every hub for hours.
    rd = _write_run(
        tmp_path,
        "runs",
        "r5",
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {"type": "approval.prompt", "id": "a1", "prompt": "Allow run_command: pytest"},
        ],
    )
    s = summarize_session_dir(rd)
    assert (s.status, s.reason) == ("waiting", "needs answer")
    # Once answered, the run is running again (the approver appends the answer).
    with (rd / "logs.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"type": "approval.answer", "id": "a1", "approved": true}\n')
    assert summarize_session_dir(rd).status == "running"


def test_summary_dead_worker_reads_stale_at_once(tmp_path: pathlib.Path) -> None:
    # A killed run (worker.pid points at a dead process, no session.end) must not
    # read "running" for the whole silence window; the pid probe settles it now.
    rd = _write_run(tmp_path, "runs", "r3", [{"type": "session.start", "mode": "run"}])
    (rd / "worker.pid").write_text("999999999", encoding="utf-8")  # beyond pid_max: never alive
    assert summarize_session_dir(rd).status == "stale"


def test_summary_live_worker_with_a_silent_log_stays_running(tmp_path: pathlib.Path) -> None:
    # The converse: a live worker blocked in a long provider call emits no
    # events for minutes, and must not read stale for it.
    import os

    rd = _write_run(tmp_path, "runs", "r4", [{"type": "session.start", "mode": "run"}])
    (rd / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    assert summarize_session_dir(rd).status == "running"


def test_summary_carries_the_partial_cost_marker(tmp_path: pathlib.Path) -> None:
    """LogScan's sticky usd_partial reaches SessionSummary, so listings print the run page's ~$."""
    rd = _write_run(
        tmp_path,
        "runs",
        "r1",
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {"type": "budget.update", "usd_total": 0.0123, "usd_partial": True},
            {"type": "session.end", "all_passed": True, "reason": "finish_session"},
        ],
    )
    assert summarize_session_dir(rd).usd_partial is True
    clean = _write_run(
        tmp_path,
        "runs",
        "r2",
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {"type": "budget.update", "usd_total": 0.0123},
            {"type": "session.end", "all_passed": True, "reason": "finish_session"},
        ],
    )
    assert summarize_session_dir(clean).usd_partial is False


def test_run_is_live_finished_run_with_lingering_pid_is_not_live(tmp_path: pathlib.Path) -> None:
    """A finished run whose worker.pid survives into teardown is not live.

    session_is_live folds the log facts; fed empty facts it degenerated to worker_is_alive and
    called this run "starting".
    """
    rd = _write_run(
        tmp_path,
        "runs",
        "r1",
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {"type": "session.end", "all_passed": True, "reason": "finish_session"},
        ],
    )
    (rd / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    assert session_is_live(rd) is False


def test_run_is_live_waiting_on_an_answer_is_live(tmp_path: pathlib.Path) -> None:
    # Blocked on an unanswered approval with a live worker: the answer WILL be
    # read, so the prompt buttons and the composer stay live.
    rd = _write_run(
        tmp_path,
        "runs",
        "r1",
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {"type": "approval.prompt", "id": "a1", "prompt": "Allow run_command: pytest"},
        ],
    )
    (rd / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    assert session_is_live(rd) is True


def test_run_is_live_dead_worker_is_not_live(tmp_path: pathlib.Path) -> None:
    rd = _write_run(tmp_path, "runs", "r1", [{"type": "session.start", "mode": "run"}])
    (rd / "worker.pid").write_text("999999999", encoding="utf-8")  # beyond pid_max
    assert session_is_live(rd) is False


def test_run_is_live_unstarted_dirs(tmp_path: pathlib.Path) -> None:
    # A parked submission or fork --no-run dir: nothing polls markers -> not
    # live (resume is the offer). A launching worker (pid, no events yet) is.
    rd = tmp_path / "sessions" / "runs" / "parked"
    rd.mkdir(parents=True)
    (rd / "manifest.json").write_text(
        json.dumps({"version": 2, "parked_task": "queued work"}), encoding="utf-8"
    )
    assert session_is_live(rd) is False
    live = _write_run(tmp_path, "runs", "launching", [])
    (live / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    assert session_is_live(live) is True


def test_summary_ask_task_comes_from_transcript(tmp_path: pathlib.Path) -> None:
    rd = _write_run(
        tmp_path,
        "asks",
        "a1",
        [
            {"type": "session.start", "mode": "ask", "user_task": '<file path="a.py">\nx'},
            {"type": "session.end", "all_passed": True},
        ],
    )
    (rd / "transcript.md").write_text(
        "# agent6 ask\n\n## Question\n\nwhat is the default port?\n", encoding="utf-8"
    )
    s = summarize_session_dir(rd)
    assert task_snippet(s.task) == "what is the default port?"


def test_summary_ask_task_is_the_question_even_when_it_starts_with_a_hash(
    tmp_path: pathlib.Path,
) -> None:
    """Only the leading `#` comment lines are skipped, never the question itself."""
    rd = _write_run(
        tmp_path,
        "asks",
        "a2",
        [
            {"type": "session.start", "mode": "ask", "user_task": "q"},
            {"type": "session.end", "all_passed": True},
        ],
    )
    (rd / "transcript.md").write_text(
        "# agent6 ask\n\n## Question\n\n#include <stdio.h> fails to compile, why?\n\n"
        "## Answer\n\nThe header search path is wrong.\n",
        encoding="utf-8",
    )
    s = summarize_session_dir(rd)
    assert task_snippet(s.task) == "#include <stdio.h> fails to compile, why?"


def test_summary_no_logs(tmp_path: pathlib.Path) -> None:
    rd = tmp_path / "sessions" / "runs" / "empty"
    rd.mkdir(parents=True)
    s = summarize_session_dir(rd)
    assert (s.status, s.task) == ("created", "(no logs)")


def test_summary_torn_manifest_reads_unreadable_not_created(tmp_path: pathlib.Path) -> None:
    """A manifest that exists but fails to parse is damage, never the "created" word.

    "created" reads as "never started" and offers to resume garbage; a dir with no manifest yet
    is a legitimate created.
    """
    rd = tmp_path / "sessions" / "runs" / "torn"
    rd.mkdir(parents=True)
    (rd / "manifest.json").write_text("{not json", encoding="utf-8")
    s = summarize_session_dir(rd)
    assert s.status == "unreadable"
    # The reason is a status cell on every surface: a four-line pydantic
    # report widened every row's column and put a URL in the state line.
    (rd / "manifest.json").write_text('{"mode": 5}', encoding="utf-8")
    s = summarize_session_dir(rd)
    assert s.status == "unreadable"
    assert "\n" not in s.reason and 0 < len(s.reason) <= 60


def test_summary_plan_reads_planned_not_passed(tmp_path: pathlib.Path) -> None:
    # A plan pass ends via finish_planning (its only clean exit) with
    # all_passed=True; it gates nothing, so it must read "planned", not "passed".
    rd = _write_run(
        tmp_path,
        "runs",
        "p1",
        [
            {"type": "session.start", "mode": "plan", "user_task": "plan the refactor"},
            {"type": "session.end", "all_passed": True, "reason": "finish_planning"},
        ],
    )
    s = summarize_session_dir(rd)
    assert (s.mode, s.status, s.reason) == ("plan", "planned", "")
    # A real run still reads "passed" (finish_session + all_passed) -- unchanged.
    rd2 = _write_run(
        tmp_path,
        "runs",
        "r1",
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {"type": "session.end", "all_passed": True, "reason": "finish_session"},
        ],
    )
    assert summarize_session_dir(rd2).status == "passed"


def test_summary_manifest_only_fork_shows_mode_and_task(tmp_path: pathlib.Path) -> None:
    # A `fork --no-run` fork has a manifest (mode + task) but no logs yet; the
    # listing must show them, not a blank "? ? (no logs)".
    rd = tmp_path / "sessions" / "runs" / "child"
    rd.mkdir(parents=True)
    (rd / "manifest.json").write_text(
        json.dumps({"mode": "plan", "user_task": "carry this forward"}), encoding="utf-8"
    )
    s = summarize_session_dir(rd)
    assert (s.mode, s.task, s.status) == ("plan", "carry this forward", "created")


def test_summary_launching_run_reads_starting(tmp_path: pathlib.Path) -> None:
    # A run with no verify_command spends ~80s inferring one BEFORE session.start.
    # During it the log has a role.call (the inference LLM call) but no session.start,
    # and the worker is alive -- it must read "starting" (its real mode+task from
    # the manifest), not a blank "? / (no task) / running" that looks missing.
    rd = _write_run(tmp_path, "runs", "boot", [{"type": "role.call", "role": "verify_inferer"}])
    (rd / "manifest.json").write_text(
        json.dumps({"mode": "run", "user_task": "refactor the loop"}), encoding="utf-8"
    )
    (rd / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")  # a live worker
    s = summarize_session_dir(rd)
    assert (s.mode, s.task, s.status) == ("run", "refactor the loop", "starting")


def test_summary_pre_start_dead_worker_says_it_died_launching(tmp_path: pathlib.Path) -> None:
    """A worker killed during preflight reads with its own reason, not "created" or a bare "stale".

    Its pid file survives with preflight events and real spend; a dir with no pid file ever stays
    "created".
    """
    rd = _write_run(tmp_path, "runs", "dead", [{"type": "role.call", "role": "verify_inferer"}])
    (rd / "manifest.json").write_text(
        json.dumps({"mode": "run", "user_task": "t"}), encoding="utf-8"
    )
    (rd / "worker.pid").write_text("999999999", encoding="utf-8")  # never alive
    s = summarize_session_dir(rd)
    assert (s.status, s.reason) == ("stale", "died launching")

    never_launched = _write_run(tmp_path, "runs", "husk", [])
    (never_launched / "worker.pid").unlink(missing_ok=True)
    assert summarize_session_dir(never_launched).status == "created"


def test_a_forks_single_execution_is_one_execution(tmp_path: pathlib.Path) -> None:
    """The first execution-start of any kind begins execution 1.

    A fork's log opens with loop.resume.start, and the unconditional increment counted its single
    execution as two.
    """
    rd = _write_run(
        tmp_path,
        "runs",
        "fork-1",
        [
            {"type": "loop.resume.start", "iteration": 1},
            {"type": "budget.update", "usd_total": 0.05},
            {"type": "session.end", "all_passed": True, "reason": "finish_session"},
        ],
    )
    scan = listing.scan_session_log(rd / "logs.jsonl")
    assert scan.executions == 1
    assert scan.cost_usd == 0.05

    # A real second execution still counts (and banks the first execution's spend).
    rd2 = _write_run(
        tmp_path,
        "runs",
        "fork-2",
        [
            {"type": "loop.resume.start", "iteration": 1},
            {"type": "budget.update", "usd_total": 0.05},
            {"type": "loop.resume.start", "iteration": 5},
            {"type": "budget.update", "usd_total": 0.01},
        ],
    )
    scan2 = listing.scan_session_log(rd2 / "logs.jsonl")
    assert scan2.executions == 2
    assert scan2.cost_usd == pytest.approx(0.06)


def test_a_forks_log_carries_its_mode_so_its_gate_verdict_is_read(tmp_path: pathlib.Path) -> None:
    """The scan reads `mode` off loop.resume.start as off session.start.

    A passed fork listed `verify_ok: null`, ranked below any `true` by `sessions compare`.
    """
    import json

    rd = _write_run(
        tmp_path,
        "runs",
        "fork-3",
        [
            {"type": "loop.resume.start", "mode": "run", "iteration": 1},
            {"type": "verify.end", "cmd": ["pytest"], "exit_code": 0, "duration_s": 1.0},
            {"type": "session.end", "all_passed": True, "reason": "finish_session"},
        ],
    )
    (rd / "manifest.json").write_text(
        json.dumps({"session_id": "fork-3", "mode": "run", "user_task": "t", "base_sha": ""}),
        encoding="utf-8",
    )
    scan = listing.scan_session_log(rd / "logs.jsonl")
    assert scan.mode == "run"
    assert scan.verify_verdict() is True
    assert listing.summary_row(summarize_session_dir(rd))["verify_ok"] is True


def test_summary_cost_sums_across_resume_executions(tmp_path: pathlib.Path) -> None:
    # Each resume execution starts a fresh budget (usd_total resets to 0). The listing
    # total must be the cumulative spend across executions, not just the latest execution's.
    rd = _write_run(
        tmp_path,
        "runs",
        "r1",
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {"type": "budget.update", "usd_total": 0.01},
            {"type": "budget.update", "usd_total": 0.02},  # execution 1 ends at $0.02
            {"type": "session.end", "all_passed": False, "reason": "budget_exhausted"},
            {"type": "loop.resume.start", "iteration": 3},
            {"type": "budget.update", "usd_total": 0.003},
            {"type": "budget.update", "usd_total": 0.007},  # execution 2 ends at $0.007
            {"type": "session.end", "all_passed": True, "reason": "finish_session"},
        ],
    )
    s = summarize_session_dir(rd)
    assert abs(s.cost_usd - 0.027) < 1e-9  # 0.02 (execution 1) + 0.007 (execution 2), not 0.007


def test_is_run_husk(tmp_path: pathlib.Path) -> None:
    # Neither manifest nor logs: never started, a husk.
    husk = tmp_path / "husk"
    husk.mkdir()
    assert is_session_husk(husk)
    # Either file makes it a real run.
    with_logs = tmp_path / "with-logs"
    with_logs.mkdir()
    (with_logs / "logs.jsonl").write_text("", encoding="utf-8")
    assert not is_session_husk(with_logs)
    with_manifest = tmp_path / "with-manifest"
    with_manifest.mkdir()
    (with_manifest / "manifest.json").write_text("{}", encoding="utf-8")
    assert not is_session_husk(with_manifest)
    # A dir with neither file but a LIVE worker.pid is a launching run in its
    # pre-manifest preflight window, not a husk -- keep it listed (as "starting").
    launching = tmp_path / "launching"
    launching.mkdir()
    (launching / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    assert not is_session_husk(launching)
    # ... but a dead worker.pid with no files is still a husk.
    dead = tmp_path / "dead-husk"
    dead.mkdir()
    (dead / "worker.pid").write_text("999999999", encoding="utf-8")
    assert is_session_husk(dead)


def test_summary_survives_a_valid_json_non_object_line(tmp_path: pathlib.Path) -> None:
    # A valid-JSON line that isn't an object (a torn or adversarial writer) must
    # not crash the listing fold -- one bad line otherwise took down the whole
    # hub / `sessions list` / TUI home. It's skipped like an unparseable line.
    rd = tmp_path / "sessions" / "runs" / "weird"
    rd.mkdir(parents=True)
    (rd / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "user_task": "do a thing"})
        + "\n"
        + "[1, 2, 3]\n"  # valid JSON, not a dict
        + '"a bare string"\n'
        + json.dumps({"type": "session.end", "all_passed": True, "reason": "finish_session"})
        + "\n",
        encoding="utf-8",
    )
    s = summarize_session_dir(rd)  # must not raise
    assert s.task == "do a thing"
    assert s.status == "passed"


def test_summary_survives_a_malformed_usd_total(tmp_path: pathlib.Path) -> None:
    # budget.update is agent6-written, but a torn write or hand-edited log can
    # leave usd_total non-numeric; the scan keeps the last good figure instead
    # of aborting the whole listing (same degradation the typed fold applies).
    # Falsy junk ('', False) counts: an `or 0.0` fallback silently reset it.
    rd = tmp_path / "sessions" / "runs" / "torn-usd"
    rd.mkdir(parents=True)
    (rd / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "user_task": "t"})
        + "\n"
        + json.dumps({"type": "budget.update", "usd_total": 0.25})
        + "\n"
        + json.dumps({"type": "budget.update", "usd_total": "garbage"})
        + "\n"
        + json.dumps({"type": "budget.update", "usd_total": [1, 2]})
        + "\n"
        + json.dumps({"type": "budget.update", "usd_total": ""})
        + "\n"
        + json.dumps({"type": "budget.update", "usd_total": False})
        + "\n"
        + json.dumps({"type": "session.end", "all_passed": True, "reason": "finish_session"})
        + "\n",
        encoding="utf-8",
    )
    s = summarize_session_dir(rd)  # must not raise
    assert s.status == "passed"
    assert s.cost_usd == 0.25  # the last good figure, not 0 and not a crash


def test_summary_gateless_settle_reads_finished_unverified(tmp_path: pathlib.Path) -> None:
    # A gateless run's quiet finish committed real work but nothing verified
    # it: "finished · unverified", deliberately neither green nor "failed".
    # ("unverified", not "no verify": a command may exist via mid-run adoption
    # and simply never have passed.)
    rd = _write_run(
        tmp_path,
        "runs",
        "g1",
        [
            {"type": "session.start", "mode": "run", "user_task": "build it"},
            {"type": "session.end", "all_passed": False, "reason": "settled"},
        ],
    )
    s = summarize_session_dir(rd)
    assert (s.status, s.reason) == ("finished", "unverified")


def test_summary_settle_after_a_red_gate_reads_gate_red(tmp_path: pathlib.Path) -> None:
    """Settling after a failed reverify is a deliberate red-gated end, not an unverified one."""
    rd = _write_run(
        tmp_path,
        "runs",
        "red-settle",
        [
            {"type": "session.start", "mode": "run", "user_task": "fix it"},
            {"type": "verify.end", "cmd": ["pytest"], "exit_code": 1},
            {"type": "session.end", "all_passed": False, "reason": "settled"},
        ],
    )
    s = summarize_session_dir(rd)
    assert (s.status, s.reason, s.verify_ok) == ("finished", "gate red", False)


def test_summary_second_run_start_reads_running(tmp_path: pathlib.Path) -> None:
    """An ask REPL follow-up on the same log reads "running" while it streams, not "answered"."""
    rd = _write_run(
        tmp_path,
        "asks",
        "ask-repl",
        [
            {"type": "session.start", "mode": "ask", "user_task": "q"},
            {"type": "session.end", "all_passed": True, "reason": "answered"},
            {"type": "session.start", "mode": "ask", "user_task": "q2"},
            {"type": "role.call", "role": "worker", "model": "m"},
        ],
    )
    assert summarize_session_dir(rd).status == "running"


def test_newest_run_dir_skips_husks_that_no_listing_shows(tmp_path: pathlib.Path) -> None:
    """The recency query hides a husk like every listing does.

    A bare `attach`, `sessions show` or `stop` targeted a phantom and could miss a live run.
    """
    bucket = tmp_path / "sessions" / "runs"
    bucket.mkdir(parents=True)
    real = bucket / "real-run-0001"
    real.mkdir()
    (real / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "t"}) + "\n",
        encoding="utf-8",
    )
    os.utime(real, (time.time() - 7200, time.time() - 7200))
    os.utime(real / "logs.jsonl", (time.time() - 7200, time.time() - 7200))
    husk = bucket / "zz-husk-0002"  # newer, but nothing ever ran
    husk.mkdir()

    assert listing.newest_session_dir([bucket]) == real


def test_summary_forked_execution_reads_mode_and_task_from_manifest(tmp_path: pathlib.Path) -> None:
    """The manifest fallback gates on a missing mode, not on saw_start.

    A resumed execution's log holds only loop.resume.start, so the row blanked to "? (no logs)".
    """
    rd = _write_run(tmp_path, "runs", "forked-0001", [{"type": "loop.resume.start"}])
    (rd / "manifest.json").write_text(
        json.dumps(
            {"version": 2, "session_id": "forked-0001", "mode": "run", "user_task": "carry on"}
        ),
        encoding="utf-8",
    )
    s = summarize_session_dir(rd)
    assert (s.mode, s.task) == ("run", "carry on")


def test_scan_counts_a_non_string_prompt_id_as_blocking(tmp_path: pathlib.Path) -> None:
    """The prompt side coerces ids to str like the answer side, so an int id never reads running."""
    log = tmp_path / "logs.jsonl"
    log.write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "t"})
        + "\n"
        + json.dumps({"type": "approval.prompt", "id": 7})
        + "\n",
        encoding="utf-8",
    )
    assert listing.scan_session_log(
        log
    ).operator_blocked  # the int id still registers as unanswered

    log.write_text(
        log.read_text(encoding="utf-8") + json.dumps({"type": "approval.answer", "id": 7}) + "\n",
        encoding="utf-8",
    )
    assert not listing.scan_session_log(log).operator_blocked  # answered by the same int id


def test_a_crashed_run_reads_dead_at_once(tmp_path: pathlib.Path) -> None:
    """A loop that escaped with a fault records session.end reason=crashed, so every surface agrees.

    Without it the dying process cleared worker.pid and every surface showed a dead run as
    "running" for ten minutes; a SIGKILLed run leaves its pid file and reads stale at once.
    """
    session_dir = tmp_path / "sessions" / "runs" / "gone"
    session_dir.mkdir(parents=True)
    (session_dir / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "t"})
        + "\n"
        + json.dumps({"type": "session.end", "reason": "crashed", "all_passed": False})
        + "\n",
        encoding="utf-8",
    )
    summary = summarize_session_dir(session_dir)
    assert (summary.status, summary.reason) == ("failed", "crashed")
    assert session_is_live(session_dir) is False


def test_summary_ungated_end_reads_finished_and_absent_key_reads_as_before(
    tmp_path: pathlib.Path,
) -> None:
    """session.end's all_passed is a tri-state: an explicit null is "finished", an absent key False.

    A gateless silent finish listed as "passed" for a tree nothing verified.
    """
    ungated = _write_run(
        tmp_path,
        "runs",
        "r-ungated",
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {"type": "session.end", "reason": "silent_finish", "all_passed": None},
        ],
    )
    assert (summarize_session_dir(ungated).status, summarize_session_dir(ungated).reason) == (
        "finished",
        "",
    )
    legacy = _write_run(
        tmp_path,
        "runs",
        "r-legacy",
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {"type": "session.end", "reason": "went_quiet"},
        ],
    )
    assert summarize_session_dir(legacy).status == "failed"


def _summary(session_id: str, *, mtime: float, coordinator: str = "", lane: int | None = None):
    return listing.SessionSummary(
        session_id=session_id,
        mode="run",
        task="t",
        status="passed",
        reason="",
        cost_usd=0.0,
        usd_partial=False,
        mtime=mtime,
        coordinator=coordinator,
        lane=lane,
    )


def test_a_lane_nests_under_its_coordinator_and_an_orphan_stays_a_row() -> None:
    """A lane nests under the session that dispatched it, and the group sorts by its latest lane.

    A lane whose coordinator is not listed has nothing to nest under; a lane that dispatched
    its own group carries its lanes one level deeper, every session once.
    """
    rows = listing.nested_rows(
        [
            _summary("other", mtime=50.0),
            _summary("fan", mtime=10.0),
            _summary("fan-l2", mtime=30.0, coordinator="fan", lane=2),
            _summary("fan-l1", mtime=20.0, coordinator="fan", lane=1),
            _summary("gone-l1", mtime=40.0, coordinator="gone", lane=1),
            _summary("fan-l1-p1-l1", mtime=35.0, coordinator="fan-l1", lane=1),
        ]
    )

    def shape(row: object) -> object:

        assert isinstance(row, listing.ListingRow)
        return (row.summary.session_id, [shape(ln) for ln in row.lanes])

    assert [shape(r) for r in rows] == [
        ("other", []),
        ("gone-l1", []),
        ("fan", [("fan-l1", [("fan-l1-p1-l1", [])]), ("fan-l2", [])]),
    ]
    assert [r.mtime for r in rows] == [50.0, 40.0, 35.0]
    fan = listing.row_json(rows[2], winners={"fan-l2"})
    assert fan["mtime"] == 35.0  # the group's latest activity, as the row sorts
    lanes: Any = fan["lanes"]
    assert [ln["session_id"] for ln in lanes] == ["fan-l1", "fan-l2"]
    assert lanes[1]["winner"] is True and lanes[1]["lane"] == 2
    assert lanes[1]["coordinator"] == "fan" and lanes[1]["lanes"] == []
    assert lanes[0]["lanes"][0]["session_id"] == "fan-l1-p1-l1"
    assert fan["lane"] is None and fan["coordinator"] == ""


def test_a_never_started_run_reads_at_the_parked_level() -> None:
    """A `fork --no-run` dir's word warns like "parked" does."""
    assert format.status_level("created") == format.status_level("parked") == "warn"


def test_scan_carries_the_cached_tokens_the_budget_reports(tmp_path: pathlib.Path) -> None:
    """The scan keeps the cached input side, so a 500k-token run never reads as a few dozen tokens.

    Journals written before the fields existed read as None.
    """
    logs = tmp_path / "logs.jsonl"
    logs.write_text(
        json.dumps({"type": "session.start", "session_id": "s", "mode": "run", "user_task": "t"})
        + "\n"
        + json.dumps(
            {
                "type": "budget.update",
                "input_total": 18,
                "output_total": 2194,
                "cache_read_total": 42486,
                "cache_creation_total": 22617,
                "usd_total": 0.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    scan = listing.scan_session_log(logs)
    assert (scan.input_tokens, scan.output_tokens) == (18, 2194)
    assert (scan.cache_read_tokens, scan.cache_creation_tokens) == (42486, 22617)
    logs.write_text(
        json.dumps({"type": "budget.update", "input_total": 1, "output_total": 2}) + "\n",
        encoding="utf-8",
    )
    old = listing.scan_session_log(logs)
    assert (old.cache_read_tokens, old.cache_creation_tokens) == (None, None)
    # The four travel as one group: an aggregate event without the cached side
    # (a draft's summed attempts) shows none, not an earlier event's figures.
    logs.write_text(
        json.dumps(
            {
                "type": "budget.update",
                "input_total": 18,
                "output_total": 2194,
                "cache_read_total": 42486,
                "cache_creation_total": 22617,
            }
        )
        + "\n"
        + json.dumps({"type": "budget.update", "input_total": 250, "output_total": 50})
        + "\n",
        encoding="utf-8",
    )
    aggregate = listing.scan_session_log(logs)
    assert (aggregate.input_tokens, aggregate.output_tokens) == (250, 50)
    assert (aggregate.cache_read_tokens, aggregate.cache_creation_tokens) == (None, None)


def test_summary_names_the_questions_nobody_answered(tmp_path: pathlib.Path) -> None:
    """A row names a question that waited unanswered in the transcript, not a bare "passed"."""
    rd = _write_run(
        tmp_path,
        "runs",
        "r9",
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {"type": "question.prompt", "id": "q1", "questions": [{"question": "Which?"}]},
            {
                "type": "question.answer",
                "id": "q1",
                "answers": [""],
                "source": "headless-default",
                "unseen": True,
            },
            {"type": "question.prompt", "id": "q2", "questions": [{"question": "And?"}]},
            {
                "type": "question.answer",
                "id": "q2",
                "answers": ["b"],
                "source": "frontend",
                "unseen": False,
            },
            {"type": "session.end", "reason": "finish_session", "all_passed": True},
        ],
    )

    (rd / "worker.pid").unlink()
    assert listing.scan_session_log(rd / "logs.jsonl").unattended_questions == 1
    s = summarize_session_dir(rd)
    assert (s.status, s.reason) == ("passed", "1 question unanswered")
