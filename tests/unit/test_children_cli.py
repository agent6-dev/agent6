# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A parent run and its real children, end to end through `agent6 run`.

The parent runs in this process through `cli_main`; each child is the detached process
`agent6 run` spawns; one scripted provider answers them all, by the task each was given.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import threading
from typing import Any

import pytest

from agent6 import paths
from agent6.config import Config, layer
from agent6.sessions import child_dispatch
from agent6.ui.cli import cli_main
from tests import launch_env

# A child's session id: its parent's, the task's group and the child's number.
_CHILD_ID = re.compile(r"-p\d+-l\d+$")


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _text(message: dict[str, Any]) -> str:
    """Return a message's text, whether the wire carries a string or a list of parts."""
    content = message.get("content") or ""
    if isinstance(content, str):
        return content
    return "\n".join(str(part.get("text", "")) for part in content if isinstance(part, dict))


def _first_task(body: dict[str, Any]) -> str:
    """Return the first user message of a request: the session's own task."""
    return _text(next(m for m in body["messages"] if m["role"] == "user"))


def _turn(body: dict[str, Any]) -> int:
    """Return how many turns the session has taken: its assistant messages so far."""
    return sum(1 for m in body["messages"] if m["role"] == "assistant")


def _started(body: dict[str, Any]) -> list[str]:
    """Return the child ids the parent's `parallel` calls answered with, in call order."""
    ids: list[str] = []
    for message in body["messages"]:
        if message["role"] != "tool":
            continue
        answer = json.loads(_text(message))
        if isinstance(answer, dict) and answer.get("dispatched"):
            ids.extend(answer["children"])
    return ids


def test_a_parent_starts_two_tasks_keeps_working_reads_a_child_and_each_lands_as_it_ends(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two `parallel` calls in one turn start two real children; the parent takes turns meanwhile.

    The second task's child ends first and is read; its work lands at the next boundary while
    the first task's child still runs, and the parent finishes after reading both landings.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    parent_waits_for_a = threading.Event()
    seen: list[tuple[str, int, dict[str, Any]]] = []
    lock = threading.Lock()

    def reply(body: dict[str, Any]) -> launch_env.Reply:
        task, turn = _first_task(body), _turn(body)
        who = next(name for name in ("CHILD-A", "CHILD-B", "PARENT") if name in task)
        with lock:
            seen.append((who, turn, body))
        if who == "PARENT":
            first, second = (*_started(body), "", "")[:2]
            if turn == 2:
                parent_waits_for_a.set()
            return [
                [
                    ("parallel", {"task": "CHILD-A: write a.txt"}),
                    ("parallel", {"task": "CHILD-B: write b.txt"}),
                ],
                [("read_background", {"id": second, "wait_s": 120})],
                [("read_background", {"id": first, "wait_s": 120})],
                [("read_file", {"path": "README.md"})],
                [("finish_session", {"summary": "both tasks landed"})],
            ][min(turn, 4)]
        name = "a.txt" if who == "CHILD-A" else "b.txt"
        if turn > 0:
            return [("finish_session", {"summary": f"wrote {name}"})]
        if who == "CHILD-A":
            # The first task's child stays running until the parent has read the second's.
            assert parent_waits_for_a.wait(120)
        return [("apply_edit", {"path": name, "old_string": "", "new_string": f"{who}\n"})]

    with launch_env.scripted_provider(reply_for=reply) as (base_url, _calls):
        cfg = Config.model_validate(
            {
                "providers": {"p": {"backend": "openai", "base_url": base_url}},
                "models": {"default": {"model": "p/m"}},
                "git": {"worktree": {"base_dir": str(tmp_path / "clones")}},
                "sandbox": {"isolation": "none", "run_commands": "allow"},
                "harness": {"parallel": {"tool": True}, "verify": {"infer": False}},
                "review": {"when": "never"},
            }
        )
        config = tmp_path / "config.toml"
        config.write_text(layer.materialize(cfg, explicit_leaves=frozenset()))
        monkeypatch.chdir(repo)
        code = cli_main(
            ["--config", str(config), "run", "--auto-approve", "PARENT: split the work in two"]
        )

    out = capsys.readouterr()
    assert code == 0, out.out[-3000:] + out.err[-3000:]
    runs = paths.state_dir(repo) / "sessions" / "runs"
    (session_dir,) = [d for d in runs.iterdir() if not _CHILD_ID.search(d.name)]
    session = session_dir.name
    one, two = f"{session}-p1-l1", f"{session}-p2-l1"
    parent = [body for who, _turn_no, body in seen if who == "PARENT"]
    assert [turn for who, turn, _body in seen if who == "PARENT"] == [0, 1, 2, 3, 4]
    assert _started(parent[1]) == [one, two]

    def notices(body: dict[str, Any]) -> list[str]:
        return [
            line
            for message in body["messages"]
            if message["role"] == "user"
            for line in _text(message).splitlines()
            if line.startswith("[parallel] ")
        ]

    # The parent's third call read the second task's end and its landing: the first task ran on.
    assert notices(parent[2]) == [
        f"[parallel] child {two} finished.",
        f"[parallel] group {session}-p2 complete (1 child(ren)):",
    ]
    assert "summary: wrote b.txt" in json.dumps(parent[2])
    order = [(who, turn) for who, turn, _body in seen]
    assert order.index(("PARENT", 2)) < order.index(("CHILD-A", 1))
    # Its fourth call read the first task's end and its landing.
    assert notices(parent[3]) == [
        f"[parallel] child {two} finished.",
        f"[parallel] group {session}-p2 complete (1 child(ren)):",
        f"[parallel] child {one} finished.",
        f"[parallel] group {session}-p1 complete (1 child(ren)):",
    ]
    assert "summary: wrote a.txt" in json.dumps(parent[3])
    # The read of the ended child returned its line and transcript while the other still ran.
    read_b = json.loads(
        _text(next(m for m in reversed(parent[2]["messages"]) if m["role"] == "tool"))
    )
    assert read_b["output"].splitlines() == [
        f"[{two}] ended, not landed: CHILD-B: write b.txt",
        "",
        "user: CHILD-B: write b.txt",
        "",
        "[tool] apply_edit",
        "",
        "[tool] finish_session",
    ]
    assert read_b["children"][0] == f"[{one}] running: CHILD-A: write a.txt"
    # Both children's work is in the parent's tree and on its chain, the first on the second.
    assert (repo / "a.txt").read_text(encoding="utf-8") == "CHILD-A\n"
    assert (repo / "b.txt").read_text(encoding="utf-8") == "CHILD-B\n"
    first, second = child_dispatch.read_records(session_dir)
    assert [(r.source, r.state, r.delivered) for r in (first, second)] == [
        ("model", "complete", True)
    ] * 2
    a_join, b_join = first.children[0].join, second.children[0].join
    assert a_join is not None and b_join is not None
    assert (a_join.status, b_join.status) == ("joined", "joined")
    assert a_join.target_before == b_join.target_after
    assert first.children[0].limits is None, "a child the model starts has no cap of its own"
    rows = [json.loads(line) for line in (session_dir / "logs.jsonl").read_text().splitlines()]
    assert [row["group"] for row in rows if row["type"] == "parallel.joined"] == [
        f"{session}-p2",
        f"{session}-p1",
    ]
    assert [row["reason"] for row in rows if row["type"] == "session.stop"] == ["finish_session"]
