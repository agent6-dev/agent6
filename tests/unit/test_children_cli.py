# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A parent run and its real children, end to end through `agent6 run`.

The parent runs in this process through `cli_main`; each child is the detached process
`agent6 run` spawns; one scripted provider answers them all, by the task each was given.
"""

from __future__ import annotations

import contextlib
import json
import pathlib
import re
import subprocess
import sys
import threading
from typing import Any

import pytest

from agent6 import paths
from agent6.config import Config, layer
from agent6.harness.parallel import merge as parallel_merge
from agent6.sessions import child_dispatch
from agent6.tools import schema
from agent6.ui.cli import entry
from agent6.ui.web import server as web_server
from tests import launch_env
from tests.web import signin

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


def test_three_listed_tasks_two_children_and_no_refill(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A handoff holds the listed task until return, without taking the parent's next task."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(
        repo,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.com",
        "commit",
        "--allow-empty",
        "-m",
        "base",
    )
    base = _git(repo, "rev-parse", "HEAD")
    agent6 = pathlib.Path(sys.executable).with_name("agent6")
    monkeypatch.chdir(repo)
    stages: list[list[tuple[str, str, list[str]]]] = []
    release = {name: threading.Event() for name in ("Apples", "Pears")}
    briefs: list[str] = []
    parent_calls: list[dict[str, Any]] = []
    live_graph: list[str] = []
    live_children: list[str] = []
    web: dict[str, Any] = {}

    def reply(body: dict[str, Any]) -> launch_env.Reply:  # noqa: PLR0911  # one reply per scripted turn
        task = _first_task(body)
        if "Fruit inventory" not in task:
            if _turn(body) == 0:
                briefs.append(task)
                name = next(name for name in release if name in task)
                assert release[name].wait(45)
                return ("apply_edit", {"path": name, "old_string": "", "new_string": "counted\n"})
            if _turn(body) == 1:
                return _commit()
            return launch_env.FINISH
        turn = len(parent_calls)
        parent_calls.append(body)
        answers = [_text(m) for m in body["messages"] if m["role"] == "tool"]
        if answers and (rows := json.loads(answers[-1]).get("tasks")):
            stages.append([(r["id"], r["status"], r.get("held_by", [])) for r in rows])
        one, two = (*_started(body), "", "")[:2]
        if turn == 0:
            return [
                (
                    "add_task",
                    {"title": name, "rationale": f"Count {name}", "acceptance": f"File {name}"},
                )
                for name in ("Apples", "Pears", "Total")
            ]
        if turn in (1, 3, 5, 7, 9, 11):
            return ("list_tasks", {})
        if turn == 2:
            return ("parallel", {"task_id": "0002"})
        if turn == 4:
            return ("parallel", {"task_id": "0003"})
        if turn == 6 and one and two:
            parent = one.removesuffix("-p1-l1")
            live_graph.append(
                subprocess.check_output([agent6, "sessions", "graph", parent], text=True)
            )
            live_children.append(
                subprocess.check_output([agent6, "sessions", "list", "--children"], text=True)
            )
            with web_server.WebServer(("127.0.0.1", 0), repo, "", secret=signin.SECRET) as srv:
                threading.Thread(target=srv.serve_forever, daemon=True).start()
                try:
                    with contextlib.closing(signin.SignedIn(srv.server_address[1])) as conn:
                        conn.request("GET", f"/api/session/{parent}")
                        response = conn.getresponse()
                        assert response.status == 200
                        web.update(json.load(response))
                finally:
                    srv.shutdown()
            release["Pears"].set()
            return ("read_background", {"id": two, "wait_s": 60})
        if turn == 8 and one:
            release["Apples"].set()
            return ("read_background", {"id": one, "wait_s": 60})
        if turn == 10:
            return ("update_task", {"id": "0004", "status": "passed"})
        return ("finish_session", {"summary": "inventory complete"})

    with launch_env.scripted_provider(reply_for=reply) as (url, _calls):
        cfg = Config.model_validate(
            {
                "providers": {"p": {"backend": "openai", "base_url": url}},
                "models": {"default": {"model": "p/m"}},
                "git": {"control": "model", "worktree": {"base_dir": str(tmp_path / "clones")}},
                "sandbox": {"isolation": "none", "run_commands": "allow"},
                "harness": {"parallel": {"tool": True}, "verify": {"infer": False}},
                "review": {"when": "never"},
            }
        )
        config = tmp_path / "config.toml"
        config.write_text(layer.materialize(cfg, explicit_leaves=frozenset()))
        subprocess.run(
            [agent6, "--config", config, "run", "--auto-approve", "Fruit inventory"],
            check=True,
            timeout=150,
        )

    one, two = _started(parent_calls[-1])
    assert all(s[0] == ("0001", "pending", []) for s in stages)
    assert [s[1:] for s in stages] == [
        [("0002", "in_progress", []), ("0003", "pending", []), ("0004", "pending", [])],
        [("0002", "in_progress", [one]), ("0003", "in_progress", []), ("0004", "pending", [])],
        [
            ("0002", "in_progress", [one]),
            ("0003", "in_progress", [two]),
            ("0004", "in_progress", []),
        ],
        [("0002", "in_progress", [one]), ("0003", "passed", []), ("0004", "in_progress", [])],
        [("0002", "passed", []), ("0003", "passed", []), ("0004", "in_progress", [])],
        [("0002", "passed", []), ("0003", "passed", []), ("0004", "passed", [])],
    ]
    for name, task_id, child in (("Apples", "0002", one), ("Pears", "0003", two)):
        assert any(f"{name}\nRationale: Count {name}\nAcceptance: File {name}" in b for b in briefs)
        assert f"held by {child}" in live_graph[0]
        assert f"holds task {task_id}" in live_children[0]
        assert any(c["session_id"] == child and c["holds_task"] == task_id for c in web["children"])
        assert any(t["id"] == task_id and f"held by {child}" in t["note"] for t in web["tasks"])
        assert _git(repo, "show", f"refs/agent6/{child}/head:{name}") == "counted"
    assert _git(repo, "rev-parse", "HEAD") == base


@pytest.mark.parametrize("control", ["agent6", "model"])
def test_a_parent_reads_two_tasks_and_each_completes_as_it_ends(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    control: str,
) -> None:
    """Two `parallel` calls in one turn start two real children; the parent takes turns meanwhile.

    Each task completes at its next boundary, merging only under agent6-owned git.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    child_a_started = threading.Event()
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
                assert child_a_started.wait(120)
                assert entry.cli_main(["steer", first, "Use one line."]) == 0
                parent_waits_for_a.set()
            return [
                [
                    ("parallel", {"task": "CHILD-A: write a.txt"}),
                    ("parallel", {"task": "CHILD-B: write b.txt"}),
                    ("finish_session", {"summary": "too early"}),
                ],
                [("read_background", {"id": second, "wait_s": 120})],
                [("read_background", {"id": first, "wait_s": 120})],
                [("read_file", {"path": "README.md"})],
                [("finish_session", {"summary": "both tasks completed"})],
            ][min(turn, 4)]
        name = "a.txt" if who == "CHILD-A" else "b.txt"
        if turn == 1 and control == "model":
            return _commit()
        if turn > 0:
            return [("finish_session", {"summary": f"wrote {name}"})]
        if who == "CHILD-A":
            # The first task's child stays running until the parent has read the second's.
            child_a_started.set()
            assert parent_waits_for_a.wait(120)
        return [("apply_edit", {"path": name, "old_string": "", "new_string": f"{who}\n"})]

    with launch_env.scripted_provider(reply_for=reply) as (base_url, _calls):
        cfg = Config.model_validate(
            {
                "providers": {"p": {"backend": "openai", "base_url": base_url}},
                "models": {"default": {"model": "p/m"}},
                "git": {"control": control, "worktree": {"base_dir": str(tmp_path / "clones")}},
                "sandbox": {"isolation": "none", "run_commands": "allow"},
                "harness": {"parallel": {"tool": True}, "verify": {"infer": False}},
                "review": {"when": "never"},
            }
        )
        config = tmp_path / "config.toml"
        config.write_text(layer.materialize(cfg, explicit_leaves=frozenset()))
        monkeypatch.chdir(repo)
        code = entry.cli_main(
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

    def finished(sid: str) -> str:
        ref = f"agent6/{sid}" if control == "agent6" else f"refs/agent6/{sid}/head"
        return f"[parallel] child {sid} finished. {ref} holds 1 commit."

    # The parent's third call read the second task's end and its merge: the first task ran on.
    assert notices(parent[2]) == [
        finished(two),
        f"[parallel] group {session}-p2 complete (1 child(ren)):",
    ]
    assert "summary: wrote b.txt" in json.dumps(parent[2])
    order = [(who, turn) for who, turn, _body in seen]
    assert order.index(("PARENT", 2)) < order.index(("CHILD-A", 1))
    # Its fourth call read the first task's end and its merge.
    assert notices(parent[3]) == [
        finished(two),
        f"[parallel] group {session}-p2 complete (1 child(ren)):",
        finished(one),
        f"[parallel] group {session}-p1 complete (1 child(ren)):",
    ]
    assert "summary: wrote a.txt" in json.dumps(parent[3])
    # One task still ran when the read returned the other child's fields and transcript.
    read_b = json.loads(
        _text(next(m for m in reversed(parent[2]["messages"]) if m["role"] == "tool"))
    )
    transcript = read_b.pop("output").splitlines()
    assert read_b == {
        "id": two,
        "attempt": 1,
        "state": "ended, task pending",
        "task": "CHILD-B: write b.txt",
        "running": False,
        "mergeable": False,
        "commits": 1,
    }
    expected_read = [
        "user: CHILD-B: write b.txt",
        "",
        r"[tool] apply_edit path=b.txt old_string= new_string=CHILD-B\n",
        "",
        "[tool] apply_edit result: applied=['create'] path=b.txt",
    ]
    if control == "model":
        command, _, result = transcript[6:9]
        assert command.startswith('[tool] run_command argv=["sh", "-c", "git add')
        assert result.startswith("[tool] run_command result: exit=0 in ")
        expected_read.extend(["", command, "", result])
    expected_read.extend(
        ["", "[tool] finish_session summary=wrote b.txt", "", "[tool] finish_session result: ok"]
    )
    assert transcript == expected_read
    # Agent6-owned git merges; model-owned git retains each ref without changing the checkout.
    assert (repo / "a.txt").exists() is (control == "agent6")
    assert (repo / "b.txt").exists() is (control == "agent6")
    first, second = child_dispatch.read_records(session_dir)
    assert [(r.source, r.state, r.delivered) for r in (first, second)] == [
        ("model", "complete", True)
    ] * 2
    a_merge, b_merge = first.children[0].merge, second.children[0].merge
    assert a_merge is not None and b_merge is not None
    expected = "merged" if control == "agent6" else "retained"
    assert (a_merge.status, b_merge.status) == (expected, expected)
    assert _git(repo, "show", f"{a_merge.branch}:a.txt") == "CHILD-A"
    assert _git(repo, "show", f"{b_merge.branch}:b.txt") == "CHILD-B"
    if control == "model":
        assert a_merge.branch in json.dumps(parent[3]) and b_merge.branch in json.dumps(parent[2])
    assert a_merge.target_before == b_merge.target_after
    assert first.children[0].limits is None, "a child the model starts has no cap of its own"
    rows = [json.loads(line) for line in (session_dir / "logs.jsonl").read_text().splitlines()]
    for sid in (one, two):
        assert [
            r["activity"]
            for r in rows
            if r["type"] == "parallel.child_activity" and r["child"] == sid
        ] == (["started", "steered", "finished"] if sid == one else ["started", "finished"])
    assert entry.cli_main(["attach", session]) == 0
    attached = capsys.readouterr().out
    for rendered in (out.out + out.err, attached):
        assert rendered.count(f"child {one} steered: Use one line.") == 1
        for sid in (one, two):
            assert f"child {sid} started" in rendered
            assert f"child {sid} finished" in rendered
            assert (
                f"{'merged into parent chain' if control == 'agent6' else 'retained'}  {sid}"
                in rendered
            )
    assert [row["group"] for row in rows if row["type"] == "parallel.returned"] == [
        f"{session}-p2",
        f"{session}-p1",
    ]
    assert any(row["type"] == "finish.refused.children" for row in rows)
    assert [row["reason"] for row in rows if row["type"] == "session.stop"] == ["finish_session"]


@pytest.mark.parametrize("control", ["agent6", "model"])
def test_a_parent_steers_its_running_child_and_the_child_reads_it_as_its_parents(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    control: str,
) -> None:
    """`steer_child` reaches a real child at its next boundary, marked as from its parent."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    child_started, steered = threading.Event(), threading.Event()
    seen: list[tuple[str, int, dict[str, Any]]] = []
    lock = threading.Lock()
    heading = "PARENT STEERING (an instruction from the run that started this child):"

    def reply(body: dict[str, Any]) -> launch_env.Reply:
        task, turn = _first_task(body), _turn(body)
        who = "PARENT" if "PARENT" in task else "CHILD"
        with lock:
            seen.append((who, turn, body))
        if who == "PARENT":
            (child, *_rest) = (*_started(body), "")
            if turn == 1:
                # The child has a session dir once it has asked for its first turn.
                assert child_started.wait(120)
            if turn == 2:
                steered.set()
            return [
                [("parallel", {"task": "CHILD: write a.txt"})],
                [("steer_child", {"id": child, "text": "also write c.txt"})],
                [("read_background", {"id": child, "wait_s": 120})],
                [("finish_session", {"summary": "the child wrote both"})],
            ][min(turn, 3)]
        if turn == 0:
            child_started.set()
            assert steered.wait(120)
            return [("apply_edit", {"path": "a.txt", "old_string": "", "new_string": "a\n"})]
        if turn == 1 and heading in json.dumps(body["messages"]):
            return [("apply_edit", {"path": "c.txt", "old_string": "", "new_string": "c\n"})]
        if turn == 2 and control == "model":
            return _commit()
        return [("finish_session", {"summary": "wrote what was asked"})]

    with launch_env.scripted_provider(reply_for=reply) as (base_url, _calls):
        cfg = Config.model_validate(
            {
                "providers": {"p": {"backend": "openai", "base_url": base_url}},
                "models": {"default": {"model": "p/m"}},
                "git": {"control": control, "worktree": {"base_dir": str(tmp_path / "clones")}},
                "sandbox": {"isolation": "none", "run_commands": "allow"},
                "harness": {"parallel": {"tool": True}, "verify": {"infer": False}},
                "review": {"when": "never"},
            }
        )
        config = tmp_path / "config.toml"
        config.write_text(layer.materialize(cfg, explicit_leaves=frozenset()))
        monkeypatch.chdir(repo)
        code = entry.cli_main(
            ["--config", str(config), "run", "--auto-approve", "PARENT: delegate"]
        )

    out = capsys.readouterr()
    assert code == 0, out.out[-3000:] + out.err[-3000:]
    runs = paths.state_dir(repo) / "sessions" / "runs"
    (session_dir,) = [d for d in runs.iterdir() if not _CHILD_ID.search(d.name)]
    child = f"{session_dir.name}-p1-l1"
    parent = [body for who, _turn_no, body in seen if who == "PARENT"]
    # The steer answered at once, while the child was still in its first turn.
    answer = json.loads(
        _text(next(m for m in reversed(parent[2]["messages"]) if m["role"] == "tool"))
    )
    assert answer == {
        "steered": child,
        "message": f"[parallel] child {child}: steer queued for another turn, if any.",
    }
    # The child's second request carries the steer under its parent's heading, once.
    second = next(body for who, turn, body in seen if (who, turn) == ("CHILD", 1))
    told = [_text(m) for m in second["messages"] if m["role"] == "user"]
    assert sum(text.count(f"{heading}\nalso write c.txt") for text in told) == 1
    assert not any("OPERATOR STEERING" in text for text in told)
    # Both settings retain the work the parent asked for.
    assert (repo / "a.txt").exists() is (control == "agent6")
    assert (repo / "c.txt").exists() is (control == "agent6")
    merge = child_dispatch.read_records(session_dir)[0].children[0].merge
    assert merge is not None
    assert _git(repo, "show", f"{merge.branch}:c.txt") == "c"
    rows = [json.loads(line) for line in (runs / child / "logs.jsonl").read_text().splitlines()]
    (steer,) = [row for row in rows if row["type"] == "steer.parent"]
    assert (steer["text"], steer["chars"]) == ("also write c.txt", 16)
    assert f"child {child} steered: also write c.txt" in out.out + out.err
    assert entry.cli_main(["attach", session_dir.name]) == 0
    assert f"child {child} steered: also write c.txt" in capsys.readouterr().out
    assert not [row for row in rows if row["type"] == "steer.injected"]


@pytest.mark.parametrize("control", ["agent6", "model"])
def test_a_reviewer_ranks_real_attempts_under_both_git_settings(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    control: str,
) -> None:
    """Two real attempts, then a real judge: a reviewer session whose answer is the order.

    The judge's clone holds both attempts, its opening lists them, and each of its turns
    opens with its reply count.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    seen: list[tuple[str, int, dict[str, Any]]] = []
    lock = threading.Lock()
    attempts = iter(("first", "second"))

    def reply(body: dict[str, Any]) -> launch_env.Reply:  # noqa: PLR0911  # one reply per scripted turn
        task, turn = _first_task(body), _turn(body)
        who = "JUDGE" if "\nATTEMPT " in task else "PARENT" if "PARENT" in task else "CHILD"
        with lock:
            seen.append((who, turn, body))
            if who == "CHILD" and turn == 0:
                text = next(attempts)
                return [("apply_edit", {"path": "x.txt", "old_string": "", "new_string": text})]
        if who == "JUDGE":
            return judge_reply(task, turn)
        if who == "CHILD":
            if turn == 1 and control == "model":
                return _commit()
            return [("finish_session", {"summary": "wrote x.txt"})]
        if turn == 0:
            return [("parallel", {"task": "CHILD: write x.txt", "spec": "2"})]
        if "[parallel] group" in json.dumps(body["messages"]):
            return [("finish_session", {"summary": "the ranked task completed"})]
        return [("read_background", {"wait_s": 120})]

    def judge_reply(task: str, turn: int) -> launch_env.Reply:
        if turn == 0:
            # The clone holds each attempt as a branch: its log is read before ranking.
            branch = re.findall(r"^branch: (\S+)", task, re.MULTILINE)[-1]
            return [("run_command", {"argv": ["git", "log", "--format=%s", "-1", branch]})]
        ids = re.findall(r"^ATTEMPT (\S+)$", task, re.MULTILINE)
        return json.dumps({"order": list(reversed(ids)), "summary": "the second reads best"})

    with launch_env.scripted_provider(
        reply_for=reply, usage={"prompt_tokens": 9, "completion_tokens": 3}, cost_usd=0.01
    ) as (base_url, _calls):
        cfg = Config.model_validate(
            {
                "providers": {"p": {"backend": "openai", "base_url": base_url}},
                "models": {"default": {"model": "p/m"}},
                "git": {"control": control, "worktree": {"base_dir": str(tmp_path / "clones")}},
                "sandbox": {"isolation": "none", "run_commands": "allow"},
                "harness": {
                    "parallel": {"tool": True, "judges": ["default"]},
                    "verify": {"infer": False},
                },
                "review": {"when": "never", "max_turns": 5},
            }
        )
        config = tmp_path / "config.toml"
        config.write_text(layer.materialize(cfg, explicit_leaves=frozenset()))
        monkeypatch.chdir(repo)
        code = entry.cli_main(
            ["--config", str(config), "run", "--auto-approve", "PARENT: try it twice"]
        )

    out = capsys.readouterr()
    assert code == 0, out.out[-3000:] + out.err[-3000:]
    state = paths.state_dir(repo)
    (session_dir,) = [
        d for d in (state / "sessions" / "runs").iterdir() if not _CHILD_ID.search(d.name)
    ]
    session = session_dir.name
    one, two = f"{session}-p1-l1", f"{session}-p1-l2"
    judge = [body for who, _turn_no, body in seen if who == "JUDGE"]
    # The judge is one reviewer session: its opening lists both attempts, in child order.
    opening = _first_task(judge[0])
    assert re.findall(r"^ATTEMPT (\S+)$", opening, re.MULTILINE) == [one, two]
    assert "verify: not_applicable" in opening
    assert "DIFF: none" not in opening
    assert len(re.findall(r"^branch: ", opening, re.MULTILINE)) == 2
    assert "[harness] reply 1 of 5." in json.dumps(judge[0]["messages"])
    assert "[harness] reply 2 of 5." in json.dumps(judge[1]["messages"])
    ran = json.loads(_text(next(m for m in reversed(judge[1]["messages"]) if m["role"] == "tool")))
    subject = "agent6 turn 1: update x.txt" if control == "agent6" else "child work"
    assert (ran["returncode"], ran["stdout"].strip()) == (0, subject)
    assert (state / "sessions" / "reviewers" / f"{session}-rp1-l1" / "completion.json").is_file()
    # The bill counts both attempts and the ranking reviewer, not just the parent's calls.
    parent_calls = sum(who == "PARENT" for who, _, _ in seen)
    child_calls = len(seen) - parent_calls
    assert entry.cli_main(["sessions", "receipt", session]) == 0
    for surface in (out.err + out.out, capsys.readouterr().out):
        assert f"Parent: in={9 * parent_calls} out={3 * parent_calls}" in surface
        assert f"Children: in={9 * child_calls} out={3 * child_calls}" in surface
        assert f"TOTAL: in={9 * len(seen)} out={3 * len(seen)}" in surface
    assert f"cost=${0.01 * len(seen):.2f}" in out.err + out.out
    # The second attempt wins; only agent6-owned git merges it.
    task, judges = (
        child_dispatch.read_records(session_dir)[-1],
        child_dispatch.read_records(session_dir)[0],
    )
    assert (judges.identity, judges.source, judges.state) == (f"{session}-rp1", "panel", "complete")
    assert task.ranking is not None
    assert (task.ranking.order, task.ranking.ranked_by, task.ranking.winner) == (
        (two, one),
        "judge",
        two,
    )
    assert task.ranking.rationale == "the second reads best"
    assert [child.merge.status for child in task.children if child.merge] == [
        "ranked" if control == "agent6" else "retained",
        "merged" if control == "agent6" else "retained",
    ]
    assert (repo / "x.txt").exists() is (control == "agent6")
    for child in task.children:
        assert child.merge is not None
        assert _git(repo, "show", f"{child.merge.branch}:x.txt") in ("first", "second")
    parent_last = next(body for who, _t, body in reversed(seen) if who == "PARENT")
    assert "Ranked by the judges: the second reads best" in json.dumps(parent_last["messages"])
    assert (
        "ranked candidates (best first):" in out.out and "judge: the second reads best" in out.out
    )


def _commit() -> launch_env.Reply:
    """Commit a model-owned child's work with an explicit test identity."""
    return [
        (
            "run_command",
            {
                "argv": [
                    "sh",
                    "-c",
                    "git add -A && git -c user.name=test -c user.email=test@example.com"
                    " commit -m 'child work'",
                ]
            },
        )
    ]


def test_child_tool_text_and_notices_state_both_git_policies() -> None:
    """A model-owned run is never promised an automatic child merge."""
    text = schema.ParallelInput.TOOL_DESCRIPTION
    assert 'git.control = "model"' in text and "names the ref" in text
    for ids in (["one"], ["one", "two"]):
        notice = parallel_merge.started_text(ids)
        assert "task completes" in notice
        assert "first merges" not in notice
