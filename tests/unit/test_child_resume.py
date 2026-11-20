"""The same child conversation and clone continue only under their owning parent."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import threading
import tracemalloc
from collections.abc import Callable
from typing import Any

import pytest

from agent6 import child_records, paths
from agent6.app import child_launch, child_resume, child_wait, child_watch
from agent6.app import children as app_children
from agent6.config import Config, layer
from agent6.harness import child_repo
from agent6.sessions import child_dispatch, event_log, ipc, journal, lock, process
from agent6.sessions import manifest as sessions_manifest
from agent6.tools import operator_prompts
from agent6.ui.cli import cli_main
from tests import child_env, launch_env
from tests.unit import test_children_background as background
from tests.unit import test_children_cli as children_cli


@pytest.mark.parametrize("control", ["agent6", "model"])
def test_the_cli_continues_the_same_child_and_refuses_direct_resume(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    control: str,
) -> None:
    """A parent restarts its stopped child without losing its conversation or committed work."""
    repo = tmp_path / "repo"
    background._init_repo(repo)
    ready = threading.Event()
    seen: list[dict[str, Any]] = []
    parents: list[dict[str, Any]] = []

    def reply(body: dict[str, Any]) -> launch_env.Reply:  # noqa: PLR0911  # one reply per scripted turn
        task = children_cli._first_task(body)
        turn = children_cli._turn(body)
        if "PARENT:" in task:
            parents.append(body)
            ids = children_cli._started(body)
            sid = ids[0] if ids else ""
            if turn == 1:
                assert ready.wait(120)
            return [
                [("parallel", {"task": "CHILD: start the work"})],
                [("stop_background", {"id": sid})],
                [("read_background", {"id": sid, "wait_s": 120})],
                [("steer_child", {"id": sid, "text": "CONTINUE-OWNED: finish"})],
                [("read_background", {"id": sid, "wait_s": 120})],
                [("finish_session", {"summary": "child continued"})],
            ][min(turn, 5)]
        seen.append(body)
        text = json.dumps(body["messages"])
        if "CONTINUE-OWNED" in text:
            if "continued.txt" not in text:
                return [
                    ("read_file", {"path": "child.txt"}),
                    ("apply_edit", {"path": "continued.txt", "new_string": "continued\n"}),
                ]
            if control == "model" and "commit -m continuation" not in text:
                return [
                    (
                        "run_command",
                        {
                            "argv": [
                                "sh",
                                "-c",
                                "git add continued.txt && git -c user.name=test"
                                " -c user.email=test@example.com commit -m continuation",
                            ]
                        },
                    )
                ]
            return [("finish_session", {"summary": "same child finished"})]
        if turn == 0:
            return [("apply_edit", {"path": "child.txt", "new_string": "original work\n"})]
        if turn == 1 and control == "model":
            return [
                (
                    "run_command",
                    {
                        "argv": [
                            "sh",
                            "-c",
                            "git add child.txt && git -c user.name=test"
                            " -c user.email=test@example.com commit -m child",
                        ]
                    },
                )
            ]
        ready.set()
        return [("run_command", {"argv": ["sleep", "60"]})]

    with launch_env.scripted_provider(reply_for=reply) as (url, _calls):
        cfg = Config.model_validate(
            {
                "providers": {"p": {"backend": "openai", "base_url": url}},
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
        code = cli_main(
            ["--config", str(config), "run", "--auto-approve", "PARENT: continue child"]
        )
    output = capsys.readouterr()
    assert code == 0, output.out[-3000:] + output.err[-3000:]
    runs = paths.state_dir(repo) / "sessions" / "runs"
    (parent,) = [d for d in runs.iterdir() if not children_cli._CHILD_ID.search(d.name)]
    (record,) = child_dispatch.read_records(parent)
    (child,) = record.children
    assert child.continuation is not None and child.continuation.sequence == 1
    assert child.join is not None
    assert child.join.status == ("joined" if control == "agent6" else "retained")
    continued = [b for b in seen if "CONTINUE-OWNED" in json.dumps(b["messages"])]
    assert continued
    stopped = json.dumps(parents[3]["messages"])
    assert "conversation at last call: 1 tokens; last ran" in stopped
    assert "last ran: unknown" not in stopped
    assert children_cli._turn(continued[0]) > 0
    assert "original work" in json.dumps(continued[-1])
    assert (repo / "continued.txt").exists() is (control == "agent6")
    assert "continued" in background._git(repo, "show", f"{child.join.branch}:continued.txt")
    assert "parallel.child_continued" in (parent / "logs.jsonl").read_text()
    assert cli_main(["--config", str(config), "resume", child.session_id]) == 2
    output = capsys.readouterr()
    assert "steer_child" in output.out + output.err


@pytest.mark.parametrize("continued", [False, True], ids=["start", "resume"])
def test_a_child_cannot_share_its_clone_with_another_writer(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, continued: bool
) -> None:
    repo, clone = tmp_path / "repo", tmp_path / "clone"
    child_env.init_repo(repo)
    with launch_env.scripted_provider() as (url, calls):
        path, launch = child_env.launch_worker(repo, clone, child_env.config(url), monkeypatch)

        def run() -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                child_launch.bootstrap_argv(),
                cwd=clone,
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
            )

        if continued:
            first = run()
            assert first.returncode == 0, first.stdout + first.stderr
            child_launch.write(path.parent, launch.model_copy(update={"continuation": "finish"}))

        fd = lock.acquire_repo_writer(paths.state_dir(clone), clone, "standalone")
        assert fd is not None
        try:
            child = run()
        finally:
            lock.release_single_writer(fd)
        assert child.returncode == 2, child.stdout + child.stderr
        assert "run 'standalone' is already driving this checkout" in child.stdout + child.stderr
        assert len(calls) == int(continued)


@pytest.mark.parametrize("foreign", ["parent", "group", "number", "session", "clone", "reviewer"])
def test_a_private_launch_cannot_continue_another_child(
    tmp_path: pathlib.Path, foreign: str
) -> None:
    """A launch must match the record's complete child identity and its execution's clone."""
    launch = child_launch.ChildLaunch(
        worktree=tmp_path,
        record_state=tmp_path / "state",
        session_id="root-p1-l1",
        task="work",
        config=tmp_path / "config.toml",
        parent="root",
        group="root-p1",
        number=1,
        continuation="finish",
    )
    record = sessions_manifest.SessionManifest.model_validate(
        {
            "session_id": launch.session_id,
            "task": "work",
            "mode": "run",
            "parallel": {"parent": "root", "group": "root-p1", "number": 1},
        }
    )
    assert child_resume.authorized(launch, record, session_id=launch.session_id, root=tmp_path)
    edits: dict[str, Any] = {
        "parent": {"parent": "other"},
        "group": {"group": "other-p1"},
        "number": {"number": 2},
        "session": {"session_id": "root-p1-l2"},
        "clone": {"worktree": tmp_path / "other"},
        "reviewer": {"contract": object()},
    }
    assert not child_resume.authorized(
        launch.model_copy(update=edits[foreign]),
        record,
        session_id=launch.session_id,
        root=tmp_path,
    )


@pytest.mark.parametrize("redirect", [True, False], ids=["redirected", "recreated"])
def test_a_changed_clone_cannot_continue_the_bound_child(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, redirect: bool
) -> None:
    repo, clone = tmp_path / "repo", tmp_path / "clone"
    child_env.init_repo(repo)
    with launch_env.scripted_provider() as (url, calls):
        path, _ = child_env.launch_worker(repo, clone, child_env.config(url), monkeypatch)
        argv = child_launch.bootstrap_argv()
        first = subprocess.run(
            argv, cwd=clone, capture_output=True, text=True, timeout=90, check=False
        )
        assert first.returncode == 0, first.stdout + first.stderr
        launch = child_launch.read_file(path)
        child_launch.write(path.parent, launch.model_copy(update={"continuation": "continue"}))
        clone.rename(tmp_path / "retired")
        target = tmp_path / "other" if redirect else clone
        child_repo.clone_worktree(repo, target)
        if redirect:
            clone.symlink_to(target, target_is_directory=True)
        child = subprocess.run(
            argv, cwd=clone, capture_output=True, text=True, timeout=90, check=False
        )
        assert child.returncode == 2 and "clone" in child.stderr, child.stdout + child.stderr
        assert len(calls) == 1
        directory = launch.record_state / "sessions" / "runs" / launch.session_id
        assert not any(
            json.loads(row)["type"] == "session.resume"
            for row in (directory / "logs.jsonl").read_text().splitlines()
        )


@pytest.fixture
def retained_child(tmp_path: pathlib.Path) -> child_records.ChildSpec:
    """Keep the stopped child's clone, private launch and journal for a continuation."""
    repo = tmp_path / "repo"
    background._init_repo(repo)
    state = paths.state_dir(repo)
    spec = child_records.ChildSpec(1, "root-p1-l1", tmp_path / "clone", None)
    child_repo.clone_worktree(repo, spec.worktree)
    private = child_launch.launch_dir(state, "root-p1")
    launch = child_launch.ChildLaunch(
        worktree=spec.worktree,
        clone=child_launch.CloneReceipt.capture(spec.worktree),
        record_state=state,
        session_id=spec.session_id,
        task="work",
        config=tmp_path / "config.toml",
        parent="root",
        group="root-p1",
        number=1,
    )
    child_launch.write(private, launch)
    directory = child_wait.child_dir(state, spec.session_id, mode=spec.mode)
    directory.mkdir(parents=True)
    (directory / "loop_state.json").write_text("{}")
    sink = event_log.EventSink(directory / "logs.jsonl")
    sink.emit(journal.SessionStart(user_task="work", mode="run", execution_id="original"))
    sink.emit(journal.SessionStop(execution_id="original", reason="interrupted", turns=1))
    sink.emit(
        journal.SessionResume(
            session_id=spec.session_id,
            execution_id="earlier",
            steer="work",
            mode="run",
            user_task="work",
            turn=1,
            messages=2,
        )
    )
    sink.emit(
        journal.ApprovalPrompt(
            id="approval-1", prompt="Previous setup?", standing=True, scope="command"
        )
    )
    sink.emit(journal.SessionStop(execution_id="earlier", reason="interrupted", turns=1))
    return spec


def test_child_launch_requires_a_plain_git_directory(
    retained_child: child_records.ChildSpec,
) -> None:
    """A common-directory pointer is not part of an independent child clone."""
    clone = retained_child.worktree
    (clone / ".git" / "commondir").write_text(".\n", encoding="utf-8")
    with pytest.raises(child_launch.LaunchError, match="independent child clone"):
        child_launch.CloneReceipt.capture(clone)


def test_parent_does_not_rebind_a_continuation_to_a_replaced_clone(
    tmp_path: pathlib.Path, retained_child: child_records.ChildSpec
) -> None:
    repo = tmp_path / "repo"
    state = paths.state_dir(repo)
    private = child_launch.launch_dir(state, "root-p1") / "child-1.json"
    before = private.read_bytes()
    retained_child.worktree.rename(tmp_path / "original")
    child_repo.clone_worktree(repo, retained_child.worktree)

    def unexpected_spawn(*_args: object, **_kwargs: object) -> tuple[None, int, str]:
        pytest.fail("a replaced clone cannot start")

    result = app_children.bridge_spawner(
        retained_child,
        "work",
        cfg=Config(),
        explicit_leaves=frozenset(),
        origin=repo,
        record_state=state,
        max_usd=None,
        group_id="root-p1",
        parent="root",
        continuation="finish",
        runtime=app_children.ChildRuntime(spawn=unexpected_spawn),
    )
    assert not result.ok and "clone" in result.error
    assert private.read_bytes() == before


def test_a_continuation_skips_complete_history_with_bounded_reads(
    tmp_path: pathlib.Path, retained_child: child_records.ChildSpec
) -> None:
    repo = tmp_path / "repo"
    state = paths.state_dir(repo)
    child = child_wait.child_dir(state, retained_child.session_id, mode=retained_child.mode)
    with (child / "logs.jsonl").open("ab") as fh:
        fh.write(b'{"type":"loop.note","text":"' + b"x" * (8 << 20) + b'"}\n')
        fh.write(b"{" + b" " * (2 << 20))

    def finished(*_args: object, **_kwargs: object) -> tuple[None, int, str]:
        return None, 1, "ended before report"

    tracemalloc.start()
    try:
        app_children.bridge_spawner(
            retained_child,
            "work",
            cfg=Config(),
            explicit_leaves=frozenset(),
            origin=repo,
            record_state=state,
            max_usd=None,
            group_id="root-p1",
            parent="root",
            continuation="finish",
            runtime=app_children.ChildRuntime(spawn=finished),
        )
        assert tracemalloc.get_traced_memory()[1] < (4 << 20)
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize("control", ["agent6", "model"])
@pytest.mark.parametrize("phase", ["unstarted", "short", "live", "finished", "other"])
def test_continuation_readiness_distinguishes_executions(
    tmp_path: pathlib.Path, retained_child: child_records.ChildSpec, control: str, phase: str
) -> None:
    """A new execution stays recognizable after its worker pid disappears."""
    spec = retained_child
    repo = tmp_path / "repo"
    state = paths.state_dir(repo)
    directory = child_wait.child_dir(state, spec.session_id, mode=spec.mode)
    sink = event_log.EventSink(directory / "logs.jsonl")

    def spawn(
        argv: list[str],
        cwd: pathlib.Path,
        *,
        ready: Callable[[int], pathlib.Path | None],
        stop_reason: Callable[[int], str],
        env: dict[str, str],
    ) -> tuple[pathlib.Path | None, int, str]:
        assert argv == child_launch.bootstrap_argv() and cwd == spec.worktree
        assert not stop_reason(os.getpid())
        process.write_worker_pid(directory, os.getpid())
        assert ready(os.getpid()) is None, "a live pid with the old journal is not a new start"
        assert child_launch.read_file(pathlib.Path(env[child_launch.ENV])).continuation == "finish"
        if phase == "short":
            lines = sink.path.read_bytes().splitlines(keepends=True)
            sink.path.write_bytes(b"".join(lines[:-1]))
            assert ready(os.getpid()) is None, "an earlier resume is not this launch's report"
            return None, 0, "no new execution"
        if phase == "unstarted":
            return None, 0, "no new execution"
        process.emit_session_start(
            sink,
            directory,
            journal.SessionResume(
                session_id=spec.session_id,
                user_task="work",
                steer="finish",
                mode="run",
                turn=1,
                messages=2,
                execution_id="continued",
            ),
        )
        if phase == "other":
            process.clear_worker_pid(directory)
            assert ready(os.getppid()) is None, "the other worker's report cannot start this child"
            return None, os.getppid(), "no matching report"
        if phase == "finished":
            sink.emit(
                journal.SessionStop(execution_id="continued", reason="finish_session", turns=2)
            )
            process.clear_worker_pid(directory)
        try:
            assert ready(os.getpid()) == directory
            return directory, os.getpid(), ""
        finally:
            process.clear_worker_pid(directory)

    result = app_children.bridge_spawner(
        spec,
        "work",
        cfg=Config.model_validate({"git": {"control": control}}),
        explicit_leaves=frozenset(),
        origin=repo,
        record_state=state,
        max_usd=None,
        group_id="root-p1",
        parent="root",
        continuation="finish",
        runtime=app_children.ChildRuntime(spawn=spawn),
    )
    assert result.ok is (phase in {"live", "finished"})


@pytest.mark.parametrize("owner", ["ours", "other", "changed"])
def test_only_the_current_workers_preflight_can_receive_an_answer(
    tmp_path: pathlib.Path, retained_child: child_records.ChildSpec, owner: str
) -> None:
    """Only this launch's prompt can be relayed, while the same worker still owns it."""
    repo = tmp_path / "repo"
    state = paths.state_dir(repo)
    directory = child_wait.child_dir(state, retained_child.session_id, mode=retained_child.mode)
    parent = directory.with_name("root")
    parent.mkdir()
    if owner == "other":
        assert ipc.record_answer(parent, "session", "command")
    asked: list[str] = []

    def approve(request: operator_prompts.ApprovalRequest, /) -> operator_prompts.ApprovalAnswer:
        asked.append(request.prompt)
        if owner == "changed":
            process.write_worker_pid(directory, os.getppid())
        return operator_prompts.ApprovalAnswer(approved=True, source="frontend")

    prompts = operator_prompts.OperatorPrompts(session_dir=parent, approver=approve)

    def spawn(
        argv: list[str],
        cwd: pathlib.Path,
        *,
        ready: Callable[[int], pathlib.Path | None],
        stop_reason: Callable[[int], str],
        env: dict[str, str],
    ) -> tuple[pathlib.Path | None, int, str]:
        pid = os.getpid()
        process.write_worker_pid(directory, os.getppid() if owner == "other" else pid)
        assert ready(pid) is None
        if owner == "ours":
            assert not stop_reason(pid)
            assert not asked and not ipc.answer_written(directory, "approval-1")
        event_log.EventSink(directory / "logs.jsonl").emit(
            journal.ApprovalPrompt(
                id="approval-1", prompt="Current setup?", standing=True, scope="command"
            )
        )
        assert not stop_reason(pid)
        return None, pid, "exited before start"

    result = app_children.run_child_to_completion(
        retained_child,
        "work",
        cfg=Config(),
        explicit_leaves=frozenset(),
        origin=repo,
        record_state=state,
        group="root-p1",
        parent="root",
        continuation="finish",
        runtime=app_children.ChildRuntime(spawn=spawn),
        watch=child_watch.ChildWatch(prompts, parent),
    )
    assert not result.ok and result.error == "exited before start"
    assert ipc.answer_written(directory, "approval-1") is (owner == "ours")
    assert asked == ([] if owner == "other" else ["[attempt 1 root-p1-l1] Current setup?"])
