# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One ACP prompt becoming one agent6 run, driven over a real connection."""

from __future__ import annotations

import io
import json
import os
import select
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest

from agent6.app.reporter import Reporter
from agent6.config.model import ConfigError
from agent6.runs.layout import RunLayout
from agent6.ui.acp import runner
from agent6.ui.acp import session as session_mod
from agent6.ui.acp.runner import STDERR_REPORTER, RunBridge, option_kind, stop_reason
from agent6.ui.acp.server import ACPServer


class _Wire:
    """A live connection: messages in one pipe, replies out the other."""

    def __init__(self) -> None:
        self._in_r, self._in_w = os.pipe()
        self._out_r, self._out_w = os.pipe()
        self.server = ACPServer(
            stdin=os.fdopen(self._in_r, "rb"),
            stdout=os.fdopen(self._out_w, "wb"),
        )
        self.server.sessions = RunBridge(server=self.server).sessions()
        # Unbuffered, so `select` telling us nothing is waiting is the truth.
        self._reader = os.fdopen(self._out_r, "rb", buffering=0)
        self._thread = threading.Thread(target=self.server.serve, daemon=True)
        self._thread.start()

    def send(self, **message: Any) -> None:
        os.write(self._in_w, json.dumps({"jsonrpc": "2.0", **message}).encode() + b"\n")

    def recv(self, timeout: float = 5.0) -> dict[str, Any]:
        if not select.select([self._out_r], [], [], timeout)[0]:
            raise AssertionError("the editor got nothing back")
        return json.loads(self._reader.readline())

    def until(self, method: str, timeout: float = 5.0) -> dict[str, Any]:
        for _ in range(50):
            message = self.recv(timeout=timeout)
            if message.get("method") == method or method == "":
                return message
        raise AssertionError(f"no {method} arrived")

    def close(self) -> None:
        os.close(self._in_w)
        self._thread.join(timeout=5.0)

    def new_session(self, cwd: Path) -> str:
        self.send(id=1, method="initialize", params={"clientCapabilities": {}})
        self.recv()
        self.send(id=2, method="session/new", params={"cwd": str(cwd)})
        return str(self.recv()["result"]["sessionId"])

    def prompt(self, session_id: str, text: str, *, req_id: int = 3) -> None:
        self.send(
            id=req_id,
            method="session/prompt",
            params={"sessionId": session_id, "prompt": [{"type": "text", "text": text}]},
        )


def _ignore(_path: Path) -> None:
    """A cancel that has nowhere to write is still a cancel."""


def _repo(path: Path) -> Path:
    """A session's cwd has to pass the same git-repo wall `agent6 run` uses."""
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    return path


def test_the_reporter_never_writes_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    """stdout IS the protocol stream. One status line on it desynchronises the
    connection irrecoverably, and no editor recovers from that."""
    STDERR_REPORTER.out("a status line")
    STDERR_REPORTER.err("a warning")
    captured = capsys.readouterr()
    assert captured.out == "", "the wire must carry nothing but JSON-RPC"
    assert "a status line" in captured.err and "a warning" in captured.err


def test_a_cancel_reaches_the_run_it_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The run id is minted BEFORE the run starts, so the session has a handle
    to address. Letting the lifecycle mint its own left `run_id` empty: the
    cancel reported success while the run continued to completion, spending
    budget and making commits."""
    stopped: list[Path] = []
    monkeypatch.setattr(session_mod, "request_stop", stopped.append)
    monkeypatch.chdir(tmp_path)

    started, release = threading.Event(), threading.Event()

    def _blocking_run(*_a: object, **kw: object) -> int:
        started.set()
        release.wait(timeout=5.0)
        return 0

    monkeypatch.setattr(runner, "run_task", _blocking_run)

    wire = _Wire()
    try:
        session_id = wire.new_session(_repo(tmp_path))
        wire.prompt(session_id, "do the thing")
        assert started.wait(timeout=5.0), "the run never started"
        wire.send(method="session/cancel", params={"sessionId": session_id})
        for _ in range(100):
            if stopped:
                break
            threading.Event().wait(0.05)
        assert stopped, "the stop marker never reached a run directory"
        assert stopped[0].parent.name == "runs", stopped[0]
        assert stopped[0].name, "the run id was empty, so the cancel addressed nothing"
    finally:
        release.set()
        wire.close()


def test_a_cancelled_turn_says_so(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(session_mod, "request_stop", _ignore)
    monkeypatch.chdir(tmp_path)
    started, release = threading.Event(), threading.Event()

    def _blocking_run(*_a: object, **kw: object) -> int:
        started.set()
        release.wait(timeout=5.0)
        return 0

    monkeypatch.setattr(runner, "run_task", _blocking_run)
    wire = _Wire()
    try:
        session_id = wire.new_session(_repo(tmp_path))
        wire.prompt(session_id, "do the thing")
        assert started.wait(timeout=5.0)
        wire.send(method="session/cancel", params={"sessionId": session_id})
        release.set()
        answer = wire.until("")
        assert answer["result"]["stopReason"] == "cancelled"
    finally:
        release.set()
        wire.close()


def test_the_runs_journal_streams_out_as_session_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tail is the whole live view: without it an editor sees a turn that
    starts, says nothing for minutes, and then answers.

    Driven by the RECORDED journal the fold's golden test uses, not by
    hand-written events: a fabricated shape the engine never emits is how a
    surface tests green while rendering nothing.
    """
    monkeypatch.chdir(tmp_path)
    recorded = Path(__file__).parent.parent / "unit" / "data" / "golden_run_logs.jsonl"

    def _writing_run(*_a: object, **kw: object) -> int:
        run_id = kw["run_id"]
        assert isinstance(run_id, str)
        layout = RunLayout(state_dir=runner.resolved_state_dir(tmp_path), run_id=run_id)
        layout.run_dir.mkdir(parents=True, exist_ok=True)
        layout.logs_path.write_bytes(recorded.read_bytes())
        return 0

    monkeypatch.setattr(runner, "run_task", _writing_run)
    wire = _Wire()
    try:
        session_id = wire.new_session(_repo(tmp_path))
        wire.prompt(session_id, "do the thing")
        seen: list[str] = []
        for _ in range(40):
            message = wire.recv()
            if message.get("method") != "session/update":
                continue
            assert message["params"]["sessionId"] == session_id
            seen.append(json.dumps(message["params"]["update"]))
            if any("Run passed" in body for body in seen):
                break
        assert any("Let me read the file." in body for body in seen), "no thinking reached it"
        assert any('"tool_call"' in body for body in seen), "no tool call reached it"
        assert any("Run passed" in body for body in seen), "the ending never reached it"
    finally:
        wire.close()


def test_a_run_that_cannot_start_says_why(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken config is the ordinary case, and it raises before the run has a
    journal to carry the reason. The editor would otherwise see a turn end with
    a stop reason and no words at all."""
    monkeypatch.chdir(tmp_path)

    def _broken(*_a: object, **_kw: object) -> object:
        raise ConfigError("Config file is not valid TOML (agent6.toml)")

    monkeypatch.setattr(runner, "load_effective", _broken)
    wire = _Wire()
    try:
        session_id = wire.new_session(_repo(tmp_path))
        wire.prompt(session_id, "do the thing")
        said = wire.until("session/update")
        text = said["params"]["update"]["content"]["text"]
        assert "could not start" in text and "agent6.toml" in text, text
        assert wire.until("")["result"]["stopReason"] == "refusal"
    finally:
        wire.close()


def _bridge(answer: dict[str, Any]) -> RunBridge:
    bridge = RunBridge(server=ACPServer(stdin=io.BytesIO(), stdout=io.BytesIO()))

    def _answer(*_a: object, **_kw: object) -> dict[str, Any]:
        return answer

    bridge.server.request = _answer  # pyright: ignore[reportAttributeAccessIssue]
    return bridge


def test_an_approval_round_trips_through_the_editor() -> None:
    bridge = _bridge({"outcome": {"outcome": "selected", "optionId": "allow"}})
    session = session_mod.Session(id="s", cwd=Path("/x"))
    assert bridge.ask(session, "Allow run_command: ls", ("allow", "deny"), True) == "allow"


@pytest.mark.parametrize(
    "answer",
    [
        {"outcome": {"outcome": "cancelled"}},
        {},
        {"outcome": {"outcome": "selected", "optionId": "allow-everything"}},
        {"outcome": {"outcome": "selected"}},
    ],
)
def test_only_an_option_we_offered_is_an_answer(answer: dict[str, Any]) -> None:
    """A timeout, a cancel and an echoed string are all "no answer". Treating
    an unknown string as one would let it become an allow by prefix, and the
    seam reads a None as the cautious answer."""
    bridge = _bridge(answer)
    session = session_mod.Session(id="s", cwd=Path("/x"))
    assert bridge.ask(session, "Allow run_command: rm -rf /", ("allow", "deny"), True) is None


def test_the_option_kinds_carry_what_the_editor_may_remember() -> None:
    """`allow once` is the fetch tool's off-list host, where an editor that
    remembers the answer would silently cover a different host."""
    assert option_kind("allow", True) == "allow_always"
    assert option_kind("allow once", False) == "allow_once"
    assert option_kind("deny", True) == "reject_once"
    assert option_kind("dark", None) == "allow_once"
    # The MODEL writes a question's options. Keying on the text let it name one
    # "allow" and have it advertised as a permission the editor may REMEMBER.
    assert option_kind("allow", None) == "allow_once"


def test_the_stop_reason_is_one_acp_defines() -> None:
    assert stop_reason(0) == "end_turn"
    assert stop_reason(1) == "refusal"
    assert stop_reason(2) == "refusal"
    assert stop_reason(130) == "cancelled"


def test_an_editor_driven_run_is_not_refused_for_lack_of_a_terminal() -> None:
    """`agent6 acp`'s stdin is the protocol pipe, never a tty.

    The refusal used to test the tty rather than the surface's own declaration,
    so with the stock `run_commands = "ask"` EVERY editor-driven run was
    refused before it started -- and the whole `session/request_permission`
    path it refused on behalf of was unreachable.
    """
    from agent6.app.preflight import headless_approval_refusal
    from agent6.config import Config
    from agent6.ui.acp.server import capabilities_from

    cfg = Config.model_validate({"sandbox": {"run_commands": "ask"}})
    caps = capabilities_from({})  # a client that declared nothing
    assert caps.can_ask is True, "every ACP client must answer session/request_permission"
    assert headless_approval_refusal(cfg, tui_enabled=False, away="", can_ask=caps.can_ask) is None


def test_a_turn_cancelled_while_queued_never_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runs are serialised, so a second turn can wait minutes for the lock.

    Minting the run id inside the lock left that whole window addressing
    nothing: the cancel wrote no stop marker, the queued turn then ran to
    completion spending budget and making commits, and the editor was told
    "cancelled" the entire time.
    """
    monkeypatch.setattr(session_mod, "request_stop", _ignore)
    monkeypatch.chdir(tmp_path)
    ran: list[str] = []
    first_started, release = threading.Event(), threading.Event()

    def _blocking_run(*_a: object, **kw: object) -> int:
        ran.append(str(kw["run_id"]))
        first_started.set()
        release.wait(timeout=5.0)
        return 0

    monkeypatch.setattr(runner, "run_task", _blocking_run)
    wire = _Wire()
    try:
        first = wire.new_session(_repo(tmp_path))
        wire.send(id=9, method="session/new", params={"cwd": str(tmp_path)})
        second = str(wire.recv()["result"]["sessionId"])
        wire.prompt(first, "the long one", req_id=3)
        assert first_started.wait(timeout=5.0)
        wire.prompt(second, "the queued one", req_id=4)
        wire.send(method="session/cancel", params={"sessionId": second})
        release.set()
        answers = {}
        for _ in range(2):
            reply = wire.until("")
            answers[reply["id"]] = reply["result"]["stopReason"]
        assert len(ran) == 1, f"the cancelled turn started anyway: {ran}"
        assert answers[4] == "cancelled", answers
        assert answers[3] == "end_turn", answers
    finally:
        release.set()
        wire.close()


def test_a_session_outside_a_git_repo_is_refused(tmp_path: Path) -> None:
    """`cwd` arrives over the wire and becomes what the jail mounts WRITABLE.

    `agent6 run` walls this with the same check; the ACP path was the one
    caller that never ran it, so a client could point a run at any absolute
    path -- and `$HOME` on a machine with dotfiles under git would hand the
    model the whole home directory as its workspace.
    """
    wire = _Wire()
    try:
        wire.send(id=1, method="initialize", params={"clientCapabilities": {}})
        wire.recv()
        wire.send(id=2, method="session/new", params={"cwd": str(tmp_path)})
        reply = wire.recv()
        assert "result" not in reply, "a non-repo directory became a workspace"
        assert "not a git repository" in reply["error"]["message"]
    finally:
        wire.close()


def test_a_relative_or_missing_cwd_is_refused(tmp_path: Path) -> None:
    wire = _Wire()
    try:
        wire.send(id=1, method="initialize", params={"clientCapabilities": {}})
        wire.recv()
        for cwd in ("relative/path", None):
            wire.send(id=2, method="session/new", params={"cwd": cwd})
            assert "absolute" in wire.recv()["error"]["message"]
    finally:
        wire.close()


def test_a_refusal_that_never_reached_a_journal_still_says_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """About a dozen lifecycle paths `return 2` after writing their reason to
    the reporter and nowhere else. With no `run.end` the fold produces no
    ending, so the editor saw a turn stop with a stop reason and no words."""
    monkeypatch.chdir(tmp_path)

    def _refusing_run(*_a: object, **kw: object) -> int:
        reporter = kw["reporter"]
        assert isinstance(reporter, Reporter)
        reporter.err("REFUSING: another writer holds this repository")
        return 2

    monkeypatch.setattr(runner, "run_task", _refusing_run)
    wire = _Wire()
    try:
        session_id = wire.new_session(_repo(tmp_path))
        wire.prompt(session_id, "do the thing")
        said = wire.until("session/update")["params"]["update"]["content"]["text"]
        assert "another writer holds this repository" in said
        assert wire.until("")["result"]["stopReason"] == "refusal"
    finally:
        wire.close()


def test_a_closed_editor_stops_waiting_for_answers_it_will_never_get() -> None:
    """The read loop is the only thing that delivers a client's answer, so once
    it is gone a worker parked on an approval waits the full permission
    timeout -- far longer than the EOF grace, so the process always exited and
    killed the run it was trying to let finish."""
    server = ACPServer(stdin=io.BytesIO(), stdout=io.BytesIO())
    answered: list[dict[str, Any]] = []
    asking = threading.Thread(
        target=lambda: answered.append(
            server.request("session/request_permission", {}, timeout_s=30.0)
        ),
        daemon=True,
    )
    asking.start()
    for _ in range(100):  # let it register its pending slot
        if server._pending:  # pyright: ignore[reportPrivateUsage]
            break
        threading.Event().wait(0.01)
    server.abandon_pending()
    asking.join(timeout=5.0)
    assert answered == [{}], "the worker was left waiting"


def test_a_question_with_no_buttons_is_not_put_to_the_editor() -> None:
    """ACP v1 carries a question as a permission request, whose options ARE
    the buttons. A free-form `ask_user` has none, so there was nothing to
    press: it stalled the full 300s timeout and then answered "said nothing"
    anyway -- up to eight times in one call, sequentially."""
    sent: list[tuple[str, dict[str, Any]]] = []
    bridge = _bridge({"outcome": {"outcome": "cancelled"}})

    def _record(method: str, params: dict[str, Any], **_kw: object) -> dict[str, Any]:
        sent.append((method, params))
        return {}

    bridge.server.request = _record  # pyright: ignore[reportAttributeAccessIssue]
    session = session_mod.Session(id="s", cwd=Path("/x"))
    assert bridge.ask(session, "What should the theme be?", (), None) is None
    assert sent == [], "an unanswerable prompt was still shown"
    assert bridge.ask(session, "Theme?", ("dark", "light"), None) is None
    assert len(sent) == 1, "a question WITH options still goes out"


def test_an_approval_closes_the_tool_call_it_announced() -> None:
    """`toolCall` is required on a permission request, so an ask announces one.
    ACP models a tool call as an entity with a lifecycle, so an editor kept one
    PENDING entry per approval for the life of the session."""
    sent: list[dict[str, Any]] = []
    bridge = _bridge({"outcome": {"outcome": "selected", "optionId": "allow"}})
    bridge.server.notify_raw = sent.append  # pyright: ignore[reportAttributeAccessIssue]
    session = session_mod.Session(id="s", cwd=Path("/x"))
    assert bridge.ask(session, "Allow run_command: ls", ("allow", "deny"), True) == "allow"
    closes = [m for m in sent if m["params"]["update"]["sessionUpdate"] == "tool_call_update"]
    assert len(closes) == 1 and closes[0]["params"]["update"]["status"] == "completed"


def test_a_malformed_frame_cannot_answer_an_outstanding_approval() -> None:
    """`_deliver` keyed on "has an id and no method", so any junk carrying an
    outstanding id became that approval's answer -- and an unreadable answer
    denies, so a stray frame could silently refuse a command."""
    server = ACPServer(stdin=io.BytesIO(), stdout=io.BytesIO())
    answered: list[dict[str, Any]] = []
    asking = threading.Thread(
        target=lambda: answered.append(
            server.request("session/request_permission", {}, timeout_s=3)
        ),
        daemon=True,
    )
    asking.start()
    for _ in range(100):
        if server._pending:  # pyright: ignore[reportPrivateUsage]
            break
        threading.Event().wait(0.01)
    req_id = next(iter(server._pending))  # pyright: ignore[reportPrivateUsage]
    assert server._deliver(req_id, {"id": req_id}) is False  # pyright: ignore[reportPrivateUsage]
    assert server._pending, "the slot was consumed by a non-response"  # pyright: ignore[reportPrivateUsage]
    server.abandon_pending()
    asking.join(timeout=5.0)
