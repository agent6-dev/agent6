# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Drive `agent6 acp` over stdio as an editor of each protocol version would.

A scripted client talks to the installed console script, and a local scripted endpoint
stands in for the provider. Every frame is checked against that version's schema.
"""

from __future__ import annotations

import collections
import contextlib
import http.server
import json
import os
import pathlib
import select
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from typing import Any, override

import pytest

from tests.acp import schema_check

Frame = dict[str, Any]
_WAIT_S = 60.0


def _chunk(name: str, arguments: dict[str, Any]) -> Frame:
    """Return one streamed model reply that calls a tool."""
    call: Frame = {"index": 0, "id": f"call-{name}", "type": "function"}
    call["function"] = {"name": name, "arguments": json.dumps(arguments)}
    choice = {"index": 0, "delta": {"tool_calls": [call]}, "finish_reason": "tool_calls"}
    usage = {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
    return {"choices": [choice], "usage": usage}


FINISH = _chunk("finish_session", {"summary": "The recorded model summary."})
LIST = _chunk("list_dir", {"path": "."})


class _Provider(http.server.ThreadingHTTPServer):
    """Answer each model call with the script's next step.

    Attributes:
        script: The replies, in order: a chunk, or an HTTP status to fail with.
        hold: While it is clear, a call waits here after its first words.
        asked: Set each time a call has streamed its first words.
        requests: The body of every call, in order.
    """

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Call)
        self.script: collections.deque[Frame | int] = collections.deque()
        self.hold = threading.Event()
        self.hold.set()
        self.asked = threading.Event()
        self.requests: list[str] = []


class _Call(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        provider = self.server
        assert isinstance(provider, _Provider)
        provider.requests.append(self.rfile.read(int(self.headers["Content-Length"])).decode())
        step = provider.script.popleft() if provider.script else FINISH
        if isinstance(step, int):
            body = json.dumps({"error": {"message": "the scripted provider refused"}}).encode()
            self.send_response(step)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        # The body ends when the connection closes, so a held call is a stream in flight.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        with contextlib.suppress(OSError):  # a cancelled call's reader is gone
            self._say({"choices": [{"index": 0, "delta": {"content": "Thinking. "}}]})
            provider.asked.set()
            while not provider.hold.wait(0.2):
                # A live stream keeps sending: the agent notices a cancel at the next byte.
                self.wfile.write(b": waiting\n\n")
                self.wfile.flush()
            self._say(step)
            self.wfile.write(b"data: [DONE]\n\n")

    def _say(self, chunk: Frame) -> None:
        self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
        self.wfile.flush()

    @override
    def log_message(self, *args: object, **kwargs: object) -> None:
        pass


class _Editor:
    """One connection to `agent6 acp`, speaking one protocol version."""

    def __init__(self, exe: pathlib.Path, config: pathlib.Path, repo: pathlib.Path, version: int):
        self.version = version
        self.stderr = repo.parent / f"stderr-{version}"
        root = ["--allow-root"] if os.geteuid() == 0 else []
        self._log = self.stderr.open("ab")
        self._proc = subprocess.Popen(
            [str(exe), *root, "--config", str(config), "acp"],
            cwd=repo,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            bufsize=0,
        )
        self._next = 0
        self.frames: list[Frame] = []

    def send(self, method: str, params: Frame) -> int:
        """Send one request and return its id."""
        self._next += 1
        self._write({"id": self._next, "method": method, "params": params})
        return self._next

    def notify(self, method: str, params: Frame) -> None:
        self._write({"method": method, "params": params})

    def _write(self, message: Frame) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(json.dumps({"jsonrpc": "2.0", **message}).encode() + b"\n")

    def until(self, done: Callable[[Frame], bool]) -> list[Frame]:
        """Read frames until one satisfies `done`; return those read, each schema-checked."""
        assert self._proc.stdout is not None
        got: list[Frame] = []
        while True:
            ready = select.select([self._proc.stdout], [], [], _WAIT_S)[0]
            assert ready, f"no frame arrived; got {got}\n{self.stderr.read_text()}"
            line = self._proc.stdout.readline()
            assert line, f"the agent closed its output; got {got}\n{self.stderr.read_text()}"
            frame = json.loads(line)
            self._check(frame)
            got.append(frame)
            self.frames.append(frame)
            if done(frame):
                return got

    def reply_to(self, req_id: int) -> Frame:
        """Read until the reply to a request and return it."""
        return self.until(lambda frame: frame.get("id") == req_id)[-1]

    def _check(self, frame: Frame) -> None:
        if frame.get("method") != "session/update":
            return
        found = (
            schema_check.errors("SessionNotification", frame["params"])
            if self.version == 1
            else schema_check.v2_errors("UpdateSessionNotification", frame["params"])
        )
        assert not found, f"{found}\n{json.dumps(frame)}"

    def close(self) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.close()
        try:
            self._proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        self._log.close()


@pytest.fixture
def scripted(tmp_path: pathlib.Path) -> Iterator[tuple[_Provider, Callable[[int], _Editor]]]:
    """Yield the scripted provider and a factory of editors on one repository."""
    exe = pathlib.Path(sys.executable).with_name("agent6")
    if not exe.is_file():
        pytest.skip("agent6 console script is not installed beside the test interpreter")
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    identity = ["-c", "user.name=t", "-c", "user.email=t@t"]
    seed = ["commit", "-q", "--allow-empty", "-m", "seed"]
    subprocess.run(["git", "-C", str(repo), *identity, *seed], check=True)
    provider = _Provider()
    serving = threading.Thread(target=provider.serve_forever, daemon=True)
    serving.start()
    config = tmp_path / "config.toml"
    config.write_text(
        '[providers.recorded]\nbackend = "openai"\nauth_style = "none"\n'
        f'base_url = "http://127.0.0.1:{provider.server_port}/v1"\n'
        '[models.default]\nmodel = "recorded/recorded"\n'
        '[sandbox]\nisolation = "none"\nrun_commands = "deny"\n',
        encoding="utf-8",
    )
    editors: list[_Editor] = []

    def connect(version: int) -> _Editor:
        editors.append(_Editor(exe, config, repo, version))
        return editors[-1]

    try:
        yield provider, connect
    finally:
        provider.hold.set()
        for editor in editors:
            editor.close()
        provider.shutdown()
        provider.server_close()
        serving.join(timeout=5)


def _text(text: str) -> list[Frame]:
    return [{"type": "text", "text": text}]


def _updates(frames: list[Frame], kind: str) -> list[Frame]:
    """Return the bodies of the `session/update` frames of one kind."""
    bodies = [f["params"]["update"] for f in frames if f.get("method") == "session/update"]
    return [body for body in bodies if body["sessionUpdate"] == kind]


def steered_id(frames: list[Frame], req_id: int) -> str:
    """Return the message id a prompt's reply carried."""
    return str(next(f for f in frames if f.get("id") == req_id)["result"]["messageId"])


def _idle(frame: Frame) -> bool:
    update = frame.get("params", {}).get("update", {})
    return update.get("sessionUpdate") == "state_update" and update.get("state") == "idle"


def _open(editor: _Editor, repo: pathlib.Path) -> str:
    """Initialize the connection and return a new session's id."""
    info = {"name": "scripted", "version": "0"}
    shaped = (
        {"clientCapabilities": {}} if editor.version == 1 else {"info": info, "capabilities": {}}
    )
    hello = editor.reply_to(
        editor.send("initialize", {"protocolVersion": editor.version, **shaped})
    )
    assert hello["result"]["protocolVersion"] == editor.version
    created = editor.reply_to(editor.send("session/new", {"cwd": str(repo)}))["result"]
    if editor.version == 2:
        assert not schema_check.v2_errors("NewSessionResponse", created)
        assert {"pin", "receipt"} <= {c["name"] for c in created["availableCommands"]}
    return str(created["sessionId"])


def test_a_v1_editor_drives_a_run_to_each_of_its_ends(
    scripted: tuple[_Provider, Callable[[int], _Editor]], tmp_path: pathlib.Path
) -> None:
    """v1: the prompt's reply is the turn, a failure is its error, `session/load` replays."""
    provider, connect = scripted
    repo = tmp_path / "repo"
    editor = connect(1)
    session = _open(editor, repo)
    prompt = {"sessionId": session, "prompt": _text("Finish.")}

    first = editor.send("session/prompt", prompt)
    ended = editor.until(lambda f: f.get("id") == first)
    assert ended[-1]["result"] == {"stopReason": "end_turn"}
    assert not _updates(ended, "user_message_chunk"), "v1 never echoes the editor's own prompt"

    provider.hold.clear()
    provider.asked.clear()
    held = editor.send("session/prompt", {"sessionId": session, "prompt": _text("Wait here.")})
    assert provider.asked.wait(_WAIT_S)
    busy = editor.reply_to(editor.send("session/prompt", prompt))
    assert busy["error"]["code"] == -32602, "a v1 session refuses a prompt during a run"
    editor.notify("session/cancel", {"sessionId": session})
    assert editor.reply_to(held)["result"] == {"stopReason": "cancelled"}
    provider.hold.set()

    provider.script.append(400)
    failed = editor.reply_to(editor.send("session/prompt", prompt))
    assert failed["error"]["code"] == -32603, failed
    assert "refusal" not in json.dumps(failed)

    again = connect(1)
    again.reply_to(again.send("initialize", {"protocolVersion": 1, "clientCapabilities": {}}))
    load = again.send("session/load", {"sessionId": session, "cwd": str(repo), "mcpServers": []})
    replayed = again.until(lambda f: f.get("id") == load)
    assert "configOptions" in replayed[-1]["result"]
    said = [u["content"]["text"] for u in _updates(replayed, "user_message_chunk")]
    assert said[:2] == ["Finish.", "Wait here."]
    for frames in (replayed, editor.frames):
        words = "".join(u["content"]["text"] for u in _updates(frames, "agent_message_chunk"))
        assert words.count("The recorded model summary.") == 1, words
        assert words.count("model call was cut short") == 1, words
        assert words.count("(HTTP 400)") == 1, words


def test_a_v2_editor_drives_a_run_to_each_of_its_ends(
    scripted: tuple[_Provider, Callable[[int], _Editor]], tmp_path: pathlib.Path
) -> None:
    """v2: the reply names the added message, the state carries each end, a run takes a steer."""
    provider, connect = scripted
    repo = tmp_path / "repo"
    editor = connect(2)
    session = _open(editor, repo)
    prompt = {"sessionId": session, "prompt": _text("Finish.")}
    keyed = prompt | {"idempotencyKey": "key-of-the-first-prompt"}

    first = editor.send("session/prompt", keyed)
    turn = editor.until(_idle)
    (reply,) = [f for f in turn if f.get("id") == first]
    assert not schema_check.v2_errors("PromptResponse", reply["result"])
    states = [(u["state"], u.get("stopReason")) for u in _updates(turn, "state_update")]
    assert states == [("running", None), ("idle", "end_turn")]
    (echo,) = _updates(turn, "user_message")
    assert echo["messageId"] == reply["result"]["messageId"]
    assert echo["content"] == _text("Finish.")
    assert _updates(turn, "agent_message"), "the run's ending never reached the editor"

    provider.script.extend([LIST, FINISH])
    provider.hold.clear()
    provider.asked.clear()
    long = editor.send("session/prompt", {"sessionId": session, "prompt": _text("Look around.")})
    editor.reply_to(long)
    assert provider.asked.wait(_WAIT_S)
    hi = {"sessionId": session, "prompt": _text("Also say hi."), "idempotencyKey": "key-of-hi"}
    steer = editor.send("session/prompt", hi)
    steered = editor.until(lambda f: f.get("id") == steer)
    retry = editor.send("session/prompt", hi)
    steered += editor.until(lambda f: f.get("id") == retry)
    assert steered_id(steered, retry) == steered_id(steered, steer)
    provider.hold.set()
    steered += editor.until(_idle)
    assert [u["state"] for u in _updates(steered, "state_update")] == ["idle"]
    echoes = [u for u in _updates(steered, "user_message") if u["content"] == _text("Also say hi.")]
    assert [u["messageId"] for u in echoes] == [steered_id(steered, steer)]
    assert "Also say hi." in provider.requests[-1], "the run never took the steer"
    assert "Also say hi." not in provider.requests[-2]

    provider.hold.clear()
    provider.asked.clear()
    held = editor.send("session/prompt", {"sessionId": session, "prompt": _text("Wait here.")})
    started = editor.until(lambda f: f.get("id") == held)
    assert "messageId" in started[-1]["result"]
    assert provider.asked.wait(_WAIT_S)
    editor.notify("session/cancel", {"sessionId": session})
    cancelled = editor.until(_idle)
    assert _updates(cancelled, "state_update")[-1]["stopReason"] == "cancelled"
    provider.hold.set()

    provider.script.append(400)
    failing = editor.send("session/prompt", prompt)
    broken = editor.until(_idle)
    assert "messageId" in next(f for f in broken if f.get("id") == failing)["result"]
    ending = _updates(broken, "state_update")[-1]
    assert ending["stopReason"] == "error"
    assert ending["error"]["code"] == -32603, ending

    again = connect(2)
    info = {"name": "scripted", "version": "0"}
    again.reply_to(again.send("initialize", {"protocolVersion": 2, "info": info}))
    cursor = {"sessionId": session, "cwd": str(repo), "replayFrom": {"type": "start"}}
    resume = again.send("session/resume", cursor)
    replayed = again.until(lambda f: f.get("id") == resume)
    assert not schema_check.v2_errors("ResumeSessionResponse", replayed[-1]["result"])
    kinds = [f["params"]["update"]["sessionUpdate"] for f in replayed[:-1]]
    assert kinds[-2:] == ["replay_complete", "state_update"]
    said = [(u["content"][0]["text"], u["messageId"]) for u in _updates(replayed, "user_message")]
    assert [text for text, _id in said[:4]] == [
        "Finish.",
        "Look around.",
        "Also say hi.",
        "Wait here.",
    ]
    assert said[0][1] == reply["result"]["messageId"], "a replayed message kept its id"
    assert said[2][1] == steered_id(steered, steer)

    asked = len(provider.requests)
    retried = again.send("session/prompt", keyed)
    listed = again.send("session/list", {"cwd": str(repo)})
    after = again.until(lambda f: f.get("id") == listed)
    assert [f.get("id") for f in after] == [retried, listed], "the retry started work"
    assert after[0]["result"] == {"messageId": reply["result"]["messageId"]}
    assert len(provider.requests) == asked


def test_a_waiting_steer_precedes_the_next_prompt_on_its_first_model_call(
    scripted: tuple[_Provider, Callable[[int], _Editor]], tmp_path: pathlib.Path
) -> None:
    """A run that ends on the call a steer arrived in leaves the steer queued for a later one."""
    provider, connect = scripted
    editor = connect(2)
    session = _open(editor, tmp_path / "repo")

    def prompt(text: str) -> Frame:
        return {"sessionId": session, "prompt": _text(text)}

    provider.hold.clear()
    provider.asked.clear()
    editor.reply_to(editor.send("session/prompt", prompt("Wrap up.")))
    assert provider.asked.wait(_WAIT_S)
    late = editor.send("session/prompt", prompt("One more thing."))
    first = editor.until(lambda f: f.get("id") == late)
    provider.hold.set()
    first += editor.until(_idle)
    assert len(provider.requests) == 1, "the run ended on the call the steer arrived in"

    provider.script.append(FINISH)
    editor.send("session/prompt", prompt("And then stop."))
    second = editor.until(_idle)
    assert len(provider.requests) == 2
    first_call = provider.requests[1]
    assert first_call.index("One more thing.") < first_call.index("And then stop.")
    said = [u["content"][0]["text"] for u in _updates(second, "user_message")]
    assert said == ["And then stop."], "the steer was reported when it was sent, not again"
    assert steered_id(first, late)
