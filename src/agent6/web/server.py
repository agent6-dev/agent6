# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `agent6 web` server: a stdlib HTTP front-end over the shared read-side.

One self-contained page (web.page) rendered by a browser, fed by:
  - plain GET JSON endpoints (the same wire form as `agent6 watch --json`), and
  - SSE (`text/event-stream`) streams that re-fold logs.jsonl / the machine
    journal on each change and push a fresh snapshot.

Zero dependencies: `http.server.ThreadingHTTPServer` only, no framework, no build
step. Binds loopback by default; a non-loopback bind is opt-in (see the `[web]`
config section) and widens the inbound network surface. The server only ever
renders folded read-state and (in the write phase) drives the typed
`agent6.frontend` contracts; it never serves secrets and never executes
arbitrary input.
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, ConfigDict, ValidationError

from agent6 import __version__
from agent6.frontend.approval import TUI_PID_FILE, clear_tui_pid, write_tui_pid
from agent6.machine import MachineError
from agent6.viewmodel import apply_event, initial_state, run_state_as_dict, tail_events
from agent6.web import actions, model
from agent6.web.page import PAGE_HTML

# SSE tuning: coalesce high-frequency streaming deltas, heartbeat idle streams so
# a disconnected client is noticed and its worker thread exits.
_DELTA_COALESCE_S = 0.15
_HEARTBEAT_S = 15.0
_MACHINE_POLL_S = 0.5
_STREAMING_DELTAS = frozenset({"role.text_delta", "role.thinking_delta"})


# Typed POST bodies (pydantic only at this HTTP trust boundary; extra keys are
# rejected so a malformed request fails loudly rather than silently ignoring a
# misspelled field).
class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NewWorkBody(_Body):
    mode: str
    task: str
    profile: str = ""


class SteerBody(_Body):
    text: str = ""


class ApproveBody(_Body):
    id: str
    approved: bool


class AnswerBody(_Body):
    id: str
    answer: str


class MergeBody(_Body):
    strategy: str = ""


class MachineCreateBody(_Body):
    task: str


class MachineRunBody(_Body):
    file: str


class ConfigSetBody(_Body):
    key: str
    value: str
    repo: bool = False


class WebServer(ThreadingHTTPServer):
    """A ThreadingHTTPServer that carries the repo cwd its handlers read from,
    and tracks which runs a browser is actively watching so it can register this
    process as the answering front-end (tui.pid) only while someone is looking."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr: tuple[str, int], cwd: Path, target: str) -> None:
        super().__init__(addr, _Handler)
        self.cwd = cwd
        self.target = target
        self._pid_lock = threading.Lock()
        self._watch_counts: dict[str, int] = {}

    def claim_run(self, run_dir: Path) -> None:
        """A browser opened this run's stream: become the run's answer front-end
        (write our pid to tui.pid) so its approval/question/steer prompts bridge
        here. Reference-counted across concurrent viewers."""
        key = str(run_dir)
        with self._pid_lock:
            n = self._watch_counts.get(key, 0) + 1
            self._watch_counts[key] = n
            if n == 1:
                write_tui_pid(run_dir, os.getpid())

    def release_run(self, run_dir: Path) -> None:
        """The last browser watching this run went away: stop claiming its prompts
        (clear tui.pid, but only if it still points at us) so the run falls back to
        its headless behaviour instead of blocking on answers no one will give."""
        key = str(run_dir)
        with self._pid_lock:
            n = self._watch_counts.get(key, 1) - 1
            if n > 0:
                self._watch_counts[key] = n
                return
            self._watch_counts.pop(key, None)
        try:
            owned = (run_dir / TUI_PID_FILE).read_text(encoding="utf-8").strip() == str(os.getpid())
        except OSError:
            owned = False
        if owned:
            clear_tui_pid(run_dir)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: WebServer  # type: ignore[assignment]

    def log_message(self, format: str, *args: Any) -> None:  # match the stdlib signature
        pass  # quiet; we are not a logging server

    @property
    def cwd(self) -> Path:
        return self.server.cwd

    # -- routing --------------------------------------------------------------

    def do_GET(self) -> None:  # BaseHTTPRequestHandler dispatch contract (method name fixed)
        path = unquote(urlsplit(self.path).path)
        try:
            self._route(path)
        except BrokenPipeError:
            pass  # client went away mid-response
        except Exception as exc:  # never take the whole server down for one bad request
            self._send_json({"error": str(exc)}, status=500)

    def do_POST(self) -> None:  # BaseHTTPRequestHandler dispatch contract (method name fixed)
        path = unquote(urlsplit(self.path).path)
        try:
            self._route_post(path)
        except BrokenPipeError:
            pass
        except ValidationError as exc:
            self._send_json({"error": f"bad request: {exc.errors()}"}, status=400)
        except Exception as exc:  # never take the whole server down for one bad request
            self._send_json({"error": str(exc)}, status=500)

    def _read_body(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(n) if n else b""
        if not raw:
            return {}
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("request body must be a JSON object")
        return obj

    def _route_post(self, path: str) -> None:
        parts = path.strip("/").split("/")
        # /api/new  /api/runs/prune  /api/config  /api/machine/create  /api/machine/run
        if path == "/api/new":
            body = NewWorkBody.model_validate(self._read_body())
            run_id, err = actions.spawn_new_work(self.cwd, body.mode, body.task, body.profile)
            self._ok_or_err(run_id is not None, {"run_id": run_id}, err)
            return
        if path == "/api/runs/prune":
            ok, msg = actions.prune_runs(self.cwd)
            self._ok_or_err(ok, {"message": msg}, msg)
            return
        if path == "/api/config":
            body = ConfigSetBody.model_validate(self._read_body())
            ok, msg = actions.set_config(self.cwd, body.key, body.value, repo=body.repo)
            self._ok_or_err(ok, {"message": msg}, msg)
            return
        if path == "/api/machine/create":
            body = MachineCreateBody.model_validate(self._read_body())
            draft, err = actions.spawn_machine_create(self.cwd, body.task)
            self._ok_or_err(draft is not None, {"draft": draft}, err)
            return
        if path == "/api/machine/run":
            body = MachineRunBody.model_validate(self._read_body())
            ok, msg = actions.spawn_machine_run(self.cwd, body.file)
            self._ok_or_err(ok, {"message": msg}, msg)
            return
        # /api/run/<id>/<verb>
        if len(parts) == 4 and parts[0] == "api" and parts[1] == "run":
            self._route_run_post(parts[2], parts[3])
            return
        self._send_json({"error": f"not found: {path}"}, status=404)

    def _route_run_post(self, run_id: str, verb: str) -> None:
        if verb == "steer":
            body = SteerBody.model_validate(self._read_body())
            ok, msg = actions.steer(self.cwd, run_id, body.text)
        elif verb == "approve":
            ab = ApproveBody.model_validate(self._read_body())
            ok, msg = actions.approve(self.cwd, run_id, ab.id, ab.approved)
        elif verb == "answer":
            qb = AnswerBody.model_validate(self._read_body())
            ok, msg = actions.answer_question(self.cwd, run_id, qb.id, qb.answer)
        elif verb == "merge":
            mb = MergeBody.model_validate(self._read_body())
            ok, msg = actions.merge_run(self.cwd, run_id, mb.strategy)
        else:
            self._send_json({"error": f"not found: run/{run_id}/{verb}"}, status=404)
            return
        self._ok_or_err(ok, {"message": msg}, msg)

    def _ok_or_err(self, ok: bool, payload: dict[str, Any], err: str) -> None:
        if ok:
            self._send_json({"ok": True, **payload})
        else:
            self._send_json({"ok": False, "error": err}, status=422)

    def _route(self, path: str) -> None:
        if path == "/":
            self._send_bytes(PAGE_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/meta":
            self._send_json({"version": __version__, "target": self.server.target})
            return
        if path == "/api/hub":
            self._send_json(model.hub_payload(self.cwd))
            return
        if path == "/api/config":
            self._send_json(model.config_payload(self.cwd))
            return
        parts = path.strip("/").split("/")
        # /api/run/<id>[/transcript|/events]
        if len(parts) >= 3 and parts[0] == "api" and parts[1] == "run":
            self._route_run(parts[2], parts[3] if len(parts) > 3 else "")
            return
        # /api/machine/<name>[/reasoning|/events]
        if len(parts) >= 3 and parts[0] == "api" and parts[1] == "machine":
            self._route_machine(parts[2], parts[3] if len(parts) > 3 else "")
            return
        self._send_json({"error": f"not found: {path}"}, status=404)

    def _route_run(self, run_id: str, sub: str) -> None:
        run_dir = model.run_dir_for(self.cwd, run_id)
        if run_dir is None:
            self._send_json({"error": f"no run {run_id!r}"}, status=404)
            return
        if sub == "":
            self._send_json(model.run_snapshot(run_dir))
        elif sub == "transcript":
            self._send_json(model.transcript_payload(run_dir))
        elif sub == "events":
            self._sse_run(run_dir)
        else:
            self._send_json({"error": f"not found: run/{run_id}/{sub}"}, status=404)

    def _route_machine(self, name: str, sub: str) -> None:
        machine_dir = model.machine_dir_for(self.cwd, name)
        if machine_dir is None:
            self._send_json({"error": f"no machine {name!r}"}, status=404)
            return
        try:
            if sub == "":
                self._send_json(model.machine_snapshot(machine_dir))
            elif sub == "reasoning":
                self._send_json(model.machine_reasoning_snapshot(machine_dir))
            elif sub == "events":
                self._sse_machine(machine_dir)
            else:
                self._send_json({"error": f"not found: machine/{name}/{sub}"}, status=404)
        except MachineError as exc:
            self._send_json({"error": f"machine {name!r}: {'; '.join(exc.problems)}"}, status=422)

    # -- plain responses ------------------------------------------------------

    def _send_json(self, payload: Any, *, status: int = 200) -> None:
        self._send_bytes(
            json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8", status=status
        )

    def _send_bytes(self, body: bytes, ctype: str, *, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # -- SSE ------------------------------------------------------------------

    def _begin_sse(self) -> None:
        # Close-framed, not keep-alive: an SSE body has no Content-Length, so the
        # socket closing is what tells the client the stream ended (and lets a
        # finished run's EventSource stop). close_connection makes the handler
        # close the socket when the stream loop returns.
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")  # tell any proxy not to buffer SSE
        self.end_headers()

    def _sse_send(self, obj: Any) -> bool:
        """Write one SSE data frame. Returns False if the client has gone away."""
        try:
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            self.wfile.flush()
        except OSError:
            return False
        return True

    def _sse_ping(self) -> bool:
        try:
            self.wfile.write(b": ping\n\n")
            self.wfile.flush()
        except OSError:
            return False
        return True

    def _sse_run(self, run_dir: Path) -> None:
        """Stream a run: fold logs.jsonl incrementally, push a fresh RunState
        snapshot on each event (coalescing streaming deltas). A background tailer
        feeds a queue so the response loop can heartbeat idle periods and exit
        promptly when the client disconnects. While connected we register as the
        run's answer front-end so its approval/steer prompts bridge to the browser."""
        self._begin_sse()
        self.server.claim_run(run_dir)
        try:
            self._sse_run_loop(run_dir)
        finally:
            self.server.release_run(run_dir)

    def _sse_run_loop(self, run_dir: Path) -> None:
        events: queue.Queue[dict[str, Any] | None] = queue.Queue()

        def tail() -> None:
            for ev in tail_events(run_dir / "logs.jsonl", follow=True, stop_when_finished=True):
                events.put(ev)
            events.put(None)  # sentinel: run ended, tailer done

        threading.Thread(target=tail, daemon=True).start()

        state = initial_state()
        last_delta_emit = 0.0
        while True:
            try:
                ev = events.get(timeout=_HEARTBEAT_S)
            except queue.Empty:
                if not self._sse_ping():
                    return
                continue
            if ev is None:  # run ended: send the final snapshot and close
                self._sse_send(run_state_as_dict(state))
                return
            state = apply_event(state, ev)
            now = time.monotonic()
            if ev.get("type") in _STREAMING_DELTAS and (now - last_delta_emit) < _DELTA_COALESCE_S:
                continue  # coalesce bursts of text/thinking deltas
            if not self._sse_send(run_state_as_dict(state)):
                return
            last_delta_emit = now

    def _sse_machine(self, machine_dir: Path) -> None:
        """Stream a machine: re-fold the journal + the current agent state's
        reasoning on a poll, pushing the combined snapshot when it changes."""
        self._begin_sse()
        prev = ""
        idle = 0.0
        while True:
            try:
                payload = {
                    "machine": model.machine_snapshot(machine_dir),
                    "reasoning": model.machine_reasoning_snapshot(machine_dir),
                }
            except MachineError as exc:
                self._sse_send({"error": "; ".join(exc.problems)})
                return
            blob = json.dumps(payload, sort_keys=True)
            if blob != prev:
                if not self._sse_send(payload):
                    return
                prev = blob
                idle = 0.0
            else:
                idle += _MACHINE_POLL_S
                if idle >= _HEARTBEAT_S and not self._sse_ping():
                    return
                if idle >= _HEARTBEAT_S:
                    idle = 0.0
            time.sleep(_MACHINE_POLL_S)


def run_web(target: str, *, host: str, port: int, cwd: Path | None = None) -> int:
    """Serve the web UI on host:port until interrupted. `target` deep-links the
    page to a run id or machine name on load (empty opens the hub)."""
    workdir = cwd or Path.cwd()
    try:
        server = WebServer((host, port), workdir, target)
    except OSError as exc:
        print(f"agent6 web: cannot bind {host}:{port}: {exc}", file=sys.stderr)
        return 2
    wildcard = {"0.0.0.0", "::"}  # noqa: S104 - literals for display only; the bind host is operator-chosen
    shown = "127.0.0.1" if host in wildcard else host
    print(f"agent6 web: serving on http://{shown}:{port}  (Ctrl-C to stop)", file=sys.stderr)
    if host not in {"127.0.0.1", "localhost", "::1"}:
        print(
            "agent6 web: WARNING bound to a non-loopback address; anyone who can reach"
            f" {host}:{port} can drive this agent. Prefer `tailscale serve` in front of a"
            " loopback bind.",
            file=sys.stderr,
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nagent6 web: stopped", file=sys.stderr)
    finally:
        server.server_close()
    return 0
