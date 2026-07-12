# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `agent6 web` server: a stdlib HTTP front-end over the shared read-side.

Serves web.page to a browser, fed by:
  - plain GET JSON endpoints (the same wire form as `agent6 watch --json`), and
  - SSE (`text/event-stream`) streams that re-fold logs.jsonl / the machine
    journal on each change and push a fresh snapshot.

Uses the stdlib `http.server.ThreadingHTTPServer`. Binds loopback by default; a
non-loopback bind is opt-in (see the `[web]` config section) and widens the
inbound network surface. The server only ever renders folded read-state and (in
the write phase) drives the typed `agent6.ui.bridge` contracts; it never serves
secrets and never executes arbitrary input.
"""

from __future__ import annotations

import json
import os
import queue
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import ip_address
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, ConfigDict, ValidationError

from agent6 import __version__
from agent6.config import is_loopback_host
from agent6.machine import MachineError
from agent6.ui.bridge.approval import (
    FRONTEND_PID_FILE,
    clear_frontend_pid,
    read_worker_pid,
    worker_is_alive,
    write_frontend_pid,
)
from agent6.ui.viewmodel import apply_event, initial_state, run_state_as_dict, tail_events
from agent6.ui.web import actions, model
from agent6.ui.web.page import ICON_SVG, MANIFEST_JSON, PAGE_HTML, SERVICE_WORKER_JS

# SSE tuning: coalesce high-frequency streaming deltas, heartbeat idle streams so
# a disconnected client is noticed and its worker thread exits.
_DELTA_COALESCE_S = 0.15
_HEARTBEAT_S = 15.0
_MACHINE_POLL_S = 0.5
_STREAMING_DELTAS = frozenset({"role.text_delta", "role.thinking_delta"})

# POST body cap. The typed bodies are a few strings (a task, an answer, a config
# value); 1 MiB is generous. An uncapped Content-Length would let one request
# buffer arbitrary bytes in this process.
_MAX_BODY_BYTES = 1 << 20


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
    # For a machine: the per-state dir name the prompt was rendered from, so the
    # answer routes to that state even if the machine has since advanced. Empty
    # (the default, and always for a run) routes to the newest state.
    state: str = ""


class ApproveBody(_Body):
    id: str
    approved: bool
    session: bool = False  # "allow session": approve every later run_command too
    state: str = ""


class AnswerBody(_Body):
    id: str
    answers: list[str]  # one per question in the ask_user prompt, by index
    state: str = ""


class MergeBody(_Body):
    strategy: str = ""


class MachineCreateBody(_Body):
    task: str


class MachineRunBody(_Body):
    file: str


class MachinePokeBody(_Body):
    # A JSON `data` payload wins over a `message` string; neither = a bare wake.
    message: str = ""
    data: Any = None


class ConfigSetBody(_Body):
    key: str
    value: str
    repo: bool = False


class WebServer(ThreadingHTTPServer):
    """A ThreadingHTTPServer that carries the repo cwd its handlers read from,
    and tracks which runs a browser is actively watching so it can register this
    process as the answering front-end (frontend.pid) only while someone is looking."""

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
        (write our pid to frontend.pid) so its approval/question/steer prompts
        bridge here. Reference-counted across concurrent viewers."""
        key = str(run_dir)
        with self._pid_lock:
            n = self._watch_counts.get(key, 0) + 1
            self._watch_counts[key] = n
            if n == 1:
                write_frontend_pid(run_dir, os.getpid())

    def release_run(self, run_dir: Path) -> None:
        """The last browser watching this run went away: stop claiming its prompts
        (clear frontend.pid, but only if it still points at us) so the run falls
        back to its headless behaviour instead of blocking on answers no one gives."""
        key = str(run_dir)
        with self._pid_lock:
            n = self._watch_counts.get(key, 1) - 1
            if n > 0:
                self._watch_counts[key] = n
                return
            self._watch_counts.pop(key, None)
            # Read + clear under the same lock: otherwise a concurrent claim_run
            # could re-write frontend.pid (to our own pid) between the count
            # hitting 0 and the clear, and the owned-check would then wrongly
            # unbridge a viewer that just started streaming.
            try:
                owned = (run_dir / FRONTEND_PID_FILE).read_text(encoding="utf-8").strip() == str(
                    os.getpid()
                )
            except OSError:
                owned = False
            if owned:
                clear_frontend_pid(run_dir)


class _IPv6WebServer(WebServer):
    address_family = socket.AF_INET6


def _bind_host(host: str) -> str:
    """Normalize URL-style bracketed IPv6 literals to socket bind addresses."""
    stripped = host.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        return stripped[1:-1]
    return stripped


def _is_ipv6_literal(host: str) -> bool:
    try:
        return ip_address(_bind_host(host)).version == 6
    except ValueError:
        return False


def _display_host(host: str) -> str:
    if host == "0.0.0.0":  # noqa: S104 - display only
        return "127.0.0.1"
    if host == "::":
        return "[::1]"
    return f"[{host}]" if _is_ipv6_literal(host) else host


def _create_web_server(host: str, port: int, cwd: Path, target: str) -> WebServer:
    bind_host = _bind_host(host)
    server_cls: type[WebServer] = _IPv6WebServer if _is_ipv6_literal(bind_host) else WebServer
    return server_cls((bind_host, port), cwd, target)


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
            csrf_err = self._csrf_refusal()
            if csrf_err is not None:
                # Close the connection rather than drain an unread body under
                # HTTP/1.1 keep-alive (a partial read would desync framing).
                self.close_connection = True
                self._send_json({"error": csrf_err}, status=403)
                return
            if self.headers.get("Transfer-Encoding"):
                # Only Content-Length bodies are read; a chunked body would sit
                # unread on the connection like the early-error cases below.
                self.close_connection = True
                self._send_json({"error": "chunked bodies are not supported"}, status=411)
                return
            content_length = int(self.headers.get("Content-Length", "0") or "0")
            if content_length < 0:
                # A negative length would make rfile.read(n) read to EOF, buffering
                # arbitrary bytes and parking the worker thread; the > cap check
                # alone (n < cap) lets it through.
                self.close_connection = True
                self._send_json({"error": "negative Content-Length"}, status=400)
                return
            if content_length > _MAX_BODY_BYTES:
                self.close_connection = True
                self._send_json({"error": f"body larger than {_MAX_BODY_BYTES} bytes"}, status=413)
                return
            self._route_post(path)
        except BrokenPipeError:
            pass
        except ValidationError as exc:
            # The body was read (validation runs on the parsed body), so the
            # connection framing is intact and may stay open.
            self._send_json({"error": f"bad request: {exc.errors()}"}, status=400)
        except Exception as exc:  # never take the whole server down for one bad request
            # The body may not have been read; a keep-alive reuse would parse the
            # leftover bytes as the next request line. Close instead.
            self.close_connection = True
            self._send_json({"error": str(exc)}, status=500)

    def _csrf_refusal(self) -> str | None:
        """Reason to refuse this state-changing POST as cross-site, or None.

        The web UI has no app-level auth: on the default loopback bind the OS
        user is the trust boundary, behind `tailscale serve` the tailnet
        identity is. Neither stops a page on ANOTHER origin in the operator's
        browser from POSTing here (classic CSRF). Two standard,
        deployment-agnostic checks close it:

        - Require `Content-Type: application/json` for a body. A cross-site
          `fetch` with that type is not a CORS "simple request", so the
          browser sends a preflight we never answer and the POST is blocked.
          This shuts the hole where a JSON body rides in as `text/plain`.
        - If an `Origin` is present, its host:port must equal `Host`. Our own
          page matches; a cross-site page (Origin: https://evil.example) does
          not. A missing Origin (curl, the CLI) is allowed -- not
          browser-driven, so not a CSRF vector.

        Residual: DNS rebinding (an attacker page rebinds its own hostname to
        127.0.0.1 so its request is same-origin) is not covered here; a Host
        allow-list would break the tailnet-hostname `tailscale serve` path, so
        that vector is left to the network layer."""
        n = int(self.headers.get("Content-Length", "0") or "0")
        if n > 0:
            ctype = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
            if ctype != "application/json":
                return f"POST body must be Content-Type: application/json, not {ctype!r}"
        origin = self.headers.get("Origin")
        if origin:
            host = self.headers.get("Host", "")
            if urlsplit(origin).netloc != host:
                return f"cross-origin POST refused (Origin {origin!r} != Host {host!r})"
        return None

    def _read_body(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(n) if n > 0 else b""
        if not raw:
            return {}
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("request body must be a JSON object")
        return obj

    def _route_post(self, path: str) -> None:  # noqa: PLR0911
        parts = path.strip("/").split("/")
        # /api/new  /api/runs/prune  /api/config  /api/machine/create  /api/machine/run
        if path == "/api/new":
            body = NewWorkBody.model_validate(self._read_body())
            run_id, err = actions.spawn_new_work(self.cwd, body.mode, body.task, body.profile)
            self._ok_or_err(run_id is not None, {"run_id": run_id}, err)
            return
        if path == "/api/runs/prune":
            # Drain the body (the client posts `{}`) even though prune takes no
            # params: an unread body would sit on the keep-alive socket and the
            # next request line would be parsed with it prepended.
            self._read_body()
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
        # /api/machine/<name>/<verb>
        if len(parts) == 4 and parts[0] == "api" and parts[1] == "machine":
            self._route_machine_post(parts[2], parts[3])
            return
        self._post_not_found(path)

    def _post_not_found(self, what: str) -> None:
        """404 for a POST whose body was never read: close the connection so the
        unread body cannot be parsed as the next request on keep-alive."""
        self.close_connection = True
        self._send_json({"error": f"not found: {what}"}, status=404)

    def _route_run_post(self, run_id: str, verb: str) -> None:
        if verb == "steer":
            body = SteerBody.model_validate(self._read_body())
            ok, msg = actions.steer(self.cwd, run_id, body.text)
        elif verb == "approve":
            ab = ApproveBody.model_validate(self._read_body())
            ok, msg = actions.approve(self.cwd, run_id, ab.id, ab.approved, session=ab.session)
        elif verb == "answer":
            qb = AnswerBody.model_validate(self._read_body())
            ok, msg = actions.answer_question(self.cwd, run_id, qb.id, qb.answers)
        elif verb == "merge":
            mb = MergeBody.model_validate(self._read_body())
            ok, msg = actions.merge_run(self.cwd, run_id, mb.strategy)
        else:
            self._post_not_found(f"run/{run_id}/{verb}")
            return
        self._ok_or_err(ok, {"message": msg}, msg)

    def _route_machine_post(self, name: str, verb: str) -> None:
        if verb == "poke":
            pb = MachinePokeBody.model_validate(self._read_body())
            ok, msg = actions.machine_poke(self.cwd, name, data=pb.data, message=pb.message)
        elif verb == "steer":
            body = SteerBody.model_validate(self._read_body())
            ok, msg = actions.machine_steer(self.cwd, name, body.text, state=body.state)
        elif verb == "approve":
            ab = ApproveBody.model_validate(self._read_body())
            ok, msg = actions.machine_approve(
                self.cwd, name, ab.id, ab.approved, session=ab.session, state=ab.state
            )
        elif verb == "answer":
            qb = AnswerBody.model_validate(self._read_body())
            ok, msg = actions.machine_answer(self.cwd, name, qb.id, qb.answers, state=qb.state)
        else:
            self._post_not_found(f"machine/{name}/{verb}")
            return
        self._ok_or_err(ok, {"message": msg}, msg)

    def _ok_or_err(self, ok: bool, payload: dict[str, Any], err: str) -> None:
        if ok:
            self._send_json({"ok": True, **payload})
        else:
            self._send_json({"ok": False, "error": err}, status=422)

    def _route(self, path: str) -> None:  # noqa: PLR0911
        if path == "/":
            self._send_bytes(PAGE_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/manifest.webmanifest":
            self._send_bytes(MANIFEST_JSON.encode("utf-8"), "application/manifest+json")
            return
        if path == "/sw.js":
            self._send_bytes(SERVICE_WORKER_JS.encode("utf-8"), "text/javascript; charset=utf-8")
            return
        if path == "/icon.svg":
            self._send_bytes(ICON_SVG.encode("utf-8"), "image/svg+xml")
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
        # /api/run/<id>[/conversation|/events]
        if len(parts) in (3, 4) and parts[0] == "api" and parts[1] == "run":
            self._route_run(parts[2], parts[3] if len(parts) > 3 else "")
            return
        # /api/machine/<name>[/reasoning|/conversation|/events]
        if len(parts) in (3, 4) and parts[0] == "api" and parts[1] == "machine":
            self._route_machine(parts[2], parts[3] if len(parts) > 3 else "")
            return
        # /api/draft/<name>[/events]: a `machine create` draft, watched as a run.
        if len(parts) in (3, 4) and parts[0] == "api" and parts[1] == "draft":
            self._route_draft(parts[2], parts[3] if len(parts) > 3 else "")
            return
        self._send_json({"error": f"not found: {path}"}, status=404)

    def _route_draft(self, name: str, sub: str) -> None:
        draft_dir = model.draft_dir_for(self.cwd, name)
        if draft_dir is None:
            self._send_json({"error": f"no draft {name!r}"}, status=404)
            return
        if sub == "":
            self._send_json(model.run_snapshot(draft_dir))
        elif sub == "conversation":
            self._send_json(model.conversation_payload(draft_dir))
        elif sub == "events":
            self._sse_run(draft_dir)
        else:
            self._send_json({"error": f"not found: draft/{name}/{sub}"}, status=404)

    def _route_run(self, run_id: str, sub: str) -> None:
        run_dir = model.run_dir_for(self.cwd, run_id)
        if run_dir is None:
            self._send_json({"error": f"no run {run_id!r}"}, status=404)
            return
        if sub == "":
            self._send_json(model.run_snapshot(run_dir))
        elif sub == "conversation":
            self._send_json(model.conversation_payload(run_dir))
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
            elif sub == "conversation":
                self._send_json(model.machine_conversation_payload(machine_dir))
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
        if self.close_connection:
            # Announce the close (CSRF refusal, unread POST body): without the
            # header a keep-alive client reuses the socket we are about to shut.
            self.send_header("Connection", "close")
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
        stop = threading.Event()

        def tail() -> None:
            src = run_dir / "logs.jsonl"
            try:
                for ev in tail_events(
                    src, follow=True, stop_when_finished=True, should_stop=stop.is_set
                ):
                    events.put(ev)
            finally:
                # ALWAYS enqueue the sentinel, even if the tailer raises: without
                # it the response loop would block on heartbeats forever.
                events.put(None)  # run ended (or tail cancelled/failed), tailer done

        threading.Thread(target=tail, daemon=True).start()

        try:
            state = initial_state()
            last_delta_emit = 0.0
            while True:
                try:
                    ev: dict[str, Any] | None = events.get(timeout=_HEARTBEAT_S)
                except queue.Empty:
                    if not self._sse_ping():
                        return
                    # A run that died without a run.end (crash / went quiet) would
                    # otherwise pin this worker forever: once its worker.pid points
                    # at a dead process, send a final snapshot and close.
                    if read_worker_pid(run_dir) is not None and not worker_is_alive(run_dir):
                        self._sse_send(run_state_as_dict(state))
                        return
                    continue
                # Fold everything already queued into ONE frame. On connect the
                # tailer replays the whole history, and a full RunState frame per
                # historical event is quadratic (13 MB probed on a 502-event run).
                last_type = ""
                while ev is not None:
                    state = apply_event(state, ev)
                    last_type = str(ev.get("type", ""))
                    try:
                        ev = events.get_nowait()
                    except queue.Empty:
                        break
                if ev is None:  # run ended: send the final snapshot and close
                    self._sse_send(run_state_as_dict(state))
                    return
                now = time.monotonic()
                if last_type in _STREAMING_DELTAS and (now - last_delta_emit) < _DELTA_COALESCE_S:
                    continue  # coalesce bursts of text/thinking deltas
                if not self._sse_send(run_state_as_dict(state)):
                    return
                last_delta_emit = now
        finally:
            stop.set()  # cancel the tailer so it exits on disconnect / dead run, not just run.end

    def _sse_machine(self, machine_dir: Path) -> None:
        """Stream a machine: re-fold the journal + the current agent state's
        reasoning on a poll, pushing the combined snapshot when it changes. While
        connected we register as the answer front-end on the INSTANCE dir, so a
        machine agent state's approval/question/steer prompts bridge to the
        browser (the state's answer files live in its per-state dir; the liveness
        gate probes this instance dir)."""
        self._begin_sse()
        self.server.claim_run(machine_dir)
        try:
            self._sse_machine_loop(machine_dir)
        finally:
            self.server.release_run(machine_dir)

    def _sse_machine_loop(self, machine_dir: Path) -> None:
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
            if payload["machine"].get("ended") is not None:
                return  # machine terminated: final snapshot sent, close the stream
            # A machine that died mid-state (no MachineEnd) would pin this
            # stream forever: its worker.pid points at a dead process AND no
            # armed wait explains the absence (a parked --exit-on-wait machine
            # legitimately has no live process between scheduler ticks).
            if (
                read_worker_pid(machine_dir) is not None
                and not worker_is_alive(machine_dir)
                and not model.machine_is_parked(machine_dir)
            ):
                return
            time.sleep(_MACHINE_POLL_S)


def run_web(target: str, *, host: str, port: int, cwd: Path | None = None) -> int:
    """Serve the web UI on host:port until interrupted. `target` deep-links the
    page to a run id or machine name on load (empty opens the hub)."""
    workdir = cwd or Path.cwd()
    bind_host = _bind_host(host)
    try:
        server = _create_web_server(bind_host, port, workdir, target)
    except OSError as exc:
        print(f"agent6 web: cannot bind {bind_host}:{port}: {exc}", file=sys.stderr)
        return 2
    shown = _display_host(bind_host)
    print(f"agent6 web: serving on http://{shown}:{port}  (Ctrl-C to stop)", file=sys.stderr)
    if not is_loopback_host(bind_host):
        print(
            "agent6 web: WARNING bound to a non-loopback address; anyone who can reach"
            f" {bind_host}:{port} can drive this agent. Prefer `tailscale serve` in front of a"
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
