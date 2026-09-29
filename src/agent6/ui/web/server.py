# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Serve the web page and its JSON and SSE endpoints from a stdlib threading server.

The GET endpoints return the wire form `agent6 attach --json` prints; the SSE streams
push a fresh snapshot on each change. The server binds loopback by default (a
non-loopback bind is opt-in under `[web]`), renders folded read state, drives the
`agent6.sessions.ipc` contracts, serves no secret and executes no arbitrary input.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import ip_address
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs, unquote, urlsplit

from pydantic import BaseModel, ConfigDict, StrictBool, ValidationError

from agent6 import __version__
from agent6.config import is_loopback_host
from agent6.config.write import PROVIDER_DEFAULTS, provider_choices
from agent6.kinds import OPERATOR_MODES
from agent6.machine import MachineError
from agent6.sessions.ipc import (
    register_frontend,
    unregister_frontend,
)
from agent6.ui.spawn import spawn_new_work
from agent6.ui.web import actions, model
from agent6.ui.web._sse import SseChannel, stream_machine, stream_session
from agent6.ui.web.page import (
    FAVICON_SVG,
    ICON_SVG,
    MANIFEST_JSON,
    PAGE_HTML,
    SERVICE_WORKER_JS,
)
from agent6.viewmodel import (
    UnknownStepError,
    machine_snapshot,
    session_snapshot,
)

# The typed bodies are a few strings; an uncapped Content-Length would buffer arbitrary bytes.
_MAX_BODY_BYTES = 1 << 20


class _Body(BaseModel):
    """A typed POST body; an extra key is refused so a misspelled field fails loudly."""

    model_config = ConfigDict(extra="forbid")


class NewWorkBody(_Body):
    """The `/api/new` body: a session of `mode` on `task`, under a preset and a model."""

    mode: str
    task: str
    preset: str = ""
    model: str = ""


class SteerBody(_Body):
    """A steer body.

    Attributes:
        text: The instruction.
        state: For a machine, the state dir the prompt was rendered from; "" is the newest.
    """

    text: str = ""
    state: str = ""


class ApproveBody(_Body):
    """An approval answer.

    Attributes:
        id: The prompt's id.
        answer: The operator's literal choice; what a session answer grants is the
            asking side's to decide.
        state: For a machine, the state dir the prompt was rendered from; "" is the newest.
    """

    id: str
    answer: Literal["yes", "no", "session", "session-deny"]
    state: str = ""


class AnswerBody(_Body):
    """An `ask_user` answer: one entry per question, by index."""

    id: str
    answers: list[str]
    state: str = ""


class MergeBody(_Body):
    """A merge body; an empty strategy takes the config's."""

    strategy: str = ""


class PruneBody(_Body):
    """A prune body; `delete_squashed` is the CLI's own opt-in flag."""

    delete_squashed: bool = False


class StopBody(_Body):
    """A stop body; `after_step` lets the current step's results and auto-commit land."""

    after_step: StrictBool = False


class ResumeBody(_Body):
    """A resume body; each empty field means as the run recorded."""

    text: str = ""
    preset: str = ""
    model: str = ""


class MachineCreateBody(_Body):
    """A `machine create` body."""

    task: str


class MachineRunBody(_Body):
    """A `machine run` body naming a listed machine file."""

    file: str


class MachinePokeBody(_Body):
    """A poke body; a JSON `data` payload wins over a `message` string."""

    message: str = ""
    data: Any = None


class ProviderBody(_Body):
    """The add-provider form's fields."""

    name: str
    api_format: str
    deployment: str = ""
    base_url: str = ""
    api_key_env: str = ""
    repo: bool = False


class ConfigSetBody(_Body):
    """A config write; `unset` removes the key from the target layer instead."""

    key: str
    value: str = ""
    repo: bool = False
    unset: bool = False


def _validation_message(exc: ValidationError) -> str:
    """Return the failed fields as one line (`task: field required`)."""
    clauses: list[str] = []
    for err in exc.errors():
        field = ".".join(str(part) for part in err["loc"]) or "body"
        msg = str(err["msg"])
        clauses.append(f"{field}: {msg[:1].lower()}{msg[1:]}")
    return "; ".join(clauses)


class WebServer(ThreadingHTTPServer):
    """The threading server, carrying the repo its handlers read from.

    It counts the browsers watching each run so the process registers as an
    answering front-end only while someone is looking.
    """

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self, addr: tuple[str, int], cwd: Path, target: str, config_path: Path | None = None
    ) -> None:
        super().__init__(addr, _Handler)
        self.cwd = cwd
        self.target = target
        self.config_path = config_path
        self._pid_lock = threading.Lock()
        self._watch_counts: dict[str, int] = {}

    def handle_error(self, request: Any, client_address: Any) -> None:
        """Keep the stdlib traceback for every error but a client vanishing mid-request."""
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)

    def claim_session(self, session_dir: Path) -> None:
        """Register as the run's answer front-end on its first viewer.

        The claim file is per process, so other front-ends are never displaced.
        """
        key = str(session_dir)
        with self._pid_lock:
            n = self._watch_counts.get(key, 0)
            if n == 0:
                register_frontend(session_dir, os.getpid())
            self._watch_counts[key] = n + 1

    def release_session(self, session_dir: Path) -> None:
        """Drop the claim when the run's last viewer leaves.

        The count and the claim change under one lock, so a concurrent claim cannot
        interleave.
        """
        key = str(session_dir)
        with self._pid_lock:
            n = self._watch_counts.get(key, 1) - 1
            if n > 0:
                self._watch_counts[key] = n
                return
            self._watch_counts.pop(key, None)
            unregister_frontend(session_dir, os.getpid())


class _IPv6WebServer(WebServer):
    """The server bound to an IPv6 literal."""

    address_family = socket.AF_INET6


def _bind_host(host: str) -> str:
    """Return the socket bind address for a host, unbracketing an IPv6 literal."""
    stripped = host.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        return stripped[1:-1]
    return stripped


def _is_ipv6_literal(host: str) -> bool:
    """Return whether the host is an IPv6 literal."""
    try:
        return ip_address(_bind_host(host)).version == 6
    except ValueError:
        return False


def _display_host(host: str) -> str:
    """Return the host as a browser can open it; a wildcard bind shows loopback."""
    if host == "0.0.0.0":  # noqa: S104  # display only
        return "127.0.0.1"
    if host == "::":
        return "[::1]"
    return f"[{host}]" if _is_ipv6_literal(host) else host


def _create_web_server(
    host: str, port: int, cwd: Path, target: str, config_path: Path | None = None
) -> WebServer:
    """Return a server bound to the host and port, IPv6 when the host is an IPv6 literal."""
    bind_host = _bind_host(host)
    server_cls: type[WebServer] = _IPv6WebServer if _is_ipv6_literal(bind_host) else WebServer
    return server_cls((bind_host, port), cwd, target, config_path)


class _Handler(BaseHTTPRequestHandler):
    """One request's handler."""

    _streaming = False  # the SSE headers went out; an error is a frame now
    _body_length = 0  # this request's Content-Length, parsed once per request
    protocol_version = "HTTP/1.1"
    server: WebServer  # type: ignore[assignment]

    def log_message(self, format: str, *args: Any) -> None:
        """Log nothing; the signature is the stdlib's."""

    @property
    def cwd(self) -> Path:
        """The repository the server reads."""
        return self.server.cwd

    @property
    def config_path(self) -> Path | None:
        """The explicit config file, or None."""
        return self.server.config_path

    # -- routing --------------------------------------------------------------

    def _parse_body_length(self) -> bool:
        """Parse the request's one Content-Length, or send 400 and close the connection.

        Two values, or one that is not a plain decimal (`1_0`, `+2` and non-ASCII
        digits pass `int()`), leave the framing ambiguous.

        Returns:
            Whether the header parsed.
        """
        lengths = self.headers.get_all("Content-Length", [])
        raw = "0" if not lengths else (lengths[0].strip() if len(lengths) == 1 else "")
        if not (raw.isascii() and raw.isdigit()):
            self.close_connection = True
            self._send_json({"error": "bad Content-Length"}, status=400)
            return False
        self._body_length = int(raw)
        return True

    def do_GET(self) -> None:
        """Route a GET; the method name is the stdlib's dispatch contract."""
        path = unquote(urlsplit(self.path).path)
        if not self._parse_body_length():
            return
        if self._body_length:
            # A GET body is never read; on keep-alive it would parse as the next request.
            self.close_connection = True
            self._send_json({"error": "a GET carries no body"}, status=400)
            return
        try:
            self._route(path)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client went away mid-response
        except Exception as exc:  # one bad request never takes the server down
            if self._streaming:
                # The status line went out; a second one would land inside the event body.
                self._sse_send({"type": "error", "error": str(exc)})
            else:
                self._send_json({"error": str(exc)}, status=500)

    def do_POST(self) -> None:
        """Route a POST; the method name is the stdlib's dispatch contract."""
        path = unquote(urlsplit(self.path).path)
        # Parsed first: a bad header meets one refusal before anything is read.
        if not self._parse_body_length():
            return
        try:
            csrf_err = self._csrf_refusal()
            if csrf_err is not None:
                # Closing beats draining an unread body: a partial read desyncs the framing.
                self.close_connection = True
                self._send_json({"error": csrf_err}, status=403)
                return
            if self.headers.get("Transfer-Encoding"):
                # Only Content-Length bodies are read; a chunked one would sit unread.
                self.close_connection = True
                self._send_json({"error": "chunked bodies are not supported"}, status=411)
                return
            if self._body_length > _MAX_BODY_BYTES:
                self.close_connection = True
                self._send_json({"error": f"body larger than {_MAX_BODY_BYTES} bytes"}, status=413)
                return
            self._route_post(path)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except ValidationError as exc:
            # The body was read, so the framing is intact and the connection may stay open.
            self._send_json({"error": _validation_message(exc)}, status=400)
        except ValueError as exc:
            # Not JSON or not an object; the body was consumed, so the connection may stay open.
            self._send_json({"error": f"bad request: {exc}"}, status=400)
        except Exception as exc:  # one bad request never takes the server down
            # The body may be unread; a keep-alive reuse would parse it as the next request.
            self.close_connection = True
            self._send_json({"error": str(exc)}, status=500)

    def _csrf_refusal(self) -> str | None:
        """Return the reason to refuse this POST as cross-site, or None.

        The UI has no app-level auth: the machine (loopback) or the tailnet is the
        trust boundary, and neither stops a page on another origin from POSTing
        here. Two checks close it: a body must be `application/json` (a cross-site
        fetch with that type needs a preflight this server never answers), and a
        present `Origin` must match `Host` (a missing one is not browser-driven).
        DNS rebinding is left to the network layer: a Host allow-list would break
        `tailscale serve`.
        """
        if self._body_length > 0:
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
        """Read the request body as a JSON object.

        Returns:
            The object; empty for an empty body.

        Raises:
            ValueError: The body is not a JSON object.
        """
        n = self._body_length
        raw = self.rfile.read(n) if n > 0 else b""
        if not raw:
            return {}
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("request body must be a JSON object")
        return obj

    def _route_post(self, path: str) -> None:  # noqa: PLR0911
        """Dispatch a POST by path."""
        parts = path.strip("/").split("/")
        if path == "/api/new":
            body = NewWorkBody.model_validate(self._read_body())
            session_dir, err = spawn_new_work(
                self.cwd,
                body.mode,
                body.task,
                preset=body.preset,
                model=body.model,
                config_path=self.config_path,
            )
            session_id = session_dir.name if session_dir is not None else None
            self._ok_or_err(session_id is not None, {"session_id": session_id}, err)
            return
        if path == "/api/sessions/rm_asks":
            self._read_body()  # drains the body for keep-alive framing
            ok, msg = actions.remove_asks(self.cwd, self.config_path)
            self._ok_or_err(ok, {"message": msg}, msg)
            return
        if path == "/api/sessions/prune":
            body = PruneBody.model_validate(self._read_body())
            ok, msg = actions.prune_sessions(
                self.cwd, delete_squashed=body.delete_squashed, config_path=self.config_path
            )
            self._ok_or_err(ok, {"message": msg}, msg)
            return
        if path == "/api/config/provider":
            pb = ProviderBody.model_validate(self._read_body())
            ok, msg = actions.add_provider(
                self.cwd,
                pb.name,
                api_format=pb.api_format,
                deployment=pb.deployment,
                base_url=pb.base_url,
                api_key_env=pb.api_key_env,
                repo=pb.repo,
            )
            self._ok_or_err(ok, {"message": msg}, msg)
            return
        if path == "/api/config":
            body = ConfigSetBody.model_validate(self._read_body())
            if body.unset:
                ok, msg = actions.unset_config(
                    self.cwd, body.key, repo=body.repo, config_path=self.config_path
                )
            else:
                ok, msg = actions.set_config(
                    self.cwd, body.key, body.value, repo=body.repo, config_path=self.config_path
                )
            self._ok_or_err(ok, {"message": msg}, msg)
            return
        if path == "/api/machine/create":
            body = MachineCreateBody.model_validate(self._read_body())
            draft, err = actions.spawn_machine_create(self.cwd, body.task, self.config_path)
            self._ok_or_err(draft is not None, {"draft": draft}, err)
            return
        if path == "/api/machine/run":
            body = MachineRunBody.model_validate(self._read_body())
            ok, msg = actions.spawn_machine_run(self.cwd, body.file, self.config_path)
            self._ok_or_err(ok, {"message": msg}, msg)
            return
        if len(parts) == 4 and parts[0] == "api" and parts[1] == "session":
            self._route_session_post(parts[2], parts[3])
            return
        if len(parts) == 4 and parts[0] == "api" and parts[1] == "machine":
            self._route_machine_post(parts[2], parts[3])
            return
        self._post_not_found(f"not found: {path}")

    def _post_not_found(self, message: str) -> None:
        """Send 404 for a POST whose body was never read, closing the connection."""
        self.close_connection = True
        self._send_json({"error": message}, status=404)

    def _route_session_post(self, session_id: str, verb: str) -> None:
        """Dispatch `/api/session/<id>/<verb>`."""
        if model.session_dir_for(self.cwd, session_id) is None:
            self._post_not_found(f"no session {session_id!r}")
            return
        if verb == "steer":
            body = SteerBody.model_validate(self._read_body())
            ok, msg = actions.steer(self.cwd, session_id, body.text)
        elif verb == "approve":
            ab = ApproveBody.model_validate(self._read_body())
            ok, msg = actions.approve(self.cwd, session_id, ab.id, ab.answer)
        elif verb == "answer":
            qb = AnswerBody.model_validate(self._read_body())
            ok, msg = actions.answer_question(self.cwd, session_id, qb.id, qb.answers)
        elif verb == "merge":
            mb = MergeBody.model_validate(self._read_body())
            ok, msg = actions.merge_run(
                self.cwd, session_id, mb.strategy, config_path=self.config_path
            )
        elif verb in ("undo", "fork"):
            self._read_body()
            payload, err = (
                actions.undo_session(self.cwd, session_id)
                if verb == "undo"
                else actions.fork_run(self.cwd, session_id, self.config_path)
            )
            self._ok_or_err(payload is not None, payload or {}, err)
            return
        elif verb == "resume":
            rb = ResumeBody.model_validate(self._read_body())
            ok, msg = actions.resume_run(
                self.cwd,
                session_id,
                rb.text,
                preset=rb.preset,
                route=rb.model,
                config_path=self.config_path,
            )
        elif verb == "stop":
            sb = StopBody.model_validate(self._read_body())
            ok, msg = actions.stop_run(self.cwd, session_id, after_step=sb.after_step)
        elif verb == "compact":
            self._read_body()
            ok, msg = actions.compact_run(self.cwd, session_id)
        elif verb == "rm":
            self._read_body()
            ok, msg = actions.remove_session(self.cwd, session_id, self.config_path)
        elif verb in ("run_plan", "review"):
            self._read_body()
            act = actions.run_plan if verb == "run_plan" else actions.review_run
            payload, err = act(self.cwd, session_id, self.config_path)
            self._ok_or_err(payload is not None, payload or {}, err)
            return
        else:
            self._post_not_found(f"not found: run/{session_id}/{verb}")
            return
        self._ok_or_err(ok, {"message": msg}, msg)

    def _route_machine_post(self, name: str, verb: str) -> None:
        """Dispatch `/api/machine/<name>/<verb>`."""
        if model.machine_dir_for(self.cwd, name) is None:
            self._post_not_found(f"no machine {name!r}")
            return
        if verb == "poke":
            pb = MachinePokeBody.model_validate(self._read_body())
            ok, msg = actions.machine_poke(self.cwd, name, data=pb.data, message=pb.message)
        elif verb == "stop":
            self._read_body()
            ok, msg = actions.machine_stop(self.cwd, name)
        elif verb == "steer":
            body = SteerBody.model_validate(self._read_body())
            ok, msg = actions.machine_steer(self.cwd, name, body.text, state=body.state)
        elif verb == "approve":
            ab = ApproveBody.model_validate(self._read_body())
            ok, msg = actions.machine_approve(self.cwd, name, ab.id, ab.answer, state=ab.state)
        elif verb == "answer":
            qb = AnswerBody.model_validate(self._read_body())
            ok, msg = actions.machine_answer(self.cwd, name, qb.id, qb.answers, state=qb.state)
        else:
            self._post_not_found(f"not found: machine/{name}/{verb}")
            return
        self._ok_or_err(ok, {"message": msg}, msg)

    def _ok_or_err(self, ok: bool, payload: dict[str, Any], err: str) -> None:
        """Send the payload under `ok`, or the error as 422."""
        if ok:
            self._send_json({"ok": True, **payload})
        else:
            self._send_json({"ok": False, "error": err}, status=422)

    def _send_routes(self) -> None:
        """Send `/api/routes?mode=&preset=`, the composer's model box; an unknown mode is 422."""
        q = parse_qs(urlsplit(self.path).query)
        mode = (q.get("mode") or ["run"])[0]
        if mode not in OPERATOR_MODES:
            self._send_json({"error": f"unknown mode {mode!r}"}, status=422)
            return
        self._send_json(
            model.routes_payload(
                self.cwd, self.config_path, mode=mode, preset=(q.get("preset") or [""])[0]
            )
        )

    def _route(self, path: str) -> None:  # noqa: PLR0911, PLR0912
        """Dispatch a GET by path."""
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
        if path == "/favicon.svg":
            self._send_bytes(FAVICON_SVG.encode("utf-8"), "image/svg+xml")
            return
        if path == "/api/meta":
            self._send_json(
                {
                    "version": __version__,
                    "target": self.server.target,
                    "target_kind": self._target_kind(),
                }
            )
            return
        if path == "/api/hub":
            self._send_json(model.hub_payload(self.cwd, self.config_path))
            return
        if path == "/api/routes":
            self._send_routes()
            return
        if path in ("/api/config", "/api/config/provider_choices"):
            # The add-provider form's choices come from the schema the TUI form reads.
            self._send_json(
                model.config_payload(self.cwd, self.config_path)
                if path == "/api/config"
                else {**provider_choices(), "defaults": PROVIDER_DEFAULTS}
            )
            return
        if path.startswith("/api/config/suggest/"):
            key = path.removeprefix("/api/config/suggest/")
            self._send_json({"values": model.config_suggestions(self.cwd, key, self.config_path)})
            return
        parts = path.strip("/").split("/")
        if len(parts) in (3, 4) and parts[0] == "api" and parts[1] == "session":
            self._route_session(parts[2], parts[3] if len(parts) > 3 else "")
            return
        if len(parts) in (3, 4) and parts[0] == "api" and parts[1] == "machine":
            self._route_machine(parts[2], parts[3] if len(parts) > 3 else "")
            return
        if len(parts) in (3, 4) and parts[0] == "api" and parts[1] == "draft":
            self._route_draft(parts[2], parts[3] if len(parts) > 3 else "")
            return
        self._send_json({"error": f"not found: {path}"}, status=404)

    def _target_kind(self) -> str:
        """Return the view the CLI target deep-links to, resolved per request.

        Returns:
            "session", "draft" or "machine", or "" with no target or no match.
        """
        t = self.server.target
        if not t:
            return ""
        if model.session_dir_for(self.cwd, t) is not None:
            return "session"
        if model.draft_dir_for(self.cwd, t) is not None:
            return "draft"
        if model.machine_dir_for(self.cwd, t) is not None:
            return "machine"
        return ""

    def _route_draft(self, name: str, sub: str) -> None:
        """Dispatch `/api/draft/<name>[/<sub>]`, a `machine create` draft watched as a run."""
        draft_dir = model.draft_dir_for(self.cwd, name)
        if draft_dir is None:
            self._send_json({"error": f"no draft {name!r}"}, status=404)
            return
        if sub == "":
            step = (parse_qs(urlsplit(self.path).query).get("step") or [""])[0]
            try:
                self._send_json(session_snapshot(draft_dir, step=step))
            except UnknownStepError as e:
                self._send_json({"error": str(e)}, status=422)
        elif sub == "conversation":
            self._send_json(model.conversation_payload(draft_dir))
        elif sub == "diff":
            q = parse_qs(urlsplit(self.path).query)
            workspace = model.draft_workspace(self.cwd, name, self.config_path)
            if workspace is None:
                gone = "the drafting workspace is gone (removed with a published draft)"
                payload, why = None, gone
            else:
                payload, why = model.draft_step_diff_payload(
                    workspace,
                    (q.get("sha") or [""])[0],
                    cumulative=(q.get("cumulative") or ["0"])[0] in ("1", "true"),
                )
            if payload is None:
                self._send_json({"error": why}, status=422)
            else:
                self._send_json(payload)
        elif sub == "events":
            self._sse_session(draft_dir)
        else:
            self._send_json({"error": f"not found: draft/{name}/{sub}"}, status=404)

    def _route_session(self, session_id: str, sub: str) -> None:
        """Dispatch `/api/session/<id>[/<sub>]`."""
        session_dir = model.session_dir_for(self.cwd, session_id)
        if session_dir is None:
            self._send_json({"error": f"no session {session_id!r}"}, status=404)
            return
        if sub == "":
            step = (parse_qs(urlsplit(self.path).query).get("step") or [""])[0]
            try:
                self._send_json(session_snapshot(session_dir, repo=self.cwd, step=step))
            except UnknownStepError as e:
                self._send_json({"error": str(e)}, status=422)
        elif sub == "conversation":
            self._send_json(model.conversation_payload(session_dir))
        elif sub == "restate":
            self._send_json(model.restate_payload(session_dir))
        elif sub == "diff":
            q = parse_qs(urlsplit(self.path).query)
            payload, why = model.step_diff_payload(
                self.cwd,
                session_dir,
                (q.get("sha") or [""])[0],
                cumulative=(q.get("cumulative") or ["0"])[0] in ("1", "true"),
            )
            if payload is None:
                self._send_json({"error": why}, status=422)
            else:
                self._send_json(payload)
        elif sub == "resume_defaults":
            preset = (parse_qs(urlsplit(self.path).query).get("preset") or [""])[0]
            self._send_json(
                model.resume_defaults_payload(
                    self.cwd, self.config_path, session_dir, preset=preset
                )
            )
        elif sub == "events":
            self._sse_session(session_dir)
        else:
            self._send_json({"error": f"not found: /api/session/{session_id}/{sub}"}, status=404)

    def _route_machine(self, name: str, sub: str) -> None:
        """Dispatch `/api/machine/<name>[/<sub>]`."""
        machine_dir = model.machine_dir_for(self.cwd, name)
        if machine_dir is None:
            self._send_json({"error": f"no machine {name!r}"}, status=404)
            return
        try:
            if sub == "":
                self._send_json(machine_snapshot(machine_dir))
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
        """Send a JSON response."""
        self._send_bytes(
            json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8", status=status
        )

    def _send_bytes(self, body: bytes, ctype: str, *, status: int = 200) -> None:
        """Send one uncached response."""
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if self.close_connection:
            # Without the header a keep-alive client reuses the socket about to shut.
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    # -- SSE ------------------------------------------------------------------

    def _begin_sse(self) -> None:
        """Send the event-stream headers; the socket closing is what ends the stream."""
        self.close_connection = True
        self._streaming = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")  # a proxy must not buffer the stream
        self.end_headers()

    def _sse_send(self, obj: Any) -> bool:
        """Write one data frame.

        Returns:
            False once the client has gone.
        """
        try:
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            self.wfile.flush()
        except OSError:
            return False
        return True

    def _sse_ping(self) -> bool:
        """Write a heartbeat comment.

        Returns:
            False once the client has gone.
        """
        try:
            self.wfile.write(b": ping\n\n")
            self.wfile.flush()
        except OSError:
            return False
        return True

    def _sse_session(self, session_dir: Path) -> None:
        """Stream a run, registered as its answer front-end while connected."""
        self._begin_sse()
        self.server.claim_session(session_dir)
        try:
            stream_session(self._channel(), session_dir, repo=self.cwd)
        finally:
            self.server.release_session(session_dir)

    def _sse_machine(self, machine_dir: Path) -> None:
        """Stream a machine, registered as the answer front-end on its instance dir.

        A state's answer files live in its per-state dir; the liveness gate probes
        the instance dir.
        """
        self._begin_sse()
        self.server.claim_session(machine_dir)
        try:
            stream_machine(self._channel(), machine_dir)
        finally:
            self.server.release_session(machine_dir)

    def _channel(self) -> SseChannel:
        """Return this handler's socket writes as a channel."""
        return SseChannel(send=self._sse_send, ping=self._sse_ping)


def run_web(
    target: str,
    *,
    host: str,
    port: int,
    cwd: Path | None = None,
    config_path: Path | None = None,
) -> int:
    """Serve the web UI until interrupted.

    Args:
        target: A run id, machine name or draft the page opens on; "" opens the hub.
        host: The bind host.
        port: The bind port.
        cwd: The repository; None is the process cwd.
        config_path: An explicit config file, or None.

    Returns:
        The exit code: 2 when the bind failed.
    """
    workdir = cwd or Path.cwd()
    bind_host = _bind_host(host)
    try:
        server = _create_web_server(bind_host, port, workdir, target, config_path)
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
