"""Frame session requests and drain transport diagnostics with bounded retention."""

from __future__ import annotations

import dataclasses
import json
import os
import select
import subprocess
import threading
import time
from collections.abc import Callable
from typing import IO

from agent6.sandbox import jail

ANSWER_POLL_S = 0.2
REQUEST_LIMIT = 1024 * 1024


@dataclasses.dataclass(slots=True)
class Stderr:
    """Drain transport diagnostics independently, retaining a bounded tail."""

    pipe: IO[bytes]
    _tail: bytearray = dataclasses.field(default_factory=bytearray)
    _lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    _done: threading.Event = dataclasses.field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    _truncated: bool = False

    def start(self) -> None:
        """Start the independent stderr drain."""
        os.set_blocking(self.pipe.fileno(), False)
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _read_available(self) -> bool:
        for _ in range(128):
            try:
                chunk = os.read(self.pipe.fileno(), 8192)
            except BlockingIOError:
                return True
            if not chunk:
                return False
            self._tail.extend(chunk)
            if len(self._tail) > 65536:
                del self._tail[:-65536]
                self._truncated = True
        return True

    def _drain(self) -> None:
        while not self._done.is_set():
            if select.select([self.pipe], [], [], 0.1)[0]:
                with self._lock:
                    if not self._read_available():
                        return

    def text(self) -> str:
        """Read the retained stderr tail.

        Returns:
            Diagnostics, with a truncation marker when bytes were discarded.
        """
        with self._lock:
            if not self.pipe.closed:
                self._read_available()
            prefix = "[jail stderr truncated]\n" if self._truncated else ""
            return prefix + jail.lossy_text(bytes(self._tail)).strip()

    def close(self) -> None:
        """Stop draining without waiting for the peer to close its descriptor."""
        self._done.set()
        if self._thread is not None:
            self._thread.join(timeout=1)


def frame(request: dict[str, object]) -> bytes:
    """Serialize one bounded request.

    Args:
        request: The protocol object.

    Returns:
        UTF-8 JSON with its newline delimiter.

    Raises:
        jail.JailUnavailableError: The request is oversized or not JSON-serializable.
    """
    try:
        data = (json.dumps(request, allow_nan=False) + "\n").encode()
    except (TypeError, ValueError) as exc:
        raise jail.JailUnavailableError(f"jail session request is not valid JSON: {exc}") from exc
    if len(data) > REQUEST_LIMIT:
        raise jail.JailUnavailableError(f"jail protocol frame exceeds {REQUEST_LIMIT} bytes")
    return data


def read_frame(
    proc: subprocess.Popen[bytes],
    limit: int,
    *,
    request_id: int = 0,
    interrupted: Callable[[], bool] | None = None,
    deadline: float | None = None,
) -> bytes:
    """Read one bounded frame without blocking cancellation on a partial line.

    Args:
        proc: The protocol peer.
        limit: Maximum frame bytes, including the newline.
        request_id: The active request, or zero during readiness.
        interrupted: Polled while reading; true sends one cancellation frame.
        deadline: The readiness deadline, or None for a command response.

    Returns:
        The complete frame, including its newline.

    Raises:
        EOFError: The peer closed stdout before a complete frame.
        jail.JailUnavailableError: The frame is oversized, malformed or late.
    """
    assert proc.stdout is not None and proc.stdin is not None
    line = bytearray()
    asked = False
    while True:
        if deadline is not None and time.monotonic() > deadline:
            raise jail.JailUnavailableError("jail session handshake timed out")
        if not asked and interrupted is not None and interrupted():
            proc.stdin.write(frame({"kind": "cancel", "id": request_id}))
            proc.stdin.flush()
            asked = True
        if not select.select([proc.stdout], [], [], ANSWER_POLL_S)[0]:
            if proc.poll() is not None:
                raise EOFError(f"jail transport exited ({proc.returncode})")
            continue
        chunk = os.read(proc.stdout.fileno(), min(65536, limit + 1 - len(line)))
        if not chunk:
            raise EOFError("jail session ended before a complete protocol frame")
        line.extend(chunk)
        if len(line) > limit:
            raise jail.JailUnavailableError(f"jail protocol frame exceeds {limit} bytes")
        if b"\n" in chunk:
            if line.find(b"\n") != len(line) - 1:
                raise jail.JailUnavailableError("jail session sent data after a protocol frame")
            return bytes(line)


def write_startup(proc: subprocess.Popen[bytes], data: bytes, deadline: float) -> None:
    """Bound bootstrap transfer and policy writes even when a runtime stops reading.

    Args:
        proc: The raw process transport.
        data: The launcher bytes or one policy frame.
        deadline: The monotonic startup deadline.

    Raises:
        EOFError: The transport exited.
        jail.JailUnavailableError: The transfer timed out.
    """
    assert proc.stdin is not None
    fd = proc.stdin.fileno()
    os.set_blocking(fd, False)
    try:
        view = memoryview(data)
        while view:
            if time.monotonic() >= deadline:
                raise jail.JailUnavailableError("jail bootstrap transfer timed out")
            if proc.poll() is not None:
                raise EOFError(f"jail transport exited ({proc.returncode})")
            if select.select([], [fd], [], ANSWER_POLL_S)[1]:
                try:
                    view = view[os.write(fd, view[:65536]) :]
                except BlockingIOError:
                    continue
    finally:
        os.set_blocking(fd, True)
