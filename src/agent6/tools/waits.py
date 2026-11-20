"""Run one deadline callback while the calling thread waits for a result."""

from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import threading
import time
from collections.abc import Callable, Generator


@dataclasses.dataclass(frozen=True, slots=True)
class Deadline:
    """Share one callback across the waits in a turn.

    Attributes:
        at: The monotonic deadline.
        callback: Reports whether it changed the history.
    """

    at: float
    callback: Callable[[], bool]
    _outcome: concurrent.futures.Future[bool] = dataclasses.field(
        init=False, default_factory=concurrent.futures.Future
    )

    @contextlib.contextmanager
    def during(self, bound: float | None) -> Generator[None]:
        """Join an active callback before returning the wait's result.

        Args:
            bound: The most this wait can last in seconds, or None for no bound.

        Yields:
            While the original wait owns its thread.
        """
        if self._outcome.done() or (bound is not None and time.monotonic() + bound < self.at):
            yield
            return

        def at_limit() -> None:
            # The loop reads a failure after saving the result, never in place of it.
            try:
                self._outcome.set_result(self.callback())
            except Exception as exc:
                self._outcome.set_exception(exc)

        timer = threading.Timer(max(0, self.at - time.monotonic()), at_limit)
        timer.name = "idle-compaction"
        timer.start()
        try:
            yield
        finally:
            timer.cancel()
            timer.join()

    def result(self) -> bool:
        """Return the joined callback's outcome, or False when no wait reached the limit."""
        return self._outcome.done() and self._outcome.result()


class _Turn(threading.local):
    # Concurrent reviewers may share a dispatcher, but never a turn's history.
    deadline: Deadline | None = None


_TURN = _Turn()


@contextlib.contextmanager
def bind(deadline: Deadline | None) -> Generator[None]:
    """Bind the turn's deadline to the thread that dispatches its tools.

    Args:
        deadline: The strategy's deadline, or None for no callback.

    Yields:
        While the turn can wait.
    """
    previous = _TURN.deadline
    _TURN.deadline = deadline
    try:
        yield
    finally:
        _TURN.deadline = previous


def during(bound: float | None = None) -> contextlib.AbstractContextManager[None]:
    """Arm the turn's callback for the duration of a blocking wait.

    Args:
        bound: The most this wait can last in seconds, or None for no bound.

    Returns:
        The scope of the wait. A nonblocking poll arms nothing.
    """
    if _TURN.deadline is None or bound == 0:
        return contextlib.nullcontext()
    return _TURN.deadline.during(bound)
