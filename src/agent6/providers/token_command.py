# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Mint a short-lived bearer token by running an operator-configured command.

Some OpenAI-compatible endpoints don't take a static API key; they want a
short-lived bearer that has to be refreshed (cloud OAuth access tokens,
internal OIDC/STS gateways, ...). ``[providers.<name>].token_command`` names a
command that prints such a token to stdout; :class:`CommandToken` runs it,
caches the result for ``token_command_ttl_s`` seconds, and re-runs it on demand
so the provider stays authenticated without a human re-pasting a key.

The command runs in agent6's own process, outside any run sandbox, with the
operator's environment, the same trust level as a ``[[mcp.servers]]`` command.
It is therefore an operator-controlled config knob, never something a run can
set, and the token it prints is sent as ``Authorization: Bearer <token>``.
"""

from __future__ import annotations

import subprocess
import threading
import time
from collections.abc import Sequence

from agent6.providers.types import ProviderError

_DEFAULT_RUN_TIMEOUT_S = 30.0


class CommandToken:
    """Cached, refreshable bearer minted by running an external command.

    Thread-safe. ``token()`` returns the cached value while it is younger than
    ``ttl_s`` and otherwise re-runs the command; ``invalidate()`` forces the
    next ``token()`` to re-run (the provider calls it after a 401/403 so an
    expired token self-heals regardless of ``ttl_s``).
    """

    __slots__ = ("_argv", "_fetched_at", "_lock", "_run_timeout_s", "_token", "_ttl_s")

    def __init__(
        self,
        argv: Sequence[str],
        *,
        ttl_s: float = 300.0,
        run_timeout_s: float = _DEFAULT_RUN_TIMEOUT_S,
    ) -> None:
        self._argv = list(argv)
        self._ttl_s = ttl_s
        self._run_timeout_s = run_timeout_s
        self._lock = threading.Lock()
        self._token = ""
        self._fetched_at = 0.0  # time.monotonic() of the last successful run

    def token(self) -> str:
        """Return a fresh-enough bearer, running the command if needed.

        Raises :class:`ProviderError` if the command is missing, times out,
        exits non-zero, or prints nothing.
        """
        with self._lock:
            now = time.monotonic()
            if self._token and (now - self._fetched_at) < self._ttl_s:
                return self._token
            token = self._run()
            self._token = token
            self._fetched_at = now
            return token

    def invalidate(self) -> None:
        """Drop the cached token so the next ``token()`` re-runs the command."""
        with self._lock:
            self._token = ""
            self._fetched_at = 0.0

    def _run(self) -> str:
        try:
            proc = subprocess.run(
                self._argv,
                capture_output=True,
                text=True,
                timeout=self._run_timeout_s,
                check=False,
            )
        except FileNotFoundError as exc:
            raise ProviderError(f"token_command not found: {self._argv[0]!r}") from exc
        except subprocess.TimeoutExpired as exc:
            raise ProviderError(
                f"token_command timed out after {self._run_timeout_s:.0f}s: {self._argv}"
            ) from exc
        except OSError as exc:
            raise ProviderError(f"token_command failed to start: {exc}") from exc
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()[:500]
            detail = f": {stderr}" if stderr else ""
            raise ProviderError(f"token_command exited {proc.returncode}{detail}")
        token = (proc.stdout or "").strip()
        if not token:
            raise ProviderError(f"token_command produced no output: {self._argv}")
        return token
