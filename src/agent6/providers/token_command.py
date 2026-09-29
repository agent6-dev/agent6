# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Mint a short-lived bearer token by running an operator-configured command.

`[providers.<name>].token_command` names a command that prints a bearer to stdout
(a cloud OAuth access token, an OIDC or STS gateway); `CommandToken` caches it for
`token_command_ttl_s` seconds and re-runs on demand. The command runs outside any
sandbox with the operator's environment, the trust level of an MCP server command,
so only config sets it, never a run.
"""

from __future__ import annotations

import subprocess
import threading
import time
from collections.abc import Sequence

from agent6.providers.types import ProviderError

_DEFAULT_RUN_TIMEOUT_S = 30.0


class CommandToken:
    """A cached, refreshable bearer minted by an external command; thread-safe."""

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
        self._fetched_at = 0.0  # monotonic time of the last successful run

    def token(self) -> str:
        """Return the cached bearer while younger than the TTL, else re-run the command.

        Raises:
            ProviderError: The command is missing, times out, exits non-zero or
                prints nothing.
        """
        with self._lock:
            now = time.monotonic()
            if self._token and (now - self._fetched_at) < self._ttl_s:
                return self._token
            token = self._run()
            self._token = token
            self._fetched_at = now
            return token

    def invalidate(self, status: int = 401) -> bool:
        """Drop the cached token so the next `token()` re-runs the command.

        Both 401 and 403 re-mint: a command-minted bearer is short-lived and scoped,
        so either can mean it aged out (the OAuth credential refreshes on 401 only).

        Args:
            status: The HTTP status that prompted the drop; unused.

        Returns:
            True, the provider's signal to retry the request.
        """
        del status
        with self._lock:
            self._token = ""
            self._fetched_at = 0.0
        return True

    def _run(self) -> str:
        """Run the command once.

        Returns:
            The token the command printed, stripped.

        Raises:
            ProviderError: The command could not start, timed out, failed or was silent.
        """
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
