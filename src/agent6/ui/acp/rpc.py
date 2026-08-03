# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The JSON-RPC 2.0 vocabulary, shared by the transport and the methods.

Its own module because both need it: the transport turns an `RpcError` into a
reply, and a method raises one. Keeping the codes with the transport made the
two import each other.
"""

from __future__ import annotations

# The reserved codes, the only ones this front-end originates.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class RpcError(Exception):
    """An error to return to the client, rather than a crash."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
