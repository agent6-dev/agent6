"""Inspect a session's recorded evidence without calling a provider."""

from __future__ import annotations

import json
import pathlib

from agent6 import terminal
from agent6.ui.cli import _common
from agent6.viewmodel import receipt


def cmd_receipt(session_id: str, *, as_json: bool) -> int:
    """Print the derived receipt.

    Args:
        session_id: The session selector.
        as_json: Whether to print structured evidence.

    Returns:
        Zero on inspection, or 2 when no session matches.
    """
    saved = _common.resolve_target(session_id)
    if saved is None:
        return 2
    view = receipt.read_receipt(saved.session_dir, pathlib.Path.cwd())
    print(
        json.dumps(receipt.as_dict(view), ensure_ascii=False)
        if as_json
        else terminal.scrub_terminal_escapes(receipt.render(view))
    )
    return 0
