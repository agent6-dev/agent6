# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The web composer's slash completion mirrors directive.STEER_COMMANDS verbatim.

The client is static JS, so the shared table is pinned, not imported; /compact, /btw and
/now stay live-session offers.
"""

from __future__ import annotations

from agent6 import directive
from agent6.ui.web import page


def test_client_mirrors_the_steer_commands_verbatim() -> None:
    for cmd, help_ in directive.STEER_COMMANDS.items():
        assert f"['{cmd}', '{help_}']" in page.CLIENT_JS, f"the web client drifted from {cmd}"


def test_compact_is_gated_on_live() -> None:
    """The resume composer withholds exactly `LIVE_RUN_COMMANDS`, spelled by hand in the JS."""
    gate = next(line for line in page.CLIENT_JS.splitlines() if "liveNow() ||" in line)
    assert gate.count("c !== '") == len(directive.LIVE_RUN_COMMANDS)
    for cmd in directive.LIVE_RUN_COMMANDS:
        assert f"c !== '{cmd}'" in gate, cmd
    assert "() => finished === false" in page.CLIENT_JS  # the composer's live truth feeds the gate


def test_a_typed_stop_posts_the_one_stop() -> None:
    """`/stop` in the web composer calls the stop route, not a steer text."""
    assert "if (text === '/stop') {" in page.CLIENT_JS
    assert "postJSON('/api/session/' + encodeURIComponent(id) + '/stop', {})" in page.CLIENT_JS


def test_the_client_copies_only_the_spinner() -> None:
    """The heartbeat frames are pinned: the client animates locally between polls.

    Every other field is rendered by the server; a static client cannot import the frames.
    """
    from agent6.viewmodel import format

    assert f"'{format.SPINNER_FRAMES}'" in page.CLIENT_JS
    # The timestamp is rendered by the server, in the timezone every other surface shows.
    assert "function when(" not in page.CLIENT_JS
