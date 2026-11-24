# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The web composer's slash completion mirrors directive.STEER_COMMANDS verbatim.

The client is static JS, so the shared table is pinned, not imported; /compact, /btw and
/now stay live-session offers. An `@path` completes from the server's list of the files
the reference can read.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import threading

from agent6 import directive
from agent6.sessions import manifest
from agent6.ui.web import page, server
from tests.web import session_logs, signin


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


def test_the_refs_route_lists_the_files_of_the_workspace_a_composer_writes_to(
    tmp_path: pathlib.Path,
) -> None:
    """New work completes in the repository; a worktree session completes in its worktree."""
    worktree = tmp_path / "wt"
    for root, name in ((tmp_path, "notes.md"), (worktree, "notes-wt.md")):
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        (root / name).write_text("", encoding="utf-8")
    session = session_logs.write_run(tmp_path, "wt-run-AAAAAA", [], live=True)
    (session / manifest.MANIFEST_NAME).write_text(
        manifest.SessionManifest(mode="run", worktree=worktree).model_dump_json(),
        encoding="utf-8",
    )
    srv = server.WebServer(("127.0.0.1", 0), tmp_path, "", secret=signin.SECRET)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    def get(path: str) -> tuple[int, dict[str, object]]:
        conn = signin.SignedIn(int(srv.server_address[1]))
        try:
            conn.request("GET", path)
            resp = conn.getresponse()
            return resp.status, json.loads(resp.read())
        finally:
            conn.close()

    try:
        assert get("/api/refs?q=no") == (
            200,
            {"paths": ["notes.md"], "more": 0, "common": "notes.md"},
        )
        assert get("/api/refs?q=no&session=wt-run-AAAAAA")[1]["paths"] == ["notes-wt.md"]
        assert get("/api/refs?q=no&session=gone")[0] == 404
    finally:
        srv.shutdown()
        srv.server_close()


def test_the_client_reads_a_reference_as_the_expander_does() -> None:
    """The client's fragment and the Python one accept the same characters before and after `@`."""
    assert "/(?<![\\p{L}\\p{N}_@/])@([A-Za-z0-9_./-]*)$/u" in page.CLIENT_JS
