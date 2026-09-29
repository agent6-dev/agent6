# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`paintRun` reads the session id it was given, never a free `id`.

The step picker fetched `/api/session/<id>/diff` with a bare `id`, which is a
parameter of `renderRun`, not of `paintRun`: under "use strict" every step
selection threw ReferenceError and rendered "id is not defined", so per-step
diffs and the as-of panels were unreachable on the web.
"""

from __future__ import annotations

import re

from agent6.ui.web import page


def _paint_run_body() -> str:
    start = page.CLIENT_JS.index("function paintRun(")
    end = page.CLIENT_JS.index("function renderDiff(", start)
    return page.CLIENT_JS[start:end]


def test_paint_run_reads_no_free_id() -> None:
    body = _paint_run_body()
    # A bare `id` token: not a property, not `_id`, not a key, not the string 'id'.
    free = [m.group(0) for m in re.finditer(r"(?<![.\w'\"])id(?![\w:'\"])", body)]
    assert free == [], f"paintRun references a free `id` {len(free)} time(s)"


def test_the_step_picker_fetches_with_the_cards_own_id() -> None:
    body = _paint_run_body()
    assert body.count("cards._base") >= 2


def test_the_machine_watch_gates_on_the_shared_refusals() -> None:
    """Every machine input consumes the wire refusals, without a second gate."""
    start = page.CLIENT_JS.index("function paintMachine(")
    body = page.CLIENT_JS[start : page.CLIENT_JS.index("\nfunction ", start + 1)]
    assert "m.refusals" in body, "the machine watch derives its own gating"
    assert "notRunning" not in body
    assert "canAnswer ? (data.reasoning || {}) : {}" in body
    assert "!!refusals.steer || !cards._state" not in body
    assert "cards._steer_btn.disabled = !!refusals.steer" in body
    assert "cards._msg_btn.disabled = !!refusals.poke" in body
    assert "cards._stop_btn.disabled = !!refusals.stop" in body
    assert "cards._input.disabled = !!refusals.steer && !!refusals.poke" in body


def test_the_machine_header_paints_the_wire_status_as_a_pill() -> None:
    """The detail view renders the machine's `status` and its level as the hub row does."""
    start = page.CLIENT_JS.index("function paintMachine(")
    body = page.CLIENT_JS[start : page.CLIENT_JS.index("\nfunction ", start + 1)]
    assert "pill(m.level, m.status)" in body
    assert "const word =" not in body


def test_a_stop_and_an_end_each_notify_once_across_a_resume() -> None:
    """The worker_lost banner and the ended banner have their own flags.

    One flag for both never announced the end after a stop and a resume.
    """
    start = page.CLIENT_JS.index("function paintMachine(")
    body = page.CLIENT_JS[start : page.CLIENT_JS.index("\nfunction ", start + 1)]
    assert "m.worker_lost && !ctx.lostNotified" in body
    assert "m.ended && !ctx.endedNotified" in body
    assert "ctx.lostNotified = false" in body
    # The first paint seeds the flag: a stop the page opened on is not news.
    start = page.CLIENT_JS.index("function machineNotify(")
    seed = page.CLIENT_JS[start : page.CLIENT_JS.index("\nfunction ", start + 1)]
    assert "ctx.lostNotified = !!m.worker_lost" in seed


def test_the_machine_composer_hint_names_enter() -> None:
    """Every docked entry's hint says what Enter does, the machine composer's included."""
    start = page.CLIENT_JS.index("async function renderMachine")
    body = page.CLIENT_JS[start : page.CLIENT_JS.index("function paintMachine(", start)]
    hint = next(line for line in body.splitlines() if "el('div', 'hint'" in line)
    assert "Enter sends" in hint and "Shift+Enter" in hint


def test_the_machine_snapshot_gates_controls_before_the_stream_arrives() -> None:
    """The one-shot wire payload is painted before any EventSource frame."""
    start = page.CLIENT_JS.index("async function renderMachine")
    body = page.CLIENT_JS[start : page.CLIENT_JS.index("function paintMachine(", start)]
    assert "const initial = await getJSON(base)" in body
    assert body.index("paintMachine(") < body.index("new EventSource(")


def test_worker_loss_leaves_the_machine_event_source_ready_for_a_resume() -> None:
    """EOF reconnects to a resumed machine; only a journaled end closes EventSource."""
    start = page.CLIENT_JS.index(
        "live.onmessage = ev =>", page.CLIENT_JS.index("async function renderMachine")
    )
    body = page.CLIENT_JS[start : page.CLIENT_JS.index("if (!hbTimer)", start)]
    close_line = next(line for line in body.splitlines() if "data.machine &&" in line)
    assert ".ended" in close_line
    assert "worker_lost" not in close_line


def test_enter_submits_the_available_machine_composer_verb() -> None:
    """Enter sends the one enabled verb; Shift+Enter remains a textarea newline."""
    start = page.CLIENT_JS.index("async function renderMachine")
    body = page.CLIENT_JS[start : page.CLIENT_JS.index("function machineNotify(", start)]
    handler_start = body.index("din.onkeydown")
    key_handler = body[handler_start : body.index("\n  };", handler_start)]
    assert "e.key === 'Enter'" in key_handler
    assert "!e.shiftKey" in key_handler
    assert "e.preventDefault()" in key_handler
    assert "!steerBtn.disabled" in key_handler and "steerBtn.click()" in key_handler
    assert "!msgBtn.disabled" in key_handler and "msgBtn.click()" in key_handler


def test_the_web_approval_box_offers_every_answer() -> None:
    """The box offered three of the four.

    The answers come from `ui.keymap`, so a fifth would fail here rather than quietly go unoffered.
    """
    from agent6.ui import keymap

    start = page.CLIENT_JS.index("for (const ap of")
    body = page.CLIENT_JS[start : page.CLIENT_JS.index("for (const q of", start)]
    for entry in keymap.APPROVAL_ANSWERS:
        assert f"send('{entry.answer}')" in body, entry.answer
        # The button says the word the table says, as the CLI and TUI do.
        assert f"'{entry.label.capitalize()}')" in body, entry.label
    # One box, the approval the server will take an answer to.
    assert "ap.id !== s.open_approval" in body


def test_a_failure_toast_holds_until_it_is_dismissed() -> None:
    """Messages stack, and a captured refusal stays until dismissed.

    One 4-second toast at one position overlapped and vanished before it could be read.
    """
    start = page.CLIENT_JS.index("function toast(")
    body = page.CLIENT_JS[start : page.CLIENT_JS.index("\nfunction ", start + 1)]
    assert "setTimeout" in body, "a confirmation still clears itself"
    assert "if (bad)" in body and "t.remove()" in body, "a failure has no dismiss"
    assert "#toasts" in page.PAGE_HTML, "the stack has no container style"


def test_the_machine_page_offers_stop() -> None:
    """The page can park a machine at its next transition, as `machine stop` does."""
    start = page.CLIENT_JS.index("async function renderMachine")
    body = page.CLIENT_JS[start : page.CLIENT_JS.index("function paintMachine(", start)]
    assert "'/stop'" in body or "+ '/stop'" in body
    assert "cards._stop_btn" in page.CLIENT_JS


def test_the_in_flight_mark_needs_a_live_run() -> None:
    """A stale run's Overview card shows no working ellipsis.

    A killed worker leaves a `role.call` with no `role.result`, so `in_flight` stays true.
    """
    body = _paint_run_body()
    assert "r.in_flight && s.live" in body, "the in-flight mark must read liveness too"


def test_a_typed_stream_error_is_not_replayed_on_reconnect() -> None:
    """A server-side stream error closes that stream after showing its reason.

    Automatic reconnect would replay the same terminal frame forever.
    """
    from importlib import resources

    web = resources.files("agent6.ui.web")
    for name in ("client_run.js", "client_machine.js"):
        source = web.joinpath(name).read_text(encoding="utf-8")
        error_branch = next(line for line in source.splitlines() if "type === 'error'" in line)
        assert "closeLive()" in error_branch, name
        assert "toast(" in error_branch and ", true)" in error_branch, name


def test_a_submitted_prompt_disables_its_controls_until_repaint() -> None:
    """After a successful answer POST the prompt's controls stay disabled until a frame removes it.

    A failed POST restores the controls.
    """
    assert "async function postPrompt(" in page.CLIENT_JS
    start = page.CLIENT_JS.index("function paintPrompts(")
    body = page.CLIENT_JS[start : page.CLIENT_JS.index("function paintDetails(", start)]
    assert body.count("postPrompt(box, base +") == 2
    helper = page.CLIENT_JS[page.CLIENT_JS.index("function setPromptBusy(") : start]
    assert "querySelectorAll('button,input')" in helper
    assert "disabled = busy" in helper
    assert "setPromptBusy(box, false)" in helper


def test_the_web_tool_row_counts_the_args_lines_it_drops() -> None:
    """The web row folds every args line, as the TUI row does."""
    start = page.CLIENT_JS.index("// tools: one clipped line per call")
    body = page.CLIENT_JS[start : page.CLIENT_JS.index("// shells:", start)]
    extra = body[body.index("const extra") : body.index("\n", body.index("const extra"))]
    assert "args_preview" in extra, extra


def test_delete_is_gated_on_the_run_being_over() -> None:
    """Delete dims itself on a live run, as Merge does, instead of asking and then refusing."""
    assert "cards._rm_btn = rmBtn" in page.CLIENT_JS
    start = page.CLIENT_JS.index("function paintRun(")
    body = page.CLIENT_JS[start : page.CLIENT_JS.index("function renderDiff(", start)]
    assert "cards._rm_btn.disabled = !isDead" in body
