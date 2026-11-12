# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The web hub announces a session that starts waiting on an answer or ends."""

from __future__ import annotations

from agent6.ui.web import page


def test_the_hub_poll_announces_transitions_and_keeps_watching_when_it_may_notify() -> None:
    js = page.CLIENT_JS
    assert "announceSessions(statuses, next.sessions);" in js
    assert "statuses = seedStatuses(next.sessions);" in js
    # A hidden tab polls on only when the operator granted OS notifications.
    assert "if (document.hidden && !osNotifyOn()) return;" in js
    assert (
        "if (document.hidden) return;"
        not in js.split("async function renderHub")[1].split("// --- conversation")[0]
    )


def test_one_live_word_list_and_a_bell_on_both_hubs() -> None:
    js = page.CLIENT_JS
    assert "['running', 'starting', 'waiting'].includes" not in js
    assert js.count("LIVE_WORDS.includes(") >= 3
    # The sessions toolbar and the machine view both offer the permission button.
    assert js.count("'🔔 Notifications'") == 2
