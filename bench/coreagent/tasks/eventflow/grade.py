#!/usr/bin/env python3
"""Authoritative hidden grader for eventflow. Not shipped into the agent's repo.

Usage: python3 grade.py <worktree-dir>
Each case GROUP runs in a fresh subprocess (config/_memo are module globals;
stale state must never leak between groups). Prints one JSON line with
score / cases_passed / cases_total / components_passed / components_total.
"""

from __future__ import annotations

import json
import subprocess
import sys

# component -> list of (name, snippet). A snippet raises on failure.
PRELUDE = """
import config, report, sessions, stats
import parser as parser_mod

def ev(line):
    return parser_mod.parse_line(line)

def T(iso):
    return ev(iso + "|u|a|1").ts
"""

GROUPS: dict[str, list[tuple[str, str]]] = {
    "parser_cents": [
        ("cents_029", 'assert ev("2026-01-02T00:00:00|a|x|0.29").value_cents == 29'),
        ("cents_035", 'assert ev("2026-01-02T00:00:00|a|x|3.5").value_cents == 350'),
        ("cents_int", 'assert ev("2026-01-02T00:00:00|a|x|12").value_cents == 1200'),
        ("cents_007", 'assert ev("2026-01-02T00:00:00|a|x|0.07").value_cents == 7'),
        ("cents_1015", 'assert ev("2026-01-02T00:00:00|a|x|10.15").value_cents == 1015'),
    ],
    "parser_validation": [
        (
            "reject_3dp",
            'import contextlib\n'
            'ok = False\n'
            'try: ev("2026-01-02T00:00:00|a|x|1.234")\n'
            'except ValueError: ok = True\n'
            "assert ok",
        ),
        (
            "reject_negative",
            "ok = False\n"
            'try: ev("2026-01-02T00:00:00|a|x|-1")\n'
            "except ValueError: ok = True\n"
            "assert ok",
        ),
        (
            "whitespace_stripped",
            'e = ev("  2026-01-02T00:00:00 | alice | click | 1.00 ")\n'
            'assert e.user == "alice" and e.value_cents == 100',
        ),
        (
            "parse_lines_skips_blank",
            'evs = list(parser_mod.parse_lines(["", "  ", "2026-01-02T00:00:00|a|x|1"]))\n'
            "assert len(evs) == 1",
        ),
    ],
    "session_gap": [
        (
            "equal_gap_same_session",
            'g = config.get("session_gap_s")\n'
            'a = ev("2026-01-02T00:00:00|u|x|1")\n'
            "b = parser_mod.Event(a.ts + g, 'u', 'x', 1)\n"
            "assert len(sessions.build_sessions([a, b])) == 1",
        ),
        (
            "gap_plus_one_splits",
            'g = config.get("session_gap_s")\n'
            'a = ev("2026-01-02T00:00:00|u|x|1")\n'
            "b = parser_mod.Event(a.ts + g + 1, 'u', 'x', 1)\n"
            "assert len(sessions.build_sessions([a, b])) == 2",
        ),
        (
            "unsorted_input_ok",
            'a = ev("2026-01-02T02:00:00|u|x|1"); b = ev("2026-01-02T00:00:00|u|x|1")\n'
            "ss = sessions.build_sessions([a, b])\n"
            "assert len(ss) == 2 and ss[0].events[0].ts < ss[1].events[0].ts",
        ),
    ],
    "session_isolation": [
        (
            "distinct_event_lists",
            'a = ev("2026-01-02T00:00:00|u|x|1")\n'
            'b = parser_mod.Event(a.ts + 999999, "u", "x", 1)\n'
            "ss = sessions.build_sessions([a, b])\n"
            "assert len(ss) == 2 and len(ss[0].events) == 1 and len(ss[1].events) == 1\n"
            "assert ss[0].events is not ss[1].events",
        ),
        (
            "fresh_session_empty",
            's1 = sessions.Session("u1"); s1.events.append(1)\n'
            's2 = sessions.Session("u2")\n'
            "assert s2.events == []",
        ),
        (
            "rebuild_no_accumulation",
            'a = ev("2026-01-02T00:00:00|u|x|1")\n'
            "n1 = len(sessions.build_sessions([a])[0].events)\n"
            "n2 = len(sessions.build_sessions([a])[0].events)\n"
            "assert n1 == 1 and n2 == 1",
        ),
    ],
    "money": [
        (
            "totals_integer_cents",
            'evs = [ev("2026-01-02T00:00:00|u|x|0.29") for _ in range(3)]\n'
            "evs = [parser_mod.Event(evs[0].ts + i, 'u', 'x', e.value_cents) for i, e in enumerate(evs)]\n"
            "ss = sessions.build_sessions(evs)\n"
            'tot = stats.user_totals(ss)["u"]\n'
            "assert tot == 87 and isinstance(tot, int)",
        ),
        (
            "avg_half_up_3over2",
            "# 5 cents over 2 sessions -> 2.5 -> HALF-UP 3 (banker's round() gives 2)\n"
            'a = ev("2026-01-02T00:00:00|u|x|0.02")\n'
            "b = parser_mod.Event(a.ts + 10**6, 'u', 'x', 3)\n"
            "ss = sessions.build_sessions([a, b])\n"
            "assert stats.avg_session_value(ss, 'u') == 3",
        ),
        (
            "avg_half_up_625",
            "# 250 cents over 4 sessions -> 62.5 -> 63\n"
            "evs = [parser_mod.Event(10**6 * (i + 1), 'u', 'x', c) for i, c in enumerate((100, 100, 25, 25))]\n"
            "ss = sessions.build_sessions(evs)\n"
            "assert stats.avg_session_value(ss, 'u') == 63",
        ),
        (
            "no_sessions_raises",
            "ok = False\n"
            "try: stats.avg_session_value([], 'ghost')\n"
            "except ValueError: ok = True\n"
            "assert ok",
        ),
    ],
    "config_freshness": [
        (
            "avg_tracks_gap_change",
            "evs = [parser_mod.Event(1000 + i * 2000, 'u', 'x', 100) for i in range(3)]\n"
            "ss1 = sessions.build_sessions(evs)  # gap 1800: 3 sessions\n"
            "v1 = stats.avg_session_value(ss1, 'u')\n"
            'config.load_overrides({"session_gap_s": 10_000})\n'
            "ss2 = sessions.build_sessions(evs)  # one session now\n"
            "v2 = stats.avg_session_value(ss2, 'u')\n"
            "assert v1 == 100 and v2 == 300, (v1, v2)",
        ),
        (
            "avg_tracks_sessions_arg",
            "a = [parser_mod.Event(1, 'u', 'x', 100)]\n"
            "b = [parser_mod.Event(1, 'u', 'x', 100), parser_mod.Event(2, 'u', 'x', 100)]\n"
            "v1 = stats.avg_session_value(sessions.build_sessions(a), 'u')\n"
            "v2 = stats.avg_session_value(sessions.build_sessions(b), 'u')\n"
            "assert v1 == 100 and v2 == 200, (v1, v2)",
        ),
    ],
    "report_order": [
        (
            "ties_break_by_name",
            'lines = report.top_users({"zed": 200, "amy": 200, "bob": 300})\n'
            'assert lines == ["bob 3.00", "amy 2.00", "zed 2.00"], lines',
        ),
        (
            "cents_padded",
            'lines = report.top_users({"a": 305})\n'
            'assert lines[0] == "a 3.05", lines',
        ),
        (
            "top_n_respects_config",
            'config.load_overrides({"top_n": 1})\n'
            'lines = report.top_users({"a": 1, "b": 2, "c": 3})\n'
            'assert lines == ["b 0.02"] or lines == ["c 0.03"], lines\n'
            'assert lines == ["c 0.03"], lines',
        ),
    ],
    "end_to_end": [
        (
            "pipeline",
            "raw = [\n"
            '    "2026-01-02T00:00:00|amy|buy|0.29",\n'
            '    "",\n'
            '    "2026-01-02T00:10:00|amy|buy|0.29",\n'
            '    "2026-01-02T09:00:00|amy|buy|0.29",\n'
            '    "2026-01-02T00:00:00|bob|buy|0.87",\n'
            "]\n"
            "evs = list(parser_mod.parse_lines(raw))\n"
            "ss = sessions.build_sessions(evs)\n"
            "tot = stats.user_totals(ss)\n"
            "assert tot == {'amy': 87, 'bob': 87}, tot\n"
            "lines = report.top_users(tot)\n"
            "assert lines == ['amy 0.87', 'bob 0.87'], lines\n"
            "assert stats.avg_session_value(ss, 'amy') == 44  # 87/2 -> 43.5 -> half-up 44",
        ),
    ],
}


def main() -> int:
    worktree = sys.argv[1]
    cases_total = 0
    cases_passed = 0
    components_passed = 0
    for _component, cases in GROUPS.items():
        comp_ok = True
        for _name, snippet in cases:
            cases_total += 1
            code = PRELUDE + "\n" + snippet
            proc = subprocess.run(
                [sys.executable, "-c", code],
                cwd=worktree,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode == 0:
                cases_passed += 1
            else:
                comp_ok = False
        if comp_ok:
            components_passed += 1
    print(
        json.dumps(
            {
                "score": round(cases_passed / cases_total, 4),
                "cases_passed": cases_passed,
                "cases_total": cases_total,
                "components_passed": components_passed,
                "components_total": len(GROUPS),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
