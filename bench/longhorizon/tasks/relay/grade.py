#!/usr/bin/env python3
"""Authoritative hidden grader for relay. Not shipped into the agent's repo.

Usage: python3 grade.py <worktree-dir> [leg]
Loads the six stage modules from the worktree and runs per-stage case
batteries plus end-to-end cli runs. Components: parse, parse_lines,
validate, sessionize, metrics, report, cli. Report/cli cases compare whole
strings; stage cases isolate one contract each.
"""

from __future__ import annotations

import importlib
import io
import json
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any


def _ev(ts: int, kind: str, **payload: str) -> dict[str, Any]:
    return {"ts": ts, "kind": kind, "payload": payload}


def _field(got: Any, *path: str) -> Any:
    """Walk nested dict keys; None as soon as the shape disagrees."""
    for p in path:
        if not isinstance(got, dict) or p not in got:
            return None
        got = got[p]
    return got


def _load_modules(worktree: str) -> dict[str, Any]:
    sys.path.insert(0, worktree)
    mods: dict[str, Any] = {}
    for name in ("parse", "validate", "sessionize", "metrics", "report", "cli"):
        try:
            mods[name] = importlib.import_module(name)
        except Exception:
            mods[name] = None
    return mods


def _run_cli(cli: Any, argv: list[str]) -> tuple[Any, str]:
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            rc = cli.main(argv)
    except SystemExit as se:
        rc = se.code
    except Exception:
        return None, buf.getvalue()
    return rc, buf.getvalue()


TINY_REPORT = (
    "USER  SESS  ACTS  DURATION\nana      1     1        60\nTOTAL sessions=1 acts=1 pings=0\n"
)
EMPTY_REPORT = "USER  SESS  ACTS  DURATION\nTOTAL sessions=0 acts=0 pings=0\n"


def build_components(m: dict[str, Any], tmp: Path) -> dict[str, dict[str, bool]]:
    p, v, s = m["parse"], m["validate"], m["sessionize"]
    me, r, c = m["metrics"], m["report"], m["cli"]

    def parses(line: str, want: dict[str, Any]) -> bool:
        try:
            return p.parse_line(line) == want
        except Exception:
            return False

    def raises(line: str) -> bool:
        try:
            p.parse_line(line)
        except ValueError:
            return True
        except Exception:
            return False
        return False

    comps: dict[str, dict[str, bool]] = {}

    comps["parse"] = {
        "ping_empty": parses("0|ping|", {"ts": 0, "kind": "ping", "payload": {}}),
        "two_keys": parses(
            "1700000000|open|user=ana;src=web",
            {"ts": 1700000000, "kind": "open", "payload": {"user": "ana", "src": "web"}},
        ),
        "zeros_underscore_space": parses(
            "007|act|user=b_o;verb=see map",
            {"ts": 7, "kind": "act", "payload": {"user": "b_o", "verb": "see map"}},
        ),
        "bad_kind": raises("12|nope|"),
        "neg_ts": raises("-5|ping|"),
        "space_ts": raises(" 12|ping|"),
        "two_fields": raises("12|ping"),
        "four_fields": raises("12|ping|a=1|b=2"),
        "dup_key": raises("12|act|user=a;user=b"),
        "second_eq": raises("12|act|k=v=w"),
        "upper_key": raises("12|act|K=v"),
        "empty_value": raises("12|act|k="),
        "empty_key": raises("12|act|=v"),
        "empty_pair": raises("12|act|a=1;;b=2"),
    }

    def pl(text: str) -> Any:
        try:
            return p.parse_lines(text)
        except Exception:
            return None

    comps["parse_lines"] = {
        "mixed": pl("0|ping|\n\n12|nope|\n25|act|user=a;verb=v\n")
        == (
            [_ev(0, "ping"), _ev(25, "act", user="a", verb="v")],
            [(3, "12|nope|")],
        ),
        "empty": pl("") == ([], []),
        "blank_only": pl("   \n") == ([], []),
        "no_trailing_newline": pl("5|ping|") == ([_ev(5, "ping")], []),
    }

    def val(events: list[dict[str, Any]]) -> Any:
        try:
            return v.validate(events)
        except Exception:
            return None

    ok_open = _ev(5, "open", user="ana", src="web")
    ok_extra = _ev(6, "open", user="ana", src="web", region="eu")
    comps["validate"] = {
        "ok": val([ok_open]) == ([ok_open], []),
        "extras_ok": val([ok_extra]) == ([ok_extra], []),
        "missing_src": val([_ev(5, "open", user="ana")]) == ([], [(0, "missing:src")]),
        "missing_alpha_first": val([_ev(5, "act")]) == ([], [(0, "missing:user")]),
        "missing_reason": val([_ev(5, "close", user="a")]) == ([], [(0, "missing:reason")]),
        "ping_payload": val([_ev(5, "ping", x="1")]) == ([], [(0, "payload:ping")]),
        "sys_beats_missing": val([_ev(5, "act", sys="1")]) == ([], [(0, "forbidden:sys")]),
        "sys_beats_ping": val([_ev(5, "ping", sys="1")]) == ([], [(0, "forbidden:sys")]),
        "indexes_are_input": val([ok_open, _ev(9, "ping", x="1"), ok_extra])
        == ([ok_open, ok_extra], [(1, "payload:ping")]),
    }

    def ses(events: list[dict[str, Any]]) -> Any:
        try:
            return s.sessionize(events)
        except Exception:
            return None

    a0 = _ev(0, "act", user="ana", verb="v")
    a1800 = _ev(1800, "act", user="ana", verb="v")
    a1801 = _ev(1801, "act", user="ana", verb="v")
    t1 = _ev(50, "open", user="ana", src="web")
    t2 = _ev(50, "act", user="ana", verb="v")
    o0 = _ev(0, "open", user="ana", src="web")
    b10 = _ev(10, "open", user="bo", src="web")
    a100 = _ev(100, "act", user="ana", verb="v")
    a3600 = _ev(3600, "act", user="ana", verb="v")
    ping = _ev(7, "ping")
    comps["sessionize"] = {
        "boundary_1800_joins": ses([a0, a1800])
        == ([{"user": "ana", "events": [a0, a1800], "start": 0, "end": 1800}], 0),
        "gap_1801_splits": ses([a0, a1801])
        == (
            [
                {"user": "ana", "events": [a0], "start": 0, "end": 0},
                {"user": "ana", "events": [a1801], "start": 1801, "end": 1801},
            ],
            0,
        ),
        "tie_keeps_input_order": ses([t1, t2])
        == ([{"user": "ana", "events": [t1, t2], "start": 50, "end": 50}], 0),
        "interleaved_users": ses([o0, b10, a100])
        == (
            [
                {"user": "ana", "events": [o0, a100], "start": 0, "end": 100},
                {"user": "bo", "events": [b10], "start": 10, "end": 10},
            ],
            0,
        ),
        "same_start_user_tiebreak": ses(
            [_ev(50, "act", user="bo", verb="v"), _ev(50, "act", user="ana", verb="v")]
        )
        == (
            [
                {
                    "user": "ana",
                    "events": [_ev(50, "act", user="ana", verb="v")],
                    "start": 50,
                    "end": 50,
                },
                {
                    "user": "bo",
                    "events": [_ev(50, "act", user="bo", verb="v")],
                    "start": 50,
                    "end": 50,
                },
            ],
            0,
        ),
        "pings_counted_dropped": ses([ping, a0, ping])
        == ([{"user": "ana", "events": [a0], "start": 0, "end": 0}], 2),
        "sorts_by_ts": ses([a3600, a0])
        == (
            [
                {"user": "ana", "events": [a0], "start": 0, "end": 0},
                {"user": "ana", "events": [a3600], "start": 3600, "end": 3600},
            ],
            0,
        ),
        "empty": ses([]) == ([], 0),
    }

    def summ(sessions: list[dict[str, Any]], *args: int) -> Any:
        try:
            return me.summarize(sessions, *args)
        except Exception:
            return None

    ses_one = {
        "user": "ana",
        "events": [o0, _ev(60, "act", user="ana", verb="v")],
        "start": 0,
        "end": 60,
    }
    ses_b = {"user": "ana", "events": [_ev(0, "act", user="ana", verb="v")], "start": 0, "end": 100}
    ses_c = {
        "user": "ana",
        "events": [_ev(200, "act", user="ana", verb="v")],
        "start": 200,
        "end": 240,
    }
    ses_bo = {"user": "bo", "events": [_ev(0, "open", user="bo", src="w")], "start": 0, "end": 0}
    comps["metrics"] = {
        "single": summ([ses_one])
        == {
            "users": {"ana": {"sessions": 1, "total_duration": 60, "total_acts": 1}},
            "overall": {
                "n_sessions": 1,
                "n_users": 1,
                "total_acts": 1,
                "max_duration": 60,
                "n_pings": 0,
            },
        },
        "aggregates": summ([ses_b, ses_c])
        == {
            "users": {"ana": {"sessions": 2, "total_duration": 140, "total_acts": 2}},
            "overall": {
                "n_sessions": 2,
                "n_users": 1,
                "total_acts": 2,
                "max_duration": 100,
                "n_pings": 0,
            },
        },
        "multi_user": _field(summ([ses_one, ses_bo]), "overall", "n_users") == 2,
        "zero_duration": _field(summ([ses_bo]), "users", "bo", "total_duration") == 0,
        "empty": summ([])
        == {
            "users": {},
            "overall": {
                "n_sessions": 0,
                "n_users": 0,
                "total_acts": 0,
                "max_duration": 0,
                "n_pings": 0,
            },
        },
        "pings_passthrough": _field(summ([], 4), "overall", "n_pings") == 4,
    }

    def rend(summary: dict[str, Any]) -> Any:
        try:
            return r.render(summary)
        except Exception:
            return None

    comps["report"] = {
        "tiny": rend(
            {
                "users": {"ana": {"sessions": 1, "total_duration": 60, "total_acts": 1}},
                "overall": {
                    "n_sessions": 1,
                    "n_users": 1,
                    "total_acts": 1,
                    "max_duration": 60,
                    "n_pings": 0,
                },
            }
        )
        == TINY_REPORT,
        "empty": rend(
            {
                "users": {},
                "overall": {
                    "n_sessions": 0,
                    "n_users": 0,
                    "total_acts": 0,
                    "max_duration": 0,
                    "n_pings": 0,
                },
            }
        )
        == EMPTY_REPORT,
        "wide_user": rend(
            {
                "users": {
                    "annabelle-k": {"sessions": 2, "total_duration": 3701, "total_acts": 7},
                    "bo": {"sessions": 1, "total_duration": 45, "total_acts": 0},
                },
                "overall": {
                    "n_sessions": 3,
                    "n_users": 2,
                    "total_acts": 7,
                    "max_duration": 3600,
                    "n_pings": 2,
                },
            }
        )
        == (
            "USER         SESS  ACTS  DURATION\n"
            "annabelle-k     2     7      3701\n"
            "bo              1     0        45\n"
            "TOTAL sessions=3 acts=7 pings=2\n"
        ),
        "full_width_duration": rend(
            {
                "users": {"zed": {"sessions": 1, "total_duration": 12345678, "total_acts": 2}},
                "overall": {
                    "n_sessions": 1,
                    "n_users": 1,
                    "total_acts": 2,
                    "max_duration": 12345678,
                    "n_pings": 0,
                },
            }
        )
        == (
            "USER  SESS  ACTS  DURATION\nzed      1     2  12345678\nTOTAL sessions=1 acts=2 pings=0\n"
        ),
        "user_width_boundary": rend(
            {
                "users": {"anna": {"sessions": 1, "total_duration": 0, "total_acts": 0}},
                "overall": {
                    "n_sessions": 1,
                    "n_users": 1,
                    "total_acts": 0,
                    "max_duration": 0,
                    "n_pings": 0,
                },
            }
        )
        == (
            "USER  SESS  ACTS  DURATION\nanna     1     0         0\nTOTAL sessions=1 acts=0 pings=0\n"
        ),
    }

    tiny = tmp / "tiny.log"
    tiny.write_text("100|open|user=ana;src=web\n160|act|user=ana;verb=view\n", "utf-8")
    erry = tmp / "erry.log"
    erry.write_text("100|open|user=ana;src=web\n12|nope|\n160|act|user=ana;verb=view\n", "utf-8")
    rejy = tmp / "rejy.log"
    rejy.write_text("100|open|user=ana\n", "utf-8")
    pingy = tmp / "pingy.log"
    pingy.write_text(
        "0|ping|\n100|open|user=ana;src=web\n160|act|user=ana;verb=view\n50|ping|\n", "utf-8"
    )

    if c is None:
        comps["cli"] = {
            k: False
            for k in (
                "clean",
                "errors_flag",
                "errors_exit",
                "rejects",
                "pings_threaded",
                "usage",
                "unreadable",
            )
        }
        return comps

    comps["cli"] = {
        "clean": _run_cli(c, [str(tiny)]) == (0, TINY_REPORT),
        "errors_flag": _run_cli(c, [str(erry), "--errors"])
        == (2, TINY_REPORT + "ERR 2 12|nope|\n"),
        "errors_exit": _run_cli(c, [str(erry)]) == (2, TINY_REPORT),
        "rejects": _run_cli(c, [str(rejy), "--errors"])
        == (2, EMPTY_REPORT + "REJ 0 missing:src\n"),
        "pings_threaded": _run_cli(c, [str(pingy)])
        == (
            0,
            "USER  SESS  ACTS  DURATION\nana      1     1        60\nTOTAL sessions=1 acts=1 pings=2\n",
        ),
        "usage": _run_cli(c, []) == (2, "usage: cli.py LOGFILE [--errors]\n"),
        "unreadable": _cli_unreadable(c, tmp),
    }
    return comps


def _cli_unreadable(c: Any, tmp: Path) -> bool:
    rc, out = _run_cli(c, [str(tmp / "absent.log")])
    return rc == 3 and out.startswith("cannot read: ")


def grade(worktree: str, leg: str = "main") -> dict[str, Any]:
    mods = _load_modules(worktree)
    with tempfile.TemporaryDirectory() as td:
        components = build_components(mods, Path(td))

    results: dict[str, dict[str, int]] = {}
    component_scores: dict[str, float] = {}
    cases_passed = 0
    cases_total = 0
    components_passed = 0
    for name, cases in components.items():
        p = sum(1 for ok in cases.values() if ok)
        results[name] = {"passed": p, "total": len(cases)}
        component_scores[name] = round(p / len(cases), 4) if cases else 0.0
        cases_passed += p
        cases_total += len(cases)
        if p == len(cases):
            components_passed += 1

    return {
        "task": "relay",
        "leg": leg,
        "cases_passed": cases_passed,
        "cases_total": cases_total,
        "score": round(cases_passed / cases_total, 4) if cases_total else 0.0,
        "components": results,
        "component_scores": component_scores,
        "components_passed": components_passed,
        "components_total": len(components),
    }


if __name__ == "__main__":
    wt = sys.argv[1] if len(sys.argv) > 1 else "."
    print(json.dumps(grade(wt, sys.argv[2] if len(sys.argv) > 2 else "main")))
