"""Starter suite: a SUBSET of spec.md (the spec is the requirement)."""

import unittest

import config
import parser as parser_mod
import report
import sessions
import stats


def _ev(line):
    return parser_mod.parse_line(line)


class TestParser(unittest.TestCase):
    def test_happy_line(self):
        ev = _ev("2026-01-02T00:00:00|alice|click|3.50")
        self.assertEqual(ev.user, "alice")
        self.assertEqual(ev.value_cents, 350)

    def test_exact_cents(self):
        self.assertEqual(_ev("2026-01-02T00:00:00|bob|buy|0.29").value_cents, 29)

    def test_bad_field_count(self):
        with self.assertRaises(ValueError):
            _ev("2026-01-02T00:00:00|alice|click")


class TestSessions(unittest.TestCase):
    def test_gap_equal_to_limit_stays_one_session(self):
        gap = config.get("session_gap_s")
        evs = [
            _ev("2026-01-02T00:00:00|alice|click|1"),
            _ev(f"2026-01-02T00:{gap // 60:02d}:00|alice|click|1"),
        ]
        self.assertEqual(len(sessions.build_sessions(evs)), 1)

    def test_far_apart_splits(self):
        evs = [
            _ev("2026-01-02T00:00:00|alice|click|1"),
            _ev("2026-01-02T09:00:00|alice|click|1"),
        ]
        self.assertEqual(len(sessions.build_sessions(evs)), 2)


class TestStatsReport(unittest.TestCase):
    def test_single_session_totals(self):
        evs = [
            _ev("2026-01-02T00:00:00|carol|buy|2.00"),
            _ev("2026-01-02T00:01:00|carol|buy|1.00"),
        ]
        ss = sessions.build_sessions(evs)
        self.assertEqual(stats.user_totals(ss), {"carol": 300})

    def test_report_orders_by_total_desc(self):
        lines = report.top_users({"a": 100, "b": 300, "c": 200})
        self.assertEqual(lines[0], "b 3.00")
        self.assertEqual(lines[1], "c 2.00")


if __name__ == "__main__":
    unittest.main()
