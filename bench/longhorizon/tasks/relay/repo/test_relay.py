"""Starter tests for the relay pipeline.

A thin slice of stages 1-2 plus one tiny end-to-end run. Necessary, not
sufficient: the acceptance battery exercises every stage in isolation and
in combination. Do not modify this file or verify.sh.
"""

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from cli import main
from parse import parse_line, parse_lines
from validate import validate


class TestParseStarter(unittest.TestCase):
    def test_parse_line(self):
        self.assertEqual(
            parse_line("1700000000|open|user=ana;src=web"),
            {"ts": 1700000000, "kind": "open", "payload": {"user": "ana", "src": "web"}},
        )

    def test_parse_line_rejects_unknown_kind(self):
        with self.assertRaises(ValueError):
            parse_line("12|nope|")

    def test_parse_lines_collects_errors(self):
        events, errors = parse_lines("0|ping|\n\nbad line\n")
        self.assertEqual(len(events), 1)
        self.assertEqual(errors, [(3, "bad line")])


class TestValidateStarter(unittest.TestCase):
    def test_missing_key_rejects(self):
        events = [{"ts": 5, "kind": "open", "payload": {"user": "ana"}}]
        valid, rejects = validate(events)
        self.assertEqual(valid, [])
        self.assertEqual(rejects, [(0, "missing:src")])


class TestEndToEndStarter(unittest.TestCase):
    def test_tiny_report(self):
        with TemporaryDirectory() as td:
            log = Path(td) / "tiny.log"
            log.write_text("100|open|user=ana;src=web\n160|act|user=ana;verb=view\n", "utf-8")
            out = io.StringIO()
            with redirect_stdout(out):
                code = main([str(log)])
        self.assertEqual(code, 0)
        self.assertEqual(
            out.getvalue(),
            "USER  SESS  ACTS  DURATION\n"
            "ana      1     1        60\n"
            "TOTAL sessions=1 acts=1 pings=0\n",
        )


if __name__ == "__main__":
    unittest.main()
