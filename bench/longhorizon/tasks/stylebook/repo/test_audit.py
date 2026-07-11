"""Starter tests for the SITEREP auditor.

Necessary, not sufficient: the acceptance battery is a superset that covers
every rule independently and in combination. Do not modify this file or
verify.sh.
"""

import unittest

from audit import audit

VALID = """\
# routine survey
DATE: 2024-03-11
SITE: north-ridge
CREW: Ana Ruiz, Bo Chen
TEMP: -12C
SAMPLE: S0100 topsoil
SAMPLE: S0142 subsoil
NOTE: overcast morning
  wind picked up after noon
END:
"""


class TestAuditStarter(unittest.TestCase):
    def test_valid_report_is_clean(self):
        self.assertEqual(audit(VALID), [])

    def test_missing_separator_space(self):
        self.assertEqual(audit(VALID.replace("TEMP: -12C", "TEMP:-12C")), ["R01"])

    def test_missing_required_tag(self):
        self.assertEqual(audit(VALID.replace("CREW: Ana Ruiz, Bo Chen\n", "")), ["R02"])

    def test_impossible_month(self):
        self.assertEqual(audit(VALID.replace("2024-03-11", "2024-13-01")), ["R03"])

    def test_missing_terminator(self):
        self.assertEqual(audit(VALID.replace("END:\n", "")), ["R10"])


if __name__ == "__main__":
    unittest.main()
