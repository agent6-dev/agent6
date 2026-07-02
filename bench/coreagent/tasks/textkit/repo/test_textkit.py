"""Starter tests for textkit (basics only; runs via ./verify.sh)."""

from __future__ import annotations

import unittest

import textkit as t


class TestNormalizeWhitespace(unittest.TestCase):
    def test_basic(self) -> None:
        self.assertEqual(t.normalize_whitespace("  a\t b\n\nc  "), "a b c")

    def test_empty(self) -> None:
        self.assertEqual(t.normalize_whitespace("   \n\t "), "")


class TestWordCount(unittest.TestCase):
    def test_basic(self) -> None:
        self.assertEqual(t.word_count("  one  two\tthree "), 3)

    def test_empty(self) -> None:
        self.assertEqual(t.word_count("  \n "), 0)


class TestMostCommonWords(unittest.TestCase):
    def test_basic(self) -> None:
        text = "the cat, the dog. The bird?"
        self.assertEqual(t.most_common_words(text, 2), [("the", 3), ("bird", 1)])

    def test_nonpositive(self) -> None:
        self.assertEqual(t.most_common_words("a a b", 0), [])


class TestWrapText(unittest.TestCase):
    def test_basic(self) -> None:
        self.assertEqual(t.wrap_text("the quick brown fox", 9), ["the quick", "brown fox"])

    def test_empty(self) -> None:
        self.assertEqual(t.wrap_text("   ", 5), [])


class TestToSnakeCase(unittest.TestCase):
    def test_camel(self) -> None:
        self.assertEqual(t.to_snake_case("fooBarBaz"), "foo_bar_baz")

    def test_acronym(self) -> None:
        self.assertEqual(t.to_snake_case("HTTPServer"), "http_server")


if __name__ == "__main__":
    unittest.main()
