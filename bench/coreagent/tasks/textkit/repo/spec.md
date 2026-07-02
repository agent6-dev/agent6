# textkit — a small text-processing library

Implement the five functions in `textkit.py`. Each is independent; the test
suite (`./verify.sh`, which runs `python3 -m unittest`) checks them separately.
All functions are pure (no I/O). Use only the Python standard library.

## Components

1. `normalize_whitespace(s: str) -> str`
   Collapse every run of whitespace (spaces, tabs, newlines) into a single
   space and strip leading/trailing whitespace. `"  a\t b\n\nc  "` -> `"a b c"`.
   The empty / all-whitespace string returns `""`.

2. `word_count(s: str) -> int`
   Number of whitespace-separated words. `word_count("  one  two\tthree ")`
   is `3`. Empty / all-whitespace is `0`.

3. `most_common_words(s: str, n: int) -> list[tuple[str, int]]`
   The `n` most frequent words as `(word, count)` pairs, most frequent first.
   Lowercase every word and strip leading/trailing ASCII punctuation
   (`.,!?;:"'()[]{}`) before counting; drop tokens that are empty after
   stripping. Break count ties alphabetically (ascending). `n <= 0` returns
   `[]`. If fewer than `n` distinct words exist, return all of them.

4. `wrap_text(s: str, width: int) -> list[str]`
   Greedy word wrap. Split on whitespace, then pack words into lines so each
   line's length is `<= width`, joining words with a single space. A word
   longer than `width` goes on its own line (never split a word). Return the
   list of lines; empty input returns `[]`. Assume `width >= 1`.

5. `to_snake_case(name: str) -> str`
   Convert `camelCase`, `PascalCase`, `kebab-case`, and space-separated names
   to `snake_case`. Insert `_` at lower→upper boundaries and before the last
   capital of an acronym run that precedes a lowercase word, replace runs of
   `-`/space with `_`, lowercase the result, and collapse repeated `_`.
   Examples: `"fooBarBaz"` -> `"foo_bar_baz"`, `"HTTPServer"` ->
   `"http_server"`, `"getHTTPResponseCode"` -> `"get_http_response_code"`,
   `"foo-bar baz"` -> `"foo_bar_baz"`, `"already_snake"` -> `"already_snake"`.

## Done when

`./verify.sh` passes (exit 0). Do not edit `test_textkit.py` or `verify.sh`.
