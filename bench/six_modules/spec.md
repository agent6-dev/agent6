# Six module extensions

Keep the existing functions. Add these functions in module order. Use `ValueError` for the error cases.

1. `pairs.py`: `swap(pair)` returns the two values in reverse order as a tuple. `span(pair)` returns the absolute difference of two integers. Both reject a sequence whose length is not two.
2. `limits.py`: `clamp(n, lo, hi)` returns the nearest integer in the closed interval. `contains(n, lo, hi)` tests membership in that interval. Both reject `lo > hi`.
3. `labels.py`: `slug(s)` lowercases and joins whitespace-separated words with hyphens. `initials(s)` joins the uppercase first letters of those words without a separator. Both reject a string without words.
4. `batches.py`: `groups(items, n)` returns successive lists of up to `n` items. `needed(items, n)` returns their number. Both reject `n < 1`.
5. `clockface.py`: `parts(total)` returns `(hours, minutes)` for a nonnegative integer minute count. `display(total)` returns the same time as `H:MM`, without wrapping at midnight. Both reject negative minutes.
6. `rangeset.py`: `inclusive(lo, hi)` returns a list of every integer from `lo` through `hi`. `width(lo, hi)` returns the number of integers in that list. Both reject `lo > hi`.
