# shapes -- find and fix the bugs

`shapes.py` already contains full implementations of the seven functions below.
Several of them have bugs: the implementation looks complete but returns the
wrong answer on some inputs. Find every bug and fix it so the test suite
(`./verify.sh`, which runs `python3 -m unittest`) passes.

Do not stop after the first bug or two. A function can pass the obvious cases
and still be wrong on an edge case; check each one against the spec below. Every
function is pure (no I/O) and uses only the Python standard library. Keep the
signatures unchanged.

## Components

1. `clamp(x, lo, hi)`
   Return `x` constrained to the inclusive range `[lo, hi]`: `lo` if `x < lo`,
   `hi` if `x > hi`, otherwise `x`. Examples: `clamp(5, 0, 10)` is `5`,
   `clamp(-5, 0, 10)` is `0`, `clamp(15, 0, 10)` is `10`. A value on either
   bound returns that bound. Ranges may be negative: `clamp(-20, -10, -1)` is
   `-10`.

2. `mean(xs: list[float]) -> float`
   Arithmetic mean of `xs` as a float. `mean([1.0, 2.0])` is `1.5`. The empty
   list returns `0.0`. A non-integer mean must not be floored:
   `mean([10.0, 5.0])` is `7.5`, not `7.0`.

3. `median(xs: list[float]) -> float`
   Median of `xs`. Sort the values first. Odd length returns the middle value;
   even length averages the two middle values. `median([3.0, 1.0, 2.0])` is
   `2.0`, `median([4.0, 1.0, 2.0, 3.0])` is `2.5`. The empty list returns `0.0`.
   The input may be unsorted.

4. `gcd(a: int, b: int) -> int`
   Greatest common divisor of `a` and `b`, always non-negative. `gcd(12, 8)` is
   `4`, `gcd(0, 5)` is `5`, `gcd(0, 0)` is `0`. Negative inputs give the
   non-negative gcd: `gcd(12, -8)` is `4`, `gcd(-12, -8)` is `4`.

5. `is_prime(n: int) -> bool`
   `True` if `n` is a prime number. Numbers below `2` are not prime.
   `is_prime(2)` is `True`, `is_prime(13)` is `True`, `is_prime(1)` is `False`.
   A perfect square of a prime is not prime: `is_prime(4)`, `is_prime(9)`, and
   `is_prime(25)` are all `False`.

6. `roman(n: int) -> str`
   Roman numeral for an integer in `1..3999`. Every subtractive form must be
   used: `IV` (4), `IX` (9), `XL` (40), `XC` (90), `CD` (400), `CM` (900).
   `roman(4)` is `"IV"`, `roman(400)` is `"CD"` (not `"CCCC"`), `roman(1994)` is
   `"MCMXCIV"`.

7. `running_max(xs: list[int]) -> list[int]`
   Return a list whose element `i` is `max(xs[0..i])`. `running_max([1, 3, 2])`
   is `[1, 3, 3]`. The empty list returns `[]`. It must work with negative
   values: `running_max([-3, -1, -2])` is `[-3, -1, -1]`, and the result always
   has the same length as the input.

## Done when

`./verify.sh` passes (exit 0). Do not edit `test_shapes.py` or `verify.sh`.
