# rpn — a reverse-Polish-notation calculator

Implement the four components in `rpn.py`. They build on each other: `evaluate`
uses the tokens from `tokenize`, `evaluate_expr` chains both, and
`RPNCalculator` records a history of `evaluate_expr` results. The test suite
(`./verify.sh`, which runs `python3 -m unittest`) checks them separately. All
code is pure (no I/O). Use only the Python standard library.

In reverse Polish notation operands come before their operator, so `3 4 +`
means `3 + 4`. A binary operator pops the top two stack values `a` (pushed
first) and `b` (pushed last) and pushes `a op b`; for `-` and `/` order
matters, so `2 3 -` is `-1` and `10 2 /` is `5`.

## Components

1. `tokenize(expr: str) -> list[str]`
   Split a space-separated RPN expression into string tokens, collapsing runs
   of whitespace. Tokens stay as strings (no parsing here). `"3   4   +"` ->
   `["3", "4", "+"]`. The empty / all-whitespace string returns `[]`.

2. `evaluate(tokens: list[str]) -> float`
   Evaluate the tokens over a stack and return the single result as a `float`.
   Numbers parse with `float()`; the operators are `+ - * /`. `["3", "4", "+"]`
   -> `7.0`, `["5", "1", "2", "+", "4", "*", "+", "3", "-"]` -> `14.0`,
   `["2", "3", "-"]` -> `-1.0`. Raise `ValueError` (any message) on:
   - division by zero, e.g. `["1", "0", "/"]`;
   - an unknown token that is neither a number nor an operator, e.g.
     `["1", "foo", "+"]`;
   - stack underflow, an operator with fewer than two operands, e.g.
     `["1", "+"]`;
   - leftover operands, more than one value remaining at the end, e.g.
     `["1", "2", "3", "+"]`.

3. `evaluate_expr(expr: str) -> float`
   Convenience wrapper: `tokenize` the string, then `evaluate` the tokens.
   `"10 2 /"` -> `5.0`. Errors from either step propagate.

4. `RPNCalculator`
   A small stateful calculator.
   - `__init__(self)` starts with an empty `history: list[tuple[str, float]]`.
   - `push(self, expr: str) -> float` evaluates `expr` with `evaluate_expr`,
     appends `(expr, result)` to `history`, and returns the result.
   - `last(self) -> float | None` returns the most recent result, or `None`
     when the history is empty.
   - `clear(self) -> None` empties the history.

## Done when

`./verify.sh` passes (exit 0). Do not edit `test_rpn.py` or `verify.sh`.
