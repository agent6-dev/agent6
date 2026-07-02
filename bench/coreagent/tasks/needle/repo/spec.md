# needle - a report renderer

Implement one function in `report.py`:

```python
def render(rows: list[dict]) -> str
```

Each row is a dict with two keys: `name` (a `str`) and `value` (a `float`).
`render` returns the whole report as a single string. The function is pure
(no I/O) and uses only the Python standard library.

The precise rules are specified in `ref1.md` through `ref5.md`. Read all five
and implement every rule. spec.md does not restate them. Each ref file states
exactly one rule that the renderer must obey, and getting the output right
requires every rule, so do not skip a file.

## Done when

`./verify.sh` passes (exit 0). Do not edit `test_report.py` or `verify.sh`.
The hidden grader checks each rule independently, so implement all five.
