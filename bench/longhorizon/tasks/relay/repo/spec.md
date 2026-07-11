# relay - an event log pipeline

Build the six-stage pipeline that turns a raw relay log into the operations
report. Each stage is one module with one public function, and each stage
consumes exactly what the previous stage produces, so the interface details
in the stage specs are load-bearing: a mismatch anywhere breaks everything
downstream.

| stage | module         | function                              | spec               |
|-------|----------------|---------------------------------------|--------------------|
| 1     | `parse.py`     | `parse_line`, `parse_lines`           | `specs/01-parse.md`|
| 2     | `validate.py`  | `validate(events)`                    | `specs/02-validate.md` |
| 3     | `sessionize.py`| `sessionize(valid_events)`            | `specs/03-sessionize.md` |
| 4     | `metrics.py`   | `summarize(sessions, n_pings=0)`      | `specs/04-metrics.md` |
| 5     | `report.py`    | `render(summary)`                     | `specs/05-report.md` |
| 6     | `cli.py`       | `main(argv) -> int`                   | `specs/06-cli.md`  |

Everything is standard library, pure Python, deterministic. Read all six
stage specs before implementing; spec.md restates none of the contracts.

## Done when

`./verify.sh` passes (exit 0). The starter tests in `test_relay.py` cover a
thin slice of stages 1-2 plus one tiny end-to-end run; the acceptance
battery exercises every stage in isolation and the full pipeline, so
implement each spec completely, not just to the starter tests. Do not modify
`test_relay.py` or `verify.sh`.
