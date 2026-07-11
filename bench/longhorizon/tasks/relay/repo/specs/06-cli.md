# Stage 6: cli - wire the pipeline

Sixth of six stage specs. This stage owns argument handling, file reading,
and the exit code. It contains no logic of its own: it runs stages 1
through 5 in order and prints what they produce.

## The function

`main(argv: list[str]) -> int`, where `argv` is the arguments AFTER the
program name (call it as `main(sys.argv[1:])`). Accepted forms:

- `main([logfile])`
- `main([logfile, "--errors"])` - the flag comes after the path.

Anything else (no args, unknown flags, extra args, flag before path)
prints exactly `usage: cli.py LOGFILE [--errors]` plus a newline to stdout
and returns 2.

Read the file as UTF-8 text. If it cannot be read, print
`cannot read: <path>` plus a newline and return 3.

Then: `parse_lines` on the text, `validate` on the events, `sessionize` on
the valid events, `summarize` on the sessions AND the ping count (thread it
through; the TOTAL line depends on it), `render` on the summary, and write
the rendered report to stdout exactly as rendered (it already ends with a
newline; add nothing).

With `--errors`, after the report print one line per problem, parse errors
first in document order, then rejects in input order:

    ERR <lineno> <line>
    REJ <index> <code>

using the tuples exactly as stages 1 and 2 produced them.

## Exit code

Return 0 when there were no parse errors and no rejects; otherwise 2. The
exit code reflects the problems whether or not `--errors` printed them.

The module must also run as a script (`python3 cli.py access.log`),
exiting with `main`'s return value; keep that wiring in a
`if __name__ == "__main__":` block.
