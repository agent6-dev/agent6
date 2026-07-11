# Stage 1: parse - relay log lines to events

First of six stage specs. This stage owns the wire format and nothing else:
what a well-formed line looks like, and how a document of lines becomes
events plus errors. Meaning (which kinds need which payload keys) is stage
2's business.

## The wire format

A relay log is plain text, split on newlines. A line that is empty or only
whitespace is skipped entirely. Every other line must be an event line:

    <ts>|<kind>|<payload>

Exactly three fields separated by the pipe character - a line whose pipe
count is not exactly two is malformed. No whitespace tolerance anywhere:
a leading or trailing space is part of the field it touches and makes it
malformed.

- `ts` - one or more ASCII digits, nothing else (leading zeros are fine, a
  sign is not). The event timestamp in epoch seconds.
- `kind` - exactly one of `open`, `act`, `close`, `ping` (lowercase).
- `payload` - zero or more `key=value` pairs joined by `;`. An empty payload
  field is a valid empty payload. A key is one or more of `a-z` and `_`. A
  value is one or more characters and cannot contain `;`, `|`, `=`, or a
  newline (so every pair has exactly one `=`, with text on both sides).
  A key appearing twice in one line is malformed. An empty pair (from `;;`
  or a trailing `;`) is malformed.

## The functions

`parse_line(line: str) -> dict` parses one event line into
`{"ts": int, "kind": str, "payload": dict[str, str]}` and raises
`ValueError` on any malformation. `ts` is converted to `int`; keys and
values stay strings.

`parse_lines(text: str) -> tuple[list, list]` walks the whole document and
returns `(events, errors)`: the parsed events of every well-formed line in
document order, and one `(lineno, line)` tuple per malformed line, also in
order. Line numbers are 1-based and count EVERY line of the document,
including the blank ones that are skipped. `line` is the offending line
exactly as it appeared (without its newline). Blank lines are neither
events nor errors.

## Worked examples

`0|ping|` parses to ts 0, kind ping, empty payload.

`1700000000|open|user=ana;src=web` parses to a two-key payload.

`007|act|user=b_o;verb=see map` is well-formed: leading zeros in ts, an
underscore key, and a space inside a value are all fine.

Malformed, each raising ValueError: `12|nope|` (unknown kind), `-5|ping|`
and ` 12|ping|` (ts not pure digits), `12|ping` (two fields),
`12|act|user=a;user=b` (duplicate key), `12|act|k=v=w` (a second `=`),
`12|act|K=v` (uppercase key), `12|act|k=` (empty value), `12|act|=v`
(empty key), `12|act|a=1;;b=2` (empty pair).
