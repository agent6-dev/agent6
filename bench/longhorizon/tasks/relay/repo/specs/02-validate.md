# Stage 2: validate - which events mean something

Second of six stage specs. Input: the event dicts stage 1 produced. This
stage checks payload SEMANTICS per kind and splits the stream into events
the pipeline will use and rejects it will only report.

## Requirements per kind

| kind  | required payload keys |
|-------|-----------------------|
| open  | `user`, `src`         |
| act   | `user`, `verb`        |
| close | `user`, `reason`      |
| ping  | must have an EMPTY payload |

Extra keys beyond the required ones are allowed on open/act/close - the
relays attach all sorts of diagnostics - with one exception: the key `sys`
is reserved by the transport layer and is forbidden on EVERY kind. An event
carrying `sys` is rejected no matter what else it carries.

## The function

`validate(events: list) -> tuple[list, list]` returns `(valid, rejects)`.
`valid` is the accepted events, unchanged and in input order. `rejects` is
one `(index, code)` per rejected event, in input order, where `index` is
the event's position in the INPUT list (0-based) and `code` is a string:

- `forbidden:sys` - the payload contains the reserved key `sys`. This code
  wins over every other finding on the same event.
- `payload:ping` - a ping whose payload is not empty (and carries no `sys`).
- `missing:<key>` - a non-ping event lacking a required key (and carrying
  no `sys`). When more than one required key is missing, report the
  alphabetically first missing key, e.g. an act with an empty payload is
  `missing:user`, not `missing:verb`.

Exactly one code per rejected event. An event is judged on its own; order
and duplicates across events are no concern of this stage.

## Worked examples

`{"ts": 5, "kind": "open", "payload": {"user": "ana", "src": "web"}}` is
valid; so is the same event with extra keys like `region=eu`.

An act with payload `{"user": "ana"}` rejects as `missing:verb`; an act
with an empty payload rejects as `missing:user` (alphabetical rule).

A ping with payload `{"x": "1"}` rejects as `payload:ping`; a ping with
`{"sys": "1"}` rejects as `forbidden:sys` (precedence).

Input `[ok, bad, ok]` returns the two ok events in `valid` and one reject
`(1, code)` - indexes refer to the input list, not to the valid list.
