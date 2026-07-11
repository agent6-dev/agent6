# SITEREP Stylebook - Rule R09: Continuation Placement

## Scope

Ninth of ten stylebook rules. This document owns exactly one concern:
continuation lines - what they are, where they may stand, and what their
text may look like. It also defines ATTACHMENT, the structure rule R08
consumes when it measures a note's full length. The length limit itself is
R08's business.

## Background and motivation

Only one thing in a SITEREP is allowed to wrap: the free prose of a note.
Everything else is single-line data, and the parsers downstream depend on
that. Continuations exist so a long note can be written readably in the
field, and they are marked by indentation - two spaces - because
indentation survives every transport the reports travel through. The
failure this rule polices is the stray indented line: an indented
temperature, a note continuation separated from its note by a blank line
and thereby orphaned, an indent of three spaces that renders as a one-space
hang in the digest. Each of those has historically been silently mis-parsed
as either data or noise; the stylebook makes them loud instead.

## The rule

A continuation line (per spec.md's classification: any line starting with
two spaces, unless it is blank) continues a note, and nothing else. It must
stand in an unbroken run of continuation lines that begins immediately
below a grammar-valid NOTE record line - physical adjacency, no intervening
lines of any kind. Its text - everything after the first two characters -
must be non-empty and must not itself begin with a space (equivalently: the
line's indentation is exactly two spaces, and something follows).

RULE: Every continuation line must (a) have as its physically preceding
line either a grammar-valid NOTE record line or a continuation line
attached to one, and (b) carry text that is non-empty and does not start
with a space. A line failing either clause violates this rule.

ATTACHMENT (consumed by R08): the continuation lines attached to a NOTE are
exactly the unbroken run of continuation lines immediately following it.
Clause (b) faults do not detach a line from the run; clause (a) is what
membership in the run means.

## Worked examples

`NOTE: overcast morning` followed directly by `  wind picked up after noon`
is a validly placed continuation.

`TEMP: -12C` followed by `  and falling` violates R09: only notes continue.

`NOTE: overcast` then a blank line, then `  wind rising` violates R09: the
blank line breaks the run, so the continuation is orphaned. The same holds
if a comment line intervenes - adjacency is physical, and comments are not
transparent HERE even though every rule otherwise ignores them.

`   triple indent` after a NOTE line violates clause (b): the text after
the first two spaces is ` triple indent`, which begins with a space.

## Edge cases and clarifications

A continuation as the report's very first line violates clause (a): there
is no preceding line at all.

A line consisting only of spaces is BLANK by spec.md's precedence (blank
outranks continuation), so it is ignored entirely - it is never a
continuation, never violates this rule, and (being an intervening line of
another kind) it terminates any run it interrupts.

Continuation text is otherwise free: it may begin with `#`, may look like
`TAG: value`, may contain anything. Classification already decided the line
is a continuation; nothing re-parses its content.

A run may be any length: a note followed by three continuations is one note
with three attached lines, each individually subject to clause (b).

A run following a NOTE line that itself fails R01's grammar attaches to
nothing: the line below a malformed note is a continuation below a
non-grammar-valid record line, which fails clause (a). The author's fix is
the note's grammar; this rule's finding is the stranded indent.

A continuation directly below the terminator `END:` fails clause (a) - the
terminator is not a NOTE - and see rule R10 for the second finding that
situation produces.

## Interactions

R08 consumes the attachment structure defined here and adds the length
arithmetic; the two rules can co-fire on one note, each for its own reason
(R08's file walks through this). R01 never sees continuation lines, by
classification precedence. R10 owns what may follow the terminator.
