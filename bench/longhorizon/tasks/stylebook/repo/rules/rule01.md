# SITEREP Stylebook - Rule R01: Record Line Grammar

## Scope

This is the first of ten stylebook rules that together define a compliant
SITEREP report. This document owns exactly one concern: the shape of a single
record line. Which tags must appear, what their values mean, where lines may
sit relative to each other, and how a report ends are all owned by companion
rules. When those questions surface here, this document only points at the
owner.

## Background and motivation

SITEREP reports are written by field crews under time pressure and parsed by
downstream tooling that is deliberately dumb. The tooling does not guess: it
splits a record line at one fixed separator and takes what it finds. Every
historical parsing incident traces back to a line that a human could read but
the separator convention did not cover: a missing space after the colon, a
lowercase tag that sorted apart from its uppercase twin, a tag that was
really a sentence. The grammar below is the contract that keeps the dumb
parser and the tired human in agreement.

## The rule

A record line is a tag, a separator, and a value.

The tag is 2 to 8 uppercase ASCII letters (`A` through `Z` only - no digits,
no underscores, no other characters). The separator is a colon followed by
exactly one space. The value begins immediately after that single space: its
first character must not be a space, and the value must contain at least one
non-space character. Characters after the first are unconstrained here, and
trailing spaces at the end of the line are permitted.

RULE: Every record line must be TAG, colon, one space, value - where TAG is
2-8 uppercase ASCII letters and the value starts with a non-space character
and contains at least one non-space character. The single exemption is the
exact terminator line `END:` defined by rule R10, which is the only record
line permitted to carry no value.

## Worked examples

`TEMP: -12C` is well-formed: a 4-letter uppercase tag, colon, one space, and
a value whose first character is `-`.

`TEMP:-12C` violates R01: the space after the colon is missing.

`Temp: -12C` violates R01: the tag contains lowercase letters.

`T: -12C` and `TEMPERATURE: -12C` both violate R01: the tag must be at least
2 and at most 8 letters, and those are 1 and 11.

`TEMP:  -12C` violates R01: the value's first character (the second space)
is a space. The separator is exactly one space, and indentation inside the
value is not tolerated.

## Edge cases and clarifications

A record line whose tag region contains anything but uppercase letters is a
violation: `TAG2: x`, `TA_G: x`, and `TAG : x` (a space before the colon) all
break the grammar. So does a line with a single leading space such as
` TEMP: -12C`: by the classification order in spec.md one leading space does
not make a continuation, so the line is a record line, and a record line must
begin with its tag's first letter at column zero.

A value of only spaces is a violation: `NOTE:    ` (colon, space, then more
spaces) has no non-space character in its value. There is no such thing as an
empty record.

Unknown tags are legal. The grammar constrains a tag's shape, not its
vocabulary: `ZZ: anything` is a perfectly well-formed record line even though
no other rule gives ZZ meaning. Which tags are required is rule R02's
concern; R01 never complains about a tag it does not recognize.

Every record line in the report is held to this grammar independently. One
malformed line among fifty well-formed ones is still a violation, and fifty
malformed lines still produce the single code R01 in the audit result.

## Interactions

The terminator: rule R10 defines the report terminator, the line consisting
of exactly `END:`. That line has no value, which would fail the grammar
above, so R01 exempts the exact four-character line `END:` and defers
entirely to R10 for its handling. The exemption is precise: `END: departed`
is NOT the terminator - it is an ordinary record line with tag `END`, and it
satisfies R01's grammar (whether the END tag may be used that way is R10's
business, not ours). Only the exact line `END:` is exempt.

Continuations: by spec.md's classification precedence a line starting with
two spaces is a continuation, not a record line, so it can never violate R01
regardless of its content. Whether it is validly placed is rule R09's
concern. Note the asymmetry: two leading spaces make a continuation, one
leading space makes a malformed record line.

Blank and comment lines are ignored before classification reaches records,
so they are invisible to this rule.
