# SITEREP Stylebook - Rule R07: Sample Ids

## Scope

Seventh of ten stylebook rules. This document owns exactly one concern: the
id that opens every SAMPLE record line's value, and the ordering of those
ids across the report. The position of samples relative to the DATE line is
rule R06's business; the outer line grammar is rule R01's.

## Background and motivation

Every physical sample bag carries a pre-printed label: the letter S and four
digits. Crews log one SAMPLE line per bag, and bags are labeled in the order
they are drawn from the sleeve, so within one report the logged ids must
climb. The two failure modes this rule exists to catch are transcription
slips (a swapped digit turns S0142 into S0124, which reads fine in isolation
and corrupts the chain of custody) and duplicate logging (the same bag
entered twice, which double-counts material at the lab). Both surface
immediately as an ordering fault when ids must be strictly increasing.

## The rule

A SAMPLE value opens with the bag id, optionally followed by a free-text
description. The id token is the part of the value before the first space
(the whole value when it contains no space). A well-formed id is a capital
`S` followed by exactly four digits. Across the whole report, reading top to
bottom, the well-formed ids must be strictly increasing.

RULE: On every grammar-valid SAMPLE record line, the value's first
space-delimited token must match capital S plus exactly four digits; and the
sequence of well-formed ids, in document order, must be strictly increasing.
Both a malformed id and an ordering fault are violations of this rule.

## Worked examples

`SAMPLE: S0100 topsoil` then `SAMPLE: S0142 subsoil` is valid: both ids are
well-formed and 0142 exceeds 0100.

`SAMPLE: S0142` then `SAMPLE: S0100` violates R07: the ids decrease.

`SAMPLE: S0100 topsoil` twice violates R07: strictly increasing means a
repeat is an ordering fault (the same bag cannot be drawn twice).

`SAMPLE: S123 topsoil` violates R07: three digits. `SAMPLE: S12345` also
violates R07: that token is S plus five digits, not a well-formed id
followed by a description - the description only begins after a space.

`SAMPLE: s0100 topsoil` violates R07: the S is case-sensitive.

## Edge cases and clarifications

The description is unconstrained: it may be empty (a bare id is a fine
value), or long, or contain digits and further S-tokens - only the FIRST
token is the id. `SAMPLE: S0100 replaces S0099` is one sample with id S0100.

Since well-formed ids are fixed-width, numeric order and string order agree;
compare them however is convenient.

A malformed id is excluded from the ordering comparison. The increasing
check runs over the well-formed ids only, in the order they appear: a report
logging `S0100`, then the malformed `0100S`, then `S0200` violates R07 for
the malformed id alone - the surviving sequence 0100, 0200 still climbs, and
the broken token does not additionally poison the ordering. One fault, one
finding (and either way, the single code R07 appears once in the result).

Strictness is per report, not per site: two different reports may reuse a
range; within one report the ids climb without ties.

A SAMPLE line that violates R01's outer grammar is R01's finding alone and
takes no part here - it neither contributes an id nor breaks the sequence
(and, as rule R06 also notes, such a line does not count as a sample for
ordering against DATE either).

## Interactions

R06 consumes the position of the first grammar-valid SAMPLE line; this rule
consumes the values. They fire independently: samples before the DATE with
perfect ids is R06 alone; samples after the DATE with a duplicated id is R07
alone.
