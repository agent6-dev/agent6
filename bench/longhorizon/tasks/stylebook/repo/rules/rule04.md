# SITEREP Stylebook - Rule R04: CREW Roster Shape

## Scope

Fourth of ten stylebook rules. This document owns exactly one concern: the
shape of a CREW record line's value - the roster. Whether a CREW line exists
is rule R02's business; the record line's outer grammar is rule R01's. This
rule begins after the value has been isolated.

## Background and motivation

The roster is the report's accountability trail: who stood on the site and
will answer questions about it a year later. Two failure patterns motivated
this rule. First, solo entries - one name - kept appearing on reports from
sites where policy requires a minimum crew of two for safety, and audits
could not tell a genuine solo violation from a lazy author who listed only
the lead. Second, free-form separators (semicolons, "and", bare commas
without spaces) broke the personnel index, which splits rosters mechanically
on one exact separator. The roster is data, not prose, and this rule keeps
it that way.

## The rule

A roster is between two and six names separated by exactly comma-space. Each
name is non-empty, contains no comma, and carries no leading or trailing
space. Names must be pairwise distinct, compared exactly (case-sensitively).

RULE: The value of every grammar-valid CREW record line must split on the
exact separator ", " into 2 to 6 names, each non-empty, comma-free, and free
of leading and trailing spaces, with no two names identical.

## Worked examples

`CREW: Ana Ruiz, Bo Chen` is a valid roster of two.

`CREW: Ana Ruiz` violates R04: a roster of one is below the minimum.

`CREW: Ana, Bo, Cy, Di, Ed, Fay, Gus` violates R04: seven names exceed the
maximum of six.

`CREW: Ana Ruiz,Bo Chen` violates R04: the separator must be comma-space, so
this splits into a single would-be name containing a comma, which is
forbidden.

`CREW: Ana,  Bo` violates R04: splitting on ", " leaves ` Bo` with a leading
space.

`CREW: Ana, Ana` violates R04: duplicate names.

## Edge cases and clarifications

Names may contain internal spaces, hyphens, apostrophes, periods, digits -
anything except commas and except leading or trailing spaces. `Bo Chen-Li`
and `M. Okafor` are single names. The stylebook does not validate that a
name looks like a name; it validates that the roster splits cleanly.

Distinctness is exact: `CREW: ana, Ana` is LEGAL - the comparison is
case-sensitive, and those are two different strings. The stylebook does not
guess whether they are the same person.

A trailing separator (`CREW: Ana, Bo, `) violates R04: the split yields a
final empty name (or, with a single trailing comma and no space, a name with
a trailing comma inside it - either way the roster is malformed). Note the
outer grammar can also be in play: a value may end in spaces per R01, but
what R04 sees is the full value including them, so `Ana, Bo ` has a name
`Bo ` with a trailing space - a violation of this rule.

Two names is the floor because site policy requires a witness; six is the
ceiling because larger parties file a manifest instead of a SITEREP, and a
roster above six means the author picked the wrong form.

EVERY grammar-valid CREW line is judged independently; one malformed roster
among several is a violation. A CREW line that already violates R01's outer
grammar is R01's finding alone - this rule only examines grammar-valid
lines, so one broken line does not cascade into two codes.

## Interactions

R02 requires that at least one CREW line exists; this rule shapes each one
that does. The two fail independently. No other rule reads the roster.
