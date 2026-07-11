# SITEREP Stylebook - Rule R08: Note Length

## Scope

Eighth of ten stylebook rules. This document owns exactly one concern: how
long a note may be. What makes a continuation line valid, and what it may
attach to, is rule R09's business - but because a note's full text is spread
across its NOTE line and its continuations, this rule necessarily reads
R09's attachment structure, and the exact division of labor between the two
rules is spelled out below.

## Background and motivation

Notes are the report's only free prose, and prose grows. The report digest
that operations reads every morning renders each note into a fixed-width
summary cell of 120 characters; anything longer is truncated mid-word at
render time, silently. Truncation has already eaten the operative clause of
a warning note once ("...do NOT approach from the" - the digest ended
there). The stylebook's position is that if the digest will cut it, the
author must cut it first, at writing time, where the author can choose which
words to lose.

## The rule

A note's LOGICAL TEXT is the NOTE record line's value, extended by each
attached continuation line's text in order, joined with a single space per
continuation. Attachment is syntactic and is defined by rule R09: the
unbroken run of continuation lines physically immediately following the
NOTE line. The logical text must fit the digest cell.

RULE: For every grammar-valid NOTE record line, the logical note text - the
value, then for each attached continuation line a single space plus that
line's text (everything after its two leading spaces, taken as-is) - must
not exceed 120 characters.

## Worked examples

`NOTE: overcast morning` followed by the continuation line
`  wind picked up after noon` has the logical text
`overcast morning wind picked up after noon` - 42 characters, well within
the limit.

A NOTE line whose value alone is 121 characters violates R08 with no
continuations involved.

A NOTE line with a 100-character value and one attached continuation of 25
characters has a logical text of 100 + 1 + 25 = 126 characters: an R08
violation contributed to by the join.

## Edge cases and clarifications

Exactly 120 characters is compliant; 121 is not. The limit is the cell
width, and the cell fits the boundary case.

Counting is raw. The value's characters all count, including any trailing
spaces the author left (R01 permits them). A continuation's text is taken
as-is: if its text itself begins with a space - which is a placement fault
under R09 - the join still inserts its single space and then the text's own
leading space also counts. This rule measures; it never tidies.

Each NOTE line is measured independently. Two notes of 100 characters each
are two compliant notes, not one 201-character violation. A report may carry
any number of NOTE lines (including none - NOTE is not a required tag).

Attachment is exactly R09's syntactic run: continuations separated from the
NOTE by a blank line, a comment, or any record line are NOT attached (they
are R09 placement violations standing alone), and they do NOT count toward
any note's length. A continuation that IS in the run but carries its own
R09 fault (say, text beginning with a space) still attaches and still
counts here, as stated above: R09 reports the placement fault, R08 reports
the length if the total overflows. The two rules can co-fire on one note,
each for its own reason.

A run of continuation lines following anything other than a grammar-valid
NOTE line belongs to no note: there is nothing for this rule to measure, and
the placement fault is R09's alone.

## Interactions

R09 defines continuations and attachment; this rule consumes that structure
and adds only arithmetic. R01 governs the NOTE line's outer grammar (a NOTE
line failing R01 is not grammar-valid, so no logical text is computed for
it). No other rule reads notes.
