# SITEREP Stylebook - Rule R10: Report Terminator

## Scope

Tenth of ten stylebook rules. This document owns exactly one concern: how a
report ends. It defines the terminator line, requires exactly one, and
reserves the END tag for it. The exemption that lets the terminator bypass
the record grammar is noted in rule R01; everything else about it is here.

## Background and motivation

SITEREPs travel as plain text through radio relays, mailbox copies, and
concatenated archive spools, and every one of those transports can truncate.
A report that simply stops is indistinguishable from a report that arrived
whole unless the format carries an explicit end mark. The terminator is that
mark: a reader (human or machine) that does not see it treats the report as
damaged goods. For the mark to mean anything it must be unmistakable and
unique - a report with two end marks is as suspect as a report with none,
because one of them is lying about where the report ends.

## The rule

The terminator is the line consisting of exactly the four characters `END:`
- the END tag, the colon, and nothing after it, not even a space. Every
report carries exactly one, at the very end: after the terminator only blank
lines and comments may follow. And the END tag belongs to the terminator
alone: no other record line may use it.

RULE: A report must contain exactly one line that is exactly `END:`; that
line must be the report's last line other than blanks and comments; and no
grammar-valid record line with the tag END other than the terminator may
appear anywhere in the report.

## Worked examples

A report whose final content line is `END:` (perhaps followed by a blank
line the editor left) satisfies this rule.

A report with no `END:` line violates R10 - it reads as truncated.

`END:` on line 4 and again on line 9 violates R10: two end marks.

`END:` followed two lines later by `TEMP: -12C` violates R10: content after
the mark means the mark lied.

`END: departed at dusk` as the report's last record line violates R10 twice
over in spirit and once in code: it is a record line using the reserved END
tag (forbidden), and the report then has no true terminator (also this
rule). The audit result carries R10 once.

## Edge cases and clarifications

The match is exact. `END: ` (trailing space) is not the terminator - it is
a record line with tag END and an empty value, which is not even a
grammar-valid record line (R01 demands a value with a non-space character),
so it is an R01 finding AND leaves the report unterminated, an R10 finding.
`end:` fails the same way (lowercase tag). ` END:` (leading space) is a
malformed record line, not a terminator.

`END: departed` anywhere in the report - even nowhere near the end - is a
grammar-valid record line wearing the reserved tag, and violates this rule
on that ground alone.

Blank lines and comment lines after the terminator are the ONLY things
allowed after it. A continuation line after the terminator is content, so
it violates this rule (and rule R09's placement clause, separately - two
findings, two codes).

An empty report, or one containing only blanks and comments, has no
terminator and violates R10 (alongside whatever R02 finds missing).

The terminator does not count as a record line for any other rule's
purposes: it establishes no tag presence for R02, and it is exempt from
R01's grammar by that rule's explicit carve-out. It is punctuation, not
data.

## Interactions

R01 carves out the exact terminator from the record grammar and hands it
here; the carve-out covers ONLY the exact five characters. R09 governs the
stray continuation below the terminator jointly with this rule, as above.
The other rules do not stop at the terminator: a line that appears after it
is still judged by every rule that applies to it, each earning its own code,
and it additionally violates this rule by standing there at all.
