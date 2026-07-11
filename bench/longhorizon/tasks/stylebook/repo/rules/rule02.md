# SITEREP Stylebook - Rule R02: Required Tags

## Scope

Second of ten stylebook rules. This document owns exactly one concern: which
tags a report must contain at all. The shape of a record line is rule R01's
business, the meaning and validity of each tag's value belongs to that tag's
own rule (R03 for DATE, R04 for CREW, and so on), and ordering is rule R06's.
This rule is a pure presence check.

## Background and motivation

A SITEREP is only usable if a reader can answer three questions without
leaving the document: when was this observed, where, and by whom. Reports
that omit any of the three have to be chased back to the crew that filed
them, which in practice means the observation is lost - crews rotate out and
memories fade within days. The archive team's triage rule is blunt: a report
missing its anchors is filed as noise. This rule pushes that triage to the
moment of writing, where the author can still fix it.

## The rule

Three tags anchor every report: `DATE` (when), `SITE` (where), and `CREW`
(who). Each of the three must appear on at least one record line of the
report.

Presence is established only by a record line that satisfies rule R01's
grammar. A malformed line does not count: `DATE:2024-03-11` (missing the
space after the colon) violates R01 AND fails to establish that a DATE is
present, so a report whose only DATE line is that one violates both R01 and
R02. The stylebook does not reward a broken line by treating it as half
present.

RULE: At least one grammar-valid record line must exist for each of the tags
DATE, SITE, and CREW.

## Worked examples

A report containing `DATE: 2024-03-11`, `SITE: north-ridge`, and
`CREW: Ana Ruiz, Bo Chen` (in any order, with any other lines around them)
satisfies R02.

A report with DATE and SITE lines but no CREW line violates R02.

A report with no DATE, no SITE, and no CREW violates R02 - once. The audit
result carries the code R02 a single time no matter how many of the three
anchors are missing.

## Edge cases and clarifications

Tags are case-sensitive and exact. `Date: 2024-03-11` does not establish a
DATE: its tag is not `DATE` (and the lowercase letters independently violate
R01's grammar). `DATES: x` is a different tag entirely and establishes
nothing here.

Multiplicity is not this rule's concern. Two DATE lines, five CREW lines: all
legal as far as R02 cares. Each occurrence is still individually subject to
its value rule (every DATE line to R03, every CREW line to R04), but presence
needs only one.

Value quality is not this rule's concern either. `DATE: yesterday` is a
grammar-valid record line with tag DATE, so it DOES establish presence for
R02 - and its nonsense value is flagged by R03, not by this rule. Presence
and validity are deliberately separate failures: a report with a present but
invalid date violates R03 only; a report with no date at all violates R02
(and possibly R06, if samples exist - see that rule).

Comments and blank lines never establish presence; they are ignored before
any rule sees them. A commented-out `# DATE: 2024-03-11` is not a DATE line.

Continuation lines never establish presence: they are not record lines, even
though their text may happen to start with something tag-shaped.

## Interactions

R01 gates this rule: only grammar-valid lines count, as stated above. R03,
R04, and R05 judge the values of lines whose presence this rule counts. R06
adds a position requirement on DATE relative to SAMPLE lines - a report can
satisfy R02 (a DATE exists) and still violate R06 (it arrives after the first
sample). R10's terminator `END:` is not one of the required tags and plays no
role here.
