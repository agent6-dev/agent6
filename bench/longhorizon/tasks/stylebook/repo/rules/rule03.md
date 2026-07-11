# SITEREP Stylebook - Rule R03: DATE Validity

## Scope

Third of ten stylebook rules. This document owns exactly one concern: what a
DATE record line's value must look like and mean. Whether a DATE line exists
at all is rule R02's business; where it sits relative to sample lines is rule
R06's. This rule judges values only.

## Background and motivation

Dates written by humans drift into every format the human has ever used:
month-first, day-first, two-digit years, "yesterday". The archive sorts and
joins reports by date using plain string comparison, which only works when
every date in the archive is written in one canonical, fixed-width,
zero-padded form. And a syntactically perfect date can still be a lie the
calendar never contained - a February 30th sorts beautifully and means
nothing. Both failures have burned the archive before; both are this rule's
job to stop.

## The rule

The canonical form is ISO calendar format: four digits, hyphen, two digits,
hyphen, two digits (`YYYY-MM-DD`). Zero-padding is mandatory: March is `03`,
never `3`. And the digits must name a date that actually exists on the
Gregorian calendar, honoring month lengths and leap years.

RULE: The value of every grammar-valid DATE record line must match
`YYYY-MM-DD` exactly - four digits, hyphen, two digits, hyphen, two digits,
nothing more - and must denote a real Gregorian calendar date.

## Worked examples

`DATE: 2024-03-11` is valid: canonical form, real date.

`DATE: 2024-3-11` violates R03: the month is not zero-padded.

`DATE: 2024-13-01` violates R03: there is no thirteenth month.

`DATE: 2024-02-30` violates R03: February has never had thirty days.

`DATE: 2023-02-29` violates R03: 2023 is not a leap year. `DATE: 2024-02-29`
is valid: 2024 is.

## Edge cases and clarifications

The whole value must be the date and nothing else. `DATE: 2024-03-11 noon`
violates R03: a trailing annotation is not part of the canonical form, even
though the leading ten characters would pass alone. Likewise a leading
adornment (`DATE: on 2024-03-11`) fails.

Alternative separators fail: `2024/03/11`, `2024.03.11`, and `20240311` are
not the canonical form. So do two-digit years (`24-03-11`) and month names
(`2024-Mar-11`).

Leap years follow the full Gregorian rule: divisible by 4, except centuries,
except centuries divisible by 400. So `2000-02-29` is a real date and
`1900-02-29` is not. The Python standard library's date handling implements
exactly this and is a legitimate way to check.

EVERY grammar-valid DATE line in the report is judged independently. If a
report carries two DATE lines and either value is malformed or impossible,
the report violates R03. There is no "first one wins" here.

A DATE line that violates R01's grammar (say, missing the separator space)
is R01's finding; it is not additionally measured against this rule, because
R03 only examines grammar-valid DATE lines. One broken line should not
cascade into every rule that might have touched it.

## Interactions

R02 counts presence; this rule judges values. The two fail independently: no
DATE line at all is R02 alone (there is nothing for R03 to judge); a present
but impossible DATE is R03 alone.

R06 orders the first DATE line against the first SAMPLE line and explicitly
does not care whether the DATE's value is valid - position and validity are
separate findings. A report whose only DATE line is malformed in value but
correctly placed before the samples violates R03 and satisfies R06.
