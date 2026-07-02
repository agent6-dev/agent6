# Reporting Subsystem Design - Document 2: Per-Row Value Formatting

## Scope

This is the second of five design documents (ref1 through ref5) for the
reporting subsystem's `render` function. It specifies exactly one thing: how a
single row is turned into a single line of text. Ordering, filtering, the
summary header, and the assembly of lines into the final report are each
covered by a companion document and are out of scope here. This document is
concerned only with the shape of one rendered row.

## Why row formatting deserves its own contract

A report is read by humans, and humans read numbers badly when the numbers are
formatted inconsistently. A column of figures where one entry shows "3", the
next shows "3.1", and the next shows "3.14159" is genuinely hard to scan: the
decimal points do not line up, the eye cannot compare magnitudes at a glance,
and trailing digits of spurious precision make small numbers look more exact
than they are. Inconsistent formatting also breaks naive downstream tooling.
People paste these reports into spreadsheets, grep them, and diff them between
runs. Every one of those uses is easier when each row has a single, predictable
textual shape.

The raw `value` on a row is a floating-point number, and floating-point numbers
have no natural textual form. The same quantity can print as "3", "3.0",
"3.00", or "2.9999999999999996" depending on how it was computed and how it is
converted to text. If the renderer simply interpolated the raw float, the
report would inherit all of that noise. So the renderer takes ownership of
presentation: it decides how many digits a value shows, and it shows that many
for every row, every time.

## The line format

Each eligible row becomes exactly one line. That line names the row and then
shows its value, separated by a colon and a single space. Concretely, the line
is the row's `name`, then a colon, then one space, then the formatted value. If
a row has name "alpha" and a value that formats to "1.50", its line is
`alpha: 1.50`. There is no leading text before the name, no trailing text after
the value, and no padding or alignment beyond the single space after the colon.

The interesting part is the value. We fix the precision so that every value
shows the same number of fractional digits regardless of its magnitude. Two
fractional digits is the chosen precision: it is enough to express cents,
percentages to a hundredth, and latencies to a hundredth of a unit, which
covers the overwhelming majority of what this subsystem reports, while staying
compact enough that a column of values reads cleanly. Fewer digits would lose
meaningful precision for currency; more digits would add visual noise without
adding information the readers use.

RULE: Each row is rendered as the string `"{name}: {value}"` where the value is
formatted to exactly two decimal places, so 3 renders as `3.00`, 3.1 renders as
`3.10`, and 3.14159 renders as `3.14`.

That sentence is the entire formatting contract for a row. The rest of this
document explains what "exactly two decimal places" means at the boundaries so
that no implementer has to guess.

## What "exactly two decimal places" means

Exactly two means always two: not "up to two", not "two unless the value is a
whole number". A whole number still shows two zeros after the decimal point, so
3 becomes `3.00` and 42 becomes `42.00`. This is the property that makes a
column of values line up, and it is the most common place an implementation
goes wrong by accidentally trimming trailing zeros.

A value that already has fewer than two fractional digits is padded out to two.
A value of 3.1 has one fractional digit and renders as `3.10`. A value of 2.5
renders as `2.50`. The trailing zero is significant for presentation even
though it is not significant arithmetically.

A value that has more than two fractional digits is rounded to two. The value
3.14159 renders as `3.14`. Rounding follows the platform's standard
round-half-to-even behavior at the second decimal place, which is the same
rounding the language's default fixed-precision formatting uses; implementers
should rely on that standard formatting rather than rolling their own rounding,
because hand-rolled rounding tends to disagree with the standard library at the
exact half-way cases and produces reports that do not match expectations. A
value like 1.999 rounds up at the second place and renders as `2.00`.

## Worked examples

A row named "a" with value 3.0 renders as `a: 3.00`. The whole number gains two
zeros.

A row named "a" with value 3.14159 renders as `a: 3.14`. The excess precision
is rounded away at the second decimal.

A row named "a" with value 3.1 renders as `a: 3.10`. The single fractional
digit is padded with a trailing zero.

A row named "a" with value 1.999 renders as `a: 2.00`. Rounding at the second
decimal carries into the integer part.

A row named "a" with value 42 renders as `a: 42.00`. Larger magnitudes are
formatted the same way; precision is about fractional digits, not total width.

## Edge cases and clarifications

The separator is exactly one colon followed by exactly one space. It is not a
tab, not multiple spaces, and not a colon without a space. Downstream tools that
split a line on the first ": " depend on this being uniform, so the renderer
must not vary it.

The name is emitted verbatim. The renderer does not trim, pad, quote, escape,
or case-fold the name; whatever text the name holds is what appears before the
colon. Presentation rules in this document govern the value only.

This document says nothing about the order rows appear in, which rows are
eligible, what the header looks like, or how lines are joined, because those are
other documents' concerns. It governs the textual shape of one row and only
that. An implementer who formats every eligible row as name, colon, space, and
the value to exactly two decimal places has satisfied this contract completely.
