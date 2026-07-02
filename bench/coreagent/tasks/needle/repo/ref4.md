# Reporting Subsystem Design - Document 4: The Summary Header

## Scope

This is the fourth of five design documents (ref1 through ref5) for the
reporting subsystem's `render` function. It specifies the single summary line
that introduces every report. The other documents cover how individual rows are
formatted, the order rows appear in, which rows are eligible to appear, and how
all the lines are joined into the final string. This document defines only the
content of the leading line and the count it reports.

## Why a header line exists

A bare list of rows is missing a piece of context the reader needs immediately:
how big is this report? A reader who lands on a report wants to know at a glance
whether they are looking at three items or three hundred, both to set
expectations for how long the report is and to sanity-check it against what they
expected the run to produce. A header that states the size up front answers that
question before the reader has scrolled at all.

The header also gives every report a recognizable, uniform first line. Tools and
humans alike can identify the start of a report by that line, and an automated
check can read the declared count from the header and compare it against the
number of row lines that follow, catching a whole class of assembly bugs cheaply.
A self-describing report is easier to validate than one that just dumps rows.

## The content of the header

The header is a single line of fixed shape. It begins with the literal word
`REPORT`, followed by a single space, followed by a parenthesized count, where
the count is the number of items in the report followed by a single space and
the literal word `items`. For a report of three items the header line reads
`REPORT (3 items)`. For a report of one item it reads `REPORT (1 items)`.

The word `items` is always spelled the same way and is not pluralized
conditionally. A report of exactly one item still reads `REPORT (1 items)`, not
`REPORT (1 item)`. Special-casing the singular would add a branch that every
reader and every downstream parser would then have to account for, and the
subsystem deliberately keeps the header shape uniform across every possible
count. The grammatical awkwardness of "1 items" is a deliberate, documented
trade for a header format with no special cases.

## What the count means

The count in the header is not the number of rows the caller passed in. It is
the number of rows that actually appear in the body of the report, that is, the
number of row lines rendered beneath the header. Some of the rows handed to the
renderer may be ineligible and never rendered; the rules for which rows are
eligible live in the companion filtering document and are out of scope here. The
header counts only what survives that filtering and is rendered.

RULE: The first line of the output is `"REPORT ({n} items)"`, where `n` is the
number of rows actually rendered, counted after ineligible rows have been
removed by the filter defined in the companion document.

That sentence is the header contract in full. The rest of this document
illustrates the count so that there is no ambiguity about what is and is not
included in `n`.

## Worked examples

A report whose input yields three rendered rows has the header `REPORT (3
items)`. The three row lines follow it.

A report whose input yields exactly one rendered row has the header `REPORT (1
items)`, demonstrating the uniform, non-pluralized spelling.

A report whose input contains five rows, two of which are ineligible and
therefore not rendered, has a body of three rows and so its header reads
`REPORT (3 items)`. The count reflects the three rows that appear, not the five
that were supplied. This is the most important property of the count: it tracks
the rendered body, not the raw input.

A report whose input yields no rendered rows at all, whether because the input
was empty or because every supplied row was ineligible, has the header `REPORT
(0 items)`. The count is zero and the header still appears; the header is always
present even when there is nothing beneath it.

## Validating the count

Because the header declares the size of the body, it doubles as a built-in
checksum. A reviewer or an automated test can read the integer out of the header
and count the row lines that follow; the two numbers must agree. When they
disagree, one of two bugs is present: either a row was rendered but not counted,
or a row was counted but not rendered. Both are caught for free by anyone who
reads the report, which is precisely why the count is defined as the size of the
rendered body rather than the size of the raw input. Tying the count to the
input would let the body and the header drift apart without any visible signal,
and the whole value of a self-describing header would be lost. Implementers
should therefore compute the count from the very same collection of rows they
are about to render, after every other stage has decided what that collection
is, so that the header can never contradict the body beneath it.

## Edge cases and clarifications

The count is a plain integer with no padding, no thousands separators, and no
decimal point. A report of forty-two rendered rows reads `REPORT (42 items)`.

The header text is exactly `REPORT`, one space, an open parenthesis, the count,
one space, the word `items`, and a close parenthesis. There is no colon, no
trailing punctuation, and no extra whitespace inside the parentheses beyond the
single space between the number and `items`. The letters of `REPORT` and `items`
are in the case shown here.

The header is always the first line of the output and is always present,
including when the count is zero. How the header line is positioned relative to
the rows and how it is joined to them is the assembly document's concern; this
document only fixes the header's text and the meaning of its count. Get the
count from the rendered body, spell the line exactly as shown, and the header
contract is satisfied.
