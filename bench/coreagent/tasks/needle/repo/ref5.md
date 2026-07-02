# Reporting Subsystem Design - Document 5: Input Filtering

## Scope

This is the fifth and last of the design documents (ref1 through ref5) for the
reporting subsystem's `render` function. It specifies which of the rows handed
to the renderer are eligible to appear in the report, and which are dropped
before anything else happens. The companion documents define how an eligible
row is formatted, the order eligible rows appear in, the summary header, and how
the final string is assembled. This document defines the gate those rows must
pass through first.

## Why filter at all

The rows that reach the renderer are measurements, and measurements are not
always meaningful. Upstream systems routinely emit sentinel readings to signal
"no data", "not applicable", or "error", and by long convention those sentinels
are encoded as negative numbers in a domain where only non-negative quantities
make sense. A cost cannot be negative. A count cannot be negative. A duration
cannot be negative. When a negative value shows up in one of these fields it is
not a small quantity to be ranked below the others; it is a flag that this
particular measurement should not be treated as a real data point at all.

If the renderer passed these sentinels through, they would pollute the report in
several ways at once. They would occupy rows in the body that a reader has to
scan past. They would distort the reader's sense of how many real items the
report covers. And because they are not real measurements, any decision the
reader makes based on their presence would be based on noise. The cleanest place
to deal with them is at the very entrance to the renderer, by refusing to admit
them, so that every later stage operates only on rows that represent real data.

## The filter

The gate is a sign test on the `value` field. A row whose value is negative does
not belong in the report and is removed before any other processing. It is not
ranked, not formatted, not assembled into the output, and, crucially, not
counted. From the perspective of every other stage of the renderer, a row with a
negative value never existed.

The boundary case is zero, and zero is treated as real data. A value of exactly
zero is a legitimate measurement: zero cost, zero errors, zero elapsed time are
all meaningful things to report, and a reader often specifically wants to see
the items that came in at zero. Zero is not a sentinel and is not negative, so a
row whose value is exactly zero passes the gate and is rendered like any other
eligible row.

RULE: Any row whose `value` is negative is skipped entirely: it is not rendered
and is not counted. A row whose `value` is exactly zero is kept and rendered
like any other eligible row.

That sentence is the filtering contract in full. The rest of this document
nails down the boundary so there is no ambiguity about which side of the line
zero falls on.

## The boundary, stated precisely

The test is strictly on negativity. Values strictly less than zero are removed;
values greater than or equal to zero are kept. Zero itself is on the keep side
of the boundary. An implementer can read the rule as "keep the row when its
value is at least zero" or equivalently as "drop the row when its value is below
zero"; the two phrasings describe the same gate. The single most common mistake
here is to drop zero along with the negatives by testing for a value that is not
positive; that is wrong, because zero must be kept.

A removed row is removed completely. It does not appear as a blank line, it does
not appear with a placeholder value, and it does not reserve a slot in the
ordering. The report reads exactly as it would have read if that row had never
been supplied. This total removal is what lets the header count, defined in its
own document, reflect only the rows a reader actually sees.

## Worked examples

Given a row named "a" with value -1 and a row named "b" with value 2, the row
"a" is dropped and only "b" remains eligible. The report behaves as if "a" had
never been supplied.

Given a row named "a" with value 0 and a row named "b" with value -5, the row
"a" is kept because zero is on the keep side of the boundary, and the row "b" is
dropped because its value is negative. Only "a" remains eligible.

Given two rows both with negative values, both are dropped and no rows remain
eligible. The report has an empty body. How an empty body is presented is the
concern of the assembly and header documents; this document only establishes
that both rows are gone.

Given a row named "keep" with value 0, a row named "drop" with value -0.5, and a
row named "big" with value 9, the rows "keep" and "big" survive and the row
"drop" is removed. Two rows remain eligible.

Given a row named "a" with value 100 and a row named "b" with value -200, the
row "b" is dropped even though its magnitude is large, because the test is on
the sign of the value, not its magnitude. A large negative number is still
negative and is still a sentinel; size does not buy it a place in the report.

## Edge cases and clarifications

The filter looks only at the `value` field. The name plays no part in
eligibility; a row is admitted or rejected purely on the sign of its value.

Filtering happens before ordering, before formatting, and before the count is
taken. Every other stage of the renderer sees only the rows that survived this
gate, which is why a dropped row affects neither the order of the remaining
rows nor the number the header reports. Apply this gate first: keep every row
whose value is zero or greater, drop every row whose value is below zero, and
hand the survivors on to the rest of the pipeline.
