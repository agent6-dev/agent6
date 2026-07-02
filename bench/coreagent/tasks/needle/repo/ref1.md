# Reporting Subsystem Design - Document 1: Row Ordering

## Scope

This is the first of five design documents (ref1 through ref5) that together
specify the behavior of the `render` function in the reporting subsystem. This
document covers one concern only: the order in which rows appear in the body of
the report. Value formatting, the header line, input filtering, and final
string assembly are each specified in their own companion document. When this
document refers to "rows", it means the rows that remain to be rendered; the
question of which rows are eligible to appear at all is out of scope here and is
handled elsewhere.

## Background and motivation

The reporting subsystem exists to turn an unordered collection of measurements
into a stable, human-readable digest. Consumers of the report are almost always
scanning from the top looking for the largest contributors. A procurement lead
reading a cost report wants the most expensive line items first. An on-call
engineer reading a latency report wants the slowest endpoints first. A finance
analyst reading a revenue report wants the biggest accounts first. In every one
of these scenarios the reader's attention is a scarce resource and the report's
job is to spend it well by putting the things that matter most at the top.

Early prototypes of the subsystem preserved the caller's insertion order. This
felt simple, but it produced reports whose usefulness depended entirely on
whether the upstream system happened to hand us rows in a sensible sequence. In
practice it never did. Rows arrived in hash-map iteration order, or in the order
a database returned them, or in whatever order a directory walk produced. Two
runs over the same data could emit two different reports. Readers learned not to
trust position, which defeated the entire purpose of producing an ordered
digest in the first place.

The fix is to make ordering a property the renderer guarantees, rather than
something it inherits from its input. The renderer takes responsibility for
sorting so that callers never have to, and so that every reader can rely on the
same meaning for "near the top".

## The ordering policy

The dominant key is the numeric `value` field of each row. Larger values are
more important and must therefore come first. This is a descending sort on
`value`: the row with the greatest value is rendered at the top of the body,
the row with the least value is rendered at the bottom.

Descending order is the right default because the reports this subsystem
produces are overwhelmingly "top contributors" reports. A reader who wants the
smallest values can read from the bottom up, but the common case, the case we
optimize for, is the reader who wants the biggest values and wants them
immediately. We considered making the direction configurable and rejected it:
configuration that is almost never changed is a cost with no offsetting
benefit, and a single fixed direction keeps every report in the system
comparable to every other one.

Ties are common in real data and must be handled deterministically. Two
endpoints can have identical latency to the millisecond. Two accounts can have
identical revenue to the cent. When two rows carry equal `value`, the renderer
must still place them in a fixed, reproducible order, because a report that
shuffles tied rows between runs reintroduces exactly the nondeterminism the sort
was meant to eliminate. The tie-breaker is the row's `name`, compared as text in
ascending order, so that among rows of equal value the one whose name comes
first alphabetically is rendered first.

RULE: Rows must be sorted by `value` in descending order; when two or more rows
have equal `value`, those tied rows are ordered by `name` in ascending
(A-to-Z) order.

That single sentence is the whole of the ordering contract. Everything else in
this document is rationale and worked examples to make the intent unambiguous.

## Worked examples

Consider three rows with values 1, 3, and 2 and names "a", "b", and "c"
respectively. The descending sort on value orders them 3, then 2, then 1, so the
body lists "b" first, then "c", then "a". The names play no part here because no
two values are equal.

Now consider three rows that all share the value 5, with names "zebra", "apple",
and "mango". Because the values tie, the tie-breaker decides everything: the
names in ascending order are "apple", "mango", "zebra", so that is the order in
which the rows appear.

Finally consider a mixed case: names "b" and "a" both with value 2, plus name
"c" with value 10. The descending sort puts "c" first because 10 is the largest
value. The remaining two rows tie at 2, so they are ordered by name ascending,
giving "a" before "b". The final order is "c", "a", "b".

## Edge cases and clarifications

A single row is already sorted; the policy is a no-op for it. An empty body has
nothing to order. Neither case is special; both fall out of the general rule
without extra handling.

Equality is exact equality of the `value` field. Two values that are merely
close are not tied; only values that compare equal trigger the name
tie-breaker. The renderer does not round before comparing, and it does not
bucket nearby values together. Comparison happens on the values as given.

Name comparison is ordinary lexicographic string comparison. The names in the
reports this subsystem produces are short identifiers, and the ascending
ordering of those identifiers is the familiar dictionary ordering. The
tie-breaker exists only to make the output deterministic; it carries no
semantic meaning of its own, and readers should not infer importance from the
alphabetical position of tied rows.

The sort is total: between any two distinct rows the policy always produces a
definite order, because either their values differ (decided by the descending
value comparison) or their values are equal (decided by the ascending name
comparison). There is no input for which the order is left undefined, which is
exactly the determinism guarantee the subsystem promises its readers.

## Interaction with the rest of the pipeline

Ordering happens on the set of rows that are actually going to be rendered. The
companion documents describe how individual values are formatted into text, how
the leading summary line is built, which rows are eligible to appear, and how
all the pieces are joined into the final string. None of those concerns change
the ordering policy stated above, and the ordering policy does not change them.
Keep this concern isolated: sort the rows that will be rendered by value
descending, breaking ties by name ascending, and leave every other concern to
the document that owns it.
