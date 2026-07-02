# Reporting Subsystem Design - Document 3: Output Assembly

## Scope

This is the third of five design documents (ref1 through ref5) for the
reporting subsystem's `render` function. It specifies how the finished pieces
of a report are assembled into the single string that `render` returns. The
pieces themselves are defined elsewhere: another document specifies how each row
becomes a line of text, another specifies the order of those lines, another
specifies which rows are eligible, and another specifies the content of the
leading summary line. This document takes those pieces as given and defines only
how they are concatenated into one return value.

## Why assembly is its own concern

It is tempting to treat assembly as trivial: surely you just stick the lines
together. In practice the assembly step is where a surprising number of reports
go wrong, because the boundaries between lines and the boundary at the very end
of the output are easy to get subtly inconsistent. A report that sometimes ends
with a newline and sometimes does not will produce spurious diffs when two runs
are compared, will display a stray blank line in some viewers, and will confuse
tools that count lines. Pinning down assembly precisely is what makes the
report's text stable to the last byte, which is a property the rest of the
subsystem relies on when it is tested and when it is diffed between runs.

The return value of `render` is one string. It is not a list of lines, not a
generator, and not a string with some out-of-band structure. Whatever shape the
data had inside the function, the boundary contract is a single flat string that
the caller can print, write to a file, or compare directly.

## The structure of the output

The output has two parts in a fixed order. First comes the summary line, the
single leading line whose content is defined in the companion document on the
header. Second come the rendered rows, one line each, in the order the ordering
document specifies. The summary line always comes first, before any row line,
and the row lines follow it in their established order.

So the logical sequence of lines is: the header line, then the first row line,
then the second row line, and so on through the last row line. There is exactly
one header line and it is always present, even when there are no row lines to
follow it.

The lines are joined with the newline character. Between any two adjacent lines
there is exactly one newline: one between the header and the first row, and one
between each pair of consecutive rows. The newline is a separator between
lines, not a terminator after each line, and that distinction is the crux of
this document.

RULE: The header line comes first, the rendered row lines follow it, all of
these lines are joined into one string with a single newline (`\n`) between
adjacent lines, and the output ends WITHOUT a trailing newline.

That is the assembly contract in full. The remainder of this document explains
what it implies at the boundaries.

## No trailing newline

The output does not end with a newline. The last character of the returned
string is the last character of the last line, whether that last line is a row
or, when there are no rows, the header itself. This is the single most common
assembly mistake, because many line-oriented idioms append a newline after every
line out of habit. Here the newline lives strictly between lines, so the count
of newline characters in the output is always one fewer than the count of lines.

Concretely, when the report has a header and three row lines, the output
contains four lines and therefore exactly three newline characters: header,
newline, row, newline, row, newline, row, with no newline after that final row.
When the report has a header and a single row line, the output contains two
lines and exactly one newline between them, and it ends at the last character of
that row.

## The empty body

A report can have a header and no row lines at all, when no rows are eligible to
be rendered. In that case the output is just the header line by itself. Because
the newline is a separator between lines and there is only one line, the output
contains no newline characters at all, and it of course still does not end with
a trailing newline. The header stands alone as the entire output.

This is worth stating explicitly because an implementation that builds the
output by writing the header and then appending a newline before looping over
rows would emit a stray trailing newline in the empty case. Treat the header and
the rows as a single flat sequence of lines and join that whole sequence with
newlines; do not append a newline after the header as a separate step.

## Worked examples

With a header line and rows that render to `a: 3.00`, `b: 2.00`, and `c: 1.00`
in that order, the output is the header, then a newline, then `a: 3.00`, then a
newline, then `b: 2.00`, then a newline, then `c: 1.00`, and nothing after the
last row.

With a header line and a single row that renders to `solo: 5.00`, the output is
the header, a single newline, and `solo: 5.00`, ending there.

With a header line and no rows at all, the output is exactly the header line and
nothing else, with no trailing newline.

## Edge cases and clarifications

The separator is the single newline character. It is not a carriage return, not
a carriage-return-newline pair, and not a doubled newline that would insert a
blank line between entries. Exactly one newline sits between each pair of
adjacent lines.

Nothing is inserted before the header or after the final line. There is no
leading blank line, no leading newline, no trailing whitespace, and no trailing
newline. The first character of the output is the first character of the header,
and the last character of the output is the last character of the final line.

This document is silent on what the header says, on how each row is formatted,
on which rows survive to be rendered, and on the order of the rows, because each
of those is owned by another document. Assembly means exactly this: put the
header first, follow it with the rows, join the whole list of lines with single
newlines, and emit no trailing newline.
