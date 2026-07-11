# SITEREP Stylebook - Rule R06: DATE Precedes Samples

## Scope

Sixth of ten stylebook rules. This document owns exactly one concern: the
position of the first DATE line relative to the first SAMPLE line. It is the
stylebook's only ordering rule. What a DATE value must look like is rule
R03's business; what a SAMPLE value must look like is rule R07's; whether a
DATE exists at all is rule R02's.

## Background and motivation

Sample records inherit their collection date from the report that carries
them. The archive's ingest pass streams a report top to bottom and stamps
each sample with the most recent DATE seen so far - a deliberate one-pass
design, because reports are also diffed and excerpted line by line, and an
excerpt must carry its date above its samples to survive on its own. A
report whose samples appear before any DATE produces unstamped samples, and
unstamped samples have twice forced a re-survey of a site because nobody
could prove when the material came out of the ground. The fix is a
positional guarantee, made at writing time.

## The rule

If the report contains samples, a DATE must exist and the first DATE must
come first.

RULE: If a report contains at least one grammar-valid SAMPLE record line,
then a grammar-valid DATE record line must exist, and the first such DATE
line must appear on an earlier line of the report than the first
grammar-valid SAMPLE line.

## Worked examples

DATE on line 2, samples on lines 5 and 6: satisfied.

First SAMPLE on line 3, first DATE on line 7: violates R06, even though a
DATE exists and is valid - it arrives too late to stamp the sample above it.

Samples present, no DATE line anywhere: violates R06 (and, separately, R02 -
a required tag is missing; the two codes report two different failures of
the same omission).

No SAMPLE lines at all: this rule is vacuously satisfied, whether or not a
DATE exists (R02 still requires one for its own reasons).

## Edge cases and clarifications

Only the FIRST of each kind matters. A second DATE line after the samples is
legal and irrelevant here; a second SAMPLE line before the first DATE is a
violation because that earlier sample is the first sample. Formally: let d
be the line index of the first grammar-valid DATE line and s that of the
first grammar-valid SAMPLE line; the rule requires d < s whenever s exists.

The DATE line's VALUE plays no role here. A report whose first DATE line
reads `DATE: 2024-13-01` (an impossible month, a clear R03 violation) still
satisfies R06 if that line precedes the first sample: position and validity
are separate findings, and the streaming ingest stamps by position. The
converse also holds: a perfectly valid date placed after the samples is an
R06 violation and no R03 violation.

Lines that fail R01's outer grammar count for neither side. `DATE:2024`
(missing separator space) is not a grammar-valid DATE line, so it cannot be
"the first DATE"; `SAMPLE:S0100` likewise is not a grammar-valid SAMPLE
line and does not start the sample region. R01 flags them; this rule looks
through them.

Blank lines and comments are invisible to every rule, including this one;
"earlier line" is simply smaller line index among the lines that survive
classification.

## Interactions

R02 and this rule overlap on one input (samples present, DATE absent) and
deliberately both fire there, as the worked example states. R03 and this
rule never co-fire on account of the same property - one judges where, the
other judges what. R07 constrains sample values and ordering among
themselves and does not care about DATE at all.
