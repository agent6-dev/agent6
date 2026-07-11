# SITEREP Stylebook - Rule R05: TEMP Form and Range

## Scope

Fifth of ten stylebook rules. This document owns exactly one concern: the
value of a TEMP record line - both its written form and its permitted range.
The outer line grammar is rule R01's business, and no other rule reads
temperatures.

## Background and motivation

Field temperature is logged as a whole number of degrees Celsius. The
downstream climate join is unit-blind and format-strict: it strips one
trailing `C` and parses the rest as an integer, so every deviation the crews
ever invented - a `+` sign on positive readings, a space before the unit,
decimals from a fancier thermometer, octal-looking zero padding - has either
crashed the join or, worse, silently produced a wrong number. Separately,
the sensors the crews carry are only rated between sixty below and sixty
above zero; a reading outside that band is not a remarkable observation, it
is a broken instrument, and the archive wants it rejected at the source.

## The rule

A temperature value is an optional minus sign, then a whole number written
without leading zeros (zero itself is written `0`), then an uppercase `C`
appended directly with no intervening space. The number must lie between -60
and 60 inclusive. Zero carries no sign: `-0C` is not a temperature.

RULE: The value of every grammar-valid TEMP record line must be an optional
`-`, then `0` or a digit 1-9 followed by at most one more digit, then `C` -
and the integer so written must lie in [-60, 60]. Malformed form and
out-of-range value are both violations of this rule.

## Worked examples

`TEMP: -12C` and `TEMP: 7C` and `TEMP: 0C` are valid.

`TEMP: +7C` violates R05: positive readings carry no sign.

`TEMP: 7 C` violates R05: no space before the unit.

`TEMP: 7.5C` violates R05: whole degrees only.

`TEMP: 07C` violates R05: leading zero.

`TEMP: 61C` and `TEMP: -61C` violate R05: outside the rated band. `TEMP: 60C`
and `TEMP: -60C` are the extremes and are valid.

## Edge cases and clarifications

The unit letter is exactly uppercase `C`: `12c`, `12F`, and a bare `12` all
violate this rule. So does any trailing annotation (`12C approx`) - the
value is the temperature and nothing else.

`-0C` violates R05. There is one way to write zero and it is `0C`.

The two-digit cap in the written form is a consequence of the range: any
in-range magnitude fits in two digits, so a third digit (`100C`, `-100C`)
is simultaneously a form and a range failure - one violation either way,
since both belong to this rule. The rule text spells the form out anyway so
that `007C` (in-range magnitude, illegal padding) is unambiguously caught.

TEMP is optional. A report with no TEMP line satisfies this rule vacuously -
R02 does not list TEMP among the required tags. But EVERY grammar-valid TEMP
line that does appear is judged independently, and a report may carry
several (a morning and an afternoon reading); each must stand on its own.

A TEMP line that violates R01's outer grammar (say `TEMP:-12C`, missing the
separator space) is R01's finding alone; this rule only examines
grammar-valid lines.

## Interactions

None beyond the R01 gate above. No other rule reads TEMP values, and
temperatures play no part in ordering, presence, or termination.
