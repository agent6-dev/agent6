# Stage 5: report - render the summary

Fifth of six stage specs. Input: the summary dict stage 4 built. Output:
the fixed-width text report operations reads. The format is exact to the
character; the acceptance battery compares whole strings.

## Layout

Four columns: USER, SESS, ACTS, DURATION.

- The USER column is left-justified with width `W = max(4, longest user
  name in the summary)` (4 is the header word's own length).
- SESS and ACTS are right-justified, width 4 each. DURATION is
  right-justified, width 8.
- Every line joins its four cells with TWO spaces between columns.
- The header line is the four column names in those widths (so with short
  users it reads `USER  SESS  ACTS  DURATION`).
- One row per user, users sorted ascending. Cells carry the user's
  `sessions`, `total_acts`, and `total_duration` as plain decimal ints.
- The last line is `TOTAL sessions=<n_sessions> acts=<total_acts>
  pings=<n_pings>` - no alignment, single spaces, values straight from
  `overall`.
- Lines join with `\n` and the report ends WITH a trailing newline.

## The function

`render(summary: dict) -> str`. No I/O, no truncation, no locale anything.
With no users the report is just the header line and the TOTAL line.

## Worked example

users = {ana: {sessions 1, total_acts 1, total_duration 60}}, overall =
{n_sessions 1, n_users 1, total_acts 1, max_duration 60, n_pings 0}:

    USER  SESS  ACTS  DURATION
    ana      1     1        60
    TOTAL sessions=1 acts=1 pings=0

(and the string ends with a newline after the TOTAL line). With a user
named `annabelle-k` the USER column widens to 11, so the header becomes
`USER         SESS  ACTS  DURATION` and every row pads to match.
