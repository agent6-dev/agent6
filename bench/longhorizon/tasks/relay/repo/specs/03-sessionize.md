# Stage 3: sessionize - group events into user sessions

Third of six stage specs. Input: the VALID events stage 2 accepted. This
stage turns the flat stream into per-user sessions and counts the pings out
of the way.

## Pings

Valid pings carry no user (their payload is empty), so they belong to no
session. This stage counts them and drops them: the count travels onward
(stage 4 puts it in the summary, stage 5 prints it), so do not lose it.

## Sessions

Every open/act/close event names its user in `payload["user"]`. Per user,
order that user's events by `(ts, input position)` - input position breaks
timestamp ties, so two events with equal `ts` keep their input order. Then
split the ordered run into sessions: a gap of MORE than 1800 seconds
between consecutive events starts a new session. A gap of exactly 1800
seconds does not split.

A session is the dict:

    {"user": <user>, "events": [<events in session order>],
     "start": <ts of first event>, "end": <ts of last event>}

## The function

`sessionize(valid_events: list) -> tuple[list, int]` returns
`(sessions, n_pings)`. The sessions list covers ALL users, ordered by
`(start, user)` ascending - start first, the user name as the tiebreak - so
the output is deterministic regardless of how users interleaved in the
input. A session with one event has `start == end`.

## Worked examples

Two ana events at ts 0 and 1800: one session (the boundary gap does not
split), start 0, end 1800. At ts 0 and 1801: two sessions.

Events arriving out of time order sessionize by time, not arrival: ana at
ts 3600 then ana at ts 0 is TWO one-event sessions (0 and 3600 are 3600
apart after sorting), not one.

ana at ts 0, bo at ts 10, ana again at ts 100: two sessions, ana's
(start 0, end 100) first, bo's (start 10) second, because 0 < 10. If two
users' sessions start at the same ts, the alphabetically smaller user
comes first.

Three events where two are pings: one session and `n_pings == 2`.
