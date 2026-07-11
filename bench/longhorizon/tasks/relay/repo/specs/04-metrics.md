# Stage 4: metrics - summarize sessions

Fourth of six stage specs. Input: the sessions stage 3 built, plus its ping
count. Output: one summary dict that stage 5 renders verbatim. Numbers
only; no formatting here.

## The function

`summarize(sessions: list, n_pings: int = 0) -> dict` returns:

    {
      "users": {
        <user>: {
          "sessions": <number of sessions>,
          "total_duration": <sum of (end - start) over the user's sessions>,
          "total_acts": <count of kind == "act" events in them>,
        },
        ...
      },
      "overall": {
        "n_sessions": <total session count>,
        "n_users": <distinct users with at least one session>,
        "total_acts": <act count across all sessions>,
        "max_duration": <largest single-session duration, 0 if no sessions>,
        "n_pings": <n_pings, passed through>,
      },
    }

All values are plain ints. A one-event session contributes duration 0. A
user's opens and closes count toward nothing except session membership;
only `act` events are tallied. `n_pings` defaults to 0 so the function is
callable on sessions alone, but the cli threads the real count through -
losing it there is a pipeline bug the report makes visible.

## Worked examples

One ana session, events open at ts 0 and act at ts 60: users maps ana to
sessions 1, total_duration 60, total_acts 1; overall is n_sessions 1,
n_users 1, total_acts 1, max_duration 60, n_pings 0.

Two ana sessions (durations 100 and 40, one act each) aggregate to
sessions 2, total_duration 140, total_acts 2; overall max_duration is 100.

No sessions at all: users is `{}` and overall is all zeros (plus whatever
`n_pings` was passed).
