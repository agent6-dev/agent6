# logwindow

Implement the package in `logwindow.go` so that `go test ./...` passes.
Do NOT modify `logwindow_test.go`.

## Spec

ParseLine(s) parses one log line of the exact form:

    <RFC3339 timestamp> <LEVEL> <service>: <message>

- LEVEL is one of DEBUG, INFO, WARN, ERROR (exact, uppercase).
- service is 1+ chars of lowercase letters, digits, or '-'.
- The ": " separator after the service is required even when the
  message is empty.
- The message keeps interior whitespace but trailing whitespace is
  trimmed. Leading whitespace after ": " is kept.
- Any violation returns a non-nil error.

Window collects entries and answers questions over "entries at or after
a cutoff time" (inclusive):

- NewWindow() *Window
- (*Window) Add(e Entry)
- (*Window) ErrorRate(service string, since time.Time) float64
  fraction of that service's entries (>= since) that are level ERROR;
  0 when the service has no entries in range.
- (*Window) TopK(k int, since time.Time) []string
  services ordered by entry count (>= since) descending, count ties
  broken by service name ascending. Fewer than k services means all of
  them; k <= 0 means empty (non-nil) slice.
