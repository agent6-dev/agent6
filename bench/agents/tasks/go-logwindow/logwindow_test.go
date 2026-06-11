package logwindow

import (
	"reflect"
	"testing"
	"time"
)

func ts(t *testing.T, s string) time.Time {
	t.Helper()
	v, err := time.Parse(time.RFC3339, s)
	if err != nil {
		t.Fatalf("bad test timestamp %q: %v", s, err)
	}
	return v
}

func TestParseLineValid(t *testing.T) {
	e, err := ParseLine("2026-01-02T15:04:05Z ERROR auth-svc: token expired  ")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !e.Time.Equal(ts(t, "2026-01-02T15:04:05Z")) {
		t.Errorf("time = %v", e.Time)
	}
	if e.Level != "ERROR" || e.Service != "auth-svc" {
		t.Errorf("level/service = %q/%q", e.Level, e.Service)
	}
	if e.Message != "token expired" {
		t.Errorf("message = %q (trailing whitespace must be trimmed)", e.Message)
	}
}

func TestParseLineKeepsInteriorAndLeadingMessageSpace(t *testing.T) {
	e, err := ParseLine("2026-01-02T15:04:05Z INFO web:  spaced  out")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if e.Message != " spaced  out" {
		t.Errorf("message = %q", e.Message)
	}
}

func TestParseLineEmptyMessage(t *testing.T) {
	e, err := ParseLine("2026-01-02T15:04:05Z WARN db: ")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if e.Message != "" {
		t.Errorf("message = %q", e.Message)
	}
}

func TestParseLineErrors(t *testing.T) {
	bad := []string{
		"",
		"not-a-time ERROR auth: x",
		"2026-01-02T15:04:05Z TRACE auth: x",
		"2026-01-02T15:04:05Z error auth: x",
		"2026-01-02T15:04:05Z ERROR Auth: x",
		"2026-01-02T15:04:05Z ERROR auth_svc: x",
		"2026-01-02T15:04:05Z ERROR auth x",
		"2026-01-02T15:04:05Z ERROR auth:x",
		"2026-01-02T15:04:05Z ERROR : x",
		"2026-01-02T15:04:05Z ERROR",
	}
	for _, s := range bad {
		if _, err := ParseLine(s); err == nil {
			t.Errorf("ParseLine(%q): expected error, got nil", s)
		}
	}
}

func fill(t *testing.T) *Window {
	t.Helper()
	w := NewWindow()
	lines := []string{
		"2026-01-02T15:00:00Z ERROR auth: a",
		"2026-01-02T15:01:00Z INFO auth: b",
		"2026-01-02T15:02:00Z ERROR auth: c",
		"2026-01-02T15:03:00Z INFO web: d",
		"2026-01-02T15:04:00Z WARN web: e",
		"2026-01-02T15:05:00Z INFO db: f",
		"2026-01-02T14:00:00Z ERROR auth: old",
	}
	for _, s := range lines {
		e, err := ParseLine(s)
		if err != nil {
			t.Fatalf("ParseLine(%q): %v", s, err)
		}
		w.Add(e)
	}
	return w
}

func TestErrorRate(t *testing.T) {
	w := fill(t)
	since := ts(t, "2026-01-02T15:00:00Z")
	if got := w.ErrorRate("auth", since); got != 2.0/3.0 {
		t.Errorf("auth rate = %v, want 2/3 (cutoff is inclusive)", got)
	}
	if got := w.ErrorRate("web", since); got != 0.0 {
		t.Errorf("web rate = %v, want 0", got)
	}
	if got := w.ErrorRate("ghost", since); got != 0.0 {
		t.Errorf("missing service rate = %v, want 0 (not NaN)", got)
	}
}

func TestErrorRateOlderCutoffIncludesHistory(t *testing.T) {
	w := fill(t)
	since := ts(t, "2026-01-02T13:00:00Z")
	if got := w.ErrorRate("auth", since); got != 3.0/4.0 {
		t.Errorf("auth rate = %v, want 3/4", got)
	}
}

func TestTopK(t *testing.T) {
	w := fill(t)
	since := ts(t, "2026-01-02T15:00:00Z")
	got := w.TopK(2, since)
	want := []string{"auth", "web"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("TopK(2) = %v, want %v", got, want)
	}
}

func TestTopKTieBreaksLexicographically(t *testing.T) {
	w := fill(t)
	since := ts(t, "2026-01-02T15:03:00Z")
	// web has 2, db has 1, auth has 0 in range.
	got := w.TopK(3, since)
	want := []string{"web", "db"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("TopK(3) = %v, want %v (zero-count services are excluded)", got, want)
	}
	w2 := NewWindow()
	for _, s := range []string{
		"2026-01-02T15:00:00Z INFO bbb: x",
		"2026-01-02T15:00:00Z INFO aaa: x",
		"2026-01-02T15:00:00Z INFO ccc: x",
	} {
		e, _ := ParseLine(s)
		w2.Add(e)
	}
	got2 := w2.TopK(3, ts(t, "2026-01-02T15:00:00Z"))
	want2 := []string{"aaa", "bbb", "ccc"}
	if !reflect.DeepEqual(got2, want2) {
		t.Errorf("tie TopK = %v, want %v", got2, want2)
	}
}

func TestTopKBounds(t *testing.T) {
	w := fill(t)
	since := ts(t, "2026-01-02T15:00:00Z")
	if got := w.TopK(0, since); got == nil || len(got) != 0 {
		t.Errorf("TopK(0) = %v, want empty non-nil", got)
	}
	if got := w.TopK(99, since); len(got) != 3 {
		t.Errorf("TopK(99) returned %d services, want 3", len(got))
	}
}
