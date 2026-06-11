package logwindow

import (
	"errors"
	"sort"
	"strings"
	"time"
)

type Entry struct {
	Time    time.Time
	Level   string
	Service string
	Message string
}

func isService(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		ok := (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '-'
		if !ok {
			return false
		}
	}
	return true
}

func ParseLine(s string) (Entry, error) {
	parts := strings.SplitN(s, " ", 3)
	if len(parts) < 3 {
		return Entry{}, errors.New("malformed line")
	}
	t, err := time.Parse(time.RFC3339, parts[0])
	if err != nil {
		return Entry{}, err
	}
	level := parts[1]
	switch level {
	case "DEBUG", "INFO", "WARN", "ERROR":
	default:
		return Entry{}, errors.New("bad level")
	}
	rest := parts[2]
	idx := strings.Index(rest, ": ")
	if idx < 0 {
		return Entry{}, errors.New("missing service separator")
	}
	service := rest[:idx]
	if !isService(service) {
		return Entry{}, errors.New("bad service")
	}
	msg := strings.TrimRight(rest[idx+2:], " \t")
	return Entry{Time: t, Level: level, Service: service, Message: msg}, nil
}

type Window struct {
	entries []Entry
}

func NewWindow() *Window { return &Window{} }

func (w *Window) Add(e Entry) { w.entries = append(w.entries, e) }

func (w *Window) ErrorRate(service string, since time.Time) float64 {
	total, errs := 0, 0
	for _, e := range w.entries {
		if e.Service != service || e.Time.Before(since) {
			continue
		}
		total++
		if e.Level == "ERROR" {
			errs++
		}
	}
	if total == 0 {
		return 0
	}
	return float64(errs) / float64(total)
}

func (w *Window) TopK(k int, since time.Time) []string {
	if k <= 0 {
		return []string{}
	}
	counts := map[string]int{}
	for _, e := range w.entries {
		if !e.Time.Before(since) {
			counts[e.Service]++
		}
	}
	names := make([]string, 0, len(counts))
	for n := range counts {
		names = append(names, n)
	}
	sort.Slice(names, func(i, j int) bool {
		if counts[names[i]] != counts[names[j]] {
			return counts[names[i]] > counts[names[j]]
		}
		return names[i] < names[j]
	})
	if k < len(names) {
		names = names[:k]
	}
	return names
}
