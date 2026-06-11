// Package logwindow: see README.md for the full spec.
// Implement everything below so `go test ./...` passes. Do not modify the tests.
package logwindow

import "time"

type Entry struct {
	Time    time.Time
	Level   string
	Service string
	Message string
}

func ParseLine(s string) (Entry, error) {
	panic("unimplemented")
}

type Window struct {
	// add fields as needed
}

func NewWindow() *Window {
	panic("unimplemented")
}

func (w *Window) Add(e Entry) {
	panic("unimplemented")
}

func (w *Window) ErrorRate(service string, since time.Time) float64 {
	panic("unimplemented")
}

func (w *Window) TopK(k int, since time.Time) []string {
	panic("unimplemented")
}
