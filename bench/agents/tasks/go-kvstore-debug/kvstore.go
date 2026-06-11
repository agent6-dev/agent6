// Package kvstore: in-memory KV store with TTL + LRU eviction.
// The clock is injected for deterministic tests.
package kvstore

import (
	"container/list"
	"time"
)

type entry struct {
	key     string
	value   string
	expires time.Time // zero time means no expiry
}

type Store struct {
	capacity int
	now      func() time.Time
	items    map[string]*list.Element
	order    *list.List // front = most recently used
}

func New(capacity int, now func() time.Time) *Store {
	return &Store{
		capacity: capacity,
		now:      now,
		items:    make(map[string]*list.Element),
		order:    list.New(),
	}
}

func (s *Store) expired(e *entry) bool {
	if e.expires.IsZero() {
		return false
	}
	return s.now().After(e.expires)
}

// pruneExpired removes every expired entry.
func (s *Store) pruneExpired() {
	for el := s.order.Front(); el != nil; {
		next := el.Next()
		e := el.Value.(*entry)
		if s.expired(e) {
			s.order.Remove(el)
			delete(s.items, e.key)
		}
		el = next
	}
}

// Set stores value under key. ttl <= 0 means no expiry. Overwriting a
// key refreshes its TTL and recency. When a new key would exceed
// capacity, expired keys are pruned first, then the least recently
// used key is evicted.
func (s *Store) Set(key, value string, ttl time.Duration) {
	var expires time.Time
	if ttl > 0 {
		expires = s.now().Add(ttl)
	}
	if el, ok := s.items[key]; ok {
		e := el.Value.(*entry)
		e.value = value
		s.order.MoveToFront(el)
		return
	}
	if len(s.items) >= s.capacity {
		s.pruneExpired()
	}
	if len(s.items) >= s.capacity {
		// evict the least recently used entry
		el := s.order.Front()
		if el != nil {
			e := el.Value.(*entry)
			s.order.Remove(el)
			delete(s.items, e.key)
		}
	}
	el := s.order.PushFront(&entry{key: key, value: value, expires: expires})
	s.items[key] = el
}

// Get returns the live value for key. A hit refreshes recency.
func (s *Store) Get(key string) (string, bool) {
	el, ok := s.items[key]
	if !ok {
		return "", false
	}
	e := el.Value.(*entry)
	if s.expired(e) {
		s.order.Remove(el)
		delete(s.items, e.key)
		return "", false
	}
	s.order.MoveToFront(el)
	return e.value, true
}

// Len reports the number of live (non-expired) keys.
func (s *Store) Len() int {
	return len(s.items)
}
