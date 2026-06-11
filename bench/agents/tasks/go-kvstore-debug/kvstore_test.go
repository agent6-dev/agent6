package kvstore

import (
	"testing"
	"time"
)

type clock struct{ t time.Time }

func (c *clock) now() time.Time          { return c.t }
func (c *clock) advance(d time.Duration) { c.t = c.t.Add(d) }

func newStore(capacity int) (*Store, *clock) {
	c := &clock{t: time.Unix(1000, 0)}
	return New(capacity, c.now), c
}

func TestBasicSetGet(t *testing.T) {
	s, _ := newStore(2)
	s.Set("a", "1", 0)
	if v, ok := s.Get("a"); !ok || v != "1" {
		t.Fatalf("Get(a) = %q,%v", v, ok)
	}
	if _, ok := s.Get("missing"); ok {
		t.Fatal("missing key reported present")
	}
}

func TestExpiryIsInclusiveAtBoundary(t *testing.T) {
	s, c := newStore(2)
	s.Set("a", "1", 10*time.Second)
	c.advance(10 * time.Second)
	if _, ok := s.Get("a"); ok {
		t.Fatal("key must be expired exactly at insertion+ttl (now >= expiry)")
	}
}

func TestExpiryBeforeBoundary(t *testing.T) {
	s, c := newStore(2)
	s.Set("a", "1", 10*time.Second)
	c.advance(9 * time.Second)
	if _, ok := s.Get("a"); !ok {
		t.Fatal("key expired too early")
	}
}

func TestOverwriteRefreshesTTL(t *testing.T) {
	s, c := newStore(2)
	s.Set("a", "1", 10*time.Second)
	c.advance(8 * time.Second)
	s.Set("a", "2", 10*time.Second) // refresh: now expires at t=18s
	c.advance(8 * time.Second)      // t=16s, still live
	if v, ok := s.Get("a"); !ok || v != "2" {
		t.Fatalf("overwrite must refresh TTL; Get = %q,%v", v, ok)
	}
}

func TestOverwriteCanRemoveExpiry(t *testing.T) {
	s, c := newStore(2)
	s.Set("a", "1", 10*time.Second)
	s.Set("a", "2", 0) // no expiry now
	c.advance(time.Hour)
	if v, ok := s.Get("a"); !ok || v != "2" {
		t.Fatalf("ttl<=0 on overwrite must mean no expiry; Get = %q,%v", v, ok)
	}
}

func TestEvictsLeastRecentlyUsed(t *testing.T) {
	s, _ := newStore(2)
	s.Set("a", "1", 0)
	s.Set("b", "2", 0)
	if _, ok := s.Get("a"); !ok { // a is now most recently used
		t.Fatal("setup Get(a) failed")
	}
	s.Set("c", "3", 0) // must evict b (LRU), not a
	if _, ok := s.Get("a"); !ok {
		t.Fatal("evicted the most recently used key instead of the LRU one")
	}
	if _, ok := s.Get("b"); ok {
		t.Fatal("LRU key b should have been evicted")
	}
	if _, ok := s.Get("c"); !ok {
		t.Fatal("new key c missing")
	}
}

func TestSetRecencyCountsToo(t *testing.T) {
	s, _ := newStore(2)
	s.Set("a", "1", 0)
	s.Set("b", "2", 0)
	s.Set("a", "9", 0) // overwrite makes a most-recently-used
	s.Set("c", "3", 0) // evicts b
	if _, ok := s.Get("a"); !ok {
		t.Fatal("a should survive: overwriting refreshed its recency")
	}
	if _, ok := s.Get("b"); ok {
		t.Fatal("b should have been evicted")
	}
}

func TestExpiredPrunedBeforeLiveEvicted(t *testing.T) {
	s, c := newStore(2)
	s.Set("a", "1", 5*time.Second)
	s.Set("b", "2", 0)
	c.advance(10 * time.Second) // a expired
	s.Set("c", "3", 0)          // prune a; b must survive
	if _, ok := s.Get("b"); !ok {
		t.Fatal("live key b evicted while expired key a was available to prune")
	}
	if _, ok := s.Get("c"); !ok {
		t.Fatal("new key c missing")
	}
}

func TestLenCountsOnlyLiveKeys(t *testing.T) {
	s, c := newStore(3)
	s.Set("a", "1", 5*time.Second)
	s.Set("b", "2", 0)
	if got := s.Len(); got != 2 {
		t.Fatalf("Len = %d, want 2", got)
	}
	c.advance(10 * time.Second)
	if got := s.Len(); got != 1 {
		t.Fatalf("Len = %d, want 1 (expired keys are not live)", got)
	}
}
