use ratelimit::{parse_rate, Rate, RateError, TokenBucket};

#[test]
fn parse_valid() {
    assert_eq!(parse_rate("10/5s").unwrap(), Rate { events: 10, per_ms: 5000 });
    assert_eq!(parse_rate("3/250ms").unwrap(), Rate { events: 3, per_ms: 250 });
    assert_eq!(parse_rate("60/1m").unwrap(), Rate { events: 60, per_ms: 60_000 });
}

#[test]
fn parse_bad_format() {
    for s in ["", "10", "10/", "/5s", "10/5", "10/5h", "a/5s", "10/bs", "1/2/3s", "10 /5s"] {
        assert_eq!(parse_rate(s), Err(RateError::BadFormat), "input {s:?}");
    }
}

#[test]
fn parse_zero_events_and_window() {
    assert_eq!(parse_rate("0/5s"), Err(RateError::ZeroEvents));
    assert_eq!(parse_rate("10/0s"), Err(RateError::ZeroWindow));
    // BadFormat outranks the zero checks when both apply.
    assert_eq!(parse_rate("0/0h"), Err(RateError::BadFormat));
}

#[test]
fn bucket_starts_full_and_spends() {
    let rate = parse_rate("1/1s").unwrap();
    let mut b = TokenBucket::new(rate, 2);
    assert!(b.allow(0));
    assert!(b.allow(0));
    assert!(!b.allow(0), "burst exhausted at t=0");
}

#[test]
fn bucket_refills_continuously() {
    let rate = parse_rate("1/100ms").unwrap();
    let mut b = TokenBucket::new(rate, 1);
    assert!(b.allow(0));
    assert!(!b.allow(50), "half a token is not a token");
    assert!(!b.allow(99));
    assert!(b.allow(100), "one full window refills one token");
    assert!(!b.allow(100), "and only one");
}

#[test]
fn bucket_fractional_progress_accumulates() {
    let rate = parse_rate("3/1000ms").unwrap();
    let mut b = TokenBucket::new(rate, 1);
    assert!(b.allow(0));
    // 3 tokens per 1000ms = 1 token per 333.33ms; at 333 not yet.
    assert!(!b.allow(333));
    assert!(b.allow(334), "fractional refill progress must not be dropped");
}

#[test]
fn bucket_caps_at_burst() {
    let rate = parse_rate("10/100ms").unwrap();
    let mut b = TokenBucket::new(rate, 3);
    assert!(b.allow(0));
    // Long idle: refill far more than burst; only 3 tokens available.
    assert!(b.allow(10_000));
    assert!(b.allow(10_000));
    assert!(b.allow(10_000));
    assert!(!b.allow(10_000), "capped at burst");
}

#[test]
fn bucket_same_timestamp_no_extra_refill() {
    let rate = parse_rate("5/100ms").unwrap();
    let mut b = TokenBucket::new(rate, 1);
    assert!(b.allow(0));
    for _ in 0..10 {
        assert!(!b.allow(10), "t=10 only ever refills 0.5 tokens total");
    }
}
