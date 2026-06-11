// See README.md for the full spec. Implement so `cargo test` passes.
// Do not modify the tests.

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Rate {
    pub events: u64,
    pub per_ms: u64,
}

#[derive(Debug, PartialEq, Eq)]
pub enum RateError {
    BadFormat,
    ZeroEvents,
    ZeroWindow,
}

pub fn parse_rate(s: &str) -> Result<Rate, RateError> {
    let _ = s;
    todo!()
}

pub struct TokenBucket {
    // add fields as needed
}

impl TokenBucket {
    pub fn new(rate: Rate, burst: u64) -> Self {
        let _ = (rate, burst);
        todo!()
    }

    pub fn allow(&mut self, now_ms: u64) -> bool {
        let _ = now_ms;
        todo!()
    }
}
