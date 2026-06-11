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

fn parse_digits(s: &str) -> Result<u64, RateError> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return Err(RateError::BadFormat);
    }
    s.parse().map_err(|_| RateError::BadFormat)
}

pub fn parse_rate(s: &str) -> Result<Rate, RateError> {
    let mut parts = s.split('/');
    let (Some(ev), Some(win), None) = (parts.next(), parts.next(), parts.next()) else {
        return Err(RateError::BadFormat);
    };
    let (num, mult) = if let Some(p) = win.strip_suffix("ms") {
        (p, 1)
    } else if let Some(p) = win.strip_suffix('s') {
        (p, 1000)
    } else if let Some(p) = win.strip_suffix('m') {
        (p, 60_000)
    } else {
        return Err(RateError::BadFormat);
    };
    let events = parse_digits(ev)?;
    let window = parse_digits(num)?;
    if events == 0 {
        return Err(RateError::ZeroEvents);
    }
    if window == 0 {
        return Err(RateError::ZeroWindow);
    }
    Ok(Rate { events, per_ms: window * mult })
}

pub struct TokenBucket {
    rate: Rate,
    burst: u64,
    tokens: u64,
    // refill progress numerator, in units of 1/per_ms tokens
    frac: u64,
    last_ms: u64,
}

impl TokenBucket {
    pub fn new(rate: Rate, burst: u64) -> Self {
        TokenBucket { rate, burst, tokens: burst, frac: 0, last_ms: 0 }
    }

    pub fn allow(&mut self, now_ms: u64) -> bool {
        let elapsed = now_ms.saturating_sub(self.last_ms);
        self.last_ms = now_ms;
        if self.tokens < self.burst {
            self.frac += elapsed * self.rate.events;
            let whole = self.frac / self.rate.per_ms;
            self.frac %= self.rate.per_ms;
            self.tokens = (self.tokens + whole).min(self.burst);
            if self.tokens == self.burst {
                self.frac = 0;
            }
        }
        if self.tokens > 0 {
            self.tokens -= 1;
            true
        } else {
            false
        }
    }
}
