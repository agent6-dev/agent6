// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Eric Lesiuta
//! Versioned, request-correlated supervisor protocol over standard streams.

use std::collections::HashSet;
use std::io::{self, BufRead, Write};
use std::num::NonZeroU64;
use std::path::Path;
use std::time::Duration;

use serde::Deserialize;

use crate::{default_memory_limit_mb, default_timeout, ChildSpec};

pub const PROTOCOL_VERSION: u32 = 1;
pub const FRAME_LIMIT: usize = 1024 * 1024;
// Two 8 MiB captures, each byte escaped to at most six JSON bytes, plus metadata.
const ANSWER_LIMIT: usize = 128 * 1024 * 1024;

/// One command to run against an already-established jail.
// deny_unknown_fields, like Policy: a request naming a field this launcher
// does not know is a version skew between the Python side and this binary, and
// silently dropping it could mean dropping a confinement the caller meant to set.
// Refuse it rather than run a request we only half-understand.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChildRequest {
    id: NonZeroU64,
    argv: Vec<String>,
    #[serde(default)]
    env: Vec<(String, String)>,
    #[serde(default = "default_timeout")]
    timeout_s: f64,
    #[serde(default = "default_memory_limit_mb")]
    memory_limit_mb: u64,
    #[serde(default)]
    checkin_s: f64,
    #[serde(default)]
    log_dir: String,
}

impl ChildRequest {
    fn child_spec(&self) -> ChildSpec<'_> {
        ChildSpec {
            argv: &self.argv,
            env: &self.env,
            memory_limit_mb: self.memory_limit_mb,
            timeout_s: self.timeout_s,
            checkin_s: self.checkin_s,
            log_dir: &self.log_dir,
        }
    }
}

/// What a serving launcher is asked to do. A backgrounded command's pid is
/// namespace-local, so the launcher is the only process that can wait on it or
/// signal it: `status` and `stop` name that pid and it does the work.
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase", deny_unknown_fields)]
enum Request {
    /// Run to completion; answer with the result.
    Run(ChildRequest),
    /// Start the command and answer at once, leaving it running. When stdin closes,
    /// the PID namespace takes it down under `full`, and
    /// `sweep_backgrounded` kills it at the other levels.
    Background(ChildRequest),
    /// Report whether a backgrounded command is still running, reaping it and
    /// answering with its exit code once it is not.
    Status { id: NonZeroU64, pid: i32 },
    /// Kill a backgrounded command's process group, then reap it; in a PID
    /// namespace, also sweep what the command left outside its group.
    Stop { id: NonZeroU64, pid: i32 },
    /// End the wait for this request; it has no separate answer.
    Cancel { id: NonZeroU64 },
}

impl Request {
    fn id(&self) -> u64 {
        match self {
            Self::Run(child) | Self::Background(child) => child.id.get(),
            Self::Status { id, .. } | Self::Stop { id, .. } | Self::Cancel { id } => id.get(),
        }
    }
}

#[derive(Debug, PartialEq)]
enum Progress {
    Complete,
    Pending,
    Eof,
}

/// Keep a partial frame across WouldBlock, with a bound including the newline.
fn read_frame(reader: &mut impl BufRead, line: &mut Vec<u8>) -> io::Result<Progress> {
    loop {
        let bytes = match reader.fill_buf() {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => return Ok(Progress::Pending),
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        if bytes.is_empty() {
            return if line.is_empty() {
                Ok(Progress::Eof)
            } else {
                Err(io::Error::other("EOF in protocol frame"))
            };
        }
        let end = bytes.iter().position(|b| *b == b'\n').map(|i| i + 1);
        let count = end.unwrap_or(bytes.len());
        if line.len() + count > FRAME_LIMIT {
            return Err(io::Error::other("protocol frame exceeds 1048576 bytes"));
        }
        line.extend_from_slice(&bytes[..count]);
        reader.consume(count);
        if end.is_some() {
            return Ok(Progress::Complete);
        }
    }
}

struct Input {
    reader: io::BufReader<io::Stdin>,
    line: Vec<u8>,
    closed: bool,
    error: Option<io::Error>,
}

impl Input {
    fn next(&mut self) -> io::Result<Option<Request>> {
        match read_frame(&mut self.reader, &mut self.line)? {
            Progress::Complete => {
                let request = serde_json::from_slice(&self.line).map_err(crate::io_err)?;
                self.line.clear();
                Ok(Some(request))
            }
            Progress::Eof => {
                self.closed = true;
                Ok(None)
            }
            Progress::Pending => Ok(None),
        }
    }
}

#[derive(PartialEq)]
pub enum Control {
    Continue,
    Cancel,
    Closed,
}

/// Borrow the only stdin reader while a foreground request owns execution.
pub struct Active<'a> {
    input: &'a mut Input,
    id: u64,
}

impl Active<'_> {
    pub fn poll(&mut self) -> Control {
        let mut control = Control::Continue;
        // A stream of stale cancellations must not starve waitid or the timeout.
        for _ in 0..64 {
            match self.input.next() {
                Ok(Some(Request::Cancel { id })) => {
                    if id.get() == self.id {
                        control = Control::Cancel;
                    }
                }
                Ok(None) => break,
                other => {
                    self.input.closed = true;
                    self.input.error = Some(match other {
                        Err(e) => e,
                        _ => io::Error::other("request received while another request is active"),
                    });
                    break;
                }
            }
        }
        if self.input.closed {
            Control::Closed
        } else {
            control
        }
    }
}

pub fn write_answer(answer: &serde_json::Value) -> io::Result<()> {
    let bytes = serde_json::to_vec(answer)?;
    if bytes.len() + 1 > ANSWER_LIMIT {
        return Err(io::Error::other("response frame exceeds 134217728 bytes"));
    }
    let mut out = io::stdout().lock();
    out.write_all(&bytes)?;
    out.write_all(b"\n")?;
    out.flush()
}

/// Only this loop writes answers; command capture never writes the protocol fd.
pub fn serve(cwd: &Path, pid_namespaced: bool) -> ! {
    let mut backgrounded = Vec::new();
    let outcome = serve_requests(cwd, pid_namespaced, &mut backgrounded);
    if !pid_namespaced {
        let pids: Vec<i32> = backgrounded.iter().map(|(pid, _)| *pid).collect();
        crate::sweep_backgrounded(&pids);
    }
    if let Err(e) = outcome {
        crate::die(format!("serve: {e}"));
    }
    std::process::exit(0)
}

fn serve_requests(
    cwd: &Path,
    pid_namespaced: bool,
    backgrounded: &mut Vec<(i32, HashSet<i32>)>,
) -> io::Result<()> {
    // No second descriptor or reader: the active command polls this same stream.
    unsafe {
        let flags = libc::fcntl(0, libc::F_GETFL);
        if flags < 0 || libc::fcntl(0, libc::F_SETFL, flags | libc::O_NONBLOCK) < 0 {
            return Err(io::Error::last_os_error());
        }
    }
    let mut input = Input {
        reader: io::BufReader::new(io::stdin()),
        line: Vec::new(),
        closed: false,
        error: None,
    };
    write_answer(&serde_json::json!({"ready": true, "protocol": PROTOCOL_VERSION}))?;
    let mut last_id = 0;
    loop {
        let Some(request) = input.next()? else {
            if input.closed {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(20));
            continue;
        };
        if matches!(request, Request::Cancel { .. }) {
            continue;
        }
        let id = request.id();
        if id <= last_id {
            return Err(io::Error::other("request IDs must increase"));
        }
        last_id = id;
        let before = if pid_namespaced {
            crate::namespace_pids()
        } else {
            HashSet::new()
        };
        let outcome = match request {
            Request::Run(child) => {
                let mut active = Active {
                    input: &mut input,
                    id,
                };
                crate::run_child(&child.child_spec(), cwd, Some(&mut active)).inspect(|answer| {
                    if let Some(pid) = answer.get("pid").and_then(|p| p.as_i64()) {
                        backgrounded.push((pid as i32, before));
                    }
                })
            }
            Request::Background(child) => {
                crate::spawn_detached(&child.child_spec(), cwd).map(|pid| {
                    backgrounded.push((pid, before));
                    serde_json::json!({"pid": pid})
                })
            }
            Request::Status { pid, .. } => Ok(crate::status_result(pid)),
            Request::Stop { pid, .. } => {
                let mut answer = crate::stop_result(pid);
                if pid_namespaced {
                    answer["survivors"] = crate::sweep_escapees_of(pid, backgrounded).into();
                }
                Ok(answer)
            }
            Request::Cancel { .. } => unreachable!(),
        };
        if let Some(e) = input.error.take() {
            return Err(e);
        }
        if input.closed {
            return Ok(());
        }
        // Bad argv is a command failure, not a broken session.
        let mut answer = outcome.unwrap_or_else(|e| {
            serde_json::json!({
                "returncode": 127, "stdout": "", "stderr": format!("child execution failed: {e}"),
                "exec_failed": true,
            })
        });
        answer["id"] = id.into();
        write_answer(&answer)?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_requests_carry_an_id() {
        assert!(
            serde_json::from_str::<Request>(r#"{"kind":"run","id":1,"argv":["true"]}"#).is_ok()
        );
    }

    #[test]
    fn cancellation_carries_only_its_request_id() {
        assert!(serde_json::from_str::<Request>(r#"{"kind":"cancel","id":1}"#).is_ok());
        for raw in [
            r#"{"kind":"cancel"}"#,
            r#"{"kind":"cancel","id":0}"#,
            r#"{"kind":"cancel","id":true}"#,
            r#"{"kind":"cancel","id":1,"argv":["true"]}"#,
        ] {
            assert!(serde_json::from_str::<Request>(raw).is_err(), "{raw}");
        }
    }
}
