// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Eric Lesiuta
//! Private per-job subreapers. Only unreaped children are signal targets.

use std::fs::File;
use std::io::{self, BufRead, Read, Seek, SeekFrom, Write};
use std::os::fd::{AsRawFd, RawFd};
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::{Duration, Instant};

use landlock::RulesetCreated;
use nix::sys::wait::{waitid, waitpid, Id, WaitPidFlag, WaitStatus};
use nix::unistd::{fork, ForkResult, Pid};
use serde_json::{json, Value};

use crate::{capture, session, ChildSpec};

pub struct Children(File);

impl Children {
    /// Open before Landlock; this descriptor names only this thread's children.
    pub fn own() -> io::Result<Self> {
        if unsafe { libc::prctl(libc::PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self(File::open("/proc/thread-self/children")?))
    }

    fn pids(&mut self) -> io::Result<Vec<i32>> {
        self.0.seek(SeekFrom::Start(0))?;
        let mut text = String::new();
        self.0.read_to_string(&mut text)?;
        text.split_whitespace()
            .map(|pid| pid.parse().map_err(crate::io_err))
            .collect()
    }

    pub fn stop(&mut self) -> io::Result<Vec<i32>> {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let pids = self.pids()?;
            if pids.is_empty() || Instant::now() >= deadline {
                return Ok(pids);
            }
            for pid in pids {
                let pid = Pid::from_raw(pid);
                let held = WaitPidFlag::WEXITED | WaitPidFlag::WNOHANG | WaitPidFlag::WNOWAIT;
                // Holding the child unreaped prevents PID reuse between this check and kill.
                if waitid(Id::Pid(pid), held).is_ok() {
                    unsafe { libc::kill(pid.as_raw(), libc::SIGKILL) };
                    let _ = waitpid(pid, Some(WaitPidFlag::WNOHANG));
                }
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

pub fn restrict(ruleset: &Option<RulesetCreated>, required: bool) -> io::Result<()> {
    if let Some(ruleset) = ruleset {
        let status = ruleset
            .try_clone()?
            .restrict_self()
            .map_err(crate::io_err)?;
        if required && status.ruleset == landlock::RulesetStatus::NotEnforced {
            return Err(io::Error::other(
                "partial isolation requires Landlock; refusing to run unconfined",
            ));
        }
    }
    Ok(())
}

/// Check enforcement before readiness, without restricting the job-spawning supervisor.
pub fn validate(ruleset: &Option<RulesetCreated>, required: bool) -> io::Result<()> {
    match unsafe { fork() }.map_err(crate::io_err)? {
        ForkResult::Child => {
            let result = restrict(ruleset, required);
            if let Err(e) = &result {
                eprintln!("agent6-jail: landlock: {e}");
            }
            std::process::exit(if result.is_ok() { 0 } else { 1 });
        }
        ForkResult::Parent { child } => match waitpid(child, None).map_err(crate::io_err)? {
            WaitStatus::Exited(_, 0) => Ok(()),
            _ => Err(io::Error::other("Landlock enforcement failed")),
        },
    }
}

struct Process {
    pid: Pid,
    channel: io::BufReader<UnixStream>,
}

pub struct Job {
    process: Option<Process>,
    stopped: Option<Value>,
    observed: Option<i32>,
    capture_error: String,
    log: File,
}

impl Job {
    pub fn fd(&self) -> Option<RawFd> {
        self.process
            .as_ref()
            .map(|p| p.channel.get_ref().as_raw_fd())
    }

    fn channel(&mut self) -> io::Result<&mut io::BufReader<UnixStream>> {
        self.process
            .as_mut()
            .map(|p| &mut p.channel)
            .ok_or_else(|| io::Error::other("job supervisor is closed"))
    }

    pub fn spawn(
        spec: &ChildSpec<'_>,
        cwd: &Path,
        background: bool,
        ruleset: &Option<RulesetCreated>,
        required: bool,
        sibling_fds: &[RawFd],
    ) -> io::Result<Self> {
        // Anonymous and CLOEXEC: no command-writable path ever selects the capture inode.
        let log = File::options()
            .read(true)
            .write(true)
            .custom_flags(libc::O_TMPFILE | libc::O_CLOEXEC)
            .mode(0o600)
            .open(crate::probe::temp_dir())?;
        let (parent, child) = UnixStream::pair()?;
        match unsafe { fork() }.map_err(crate::io_err)? {
            ForkResult::Parent { child: pid } => {
                drop(child);
                parent.set_read_timeout(Some(Duration::from_secs(7)))?;
                Ok(Self {
                    process: Some(Process {
                        pid,
                        channel: io::BufReader::new(parent),
                    }),
                    stopped: None,
                    observed: None,
                    capture_error: String::new(),
                    log,
                })
            }
            ForkResult::Child => {
                drop(parent);
                // No sibling or host channel may keep another job alive or expose responses.
                for fd in sibling_fds {
                    unsafe { libc::close(*fd) };
                }
                unsafe {
                    let null = libc::open(c"/dev/null".as_ptr(), libc::O_RDWR);
                    if null < 0 || libc::dup2(null, 0) < 0 || libc::dup2(null, 1) < 0 {
                        std::process::exit(1);
                    }
                    libc::close(null);
                }
                let result = worker(child, spec, cwd, background, ruleset, required, log);
                if let Err(e) = result {
                    eprintln!("agent6-jail: job supervisor: {e}");
                }
                std::process::exit(0);
            }
        }
    }

    pub fn receive(&mut self) -> io::Result<Value> {
        let mut line = Vec::new();
        self.channel()?
            .by_ref()
            .take(session::ANSWER_LIMIT as u64 + 1)
            .read_until(b'\n', &mut line)?;
        if line.last() != Some(&b'\n') || line.len() > session::ANSWER_LIMIT {
            return Err(io::Error::other(
                "job outcome unknown: supervisor channel ended or exceeded its frame limit",
            ));
        }
        let answer: Value = serde_json::from_slice(&line)?;
        if let Some(code) = answer.get("returncode").and_then(Value::as_i64) {
            self.observed = Some(code as i32);
        }
        if let Some(error) = answer.get("capture_error").and_then(Value::as_str) {
            if !error.is_empty() {
                self.capture_error = error.to_owned();
            }
        }
        Ok(answer)
    }

    pub fn initial(&mut self, active: &mut session::Active<'_>) -> io::Result<Value> {
        loop {
            match active.poll() {
                session::Control::Continue => {}
                session::Control::Cancel => self.channel()?.get_mut().write_all(b"c")?,
                session::Control::Closed => self.channel()?.get_mut().write_all(b"s")?,
            }
            let mut fd = libc::pollfd {
                fd: self
                    .fd()
                    .ok_or_else(|| io::Error::other("job supervisor is closed"))?,
                events: libc::POLLIN,
                revents: 0,
            };
            let rc = unsafe { libc::poll(&mut fd, 1, 20) };
            if rc > 0 {
                return self.receive();
            }
            if rc < 0 && io::Error::last_os_error().kind() != io::ErrorKind::Interrupted {
                return Err(io::Error::last_os_error());
            }
        }
    }

    pub fn status(&mut self) -> Value {
        if let Some(stopped) = &self.stopped {
            return json!({"running": false, "returncode": stopped["returncode"], "error": stopped["error"].as_str().unwrap_or(""), "capture_error": stopped["capture_error"].as_str().unwrap_or("")});
        }
        self.exchange(b"p").unwrap_or_else(|e| self.lost(e))
    }

    pub fn stop(&mut self) -> Value {
        if let Some(answer) = &self.stopped {
            return answer.clone();
        }
        let answer = self.exchange(b"s").unwrap_or_else(|e| self.lost(e));
        self.stopped = Some(answer.clone());
        self.process.take();
        answer
    }

    pub fn read(&mut self, offset: u64, limit: usize) -> io::Result<Value> {
        if limit == 0 || limit > 65536 {
            return Err(io::Error::other(
                "read limit must be between 1 and 65536 bytes",
            ));
        }
        let status = self.status();
        let capture_error = ["capture_error", "error"]
            .iter()
            .filter_map(|key| status[*key].as_str())
            .filter(|error| !error.is_empty())
            .collect::<Vec<_>>()
            .join("; ");
        let size = self.log.metadata()?.len();
        if offset > size {
            return Err(io::Error::other("read offset exceeds capture size"));
        }
        let mut data = vec![0; limit.min((size - offset) as usize)];
        let count = self.log.read_at(&mut data, offset)?;
        data.truncate(count);
        Ok(
            json!({"offset": offset, "next_offset": offset + count as u64, "size": size, "data": data,
                "capture_error": capture_error}),
        )
    }

    fn exchange(&mut self, command: &[u8]) -> io::Result<Value> {
        self.channel()?.get_mut().write_all(command)?;
        self.receive()
    }

    fn lost(&self, error: io::Error) -> Value {
        json!({"running": false, "stopped": false, "returncode": self.observed,
            "error": format!("job outcome or cleanup unknown: {error}"),
            "capture_error": self.capture_error})
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        // EOF tells the private subreaper to clean up, even after the session supervisor dies.
        let _ = self.channel.get_mut().shutdown(std::net::Shutdown::Both);
        let deadline = Instant::now() + Duration::from_secs(1);
        while matches!(
            waitpid(self.pid, Some(WaitPidFlag::WNOHANG)),
            Ok(WaitStatus::StillAlive)
        ) {
            if Instant::now() >= deadline {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

fn send(channel: &mut UnixStream, value: &Value) -> io::Result<()> {
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    channel.write_all(&bytes)
}

fn control(channel: &mut UnixStream) -> session::Control {
    let mut byte = [0];
    match channel.read(&mut byte) {
        Ok(1) if byte[0] == b'c' => session::Control::Cancel,
        Err(e) if e.kind() == io::ErrorKind::WouldBlock => session::Control::Continue,
        _ => session::Control::Closed,
    }
}

fn worker(
    mut channel: UnixStream,
    spec: &ChildSpec<'_>,
    cwd: &Path,
    background: bool,
    ruleset: &Option<RulesetCreated>,
    required: bool,
    log: File,
) -> io::Result<()> {
    let mut children = Children::own()?;
    let result = (|| {
        restrict(ruleset, required)?;
        channel.set_nonblocking(true)?;
        let captured = capture::Capture::new(Some(log));
        let mut answer = crate::run_child(spec, cwd, Some(&mut || control(&mut channel)),
            Some(&mut children), &captured, background)
            .unwrap_or_else(|e| json!({"returncode": 127, "stdout": "", "stderr": format!("child execution failed: {e}"), "exec_failed": true}));
        channel.set_nonblocking(false)?;
        let Some(pid) = answer.get("pid").and_then(Value::as_i64) else {
            answer["survivors"] = children.stop()?.into();
            send(&mut channel, &answer)?;
            return Ok(());
        };
        answer.as_object_mut().unwrap().remove("pid");
        send(&mut channel, &answer)?;
        let mut final_status = None;
        loop {
            let mut command = [0];
            if channel.read(&mut command)? == 0 {
                return Ok(());
            }
            if command[0] == b'c' {
                continue;
            }
            if final_status.is_none() {
                let status = crate::status_result(pid as i32);
                if status["running"] == false {
                    final_status = Some(status);
                }
            }
            if command[0] == b's' {
                let status = final_status.unwrap_or_else(|| crate::stop_result(pid as i32));
                let survivors = children.stop()?;
                captured.finish();
                send(
                    &mut channel,
                    &json!({"stopped": survivors.is_empty(), "returncode": status["returncode"], "survivors": survivors, "capture_error": captured.error()}),
                )?;
                return Ok(());
            }
            let mut status = final_status
                .clone()
                .unwrap_or_else(|| json!({"running": true, "returncode": null}));
            status["capture_error"] = captured.error().into();
            send(&mut channel, &status)?;
        }
    })();
    let cleanup = children.stop();
    if let Ok(survivors) = &cleanup {
        if !survivors.is_empty() {
            eprintln!("agent6-jail: owned descendants survived: {survivors:?}");
        }
    }
    result.and(cleanup.map(|_| ()))
}
