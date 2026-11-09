// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Eric Lesiuta
//! Launcher-owned byte capture, separate from command completion.

use std::collections::VecDeque;
use std::fs::File;
use std::io::{Read, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub struct Capture {
    buffer: Arc<Mutex<Buffer>>,
    drained: Arc<AtomicUsize>,
}

impl Capture {
    pub fn new(log: Option<File>) -> Self {
        Self {
            buffer: Arc::new(Mutex::new(Buffer {
                log,
                ..Buffer::default()
            })),
            drained: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn drain(&self, stream: impl Read + Send + 'static, tag: Stream) {
        let sink = Arc::clone(&self.buffer);
        let done = Arc::clone(&self.drained);
        std::thread::spawn(move || {
            read_capped_into(stream, tag, &sink);
            done.fetch_add(1, Ordering::SeqCst);
        });
    }

    pub fn render(&self, stream: Stream) -> String {
        self.buffer
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .render(stream)
    }

    pub fn detach(&self) {
        let mut buffer = self.buffer.lock().unwrap_or_else(|e| e.into_inner());
        buffer.detached = true;
        buffer.head.clear();
        buffer.tail.clear();
    }

    pub fn error(&self) -> String {
        self.buffer
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .error
            .clone()
    }

    pub fn finish(&self) {
        let deadline = Instant::now() + Duration::from_secs(2);
        while self.drained.load(Ordering::SeqCst) < 2 {
            if Instant::now() >= deadline {
                self.buffer.lock().unwrap_or_else(|e| e.into_inner()).error =
                    "output capture ended early: a process still holds its pipe".into();
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

// Foreground excerpts retain a bounded head and tail; the raw log has no total byte cap.
const STREAM_RETAIN_HEAD: usize = 4 * 1024 * 1024;
const STREAM_RETAIN_TAIL: usize = 4 * 1024 * 1024;

/// The source stream for a captured chunk.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Stream {
    Out,
    Err,
}

/// Bounded foreground excerpts and a raw log, serialized by the capture mutex.
#[derive(Default)]
struct Buffer {
    head: Vec<(Stream, Vec<u8>)>,
    head_bytes: usize,
    tail: VecDeque<(Stream, Vec<u8>)>,
    tail_bytes: usize,
    // Per stream, so a flood on stdout does not stamp the cap marker onto a
    // stderr result that lost nothing.
    dropped_out: u64,
    dropped_err: u64,
    log: Option<File>,
    error: String,
    detached: bool,
}

impl Buffer {
    fn push(&mut self, stream: Stream, bytes: &[u8]) {
        if let Some(file) = self.log.as_mut() {
            if let Err(e) = file.write_all(bytes) {
                self.error = format!("output capture failed: {e}");
                self.log = None;
            }
        }
        if self.detached {
            return;
        }
        if self.head_bytes < STREAM_RETAIN_HEAD {
            self.head_bytes += bytes.len();
            self.head.push((stream, bytes.to_vec()));
            return;
        }
        self.tail_bytes += bytes.len();
        self.tail.push_back((stream, bytes.to_vec()));
        // Whole chunks: they are one read each (<=64 KiB), so the cap is coarse
        // by at most one chunk and the tagging stays intact.
        while self.tail_bytes > STREAM_RETAIN_TAIL {
            match self.tail.pop_front() {
                Some((stream, dropped)) => {
                    self.tail_bytes -= dropped.len();
                    match stream {
                        Stream::Out => self.dropped_out += dropped.len() as u64,
                        Stream::Err => self.dropped_err += dropped.len() as u64,
                    }
                }
                None => break,
            }
        }
    }

    fn render(&self, only: Stream) -> String {
        let keep = |s: &Stream| only == *s;
        let mut out: Vec<u8> = Vec::new();
        for (stream, bytes) in &self.head {
            if keep(stream) {
                out.extend_from_slice(bytes);
            }
        }
        let dropped = match only {
            Stream::Out => self.dropped_out,
            Stream::Err => self.dropped_err,
        };
        if dropped > 0 {
            out.extend_from_slice(
                format!(
                    "\n[agent6-jail] output over the retained cap; {} bytes omitted here\n",
                    dropped
                )
                .as_bytes(),
            );
        }
        for (stream, bytes) in &self.tail {
            if keep(stream) {
                out.extend_from_slice(bytes);
            }
        }
        String::from_utf8_lossy(&out).into_owned()
    }
}

/// Drain *stream* into *sink* until EOF. Runs on its own thread, which may
/// outlive the command: a grandchild outside its process group keeps the write
/// end open, and nothing can make that read return.
fn read_capped_into(mut stream: impl Read, tag: Stream, sink: &Mutex<Buffer>) {
    let mut chunk = [0u8; 64 * 1024];
    loop {
        match stream.read(&mut chunk) {
            Ok(0) => break,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => {
                sink.lock().unwrap_or_else(|e| e.into_inner()).error =
                    format!("output capture failed: {e}");
                break;
            }
            Ok(n) => sink
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(tag, &chunk[..n]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Drain to EOF and render, the shape these retention tests pin. The
    /// launcher itself never waits for EOF (see the teardown in run_child).
    fn read_capped(stream: impl Read) -> String {
        let sink = Mutex::new(Buffer::default());
        read_capped_into(stream, Stream::Out, &sink);
        let rendered = sink
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .render(Stream::Out);
        rendered
    }

    #[test]
    fn capture_read_failure_is_not_silent_eof() {
        struct Broken;
        impl Read for Broken {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("capture pipe failed"))
            }
        }
        let sink = Mutex::new(Buffer::default());
        read_capped_into(Broken, Stream::Out, &sink);
        assert!(sink.lock().unwrap().error.contains("capture pipe failed"));
    }

    #[test]
    fn read_capped_passes_small_output_through_verbatim() {
        let data = b"hello \xe9 world\n"; // non-UTF-8 byte survives lossily
        let out = read_capped(&data[..]);
        assert_eq!(out, String::from_utf8_lossy(data));
        assert!(!out.contains("omitted"));
    }

    #[test]
    fn read_capped_keeps_head_and_tail_and_marks_the_drop() {
        // 3x the total cap: the head keeps the start, the tail keeps the end,
        // and the marker names how many bytes fell in between.
        let total = 3 * (STREAM_RETAIN_HEAD + STREAM_RETAIN_TAIL);
        let mut data = vec![b'x'; total];
        data[..5].copy_from_slice(b"START");
        let end = data.len();
        data[end - 3..].copy_from_slice(b"END");
        let out = read_capped(&data[..]);
        assert!(out.starts_with("START"));
        assert!(out.ends_with("END"));
        let dropped = total - STREAM_RETAIN_HEAD - STREAM_RETAIN_TAIL;
        assert!(out.contains(&format!("{dropped} bytes omitted")));
        // Bounded: retained payload + marker, nowhere near `total`.
        assert!(out.len() < STREAM_RETAIN_HEAD + STREAM_RETAIN_TAIL + 200);
    }
}
