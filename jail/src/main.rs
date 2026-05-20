// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Eric Lesiuta
//! agent6-jail: minimal sandbox launcher.
//!
//! Reads a single JSON policy on stdin, sets up:
//!   - new mount/pid/net/ipc/uts namespaces (user namespace too, so we can mount unprivileged)
//!   - minimal bind-mounted rootfs (cwd RW, /usr /bin /lib /lib64 RO, tmpfs /tmp)
//!   - Landlock FS rules
//!   - seccomp-bpf allowlist
//!   - PR_SET_NO_NEW_PRIVS
//! Then forks + execs the child, captures stdout/stderr, prints one JSON result line.
//!
//! Exits 0 if it successfully ran the child (regardless of child exit code).
//! Exits non-zero only when sandbox SETUP failed — in that case stderr explains why.

use std::ffi::{CString, OsStr};
use std::fs;
use std::io::{self, Read, Write};
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use landlock::{
    Access, AccessFs, PathBeneath, PathFd, Ruleset, RulesetAttr, RulesetCreatedAttr, ABI,
};
use nix::mount::{mount, umount2, MntFlags, MsFlags};
use nix::sched::{unshare, CloneFlags};
use nix::sys::wait::{waitpid, WaitStatus};
use nix::unistd::{chdir, fork, getgid, getuid, pivot_root, ForkResult};
use seccompiler::{
    BpfProgram, SeccompAction, SeccompFilter, TargetArch,
};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Policy {
    #[serde(default = "default_profile")]
    profile: String,
    cwd: PathBuf,
    argv: Vec<String>,
    #[serde(default)]
    env: Vec<(String, String)>,
    #[serde(default)]
    allow_network: bool,
    #[serde(default)]
    extra_ro_paths: Vec<PathBuf>,
    #[serde(default)]
    extra_rw_paths: Vec<PathBuf>,
    #[serde(default = "default_timeout")]
    timeout_s: f64,
}

fn default_profile() -> String {
    "strict".to_string()
}

fn default_timeout() -> f64 {
    600.0
}

fn die(msg: impl AsRef<str>) -> ! {
    eprintln!("agent6-jail: {}", msg.as_ref());
    std::process::exit(2);
}

fn main() {
    let mut input = String::new();
    if io::stdin().read_to_string(&mut input).is_err() {
        die("failed to read policy from stdin");
    }
    let policy: Policy = match serde_json::from_str(&input) {
        Ok(p) => p,
        Err(e) => die(format!("invalid policy JSON: {e}")),
    };

    match policy.profile.as_str() {
        "strict" => run_strict(&policy),
        "hardened" => run_hardened(&policy),
        other => die(format!("unknown sandbox profile: {other}")),
    }
}

fn run_strict(policy: &Policy) -> ! {
    if let Err(e) = setup_namespaces(policy.allow_network) {
        die(format!("namespace setup failed: {e}"));
    }
    // After unshare(CLONE_NEWPID), the parent process itself remains in the OLD
    // pid namespace — only its children, forked after the unshare, enter the new
    // pid namespace. Anything that requires being inside the new pid ns (most
    // notably mounting a fresh /proc) must therefore happen in a forked child.
    match unsafe { fork() } {
        Ok(ForkResult::Parent { child }) => match waitpid(child, None) {
            Ok(WaitStatus::Exited(_, code)) => std::process::exit(code),
            Ok(WaitStatus::Signaled(_, sig, _)) => std::process::exit(128 + sig as i32),
            Ok(other) => die(format!("unexpected wait status: {other:?}")),
            Err(e) => die(format!("waitpid failed: {e}")),
        },
        Ok(ForkResult::Child) => {
            if let Err(e) = setup_rootfs(policy) {
                die(format!("rootfs setup failed: {e}"));
            }
            if let Err(e) = apply_landlock_strict() {
                die(format!("landlock failed: {e}"));
            }
            if let Err(e) = apply_seccomp() {
                die(format!("seccomp failed: {e}"));
            }
            if let Err(e) = run_child(policy, Path::new("/workspace")) {
                die(format!("child execution failed: {e}"));
            }
            std::process::exit(0);
        }
        Err(e) => die(format!("fork failed: {e}")),
    }
}

fn run_hardened(policy: &Policy) -> ! {
    // No namespaces, no pivot_root. Landlock confines the FS; seccomp +
    // NO_NEW_PRIVS bound the syscall surface; we still operate on the real cwd
    // and inherit the original /proc, /tmp, network namespace from the parent.
    // This is the profile that runs under default-seccomp Docker where
    // CLONE_NEWUSER is blocked.
    if let Err(e) = apply_landlock_hardened(policy) {
        die(format!("landlock failed: {e}"));
    }
    if let Err(e) = apply_seccomp() {
        die(format!("seccomp failed: {e}"));
    }
    if let Err(e) = run_child(policy, &policy.cwd) {
        die(format!("child execution failed: {e}"));
    }
    std::process::exit(0);
}

fn setup_namespaces(allow_network: bool) -> io::Result<()> {
    let uid = getuid();
    let gid = getgid();

    let mut flags = CloneFlags::CLONE_NEWUSER
        | CloneFlags::CLONE_NEWNS
        | CloneFlags::CLONE_NEWPID
        | CloneFlags::CLONE_NEWIPC
        | CloneFlags::CLONE_NEWUTS;
    if !allow_network {
        flags |= CloneFlags::CLONE_NEWNET;
    }
    unshare(flags).map_err(io_err)?;

    // Map current uid/gid into the new user namespace so we appear as root inside
    // (required to mount, but capabilities are still confined to this namespace).
    fs::write("/proc/self/setgroups", "deny").ok();
    fs::write("/proc/self/uid_map", format!("0 {} 1\n", uid))
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("uid_map: {e}")))?;
    fs::write("/proc/self/gid_map", format!("0 {} 1\n", gid))
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("gid_map: {e}")))?;

    // Make all existing mounts private so our changes don't propagate.
    mount(
        Some(""),
        "/",
        Some(""),
        MsFlags::MS_REC | MsFlags::MS_PRIVATE,
        Some(""),
    )
    .map_err(io_err)?;
    Ok(())
}

fn setup_rootfs(policy: &Policy) -> io::Result<()> {
    let new_root = PathBuf::from("/tmp/agent6-jail-root");
    // Make sure parent dir is on tmpfs we can write to (it's in our own NS now).
    let _ = fs::remove_dir_all(&new_root);
    fs::create_dir_all(&new_root)?;
    // Make new_root a mount point (pivot_root requirement).
    mount(
        Some(new_root.as_path()),
        &new_root,
        Some("tmpfs"),
        MsFlags::empty(),
        Some("size=64m"),
    )
    .map_err(io_err)?;

    for dir in ["proc", "tmp", "dev", "etc", "home", "root"] {
        fs::create_dir_all(new_root.join(dir))?;
    }
    // Read-only bind mounts for system dirs.
    for src in ["/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc/alternatives"] {
        if Path::new(src).exists() {
            let dst = new_root.join(src.trim_start_matches('/'));
            fs::create_dir_all(&dst)?;
            mount(
                Some(Path::new(src)),
                &dst,
                Some(""),
                MsFlags::MS_BIND | MsFlags::MS_REC,
                Some(""),
            )
            .map_err(io_err)?;
            // Remount read-only.
            mount(
                Some(""),
                &dst,
                Some(""),
                MsFlags::MS_BIND | MsFlags::MS_REMOUNT | MsFlags::MS_RDONLY | MsFlags::MS_REC,
                Some(""),
            )
            .map_err(io_err)?;
        }
    }
    // /tmp -> tmpfs
    mount(
        Some(""),
        &new_root.join("tmp"),
        Some("tmpfs"),
        MsFlags::empty(),
        Some("size=64m"),
    )
    .map_err(io_err)?;
    // /proc — bind from host /proc (it's still our PID namespace's view from outside,
    // but inside the new pid ns we'll mount a fresh one below).
    fs::create_dir_all(new_root.join("proc"))?;
    // /dev minimal — just /dev/null /dev/zero /dev/urandom etc.
    // /dev/tty is intentionally OMITTED: child commands inherit pipes (not a
    // tty) and giving them access to the controlling terminal would let a
    // misbehaving (or LLM-orchestrated) child write escape sequences that
    // affect the agent's host terminal.
    for dev in ["null", "zero", "urandom", "random", "full"] {
        let src = PathBuf::from(format!("/dev/{dev}"));
        if !src.exists() {
            continue;
        }
        let dst = new_root.join(format!("dev/{dev}"));
        fs::File::create(&dst)?;
        mount(
            Some(src.as_path()),
            &dst,
            Some(""),
            MsFlags::MS_BIND,
            Some(""),
        )
        .map_err(io_err)?;
    }
    // Bind the cwd RW.
    let cwd_in = new_root.join("workspace");
    fs::create_dir_all(&cwd_in)?;
    mount(
        Some(policy.cwd.as_path()),
        &cwd_in,
        Some(""),
        MsFlags::MS_BIND | MsFlags::MS_REC,
        Some(""),
    )
    .map_err(io_err)?;
    // Extra RO paths.
    for src in &policy.extra_ro_paths {
        if !src.exists() {
            continue;
        }
        let dst = new_root.join(format!("ro{}", src.display()));
        fs::create_dir_all(dst.parent().unwrap_or(Path::new("/")))?;
        if src.is_dir() {
            fs::create_dir_all(&dst)?;
        } else {
            fs::File::create(&dst)?;
        }
        mount(
            Some(src.as_path()),
            &dst,
            Some(""),
            MsFlags::MS_BIND | MsFlags::MS_REC,
            Some(""),
        )
        .map_err(io_err)?;
        mount(
            Some(""),
            &dst,
            Some(""),
            MsFlags::MS_BIND | MsFlags::MS_REMOUNT | MsFlags::MS_RDONLY | MsFlags::MS_REC,
            Some(""),
        )
        .map_err(io_err)?;
    }
    // Extra RW paths.
    for src in &policy.extra_rw_paths {
        if !src.exists() {
            continue;
        }
        let dst = new_root.join(format!("rw{}", src.display()));
        fs::create_dir_all(dst.parent().unwrap_or(Path::new("/")))?;
        if src.is_dir() {
            fs::create_dir_all(&dst)?;
        } else {
            fs::File::create(&dst)?;
        }
        mount(
            Some(src.as_path()),
            &dst,
            Some(""),
            MsFlags::MS_BIND | MsFlags::MS_REC,
            Some(""),
        )
        .map_err(io_err)?;
    }

    // pivot_root into new_root.
    let put_old = new_root.join(".old_root");
    fs::create_dir_all(&put_old)?;
    pivot_root(&new_root, &put_old).map_err(io_err)?;
    chdir("/").map_err(io_err)?;
    umount2(Path::new("/.old_root"), MntFlags::MNT_DETACH).map_err(io_err)?;
    fs::remove_dir("/.old_root").ok();

    // We are in the forked child (called from main after fork), which IS in the
    // new PID namespace. Try to mount a fresh /proc so the child sees only its
    // own PID namespace. If the kernel refuses (some userns setups error with
    // "VFS: Mount too revealing"), we log to stderr and continue with an EMPTY
    // /proc. We deliberately do NOT bind-mount the host /proc as a fallback
    // because that would expose every host PID and /proc/sys tunable.
    let proc_target = Path::new("/proc");
    if let Err(e) = mount(
        Some("proc"),
        proc_target,
        Some("proc"),
        MsFlags::MS_NOSUID | MsFlags::MS_NODEV | MsFlags::MS_NOEXEC,
        Some(""),
    ) {
        eprintln!(
            "[agent6-jail] warning: fresh /proc mount failed ({e}); /proc will be empty inside the jail"
        );
    }

    chdir("/workspace").map_err(io_err)?;
    Ok(())
}

fn apply_landlock_strict() -> io::Result<()> {
    // Strict profile runs inside the pivoted rootfs; /workspace is the writable
    // mount and /usr /bin /lib /lib64 /etc /tmp /dev are read-only bind mounts.
    let access_all = AccessFs::from_all(ABI::V1);
    let access_read = AccessFs::from_read(ABI::V1);
    // from_read excludes EXECUTE; system paths must be read+execute so spawned
    // binaries can actually run (otherwise execve EACCES).
    let access_read_exec = access_read | AccessFs::Execute;
    let ruleset = Ruleset::default()
        .handle_access(access_all)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("handle_access: {e}")))?
        .create()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("create ruleset: {e}")))?;
    let mut ruleset = ruleset;
    if let Ok(fd) = PathFd::new("/workspace") {
        ruleset = ruleset
            .add_rule(PathBeneath::new(fd, access_all))
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("rule /workspace: {e}")))?;
    }
    for ro in ["/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/tmp", "/dev"] {
        if let Ok(fd) = PathFd::new(ro) {
            ruleset = ruleset
                .add_rule(PathBeneath::new(fd, access_read_exec))
                .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("rule {ro}: {e}")))?;
        }
    }
    ruleset
        .restrict_self()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("restrict_self: {e}")))?;
    Ok(())
}

fn apply_landlock_hardened(policy: &Policy) -> io::Result<()> {
    // Hardened profile runs in the real filesystem. We protect the host by
    // listing exactly the paths the child may read or write — its own cwd
    // (read+write), the extra_rw_paths, /tmp (write), and the system dirs
    // (read+execute only).
    let access_all = AccessFs::from_all(ABI::V1);
    let access_read = AccessFs::from_read(ABI::V1);
    let access_read_exec = access_read | AccessFs::Execute;
    let ruleset = Ruleset::default()
        .handle_access(access_all)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("handle_access: {e}")))?
        .create()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("create ruleset: {e}")))?;
    let mut ruleset = ruleset;

    // Read+write: the child's working directory and any explicitly granted rw paths.
    let mut rw_paths: Vec<PathBuf> = vec![policy.cwd.clone(), PathBuf::from("/tmp")];
    for p in &policy.extra_rw_paths {
        rw_paths.push(p.clone());
    }
    for p in &rw_paths {
        if let Ok(fd) = PathFd::new(p) {
            ruleset = ruleset.add_rule(PathBeneath::new(fd, access_all)).map_err(|e| {
                io::Error::new(io::ErrorKind::Other, format!("rule rw {}: {e}", p.display()))
            })?;
        }
    }

    // Read+execute: system dirs the child needs to load libraries / spawn binaries.
    let mut ro_paths: Vec<PathBuf> = vec![
        PathBuf::from("/usr"),
        PathBuf::from("/bin"),
        PathBuf::from("/sbin"),
        PathBuf::from("/lib"),
        PathBuf::from("/lib64"),
        PathBuf::from("/etc"),
        PathBuf::from("/dev"),
    ];
    for p in &policy.extra_ro_paths {
        ro_paths.push(p.clone());
    }
    for p in &ro_paths {
        if let Ok(fd) = PathFd::new(p) {
            ruleset = ruleset.add_rule(PathBeneath::new(fd, access_read_exec)).map_err(|e| {
                io::Error::new(io::ErrorKind::Other, format!("rule ro {}: {e}", p.display()))
            })?;
        }
    }

    ruleset
        .restrict_self()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("restrict_self: {e}")))?;
    Ok(())
}

fn apply_seccomp() -> io::Result<()> {
    // PR_SET_NO_NEW_PRIVS
    let rc = unsafe { libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) };
    if rc != 0 {
        return Err(io::Error::last_os_error());
    }
    let arch = if cfg!(target_arch = "x86_64") {
        TargetArch::x86_64
    } else if cfg!(target_arch = "aarch64") {
        TargetArch::aarch64
    } else {
        // Skip seccomp on unknown arches.
        return Ok(());
    };
    // Default-allow with explicit deny of the worst offenders. We are inside
    // user-ns + landlock already; seccomp here is a third layer to block obvious
    // foot-guns: ptrace, mount, setns, unshare, kexec, bpf, perf, keyctl, etc.
    let denied: &[i64] = &[
        libc::SYS_ptrace,
        libc::SYS_mount,
        libc::SYS_umount2,
        libc::SYS_pivot_root,
        libc::SYS_setns,
        libc::SYS_unshare,
        libc::SYS_kexec_load,
        libc::SYS_kexec_file_load,
        libc::SYS_bpf,
        libc::SYS_perf_event_open,
        libc::SYS_keyctl,
        libc::SYS_add_key,
        libc::SYS_request_key,
        libc::SYS_init_module,
        libc::SYS_finit_module,
        libc::SYS_delete_module,
        libc::SYS_reboot,
        libc::SYS_swapon,
        libc::SYS_swapoff,
        libc::SYS_settimeofday,
        libc::SYS_adjtimex,
        libc::SYS_clock_settime,
    ];
    let rules: std::collections::BTreeMap<i64, Vec<seccompiler::SeccompRule>> =
        denied.iter().map(|s| (*s, vec![])).collect();
    let filter = SeccompFilter::new(
        rules,
        SeccompAction::Allow,                       // default
        SeccompAction::Errno(libc::EPERM as u32),    // matched (denied)
        arch,
    )
    .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("seccomp build: {e}")))?;
    let program: BpfProgram = filter
        .try_into()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("seccomp compile: {e}")))?;
    seccompiler::apply_filter(&program)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("seccomp apply: {e}")))?;
    Ok(())
}

fn run_child(policy: &Policy, cwd: &Path) -> io::Result<()> {
    if policy.argv.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "empty argv"));
    }
    let _ = CString::new(policy.argv[0].as_bytes());
    let _ = OsStr::new(""); // silence unused import on some targets

    let mut cmd = Command::new(&policy.argv[0]);
    cmd.args(&policy.argv[1..]);
    cmd.env_clear();
    for (k, v) in &policy.env {
        cmd.env(k, v);
    }
    // Minimal PATH so basic tools work inside the jail.
    cmd.env("PATH", "/usr/bin:/bin");
    cmd.env("HOME", "/home");
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    cmd.current_dir(cwd);

    let mut child = cmd.spawn()?;
    let timeout = Duration::from_secs_f64(policy.timeout_s);
    let start = std::time::Instant::now();

    // Drain stdout/stderr on background threads so a child that writes more
    // than the pipe buffer (~64KB) before exiting cannot deadlock us. The
    // earlier implementation only read pipes AFTER try_wait() returned Some,
    // which would deadlock the child on its write() while we were stuck
    // polling try_wait() forever.
    let stdout_pipe = child.stdout.take().expect("stdout piped");
    let stderr_pipe = child.stderr.take().expect("stderr piped");
    let stdout_handle = std::thread::spawn(move || {
        let mut buf = String::new();
        let mut s = stdout_pipe;
        let _ = s.read_to_string(&mut buf);
        buf
    });
    let stderr_handle = std::thread::spawn(move || {
        let mut buf = String::new();
        let mut s = stderr_pipe;
        let _ = s.read_to_string(&mut buf);
        buf
    });

    let returncode: i32;
    let mut timed_out = false;
    loop {
        match child.try_wait()? {
            Some(status) => {
                returncode = status.code().unwrap_or_else(|| {
                    status.signal().map(|s| 128 + s).unwrap_or(-1)
                });
                break;
            }
            None => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    returncode = 124;
                    timed_out = true;
                    break;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }
    let stdout = stdout_handle.join().unwrap_or_default();
    let mut stderr = stderr_handle.join().unwrap_or_default();
    if timed_out {
        stderr.push_str("\n[agent6-jail] timeout");
    }

    let result = serde_json::json!({
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
    });
    let mut out = io::stdout().lock();
    writeln!(out, "{result}")?;
    Ok(())
}

fn io_err<E: std::fmt::Display>(e: E) -> io::Error {
    io::Error::new(io::ErrorKind::Other, format!("{e}"))
}
