// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Eric Lesiuta
//! agent6-jail: minimal sandbox launcher.
//!
//! Reads a single JSON policy on stdin, sets up:
//!   - new mount/pid/net/ipc/uts namespaces (user namespace too, so we can mount unprivileged)
//!   - minimal bind-mounted rootfs (cwd RW, /usr /bin /lib /lib64 RO, tmpfs /tmp)
//!   - Landlock FS rules
//!   - seccomp-bpf deny-list (default-allow, EPERM on the dangerous syscalls)
//!   - PR_SET_NO_NEW_PRIVS
//!   - RLIMIT_DATA memory cap on the child (policy `memory_limit_mb`)
//! Then forks + execs the child, captures stdout/stderr, prints one JSON result line.
//!
//! Exits 0 if it successfully ran the child (regardless of child exit code).
//! Exits non-zero only when sandbox SETUP failed — in that case stderr explains why.

use std::ffi::{CString, OsStr};
use std::fs;
use std::io::{self, Read, Write};
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use landlock::{
    Access, AccessFs, PathBeneath, PathFd, Ruleset, RulesetAttr, RulesetCreatedAttr, ABI,
};
use nix::mount::{mount, umount2, MntFlags, MsFlags};
use nix::sched::{unshare, CloneFlags};
use nix::sys::statvfs::{statvfs, FsFlags};
use nix::sys::wait::{waitpid, WaitStatus};
use nix::unistd::{chdir, fork, getgid, getuid, pivot_root, ForkResult};
use seccompiler::{BpfProgram, SeccompAction, SeccompFilter, TargetArch};
use serde::Deserialize;

// deny_unknown_fields: a policy field this binary does not know (version skew
// between the Python side and a stale or pinned AGENT6_JAIL_BIN) could be a
// restriction it would silently drop, so refuse instead of running weaker.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Policy {
    #[serde(default = "default_profile")]
    profile: String,
    cwd: PathBuf,
    argv: Vec<String>,
    #[serde(default)]
    env: Vec<(String, String)>,
    #[serde(default)]
    allow_network: bool,
    /// Operator-granted extra paths, bind-mounted at their REAL locations in
    /// strict (like tool_paths and the hardened profile), so a granted
    /// toolchain works via its own absolute paths and shebangs. ro is
    /// read+execute, rw is read+write. A ro grant under a system mount is
    /// redundant (already visible read+exec) and skipped; an rw grant under
    /// one cannot be honored and fails the run loudly.
    #[serde(default)]
    extra_ro_paths: Vec<PathBuf>,
    #[serde(default)]
    extra_rw_paths: Vec<PathBuf>,
    /// Real-location RO+exec bind mounts for operator-installed tools (uv, node,
    /// ...) that live outside the system dirs -- ~/.local/bin, ~/.cargo/bin, or the
    /// /opt target a /usr/local/bin symlink resolves to. Real paths mean PATH
    /// lookups and symlinks resolve. Read+execute only, never writable.
    #[serde(default)]
    tool_paths: Vec<PathBuf>,
    /// Paths inside `cwd` to make READ-ONLY from the child's view. In
    /// strict, these are re-bound RO on top of the workspace mount. In
    /// hardened (no mount namespace), the Landlock ruleset switches from
    /// "RW on cwd" to "R on cwd + RW on each top-level entry except
    /// these" — same end result for files that exist at jail-launch time,
    /// at the cost of denying writes to new top-level entries created
    /// after the jail starts. Used to keep an LLM-driven `run_command`
    /// from rewriting `.git`. Each entry
    /// must be absolute; entries that don't exist on disk are skipped.
    #[serde(default)]
    extra_protect_paths: Vec<PathBuf>,
    #[serde(default = "default_timeout")]
    timeout_s: f64,
    /// Per-process memory cap in MiB, applied via RLIMIT_DATA in the child
    /// before exec and inherited by every descendant. 0 disables. The serde
    /// default matches the Python-side `[sandbox].memory_limit_mb` default so
    /// a policy missing the field still gets the bounded value.
    #[serde(default = "default_memory_limit_mb")]
    memory_limit_mb: u64,
}

fn default_profile() -> String {
    "strict".to_string()
}

fn default_timeout() -> f64 {
    600.0
}

fn default_memory_limit_mb() -> u64 {
    4096
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
            if let Err(e) = apply_landlock_strict(policy) {
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

/// The host dirs strict bind-mounts read-only into the rootfs. Extra-path
/// grants check against this: a ro grant under one is already visible, and an
/// rw grant under one cannot be honored (the covering bind is read-only).
const SYSTEM_BINDS: [&str; 6] = [
    "/usr",
    "/bin",
    "/sbin",
    "/lib",
    "/lib64",
    "/etc/alternatives",
];

fn under_system_bind(p: &Path) -> bool {
    SYSTEM_BINDS.iter().any(|s| p.starts_with(s))
}

/// Mount flags a bind remount must repeat: the kernel refuses (EPERM) a
/// remount that would CLEAR nosuid/nodev/noexec/atime flags the bind
/// inherited from its source filesystem (e.g. a host /tmp mounted
/// nosuid,nodev), so read them off the mounted dst and carry them over.
fn carried_mount_flags(dst: &Path) -> MsFlags {
    let mut flags = MsFlags::empty();
    if let Ok(st) = statvfs(dst) {
        let f = st.flags();
        if f.contains(FsFlags::ST_NOSUID) {
            flags |= MsFlags::MS_NOSUID;
        }
        if f.contains(FsFlags::ST_NODEV) {
            flags |= MsFlags::MS_NODEV;
        }
        if f.contains(FsFlags::ST_NOEXEC) {
            flags |= MsFlags::MS_NOEXEC;
        }
        if f.contains(FsFlags::ST_NOATIME) {
            flags |= MsFlags::MS_NOATIME;
        }
        if f.contains(FsFlags::ST_NODIRATIME) {
            flags |= MsFlags::MS_NODIRATIME;
        }
        // ST_RELATIME by raw bit (0x1000, kernel ABI): the kernel sets it in
        // f_flag on every libc, but neither nix's FsFlags nor the libc crate
        // defines the constant on musl, and the release wheel builds musl.
        const ST_RELATIME_BIT: libc::c_ulong = 0x1000;
        if f.bits() & ST_RELATIME_BIT != 0 {
            flags |= MsFlags::MS_RELATIME;
        }
    }
    flags
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
    for src in SYSTEM_BINDS {
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
    // /tmp -> tmpfs. HOME points here (dispatch sets HOME=/tmp/agent6-home so
    // toolchain caches have a writable root), and go's build cache alone needs
    // several hundred MB for stdlib artifacts -- at 64m `go test` died ENOSPC
    // and models burned budgets fighting the sandbox. 1g is a hard ceiling on
    // RAM-backed pages, not an allocation; a run that needs none uses none.
    mount(
        Some(""),
        &new_root.join("tmp"),
        Some("tmpfs"),
        MsFlags::empty(),
        Some("size=1g"),
    )
    .map_err(io_err)?;
    // Operator tool dirs (uv etc.) at their REAL locations, RO. After /tmp so a dir
    // that happens to live under it is not shadowed by the fresh tmpfs. Best-effort:
    // a dir that fails to mount just leaves that tool unreachable rather than aborting
    // the run; dispatch only passes dirs OUTSIDE the system mounts above.
    for src in &policy.tool_paths {
        if !src.exists() {
            continue;
        }
        let dst = new_root.join(src.strip_prefix("/").unwrap_or(src));
        if fs::create_dir_all(&dst).is_err() {
            continue;
        }
        if mount(
            Some(src.as_path()),
            &dst,
            Some(""),
            MsFlags::MS_BIND | MsFlags::MS_REC,
            Some(""),
        )
        .is_err()
        {
            continue;
        }
        let _ = mount(
            Some(""),
            &dst,
            Some(""),
            MsFlags::MS_BIND
                | MsFlags::MS_REMOUNT
                | MsFlags::MS_RDONLY
                | MsFlags::MS_REC
                | carried_mount_flags(&dst),
            Some(""),
        );
    }
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
    // Re-bind each protect path RO on top of the workspace mount. Subdirs
    // and individual files are both supported; non-existent entries are
    // skipped silently so a project without (e.g.) a `.git` dir is not
    // a fatal config error.
    for src in &policy.extra_protect_paths {
        // Canonicalize so a symlink at .git/ can't trick us into bind-mounting
        // a target outside cwd into the jail. If the path doesn't exist yet,
        // canonicalize fails — skip it (the protect-path is a no-op anyway).
        let canon_src = match src.canonicalize() {
            Ok(p) => p,
            Err(_) => continue,
        };
        let canon_cwd = policy
            .cwd
            .canonicalize()
            .unwrap_or_else(|_| policy.cwd.clone());
        // Reject paths outside cwd defensively (Python side filters them too,
        // but the launcher is its own trust boundary).
        let rel = match canon_src.strip_prefix(&canon_cwd) {
            Ok(r) => r,
            Err(_) => {
                eprintln!(
                    "agent6-jail: skipping protect_path {} (canonical {} not under cwd {})",
                    src.display(),
                    canon_src.display(),
                    canon_cwd.display(),
                );
                continue;
            }
        };
        let target = cwd_in.join(rel);
        // Ensure the mount point exists inside our new rootfs (it should,
        // via the cwd bind, but be defensive for first-run-on-fresh-repo).
        if canon_src.is_dir() {
            fs::create_dir_all(&target)?;
        } else if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
            if !target.exists() {
                fs::File::create(&target)?;
            }
        }
        // Bind the canonical host path onto the target inside the new rootfs,
        // then remount read-only. Binding from the host path (rather than
        // self-binding inside the new mount) avoids EPERM on kernels that
        // refuse re-binding paths already covered by a recursive parent
        // bind in a user namespace.
        mount(
            Some(canon_src.as_path()),
            &target,
            Some(""),
            MsFlags::MS_BIND,
            Some(""),
        )
        .map_err(|e| {
            io::Error::new(
                io::ErrorKind::Other,
                format!(
                    "protect bind {} -> {}: {e}",
                    canon_src.display(),
                    target.display()
                ),
            )
        })?;
        mount(
            Some(""),
            &target,
            Some(""),
            // MS_NOSUID | MS_NODEV are required on the remount in a user
            // namespace — the kernel refuses to clear them, and bind
            // remounts in older kernels do not preserve them automatically.
            MsFlags::MS_BIND
                | MsFlags::MS_REMOUNT
                | MsFlags::MS_RDONLY
                | MsFlags::MS_NOSUID
                | MsFlags::MS_NODEV,
            Some(""),
        )
        .map_err(|e| {
            io::Error::new(
                io::ErrorKind::Other,
                format!("protect remount-ro {}: {e}", target.display()),
            )
        })?;
    }
    // Extra RO paths, at their REAL locations (matching tool_paths and the
    // hardened profile, and the documented contract: a granted toolchain works
    // via its own absolute paths and shebangs). A grant under a system bind is
    // redundant (already visible read+exec) and skipped. Failures are LOUD:
    // the operator listed the path, so a broken grant must not pass silently.
    for src in &policy.extra_ro_paths {
        if !src.exists() || under_system_bind(src) {
            continue;
        }
        let dst = new_root.join(src.strip_prefix("/").unwrap_or(src));
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
            MsFlags::MS_BIND
                | MsFlags::MS_REMOUNT
                | MsFlags::MS_RDONLY
                | MsFlags::MS_REC
                | carried_mount_flags(&dst),
            Some(""),
        )
        .map_err(io_err)?;
    }
    // Extra RW paths, at their REAL locations. Under a system bind the write
    // grant cannot be honored (the covering mount is read-only): refuse loudly
    // rather than mount a dead path.
    for src in &policy.extra_rw_paths {
        if !src.exists() {
            continue;
        }
        if under_system_bind(src) {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!(
                    "extra_rw path {} sits under a read-only system mount",
                    src.display()
                ),
            ));
        }
        let dst = new_root.join(src.strip_prefix("/").unwrap_or(src));
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

    // We are in the forked child (called from main after fork), which IS in the
    // new PID namespace. Mount a fresh /proc so the child sees only its own
    // PID namespace. ORDER MATTERS: this must happen while /.old_root is still
    // attached -- the kernel permits a userns proc mount only when the mount
    // namespace already contains a fully-visible proc instance, and the host's
    // /.old_root/proc is that instance. Mounting after the detach fails EPERM
    // ("mount too revealing" rule) and left /proc EMPTY, which breaks any tool
    // that reads /proc/self (observed: go cannot resolve GOROOT via
    // /proc/self/exe). If the kernel still refuses, log and continue with an
    // empty /proc; we deliberately do NOT bind-mount the host /proc as a
    // fallback because that would expose every host PID and /proc/sys tunable.
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

    umount2(Path::new("/.old_root"), MntFlags::MNT_DETACH).map_err(io_err)?;
    fs::remove_dir("/.old_root").ok();

    chdir("/workspace").map_err(io_err)?;
    Ok(())
}

fn apply_landlock_strict(policy: &Policy) -> io::Result<()> {
    // Strict profile runs inside the pivoted rootfs; /workspace (the cwd bind)
    // and /tmp (a fresh private tmpfs, see setup_rootfs) are writable, and
    // /usr /bin /lib /lib64 /etc /dev are read-only bind mounts.
    // ABI::V2 (not V1): V2 adds LANDLOCK_ACCESS_FS_REFER. A ruleset that
    // does not handle REFER keeps ABI-1 semantics where EVERY cross-directory
    // rename/hardlink fails with EXDEV, which breaks `cargo` (hardlinks build
    // artifacts between target/ subdirs), `mv` across dirs, and similar tools
    // even inside fully-writable paths. Granting REFER on rw paths only allows
    // re-parenting within hierarchies the child can already write; the crate's
    // best-effort mode degrades gracefully on ABI-1 kernels.
    let access_all = AccessFs::from_all(ABI::V2);
    let access_read = AccessFs::from_read(ABI::V2);
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
    // /tmp is a fresh private tmpfs in this jail's own mount namespace (mounted
    // in setup_rootfs), discarded when the jail exits. Grant it RW so toolchain
    // caches that key off $HOME or TMPDIR work (go-build, cargo, pip/uv); the
    // tmpfs is isolated, so RW here cannot reach the host. Mirrors the hardened
    // profile, which already grants /tmp RW.
    if let Ok(fd) = PathFd::new("/tmp") {
        ruleset = ruleset
            .add_rule(PathBeneath::new(fd, access_all))
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("rule /tmp: {e}")))?;
    }
    // /proc is the jail's OWN freshly-mounted procfs (private PID namespace,
    // see setup_rootfs), so reading it reveals only jail-local processes and
    // read-only kernel views -- the same exposure every container runtime
    // grants. Read WITHOUT execute. Without this rule every /proc read dies
    // EACCES and toolchains fail in confusing ways: go resolves GOROOT via
    // /proc/self/exe (observed: go 1.26 "cannot find GOROOT", the model then
    // rewrites verify.sh to fight the sandbox), python reads /proc/cpuinfo,
    // ps needs the listing.
    if let Ok(fd) = PathFd::new("/proc") {
        ruleset = ruleset
            .add_rule(PathBeneath::new(fd, access_read))
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("rule /proc: {e}")))?;
    }
    for ro in ["/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/dev"] {
        if let Ok(fd) = PathFd::new(ro) {
            ruleset = ruleset
                .add_rule(PathBeneath::new(fd, access_read_exec))
                .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("rule {ro}: {e}")))?;
        }
    }
    // Operator tool dirs (mounted at real locations in setup_rootfs): read+exec.
    for tp in &policy.tool_paths {
        if let Ok(fd) = PathFd::new(tp) {
            ruleset = ruleset
                .add_rule(PathBeneath::new(fd, access_read_exec))
                .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("rule tool: {e}")))?;
        }
    }
    // Grant WriteFile on the harmless sink devices. /dev/null and /dev/full
    // are bind-mounted from the host (see setup_rootfs) and pytest's logging
    // plugin opens /dev/null O_WRONLY|O_APPEND when log_file is configured,
    // which would otherwise EACCES under the /dev read-only rule above and
    // surface as INTERNALERROR. WriteFile on these specific inodes does not
    // grant create/symlink/unlink and cannot be used to escape the jail.
    for dev in ["null", "zero", "full"] {
        let p = format!("/dev/{dev}");
        if let Ok(fd) = PathFd::new(&p) {
            ruleset = ruleset
                .add_rule(PathBeneath::new(fd, AccessFs::WriteFile))
                .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("rule {p}: {e}")))?;
        }
    }
    // Extra paths are bind-mounted by setup_rootfs at their REAL locations.
    // Without a matching Landlock rule the child would get EACCES on them
    // despite the mount, so grant the access here too. Paths that didn't
    // exist on the host were skipped at mount time, so PathFd::new simply
    // fails and is ignored; a ro grant under a system bind got no mount of
    // its own, but the rule on the real path applies to the covering bind's
    // content just the same.
    for ro in &policy.extra_ro_paths {
        if let Ok(fd) = PathFd::new(ro) {
            ruleset = ruleset
                .add_rule(PathBeneath::new(fd, access_read_exec))
                .map_err(|e| {
                    io::Error::new(
                        io::ErrorKind::Other,
                        format!("rule ro {}: {e}", ro.display()),
                    )
                })?;
        }
    }
    for rw in &policy.extra_rw_paths {
        if let Ok(fd) = PathFd::new(rw) {
            ruleset = ruleset
                .add_rule(PathBeneath::new(fd, access_all))
                .map_err(|e| {
                    io::Error::new(
                        io::ErrorKind::Other,
                        format!("rule rw {}: {e}", rw.display()),
                    )
                })?;
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
    let access_all = AccessFs::from_all(ABI::V2);
    let access_read = AccessFs::from_read(ABI::V2);
    let access_read_exec = access_read | AccessFs::Execute;
    let ruleset = Ruleset::default()
        .handle_access(access_all)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("handle_access: {e}")))?
        .create()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("create ruleset: {e}")))?;
    let mut ruleset = ruleset;

    // protect_paths: in hardened we cannot do a bind-remount-RO (no mount
    // namespace). Instead, we DON'T grant RW on cwd as a whole. We grant R
    // on cwd recursively (so .git etc. stay readable), then enumerate
    // cwd's top-level entries and grant RW only to the ones that are not
    // in the protect set. Landlock rules are purely additive within a
    // single ruleset, so if no rule grants W on a path, writes to it are
    // denied — that's what gives us the read-only carve-out.
    //
    // Limitation: new top-level entries created by the child at the root
    // of cwd are not in any RW rule and will be read-only. Anything
    // inside an existing top-level dir (src/, tests/, …) gets the full
    // recursive RW rule and behaves normally.
    let has_protect = !policy.extra_protect_paths.is_empty();
    let protect_set: std::collections::HashSet<PathBuf> = policy
        .extra_protect_paths
        .iter()
        .filter_map(|p| p.canonicalize().ok().or_else(|| Some(p.clone())))
        .collect();

    if has_protect {
        // R on cwd recursively, so protected paths remain readable.
        if let Ok(fd) = PathFd::new(&policy.cwd) {
            ruleset = ruleset
                .add_rule(PathBeneath::new(fd, access_read))
                .map_err(|e| {
                    io::Error::new(
                        io::ErrorKind::Other,
                        format!("rule r cwd {}: {e}", policy.cwd.display()),
                    )
                })?;
        }
        // RW only on non-protected top-level entries.
        let canon_cwd = policy
            .cwd
            .canonicalize()
            .unwrap_or_else(|_| policy.cwd.clone());
        let entries = match fs::read_dir(&policy.cwd) {
            Ok(it) => it,
            Err(e) => {
                return Err(io::Error::new(
                    e.kind(),
                    format!("read_dir cwd {}: {e}", policy.cwd.display()),
                ));
            }
        };
        for entry in entries.flatten() {
            let p = entry.path();
            let canon = p.canonicalize().unwrap_or_else(|_| p.clone());
            if protect_set.contains(&canon) || protect_set.contains(&p) {
                continue;
            }
            // A top-level symlink whose real target escapes cwd would otherwise
            // get a recursive RW rule on that outside inode (PathFd::new follows
            // symlinks; Landlock attaches to the resolved inode), letting the
            // child write outside the workspace and defeating cwd confinement
            // under the hardened profile. Skip any entry that does not resolve
            // inside cwd. Mirrors the strip_prefix(cwd) check in setup_rootfs.
            if !canon.starts_with(&canon_cwd) {
                eprintln!(
                    "agent6-jail: hardened: skipping rw grant on {} (resolves outside cwd to {})",
                    p.display(),
                    canon.display()
                );
                continue;
            }
            if let Ok(fd) = PathFd::new(&p) {
                ruleset = ruleset
                    .add_rule(PathBeneath::new(fd, access_all))
                    .map_err(|e| {
                        io::Error::new(
                            io::ErrorKind::Other,
                            format!("rule rw {}: {e}", p.display()),
                        )
                    })?;
            }
        }
    } else {
        // No protect set: original behavior, RW on cwd as a whole.
        if let Ok(fd) = PathFd::new(&policy.cwd) {
            ruleset = ruleset
                .add_rule(PathBeneath::new(fd, access_all))
                .map_err(|e| {
                    io::Error::new(
                        io::ErrorKind::Other,
                        format!("rule rw cwd {}: {e}", policy.cwd.display()),
                    )
                })?;
        }
    }

    // Read+write: /tmp and any explicitly granted rw paths.
    let mut rw_paths: Vec<PathBuf> = vec![PathBuf::from("/tmp")];
    for p in &policy.extra_rw_paths {
        rw_paths.push(p.clone());
    }
    for p in &rw_paths {
        // Skip any rw_path that would shadow a protect_path. Landlock
        // rules combine permissively: if any rule grants W on a path,
        // the write is allowed. A blanket RW grant on an ancestor of a
        // protected path defeats the carve-out, so drop the ancestor.
        if has_protect && protect_set.iter().any(|prot| prot.starts_with(p)) {
            eprintln!(
                "agent6-jail: hardened: skipping rw grant on {} (would shadow a protect_path)",
                p.display()
            );
            continue;
        }
        if let Ok(fd) = PathFd::new(p) {
            ruleset = ruleset
                .add_rule(PathBeneath::new(fd, access_all))
                .map_err(|e| {
                    io::Error::new(
                        io::ErrorKind::Other,
                        format!("rule rw {}: {e}", p.display()),
                    )
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
    // Operator tool dirs (uv etc.): read+exec at their real host paths (hardened
    // runs in the real filesystem, so no bind mount is needed, only the grant).
    for p in &policy.tool_paths {
        ro_paths.push(p.clone());
    }
    for p in &ro_paths {
        if let Ok(fd) = PathFd::new(p) {
            ruleset = ruleset
                .add_rule(PathBeneath::new(fd, access_read_exec))
                .map_err(|e| {
                    io::Error::new(
                        io::ErrorKind::Other,
                        format!("rule ro {}: {e}", p.display()),
                    )
                })?;
        }
    }
    // Same sink-device carve-out as the strict profile; see comment there.
    for dev in ["null", "zero", "full"] {
        let p = format!("/dev/{dev}");
        if let Ok(fd) = PathFd::new(&p) {
            ruleset = ruleset
                .add_rule(PathBeneath::new(fd, AccessFs::WriteFile))
                .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("rule {p}: {e}")))?;
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
    // libc (0.2.x) doesn't define SYS_kexec_file_load for the musl aarch64 target
    // even though the syscall exists there (arm64 #294), so name it explicitly —
    // otherwise the sandbox would silently stop blocking a new-kernel load on arm64.
    #[cfg(target_arch = "aarch64")]
    const SYS_KEXEC_FILE_LOAD: i64 = 294;
    #[cfg(not(target_arch = "aarch64"))]
    const SYS_KEXEC_FILE_LOAD: i64 = libc::SYS_kexec_file_load;
    let denied: &[i64] = &[
        libc::SYS_ptrace,
        libc::SYS_process_vm_readv,
        libc::SYS_process_vm_writev,
        libc::SYS_kcmp,
        libc::SYS_mount,
        libc::SYS_umount2,
        libc::SYS_pivot_root,
        // Modern mount API (new_mount_api, Linux 5.2+). A strict jailed child
        // is userns-root with CAP_SYS_ADMIN over its own mount namespace and
        // never drops caps, so without these it could mount_setattr(2) away the
        // MOUNT_ATTR_RDONLY on the .git protect bind (or open_tree+move_mount to
        // relocate it) and defeat protect_git. Classic mount(2) is already
        // denied above; these complete the coverage.
        libc::SYS_mount_setattr,
        libc::SYS_open_tree,
        libc::SYS_move_mount,
        libc::SYS_fsopen,
        libc::SYS_fsconfig,
        libc::SYS_fsmount,
        libc::SYS_fspick,
        libc::SYS_setns,
        libc::SYS_unshare,
        libc::SYS_kexec_load,
        SYS_KEXEC_FILE_LOAD,
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
        SeccompAction::Allow,                     // default
        SeccompAction::Errno(libc::EPERM as u32), // matched (denied)
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
    // Minimal PATH so basic tools work inside the jail; the policy may extend it so
    // operator-installed tools outside /usr/bin (e.g. /usr/local/bin, ~/.local/bin)
    // resolve. Policy env is operator-side input.
    if !policy.env.iter().any(|(k, _)| k == "PATH") {
        cmd.env("PATH", "/usr/bin:/bin");
    }
    // HOME default; the policy may override it (e.g. a writable /tmp path so
    // toolchain caches like go-build work). Policy env is operator-side input.
    if !policy.env.iter().any(|(k, _)| k == "HOME") {
        cmd.env("HOME", "/home");
    }
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    cmd.current_dir(cwd);
    // Put the child in its own process group (pgid == its pid) so we can kill
    // the whole tree on timeout/exit. Otherwise a backgrounded grandchild that
    // inherited our stdout/stderr write-end keeps the pipe open and the reader
    // threads' read_to_string() never sees EOF — hanging the launcher.
    cmd.process_group(0);
    // Memory cap: RLIMIT_DATA (brk + private writable anonymous mappings,
    // enforced for mmap since kernel 4.7 — older than any Landlock-capable
    // kernel), NOT RLIMIT_AS: an address-space cap breaks VA reservers that
    // never commit the memory (V8's 4 GiB pointer-compression cage, ASAN
    // shadow, JVM heap reserve). A runaway allocation gets ENOMEM (Python
    // MemoryError, C++ bad_alloc) instead of driving the host to the OOM
    // killer. The limit inherits across fork/exec, so each descendant is
    // individually bounded; it is per-process, not per-tree (that would take
    // a cgroup, which an unprivileged launcher cannot assume). Raising the
    // hard limit back needs CAP_SYS_RESOURCE in the INITIAL user namespace,
    // which a jailed child (even userns root under `strict`) never has.
    if policy.memory_limit_mb > 0 {
        let bytes: libc::rlim_t =
            (policy.memory_limit_mb as libc::rlim_t).saturating_mul(1024 * 1024);
        unsafe {
            cmd.pre_exec(move || {
                // Clamp to the inherited hard limit: lowering is always
                // permitted, and if the operator's shell already set a
                // stricter hard cap the stricter value wins (never EPERM).
                let mut cur = libc::rlimit {
                    rlim_cur: 0,
                    rlim_max: 0,
                };
                if libc::getrlimit(libc::RLIMIT_DATA, &mut cur) != 0 {
                    return Err(io::Error::last_os_error());
                }
                let cap = bytes.min(cur.rlim_max);
                let lim = libc::rlimit {
                    rlim_cur: cap,
                    rlim_max: cap,
                };
                if libc::setrlimit(libc::RLIMIT_DATA, &lim) != 0 {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            });
        }
    }

    let mut child = cmd.spawn()?;
    let child_pid = child.id() as i32; // == pgid, since process_group(0)
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
    // Track whether try_wait()/wait() has already reaped the direct child. Once
    // reaped, child_pid (== pgid) is free for the kernel to reuse for an
    // UNRELATED process group, so a blind post-loop killpg(child_pid) could
    // SIGKILL a stranger. The timeout branch does its own killpg BEFORE waiting
    // (the pid is still ours there), so it is safe regardless.
    let mut reaped = false;
    loop {
        match child.try_wait()? {
            Some(status) => {
                returncode = status
                    .code()
                    .unwrap_or_else(|| status.signal().map(|s| 128 + s).unwrap_or(-1));
                reaped = true;
                break;
            }
            None => {
                if start.elapsed() > timeout {
                    // Kill the whole process group, not just the direct child,
                    // so backgrounded grandchildren can't keep running / hold
                    // the pipe write-end open. This runs BEFORE wait(), while
                    // child_pid is still unambiguously our process group.
                    unsafe {
                        libc::killpg(child_pid, libc::SIGKILL);
                    }
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
    // Reap the whole process group on exit so any backgrounded fd-holder is gone
    // and the pipe write-end is closed (read_to_string() then gets EOF instead of
    // the reader-thread joins below blocking until those grandchildren exit). This
    // also, by design, means a command's process group does not outlive the
    // command: a daemon the command backgrounded is torn down here rather than
    // leaking — the sandbox-appropriate behavior (in strict mode the PID namespace
    // already enforces this; this makes hardened mode match).
    //
    // ONLY when the child was not already reaped on the normal path: once
    // try_wait() has reaped it, child_pid may have been recycled by the kernel
    // for an unrelated process group, and killpg(child_pid) would signal that
    // stranger. The normal-exit case (reaped == true) skips this; the timeout
    // case already killpg'd above before waiting.
    if !reaped {
        unsafe {
            libc::killpg(child_pid, libc::SIGKILL);
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
