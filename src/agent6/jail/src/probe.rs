// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Eric Lesiuta
//! Target observations and no-follow mount proofs, not container confinement.

use std::ffi::{CString, OsStr};
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Component, Path, PathBuf};
use std::sync::OnceLock;

struct Installation {
    owner: u32,
    parent: File,
    directory: File,
    name: CString,
    path: PathBuf,
}

static INSTALLATION: OnceLock<Installation> = OnceLock::new();

fn open_at(parent: &File, name: &OsStr, flags: i32) -> io::Result<File> {
    let name = CString::new(name.as_bytes())?;
    let fd = unsafe {
        libc::openat(
            parent.as_raw_fd(),
            name.as_ptr(),
            flags | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            0o644,
        )
    };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(unsafe { File::from_raw_fd(fd) })
}

pub fn open_directory(path: &Path) -> io::Result<File> {
    if !path.is_absolute() {
        return Err(io::Error::other("mapping path must be absolute"));
    }
    let mut dir = File::options()
        .read(true)
        .custom_flags(libc::O_PATH | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open("/")?;
    for component in path.components() {
        match component {
            Component::RootDir => (),
            Component::Normal(name) => dir = open_at(&dir, name, libc::O_PATH | libc::O_DIRECTORY)?,
            _ => {
                return Err(io::Error::other(
                    "mapping path contains a non-directory component",
                ))
            }
        }
    }
    Ok(dir)
}

pub fn random_hex(bytes: usize) -> io::Result<String> {
    let mut nonce = vec![0u8; bytes];
    if unsafe { libc::getrandom(nonce.as_mut_ptr().cast(), nonce.len(), 0) } != nonce.len() as isize
    {
        return Err(io::Error::last_os_error());
    }
    Ok(nonce.iter().map(|b| format!("{b:02x}")).collect())
}

/// The installed executable is unlinked before any command can run.
pub fn installed() -> io::Result<()> {
    let path = std::env::current_dir()?;
    let name = path
        .file_name()
        .ok_or_else(|| io::Error::other("installation path has no name"))?;
    if path.parent() != Some(Path::new("/tmp")) || !name.as_bytes().starts_with(b"a6.") {
        return Err(io::Error::other(
            "installation must be a fresh /tmp/a6.* directory",
        ));
    }
    let parent = open_directory(Path::new("/tmp"))?;
    let dir = open_at(&parent, name, libc::O_RDONLY | libc::O_DIRECTORY)?;
    let info = dir.metadata()?;
    if info.uid() != unsafe { libc::geteuid() } || info.mode() & 0o777 != 0o700 {
        return Err(io::Error::other(
            "installation directory must be user-owned mode 0700",
        ));
    }
    let executable = open_at(&dir, OsStr::new("launcher"), libc::O_RDONLY)?;
    let info = executable.metadata()?;
    let running = fs::metadata("/proc/self/exe")?;
    if !info.is_file() || (info.dev(), info.ino()) != (running.dev(), running.ino()) {
        return Err(io::Error::other("installed launcher identity changed"));
    }
    if unsafe { libc::unlinkat(dir.as_raw_fd(), c"launcher".as_ptr(), 0) } != 0 {
        return Err(io::Error::last_os_error());
    }
    INSTALLATION
        .set(Installation {
            owner: std::process::id(),
            parent,
            directory: dir,
            name: CString::new(name.as_bytes())?,
            path,
        })
        .map_err(|_| io::Error::other("installation already prepared"))?;
    std::env::set_var("TMPDIR", temp_dir());
    Ok(())
}

/// Capture storage belongs to the installation, or /tmp for a local launcher.
pub fn temp_dir() -> &'static Path {
    INSTALLATION
        .get()
        .map_or(Path::new("/tmp"), |installation| {
            installation.path.as_path()
        })
}

/// Never recursively remove command-created files or follow a replacement symlink.
pub fn cleanup_installation() {
    if let Some(install) = INSTALLATION.get() {
        if install.owner != std::process::id() {
            return;
        }
        let same = open_at(
            &install.parent,
            OsStr::from_bytes(install.name.as_bytes()),
            libc::O_RDONLY | libc::O_DIRECTORY,
        )
        .and_then(|file| file.metadata())
        .and_then(|current| {
            install
                .directory
                .metadata()
                .map(|held| (current.dev(), current.ino()) == (held.dev(), held.ino()))
        })
        .unwrap_or(false);
        if same
            && unsafe {
                libc::unlinkat(
                    install.parent.as_raw_fd(),
                    install.name.as_ptr(),
                    libc::AT_REMOVEDIR,
                )
            } != 0
        {
            eprintln!(
                "agent6-jail: target TMPDIR retained: {}",
                io::Error::last_os_error()
            );
        }
    }
}

pub fn facts() -> io::Result<serde_json::Value> {
    let mut mounts = String::new();
    File::open("/proc/self/mountinfo")?
        .take(1024 * 1024 + 1)
        .read_to_string(&mut mounts)?;
    if mounts.len() > 1024 * 1024 {
        return Err(io::Error::other("mount inventory exceeds 1 MiB"));
    }
    let mut environment = std::collections::BTreeMap::new();
    for (name, value) in std::env::vars_os() {
        let name = name
            .into_string()
            .map_err(|_| io::Error::other("environment name is not UTF-8"))?;
        let value = value
            .into_string()
            .map_err(|_| io::Error::other("environment value is not UTF-8"))?;
        environment.insert(name, value);
    }
    Ok(serde_json::json!({
        "uid": unsafe { libc::geteuid() }, "gid": unsafe { libc::getegid() },
        "architecture": std::env::consts::ARCH,
        "environment": environment,
        "mounts": mounts,
    }))
}

pub fn mapping(cwd: &Path, name: &str, nonce: &str) -> io::Result<String> {
    let suffix = name.strip_prefix(".agent6-probe-").unwrap_or("");
    if suffix.len() != 32
        || nonce.len() != 64
        || !suffix
            .bytes()
            .chain(nonce.bytes())
            .all(|c| c.is_ascii_hexdigit())
    {
        return Err(io::Error::other("invalid mapping nonce or name"));
    }
    let root = open_directory(cwd)?;
    let dir = open_at(&root, OsStr::new(name), libc::O_RDONLY | libc::O_DIRECTORY)?;
    let host = open_at(&dir, OsStr::new("host"), libc::O_RDONLY | libc::O_NONBLOCK)?;
    if !host.metadata()?.is_file() {
        return Err(io::Error::other("mapping nonce is not a regular file"));
    }
    let mut observed = String::new();
    host.take(65).read_to_string(&mut observed)?;
    if observed != nonce {
        return Err(io::Error::other("host mapping nonce differs"));
    }
    let mut target = open_at(
        &dir,
        OsStr::new("target"),
        libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL,
    )?;
    let answer = random_hex(32)?;
    target.write_all(answer.as_bytes())?;
    if unsafe { libc::fchmod(target.as_raw_fd(), 0o644) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(answer)
}
