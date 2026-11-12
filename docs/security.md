# Security

agent6 treats the LLM as untrusted.

## Reporting

Report privately through GitHub's [Security tab](https://github.com/agent6-dev/agent6/security/advisories/new).
Include: agent6 version (`agent6 --version`), kernel + distro (`uname -a`, `/etc/os-release`), `agent6 check sandbox` output, and a minimal repro (ideally a failing test under `tests/security/`).

## Threat model

Adversary: a fully malicious worker model, or an honest model prompt-injected by a file in the workspace.

The adversary controls the text of every LLM response and the content of any file the agent reads.
It also chooses the tool calls and their arguments, within the published JSON schema.
Outside its control: the kernel, the agent6 binary, the provider endpoints.

The holds below describe the local jail, `full` and `partial` `sandbox.isolation`; `none` confines nothing, and under `container` the container runtime is the boundary ([container transport boundary](#container-transport-boundary)).

**Holds**

- No writes outside the workspace
    - `sandbox.extra_write_paths` widens it, visible in `agent6 config show`
    - the per-repo memory dir (file tools only) and the jail's persistent `HOME` (below) widen it, visible in `agent6 check sandbox`
    - where `sandbox.isolation = "auto"` resolves to `partial` on Landlock ABI 1-2, truncation outside write grants is unconfined (warned)
- No reads outside the workspace and a read-only system set for installed toolchains
    - `full` `sandbox.isolation`: `/usr`, `/bin`, `/sbin`, `/lib`, `/lib64` and `/etc/alternatives` bound read-only into an otherwise empty rootfs, plus a curated `/dev`: the nodes `null`, `zero`, `urandom`, `random` and `full`, a private `/dev/shm`, and a private devpts (`/dev/pts`, `/dev/ptmx`) so a command can allocate a pseudo-terminal that reaches no host terminal
    - `partial` `sandbox.isolation`: Landlock grants `/usr`, `/bin`, `/sbin`, `/lib`, `/lib64`, `/etc` and the `/dev` nodes `null`, `zero`, `urandom`, `random` and `full`
        - `/proc`, the host's `/dev/shm` and the host's `/dev/pts` (the operator's own terminals) are not granted: `ps`, Python `multiprocessing` and pseudo-terminal allocation fail there, and work under `full` `sandbox.isolation`
    - the operator's tool dirs ([Sandbox](#2-sandbox)) and `sandbox.extra_read_paths` add more; `agent6 check sandbox` prints the resolved set
    - a child on the host network (`network = "host"`, or any `sandbox.isolation` but `full`) also reads what name resolution and TLS verification need, where the host has them: `/etc/resolv.conf` and the file it points to, `/etc/hosts`, `/etc/nsswitch.conf`, and the CA store (`/etc/ssl`, `/etc/pki`, `/etc/ca-certificates`).
      File permissions still apply: the root-only private keys beside the certificates stay unreadable to a run that is not root
- `/tmp` is writable at every level
    - `full` `sandbox.isolation`: a private tmpfs discarded with the run
    - `partial` `sandbox.isolation`: the host's `/tmp`
- `HOME` is a directory of agent6's own under a jail; under `none` `sandbox.isolation` it is the operator's
    - `full` `sandbox.isolation`: `/tmp/agent6-home`, created inside the private tmpfs and gone with the run; `[sandbox].home = "cache"` swaps in the persistent dir below, bind-mounted read-write at its real path
    - `partial` `sandbox.isolation`: `$XDG_CACHE_HOME/agent6/homes/<repo-id>` (default `~/.cache/agent6/homes/<repo-id>`), one directory per repository, created `0700` by agent6 and Landlock-granted read-write like the workspace; every other repository's directory stays ungranted
    - `none` `sandbox.isolation`: the operator's own HOME (`[sandbox].home = "host"`, what `auto` resolves to there); `[sandbox].home = "cache"` gives that persistent directory by convention only, since nothing confines the commands, and the start warning says so
    - a linked worktree (`run --worktree`, a fork) gets its repository's directory, named by the git dir the session recorded and never by the worktree's own `.git` pointer; a lane's clone and a machine state's clone get it too, named by the first directory below the workspace base directory, the owning repository's id
    - a symlink, another user's directory, a mode open to group or others, or a path inside agent6's private dirs at that location refuses the run.
      The mode is checked on every run (a jailed command can `chmod` its own HOME), and the refusal names `chmod 700`; nothing restores it silently
- agent6's own git never pushes, force-pushes, rewrites history, or `reset --hard` ([Git](#7-git))
    - a `git` the model runs through `run_command` is bounded by the sandbox instead: `protect_git` keeps `.git` unwritable under `full` `sandbox.isolation`, and push needs egress
- No persistence after the run: no daemon, cron, `.bashrc` write, or setuid binary (one create path stays open, below)
    - the exception is the jail's persistent `HOME` (`partial` `sandbox.isolation`, or `[sandbox].home = "cache"`): a file or binary written there reaches the next jailed run in that repository, and stays executable.
      The operator's own shell, tools, and dotfiles never read it
    - chmod-family syscalls (`fchmodat2` included) deny modes carrying `S_ISUID` / `S_ISGID`; ordinary chmod passes
    - so do the create paths that take a mode: `creat`, `mknod`/`mknodat`, and `open`/`openat` with `O_CREAT` or `O_TMPFILE`.
      `openat2` carries its mode behind a struct pointer, out of seccomp's reach, so a command can plant the bit through it.
      Every jail mount is `nosuid`, and the file is owned by the operator's own uid, so outside the jail it grants no more than that uid (`--allow-root` below is the exception)
    - every mount carries `nosuid` and `nodev`, except the bound `/dev` nodes (the builtin five: `null`, `zero`, `urandom`, `random`, `full`, plus any `sandbox.extra_device_paths` grant), which `nodev` would make unusable, and the private devpts, which carries `nosuid` and `noexec`
    - `/tmp` allows exec (toolchain helpers)
    - children write inside the jail's mount namespace (`full` `sandbox.isolation`) or the Landlock write grants (`partial`)
    - command-session jobs use supervisor-owned descendant cleanup at every isolation level ([Owned-process cleanup](#owned-process-cleanup))
    - a survivor fails the foreground command or background stop; unconfirmed session cleanup emits `jail.degraded`

**Does not hold**

- Under `none` `sandbox.isolation`, a command runs with the operator's own access: the filesystem and persistence bounds above do not apply, though agent6 still sets `HOME` to its own directory ([Sandbox](#2-sandbox))
- The agent process's own egress is unbounded
    - agent6 reaches the configured providers (each `[providers.*].base_url` host, plus the fixed ChatGPT OAuth authority and each `[providers.*].oauth` issuer for token grants); nothing stops the process reaching elsewhere
    - a jailed command's egress is bounded under `full` `sandbox.isolation` unless `sandbox.network = "host"` ([Network](#5-network))
- Under `partial` `sandbox.isolation`, a command can hand work to a user daemon already running (tmux, `systemd --user`): pathname unix sockets have no Landlock hook and stay nameable without a mount namespace.
  Abstract unix sockets are scoped to the jail on Landlock ABI 6 (Linux 6.12); an older kernel leaves them reachable, and the run warns
    - the environment carries no address for one.
      A jailed command gets `LANG`, `LC_ALL`, `TERM` and `CI` from the host, plus `HOME`, `PATH`, `PYTHONDONTWRITEBYTECODE` and `UV_NO_SYNC` set by agent6 (a machine tool also `AGENT6_MACHINE_DATA_DIR`), and the `[git.commit]` name and email as `GIT_AUTHOR_*` and `GIT_COMMITTER_*` when the config sets them.
      A jailed MCP server's base drops `DISPLAY`, `WAYLAND_DISPLAY`, `DBUS_SESSION_BUS_ADDRESS`, and `XDG_RUNTIME_DIR` (`curated_env(desktop=False)`)
    - `full` `sandbox.isolation` does not expose them; under `sandbox.network = "host"` a daemon on an abstract unix socket (an X server) is scoped out on Landlock ABI 6 and reachable below it, warned

## Defense layers

### 1. Trust boundary

The agent's Python process is trusted and runs unconfined at every isolation level; the model is untrusted.

- the process holds the provider keys, writes the per-repo state dir, spawns the jail
- isolation levels differ only in which jail features the launcher enables

The model reaches the machine through two surfaces:

- The in-process tools, whose paths resolve against the file boundary ([File access](#3-file-access)).
- Anything that executes, which runs in the jail ([Sandbox](#2-sandbox)).

The model sees the fixed tool set in `src/agent6/tools/schema.py`, plus the tools of operator-configured MCP servers when `[mcp].enabled` is set.

- structured edits, read-only navigation, `read_session`, `list_findings` (the run's own review findings), fixed-argv verify and metric commands, `finish_session` (`finish_planning` in plan mode), `ask_user`, a curator task notepad, `use_skill` ([Skills](#skills)), approval-gated `fetch` ([Network](#5-network)), capability-gated `run_command` with `read_background` / `stop_background`, and `agent6_docs` under `agent6 ask` only
- no `shell`, no `write_file`, no `eval`
- adding a tool needs a security review note ([AGENTS.md](https://github.com/agent6-dev/agent6/blob/master/AGENTS.md))

Under `backend = "claude_code"` the model runs inside the operator's installed Claude Code binary, unjailed, as the operator, with every built-in tool off (`--tools ""`).
Its only reach into the machine is agent6's tools served over the sdk MCP tunnel, so the dispatcher, the jail, and the approval gates are unchanged.
The binary holds the operator's Claude login exactly as an interactive `claude` does, and agent6 never reads it.

Every line the CLI prints to the operator's terminal, on stdout, stderr or `/dev/tty`, passes one scrubber (`terminal.scrub_terminal_output`).
It drops a control sequence inside text the model influenced (a file name, a commit subject, a summary, a task a plan wrote), so the model cannot write the clipboard (OSC 52), retitle the window or forge a line.

- SGR styling is the allowlist, conceal excepted; an approval prompt drops every sequence from the text under judgment
- the spinners and the interactive composer write their own erase and cursor sequences under the wrapper, the composer with its rows scrubbed
- the ACP and MCP stdio transports write protocol bytes to their peer, not a terminal; the TUI and the web render text as text
- `agent6 exec` is the exception: the operator's own command writes the terminal unscrubbed, so `agent6 exec cat FILE` prints whatever sequences a model-written file holds
    - with stdin and stdout on a terminal the command runs on a pty of its own and agent6 relays the bytes, so it never holds the operator's terminal device.
      The pty is the controlling terminal of the command's session, and the launcher puts the command's process group in its foreground before the command starts.
      A command the pty stops is resumed by the launcher, which first puts its group back in the foreground
    - otherwise it inherits agent6's stdin, stdout and stderr as they are, a terminal among them included: it can change that terminal's modes, which agent6 puts back when the command ends
    - it runs in a session of its own, so an inherited terminal is never its controlling terminal: the kernel refuses it `TIOCSCTTY` on that terminal, and it takes no foreground there

### 2. Sandbox

Every tool call that allows the model to run arbitrary commands (`run_command`, `run_verify_command`, `run_metric_command`, backgrounded commands, and spawned MCP servers without `unconfined = true`) runs in a jail at the effective isolation level.

`agent6 check mcp` and `agent6 mcp add` start a spawned MCP server under that jail with the repository bound read-only.
`check mcp` applies the run's network and config refusals first and leaves a server it cannot hold that way unstarted: `unconfined = true`, a write grant (the server's `extra_write_paths`, `sandbox.extra_write_paths`, `sandbox.extra_device_paths`), or no jail at all.
`mcp add` probes under the same jail, and skips the probe only when there is no jail at all or a run would refuse the config, saying so.
It is the operator's own invocation naming the server they are adding.

### Effective controls

An execution target describes each sandbox leaf with its configured value, effective value and behaviour, enforcement owner, probe state and any warning or refusal reason.
The owner vocabulary is `agent6`, `container`, `host file tools` or `unsupported`; local preparation uses no container owner.
`resolved` means policy selection, not a successful enforcement probe; `not probed` claims no inspection, and `prepared` means preparation was observed.
A ready launcher alone does not prove every configured path exists or every optional kernel feature works.
Startup diagnostics and live sandbox checks retain those qualifications.

The record includes command approval, fetch approvals, path and device grants, HOME, networking, git protection and the per-process memory cap.
Under `none`, file-tool restrictions remain host-enforced while commands have no filesystem, device or network confinement.
Under `partial`, Landlock grants do not provide mount masks or a private network.
Preparing the session does not grant command permission: configured deny and approval requests remain in force.
Local executable lookup runs a fixed `/bin/sh` probe through that session with its command environment; candidate names are positional arguments, not shell text.
It opens executable PATH files for reading to check target visibility and Landlock grants, without reading their contents or running candidates or a PATH-selected shell.
Confirming visibility requires read access; a builtin alone cannot satisfy a direct argv command.
Shell-wrapped builtins recognized by inference need an available `sh`, not an external namesake.
An unreadable execute-only PATH candidate leaves visibility indeterminate unless a later readable match exists.
Indeterminate visibility, a timeout or an abnormal probe exit refuses inference or adoption rather than reporting an executable absent.
An explicit command path skips this advisory lookup, not command approval or confinement.

**Modes** (`[sandbox].isolation`)

| Mode | Applies |
|---|---|
| `full` | user namespace (one uid, the operator's own) + mount + PID + IPC + UTS namespaces (hostname `agent6`), a network namespace unless `sandbox.network = "host"`, `pivot_root` rootfs, private `/proc` and `/tmp`, curated `/dev`, Landlock where the kernel has it, seccomp, `NO_NEW_PRIVS`, capability drop, timeout |
| `partial` | Landlock, seccomp, `NO_NEW_PRIVS`, capability drop, timeout. No namespaces, no rootfs |
| `none` | No confinement: no namespaces, Landlock or seccomp. The timeout and `memory_limit_mib` still apply |
| `container` | None from agent6: commands run inside the selected container through the launcher's `none` branch; the runtime and the container's mounts bound them ([container transport boundary](#container-transport-boundary)) |
| `auto` *(default)* | The strongest of `full`, `partial` and `none` this host supports; never `container` |

Under `full` `sandbox.isolation` a kernel without Landlock skips that layer, warned once per run; under `partial` Landlock is the only filesystem boundary.

**Resolution**

Probed per host: one `full` jail launch, the Landlock ABI syscall, the seccomp filter's architecture (x86_64 and aarch64).
A failed trial launch makes `full` unavailable; the refusal and automatic downgrade reason quote its failure text.
That failure does not necessarily mean user namespaces are blocked.

| Setting | Host | Effective |
|---|---|---|
| `auto` | namespaces | `full` |
| `auto` | no namespaces, Landlock ABI >= 3 | `partial`, warned |
| `auto` | no namespaces, Landlock ABI 1-2 | `partial`, warned: truncation outside write grants is unconfined |
| `auto` | no namespaces, no Landlock | `none`, warned |
| `auto` | non-Linux, or no seccomp filter for the arch | `none`, warned |
| `auto` | no `agent6-jail` launcher installed | `none`, warned |
| `full` | no namespaces | refused |
| `partial` | Landlock ABI < 3 | refused |
| `full`, `partial` | non-Linux, or no seccomp filter for the arch | refused |
| `full`, `partial` | no `agent6-jail` launcher installed | refused |
| `none` | any | `none`, warned |

`--disable-sandbox` and `AGENT6_DISABLE_SANDBOX=1` force `none` `sandbox.isolation` for one invocation.
Config, flag, and env var are operator-only; the model reaches neither argv nor env.

Inside a container the same probes decide the level.
Measured 2026-10-27 on an Ubuntu 26.04 host in `mcr.microsoft.com/devcontainers/base:ubuntu`, `sandbox.isolation = "auto"` resolves to:

- `partial` under Docker 29.1.3, and `full` with both `--security-opt seccomp=unconfined` and `--security-opt apparmor=unconfined`
- `full` under rootless Podman 5.7.0
- `partial` under rootful Podman 5.7.0, and `full` with `--security-opt apparmor=unconfined`

On Ubuntu (`kernel.apparmor_restrict_unprivileged_userns = 1`), `full` under Docker or rootful Podman also needs `agent6 system apparmor install` on the host or that sysctl at 0; rootless Podman does not.
Each `--security-opt` flag on this page turns off a protection of the container itself (its seccomp filter, AppArmor profile or `/proc` masks) for every process in it.
`agent6 check sandbox` names the level and what blocks a stronger one.

**Inside the jail**

- Mounts (`full` `sandbox.isolation`): cwd and a private `/tmp` writable, system paths read-only, `extra_read_paths` / `extra_write_paths`, a persistent `HOME` under `[sandbox].home = "cache"`, and operator tool dirs at their real paths
    - every mount keeps the path it has outside
    - the workspace is mounted at its resolved path and the launcher starts the command there: under a workspace named through a symlink, `pwd` prints the resolved path
    - a grant is resolved once, when the run starts, and every jail launch of the run binds that resolved directory at the grant's configured path: `/opt/tool -> /opt/tool-1.2` shows the content of `/opt/tool-1.2` at `/opt/tool`, and a link re-pointed during the run changes nothing.
      Under `partial` `sandbox.isolation` the Landlock rule goes on the resolved directory
    - a grant that contains the workspace is mounted under it: the workspace stays writable under a read grant
    - a grant the launcher cannot bind fails the launch, and the error names the grant, the step and the OS error (`write grant /srv: remount of submount cache: EPERM`); a read-only mount inside a write grant is such a case
    - a grant on `/` or on `/dev` fails the launch by name: `/` would replace the jail's own root, and `/dev` would cover the jail's device nodes (name a device in `extra_device_paths`)
    - a grant whose path an earlier grant already shows (a file listed twice, a file inside a granted directory) is bound over it.
      Where the earlier grant shows a symlink at that path, nothing is mounted through the link: a read grant leaves the link as it is, and a write grant fails the launch by name
    - a grant that is the home directory or a directory above it lets the run's commands and file tools reach the credentials other tools keep there (`~/.ssh`, `~/.aws`, another agent's login).
      agent6 warns about such a grant at run start and in `agent6 check sandbox`.
      Under `full` `sandbox.isolation`, agent6 hides only its own config dir and state base, plus the paths in `sandbox.hide_paths`
    - the paths kept read-only inside the workspace (`.git` under `protect_git`, a machine bundle) are bound after every grant: a grant that contains the workspace does not make them writable
    - a tool shim whose target passes through a directory symlink grants the parent of the first such link read-only, so the link stays visible.
      `agent6 check sandbox` names the link in the grant's row.
      If that parent is refused, or the target has no directory link, a symlink into `<root>/bin` grants `<root>` read-only; other targets grant their directory.
      A refused root keeps the `bin` grant.
      Refused widening: a dir holding a jail PATH directory, a dir at, inside, or containing an agent6 private dir, and `$HOME` or an ancestor of it.
      The row names each refusal
    - a session started in a linked worktree also grants the repository git dir, including its `worktrees/<name>` admin entry with HEAD and index, at their real paths: a read-only bind under `full` `sandbox.isolation`, a Landlock read+exec rule under `partial`, in both cases whatever `protect_git` says.
      The main checkout's `.git` is read-only only under `full` `sandbox.isolation` with `protect_git`, so a worktree's is the more confined of the two.
      A write grant inside the recorded git dir refuses the start: the jail protects paths under the workspace only, so no grant lifts the bind's read-only mode.
    - the granted dir is the manifest's `worktree_git_dir`, recorded when agent6 adds a worktree or validates the checkout's pointer at session start, before any model command.
      A pointer outside a valid repository refuses startup in one line naming `.git`.
      Resume and same-checkout forks inherit the record rather than creating a grant from the writable pointer.
      The policy builder refuses when the pointer no longer resolves to the record.
      A policy with no record gets no grant
    - under `partial` `sandbox.isolation`, a `hide_paths` entry inside the recorded dir refuses like any other unmaskable entry
- Masked last, after every bind (`full` `sandbox.isolation`): the config dir, the state base, the workspace and project roots' `.agent6/` pack directories, `[sandbox].hide_paths`
    - each is masked under its given name and under the path it resolves to, so an entry spelled through a symlink covers the real directory
    - a hidden path inside the tree a symlinked grant resolves to is masked under the grant's spelled name too, where the jail shows that tree
    - a hidden directory reads as an empty directory and a hidden file as an empty file; a write to either fails with `Read-only file system`
    - a grant at or inside a private dir is refused at config load
    - `partial` `sandbox.isolation` cannot mask: a grant containing a private dir warns, an unmaskable `hide_paths` entry refuses
    - a `sandbox.extra_read_paths` or `extra_write_paths` entry inside a `hide_paths` entry refuses at every jailed level: the launcher binds a grant through the mask above it
    - every run, resume, machine and exec holds the checkout's workspace lease (shared) for its lifetime, and `pack` takes it alone before it writes, so neither starts under the other; a nested checkout's pack also needs the leases of the checkouts above it
    - a present `.agent6/pack.json` refuses startup at every isolation level, including `partial`.
      Run `agent6 unpack` in that workspace or delete its `.agent6/` directory
    - run preflight scans the project tree once for nested packs, without following symlinks or entering `.git`, and refuses with every pack's path.
      Trees reached through `extra_read_paths` grants are not scanned: unpack or delete a pack before granting the tree that holds it
    - a `hide_paths` entry that is the workspace or a directory above it refuses at every level, before anything runs: drop the entry, or work in a directory outside it
    - a `hide_paths` entry hides a path, not content git tracks: a tracked file still reads through git's objects (`git show HEAD:<file>`) and in every clone or worktree a lane, a fork, a `run --worktree` run or a machine state works in
    - a `hide_paths` entry inside the workspace stays out of every tree agent6 writes or compares (chain commits, review diffs, dirty checks, undo, resume) and out of the run's pre-run stash, whose change stays in place: no blob of it enters the object store through agent6
    - `XDG_CONFIG_HOME` and `XDG_STATE_HOME` relocate those two dirs, `XDG_CACHE_HOME` the persistent `HOME`; the mask, the grant validator, and the tool-mount scan all read the relocated path
- `/dev` (`full` `sandbox.isolation`): `/dev/null`, `/dev/zero`, `/dev/urandom`, `/dev/random`, `/dev/full`, a private `/dev/shm`, a private devpts at `/dev/pts` with `/dev/ptmx` pointing into it; no `/dev/tty`
    - `sandbox.extra_device_paths` binds named `/dev` nodes read-write (GPU compute).
      Each must be a char/block device on the host, or the launch refuses.
      On `partial` `sandbox.isolation` the same grant is a Landlock read+write rule on the node
    - creating one is denied at both levels: Landlock handles `MakeChar` / `MakeBlock` and grants them nowhere, and seccomp `EPERM`s the `mknod` pair by device type (below).
      `full` `sandbox.isolation` also has the user namespace (no `CAP_MKNOD` in the initial one)
    - under `partial` `sandbox.isolation` with `--allow-root`, Landlock, seccomp and the capability drop deny it
- `/` (`full` `sandbox.isolation`): Landlock permits directory listings across the private mounted tree, including its synthetic parents and empty hidden-path masks, without granting file reads, execution or writes.
- `/proc` (`full` `sandbox.isolation`): fresh and private, empty if that fails; Landlock grants it read without execute
    - the launcher, PID 1 there, hides its `/proc` entry from the command (dumpable 0, capabilities dropped) and runs with an empty environment
- seccomp: a 36-syscall deny-list returning `EPERM`.
  It covers process inspection (`ptrace`, `pidfd_getfd`, `process_vm_readv`/`writev`, `kcmp`), `io_uring_setup`, `userfaultfd`, the whole mount family (`mount`, `umount2`, `pivot_root`, `mount_setattr`, `open_tree`, `move_mount`, `fsopen`, `fsconfig`, `fsmount`, `fspick`), `setns`, `unshare`, `kexec`, `bpf`, `perf_event_open`, the keyring calls (`keyctl`, `add_key`, `request_key`), module loading, `reboot`, swap, and the clock-setting family
    - the list denies unconditionally.
      Argument-conditional rules `EPERM` two more cases on syscalls that stay allowed: a mode carrying `S_ISUID` / `S_ISGID` on `chmod` and the create family (above), and a `mknod` / `mknodat` naming a character or block device (`/dev` above)
    - anything else is allowed; the list itself is the source (`jail/src/main.rs`)
- Capabilities: cleared between fork and exec.
- Timeout: the policy's `timeout_s` (verify and metric gates set it from `[harness].verify_timeout_s`, default 600), then SIGKILL of the process group, rc=124 and `timed_out=true`.
  A command's own exit 124 is a nonzero failure, not a timeout.
    - a model's `run_command` in a run is not wall-clock killed: at `[harness].command_checkin_s` it is handed back as a background job ([config](config.md#harness)).
      Under `agent6 plan` and `agent6 ask`, in a machine state, without a jail session, or with `command_checkin_s = 0`, it is killed after 600 s
- One launcher per run at every isolation level; under `full` `sandbox.isolation` its commands share that netns, PID namespace, and private `/tmp`, and under `partial` the host's network and `/tmp`
    - EOF on the run's command channel stops the foreground job without waiting for its deadline or handing it back.
      The supervisor stops its owned jobs and descendants at every isolation level, without scanning the shared PID namespace
    - a launcher that cannot start refuses execution with its preparation error; it never falls back to another runner
- The policy arrives as JSON on a pipe from agent6, validated against a strict schema; unknown fields are refused.
- The [supervisor protocol](architecture.md#jail-supervisor-protocol) uses stdin for requests and request-correlated cancellation, stdout for responses, and stderr for launcher diagnostics.
  One nonblocking reader owns stdin even while a foreground command runs, and only the serve loop writes responses.
    - foreground and background commands receive `/dev/null` on stdin and separate stdout/stderr pipes
    - commands do not inherit the protocol streams or private job channels; dumpable 0 on both supervisor layers and the child's capability drop prevent unprivileged same-user children from reopening their descriptors through `/proc/<pid>/fd`
    - protocol version, request IDs and frame-size checks reject incompatible or malformed peers; they do not authenticate a launcher the operator selected or provide confinement under `none`
    - malformed input stops the active job before closing the session; stale cancellation IDs cannot stop another request
    - Python drains launcher stderr independently with bounded retention, so diagnostics cannot fill that pipe while it waits for a response

### Captured output

The launcher opens each capture with `O_TMPFILE | O_CLOEXEC` in its own temp directory (`/tmp` locally, the private installation directory for a container transport), before applying the job's Landlock ruleset.
Installation state selects that directory; inherited `TMPDIR` cannot redirect capture storage.
Commands inherit only their output pipes, not the capture descriptor; there is no log pathname to replace or symlink.
The supervisor reads only its owned capture by handle and bounds each response to 64 KiB of raw bytes.
Host persistence uses descriptors opened with exclusive creation and no symlink following, outside command grants.
A replaced host log path never selects bytes to read or a destination to write.
These protections do not provide filesystem confinement under `none`.

Output is untrusted data, never a completion signal.
Raw bytes are preserved on the host; tool output scrubs terminal escapes and labels tail truncation.
Background completion comes from supervisor status and stop responses; losing the channel records an unknown outcome unless an exit code was already received.
Copied bytes survive channel loss, but unread target bytes may be lost.
The per-response bound does not limit total log storage; capture and host persistence failures are reported rather than treated as command success.

### Owned-process cleanup

The serving launcher owns a table of instance-scoped job handles, not client-supplied pids.
Each job has a private subreaper process, so `setsid` and double-fork descendants stay with that job even when another job starts later.
The session supervisor also sets `PR_SET_CHILD_SUBREAPER`, including outside a private PID namespace.
No cleanup assumes PID 1 or sweeps unrelated processes in a shared namespace.

- Each subreaper opens its own `/proc/thread-self/children` before filesystem Landlock enforcement.
  The descriptor is close-on-exec and grants no command extra `/proc` access.
- The session supervisor prepares and validates the filesystem ruleset; each job subreaper applies it before spawning the command.
  The trusted session supervisor retains filesystem access needed to create later subreapers, while seccomp and the abstract-socket scope remain shared across the session.
- Cleanup signals only children held unreaped by that owner, then repeats as descendants reparent to it.
  Each cleanup pass is bounded by five seconds and reports survivors rather than claiming success.
  Process-group membership grants no cleanup authority: another job can join the same group.
- Foreground completion includes descendant cleanup; a background job retains its descendants until stop or session close, even after its leader exits.
  Status and repeated stop preserve the observed exit code even when descendant cleanup fails.
- Host command-session cleanup closes or kills only the launcher transport, never job pids.
  EOF on the host channel triggers session cleanup; EOF on a private job channel triggers that job's cleanup if the session supervisor dies.
  Host teardown does not kill the transport's process group, which also holds job subreapers.
- Completion is a supervisor response persisted by Python, not a writable command file or a transport exit code.
  Losing the channel before receiving completion leaves the command outcome unknown; a previously received exit code remains known.

Cleanup is not a guarantee against uninterruptible kernel sleep, lost signal permission, or destruction of every owning supervisor outside a private PID namespace.
`partial` and `none` do not stop same-user commands from signalling their supervisors; `none` also provides no filesystem or privilege boundary.
A lost transport leaves cleanup unconfirmed even if the remaining subreaper succeeds.
Under `full`, destruction of the namespace's init additionally makes the kernel kill its remaining processes.
The one-shot and MCP transports keep their separate host-process cleanup; they do not interpret session job handles.

**Verify**

`agent6 check sandbox` prints the `sandbox.isolation` this host resolves to and the reason when it is not `full`.
It then runs live probes at that isolation.
It exits 1 when a check fails, 0 otherwise.

| Probe | Passes when |
|---|---|
| `landlock_abi` | the kernel reports Landlock ABI >= 1 |
| `jail_true` | `/usr/bin/true` runs in the jail, rc=0 |
| `jail_blocks_network` | a jailed `getent hosts example.com` fails. n/a under `partial` `sandbox.isolation`, which has no network namespace |
| `jail_blocks_etc_write` | a jailed write to `/etc/agent6-escape` leaves no file on the host |

### 3. File access

Jailed commands in a linked worktree also receive the recorded [git-directory bind](#2-sandbox), but that grant adds no path to the file tools' boundary.

`read_file`, `list_dir`, `outline`, `find_definition`, `find_references`, `apply_edit`, and `apply_patch` run in the agent process and ask no approval.
Every path they take resolves through `Workspace`:

    workspace root
      + [sandbox].extra_read_paths (read) / extra_write_paths (write)
      - [sandbox].hide_paths
      - agent6's own config dir and state base
      - any path with a .agent6 component

- The boundary comes from config values
    - a degradation (`auto` falling back to `partial` or `none`, a host with no jail) never widens it; under `none` `sandbox.isolation` it is the only boundary
- A grant is resolved once per run, and the file tools and every jail launch hold that resolved path
    - a grant spelled inside the workspace root or a write grant, or resolving into one, is dropped: a jailed command can re-point it there.
      The run names each dropped grant in one line at its start, and the region still covers what the grant held
    - a grant reached through a link that sits in one of those regions is dropped too, though it ends outside: a command of one run could point the link elsewhere for the next.
      Its line says to grant the target by its own path.
      `/tmp` counts as one of those regions under any `sandbox.isolation` but `full`, and the file tools then lose what a grant there held
    - an MCP server's own `extra_read_paths` and `extra_write_paths` (`[mcp.servers.<name>.sandbox]`) follow the same rule, resolved once when the server is spawned; a dropped one is named at run start under its own key
    - `agent6 check sandbox` lists each grant by its resolved path, with the configured spelling when the two differ, and each dropped grant with the region it sits in
- A denied path is refused with the reason
    - `list_dir` drops the entry and reports `hidden: N`
    - under `full` `sandbox.isolation`, the jail masks the path instead (empty dir, empty file; [Sandbox](#2-sandbox))
- The symbol index skips a hidden file (an indexed one leaks symbol names and line numbers through `find_definition`)
- While `sandbox.protect_git` is on (`auto`, the default, is on except where `git.control = "model"` needs a writable `.git`) the edit tools refuse a write into the project's own `.git`, raw or symlink-resolved, at every isolation level
    - the name matches case-folded on every platform (macOS and Windows open `.GIT/config` as `.git/config`; macOS runs unsandboxed)
    - a `pyvenv.cfg` dir, a `site-packages` ancestor, and an operator protect path (a machine bundle's `.asm.toml` and `scripts/`) refuse the same way, whatever `protect_git` says
    - reads stay allowed; an editable install still imports itself inside the jail (the repo is bound at its real path)
- Inside the per-repo state dir (`$XDG_STATE_HOME/agent6/<repo-id>/`), the edit tools may write `memory/` and nothing else.
- Uncovered: with `sandbox.isolation = "none"` and `run_commands = "allow"`, a command reads a denied path directly (`run_command` argv is not screened)

**Prompt reads.** The agent process itself reads `AGENTS.md` and the manifests the verify inference uses (`pyproject.toml`, `package.json`, a Makefile and the like), and their text reaches the provider.

- A file is read only when its real path stays inside the workspace root (the git root for `AGENTS.md`), outside `.git` and outside the hidden set; anything else reads as absent.
- A symlink that stays inside the root is followed (`AGENTS.md -> docs/AGENTS.md`).
- The existence probes (a root `verify.sh`, a manifest, loose `test_*.py` files) and the model call's top-level listing leave out a file under the hidden set, by its name or its link target.
  A link out of the root still counts as present.

**Code-review rules.** The reviewer reads historical `AGENTS.md` blobs from the base tree, not the checkout.
The workspace hidden-path policy applies before reading a rule blob.
A checkout symlink cannot redirect a configured hidden path away from its historical name.
Historical symlinks resolve within the same base tree, through at most eight links, to a regular file outside the hidden set.
Current checkout links do not redirect a historical read.
A present file that cannot be read safely makes the review incomplete; missing optional files do not.

`AGENTS.md` cannot grant tools, network access or operator permissions, and cannot alter the aggregator's severity restrictions.
Model obedience is not a security boundary.

Review text notices escape control characters in rule paths and refusal reasons.
See [Code-review rules](usage.md#code-review-rules) for discovery, limits and recorded sources.

**Repo memory** is a prompt-injection persistence channel: the rendered index joins the system prompt of later runs on the same repo.
Memory (`<state_dir>/memory/`) holds facts the model may correct; rules belong in the repo's committed `AGENTS.md` or the operator rulings file `<state_dir>/decisions.md`, which the harness appends to, the operator edits with any editor, and the model sees only as read-only text injected into its prompt.

- Entries are inert data.
- The injected index is size-capped and framed as untrusted.
- The store is auditable (`agent6 memory list` / `show`).
- The jail never mounts it.
- Files are freely editable, so a hostile write can erase earlier memories.
- Sandbox, egress, and git policy come from config, so memory content cannot move them.

### 4. Commands and environment

Every command tool (`run_command`, `run_verify_command`, `run_metric_command`, `stop_background`) answers to `[sandbox].run_commands`; the first three run jailed, and `stop_background` signals a jailed child agent6 already started.
`run_commands = "deny"` withholds the verify gate too.
Under `ask`, a denied gate is withheld the same way for the rest of the run (no retry loop can discharge a refusal), and the run ends unverified.

The agent works within the environment it is given:

- Under `partial` or `full` `sandbox.isolation`, `sudo` cannot escalate, even passwordless: `NO_NEW_PRIVS` voids setuid, so jailed `sudo` fails under any `NOPASSWD` rule.
- Under `partial` or `full` `sandbox.isolation`, package installs fail: `apt` / `dnf` / `apk` need root (blocked), a route to a mirror (the default `network` has none off the box under `full` `sandbox.isolation`), and `/usr` or `/var` writes (denied).
- Host-installed toolchains compile and run; a networked build step needs `network` loosened under `full` `sandbox.isolation`.
    - a toolchain that keeps its state under your home needs that directory granted and named, because the jail's `HOME` is its own; see [Rust toolchains](#rust-toolchains) for rustup's `cargo`
- Toolchains, venvs, and deps are installed outside agent6.
  Access widens through config (`extra_read_paths`, `network`), all visible in `config show`.
- The setup command of an isolated workspace (`[workspace].setup_command`, or an AGENTS.md `## Setup command` block) is a jailed command like any other
    - Machine clones use the operator's configured command, else the AGENTS.md block at the starting commit recorded in `machine.begin`, never a block committed by a run state; read-only agent states run this setup too, under their command policy and jail
    - A machine clone needing setup refuses `sandbox.run_commands = "ask"` at preflight rather than asking; use `--auto-approve` or set it to `"allow"` to allow setup
    - it runs through the run's own jail: the isolation, network and grants every command of that run gets, the approval `run_commands = "ask"` asks, and nothing under `"deny"`
    - it has no network of its own and no read of the checkout the workspace was cut from: a setup step with a wider network would let a dependency it fetches run with it
    - an AGENTS.md block is repository content, so a repository can state the command; it reaches what a command the model chose would reach
    - under a sandbox with no route out (the default network under `full` `sandbox.isolation`), the command has to work offline
    - the recipe measured under `full` `sandbox.isolation`, for a uv project:
        1. `agent6 config set --repo sandbox.home cache`, so jailed commands in this repository share one persistent `HOME` and its tool cache
        2. warm that cache once with your own networked command, from the checkout: `agent6 --config net.toml exec --new -- uv sync --reinstall --link-mode=copy`, where `net.toml` sets `[sandbox] home = "cache"` and `network = "host"` (the host network brings the resolver and CA files with it).
           The reinstall fetches every locked package; a plain `uv sync` over a checkout whose `.venv` is complete fetches nothing, and step 3 then fails offline.
        3. `agent6 config set --repo workspace.setup_command '["uv", "sync", "--offline", "--link-mode=copy"]'`: a new worktree's `.venv` is then built from the cache with no route out (1.2 s for a two-package project), importing the worktree's own source.
           `--link-mode=copy` on both steps: the jail binds the cache `HOME` and the working tree as separate mounts, so uv cannot hardlink between them and warns on every install without it
    - the warmed cache serves every working tree agent6 makes for the repository: a `run --worktree` worktree, a fork's, a `--parallel` lane's clone and a machine state's clone all get the repository's `HOME`
- Running agent6 as root needs `--allow-root` / `AGENT6_ALLOW_ROOT=1` (plus a banner) and weakens the boundary
    - jailed children run as real root (uid 0), even under `full` `sandbox.isolation`: the uid is no boundary, and the level's layers (the table in [Sandbox](#2-sandbox)) are the whole confinement
    - a setuid bit planted through `openat2` (the one create path seccomp cannot filter) then lands on a root-owned file in the workspace.
      That is local root for whoever runs it outside the jail
    - under `full` or `partial` `sandbox.isolation`, running as root adds no write grants or network routes
    - readable files include root-only ones (`/etc/shadow` under `partial` `sandbox.isolation`; `full`'s rootfs hides it)
- Under `sudo`, agent6 reads the real user's config and secrets (`SUDO_UID` / `SUDO_USER`) and chowns config and secrets writes back, plus the state dir at run teardown, not workspace files.
    - it does not drop privileges in-process; confinement comes from the jail

#### Rust toolchains

Finding `cargo` on the jail's PATH does not mean rustup can find its toolchain.
The jail exposes `~/.cargo/bin` read-only, but its own `HOME` has neither your rustup settings nor the toolchains under `~/.rustup`.
The proxy can fail with `rustup could not choose a version of cargo to run` even when `agent6 check verify` finds it.
That check only infers a command or checks its first executable; it does not execute the gate.

For an installed rustup toolchain, grant `sandbox.extra_read_paths = ["/home/you/.rustup"]` and name that directory in the gate: `env RUSTUP_HOME=/home/you/.rustup cargo test`.
Use your actual absolute path in both places.
Test the command with `agent6 exec --new -- env RUSTUP_HOME=/home/you/.rustup cargo test` before a run.
A system-installed Rust toolchain does not need this rustup grant.

Cargo dependencies are separate from the toolchain: an offline build also needs vendored dependencies or a populated Cargo cache in the jail's HOME.
Use `sandbox.home = "cache"` to retain that private cache across runs, and populate it with an operator command whose network policy permits the fetch.
Do not grant all of your host `~/.cargo` just for its cache: it can contain registry credentials and configuration that jailed commands would then read.

### 5. Network

The agent process's own egress is unbounded.
Its own provider and credential requests follow no redirect, so a bearer goes only where the configured URL says; a request to a loopback destination ignores the proxy variables, so a plaintext loopback bearer stays on the host; a `base_url` or OAuth endpoint carries no userinfo and no fragment.
Model listings, key probes and ChatGPT plan readings use the capped provider request path.

`agent6 telemetry export` is host egress too: it dials `[telemetry].endpoint` from the agent6 process with the header values `secrets.toml` holds, outside `sandbox.network`, and only when the operator selected a signal.
The default capture carries outcomes, counts, timings and the journal's ids; `capture = "content"` adds the capped excerpts `telemetry.capture` lists ([config](config.md#telemetry)), which can hold anything a run saw.
The export dials the endpoint directly: no proxy is taken from the environment, so a plaintext loopback endpoint stays on loopback.
A plaintext endpoint is accepted on loopback only; the Prometheus textfile is written atomically to the operator's path.

A `claude_code` provider's child dials `api.anthropic.com` and `claude.ai` (login refresh) from the agent process; telemetry, crash reporting, and update hosts are disabled by its environment.
While it runs it listens on a same-uid session socket under `/tmp/cc-socks/`, reachable from a jailed command under `partial` and `none` `sandbox.isolation`, whose `/tmp` is the host's.

**`fetch`** is the model's only direct egress.

- One https URL, GET, no redirects followed, no credential, text only, 1 MiB, 20 s.
- Hosts on `sandbox.fetch_approved_hosts` are read without asking (empty by default); any other host prompts.
- Nothing resolves before that gate: a DNS query delivers the hostname to whoever runs its authoritative server, so an unapproved URL never reaches a resolver.
- Behind the gate the name resolves, and every address it yields must be global: loopback, the LAN, and `169.254.169.254` are refused, as is a URL naming one literally.
  The request dials the resolved address with the name in SNI and `Host`, so no second lookup can rebind it.
- A `content-encoding` other than identity is refused: the cap counts bytes as they arrive, and a decoded stream would expand past it.
- Hidden when a jailed command already has the host network: `network = "host"`, or any `sandbox.isolation` but `full`.

**Which network a jailed child joins.** `sandbox.network` is one of:

- `auto` *(default)*: the run's own network where the host can give one, the host's with a warning where it cannot.
- `session`: the run's own, enforced; refuses where the host cannot give one.
  Its members reach each other and nothing off the box.
- `host`: the machine's network, with the host's resolver settings and CA store readable ([Threat model](#threat-model)).

A run's commands share one launcher, so there is no per-command `sandbox.network = "none"`.
A machine state sets its own value in its `config` table, never wider than this one ([State machines](#11-state-machines)).

The run owns one session network.
A holder process creates it, the run keeps it alive with an open descriptor on `/proc/<holder>/ns/{user,net}`, and every child that asks joins those.
Entering a network namespace needs `CAP_SYS_ADMIN` in the user namespace that owns it, so a joiner enters that user namespace too.
It still gets its own mount, PID, IPC, and UTS namespaces, so two members cannot see or signal each other.

Under `full` `sandbox.isolation`, the only level with namespaces:

| jailed command | `auto` *(def)* | `session` | `host` |
|---|---|---|---|
| `run_command` | the run's session network | same | host network |
| a machine `tool` state | its own, alone | its own, alone | host network |

The column is the value the command's config resolves to: your `sandbox.network`, narrowed by the machine's `[config]` or set by the state's own `config`.

Below `full` `sandbox.isolation` there is no network namespace: `auto` degrades to the agent process's network with a warning (once per run), while `session` refuses.

**Refusals** (fail-closed)

| Configuration | When |
|---|---|
| a machine overlay sets `network = host` under your `auto` or `session` at `full` `sandbox.isolation` | `machine check`, machine start |
| explicit `network = session`, yours or a machine state's | run or machine start under `partial` or `none` `sandbox.isolation` |

**MCP servers** take their own value per server (`[mcp.servers.<name>.sandbox].network`, default `auto`):

- `auto`: a network of its own where the host can give one, degrading to the host's with a warning.
- `mcp.servers.<name>.sandbox.network = "none"`: refuses instead of degrading.
- `session`: joins the run's network, so a browser server reaches the dev server a background command started.
- `host`: the machine's network.
- A server is spawned on stdio or dialled at an operator-set `url`, outbound either way.

**Ingress.** The loop opens no accept-side socket.
`agent6 web` opens one, and only when started.

- It binds loopback (`127.0.0.1`) by default and speaks plain HTTP (for remote access run it behind `tailscale serve`; see [the web UI](web.md)).
  A non-loopback bind needs `[web].allow_non_loopback = true` for `[web].host`, or `--allow-non-loopback` for `--host`.
- Every `/api` route requires the sign-in token `POST /api/login` answers for the server's secret, sent as `Authorization: Bearer` ([sign in](web.md#sign-in)).
  Without it the answer is 401; the page shell, its static assets and the login are the only public routes.
    - The secret is minted per launch (six words of the session-id wordlist, about 44 bits) and printed on the terminal, or it is the operator's own (`--secret`, `AGENT6_WEB_SECRET`), refused under 12 characters.
    - It keeps out a local process, and a jailed command where the jail shares the host network: without it either could write the config (`sandbox.isolation = none`) or read transcripts the jail masks.
    - It does not protect the transport: there is no TLS, so on a non-loopback bind the secret and the token are readable on the network path.
    - It does not keep out whoever can read the terminal `agent6 web` prints to, the operator's environment, or the process list when `--secret` is used.
    - The login has no lockout and no rate limit; the secret's length carries it.
    - The token is an HMAC-SHA256 keyed by the secret, never the secret.
      The page keeps it in storage scoped to its origin (host and port) and sends it in a header, never a cookie: a browser sends a cookie to every port of the host, so a page a jailed command serves on a forwarded port would receive it on the first visit.
      The page shell, assets and JSON responses carry `Content-Security-Policy: frame-ancestors 'none'`, so such a page cannot frame the signed-in UI either.
- With `telemetry.metrics = "prometheus"` and `web.metrics_route = "on"` (default: `off`), `/metrics` uses the existing listener and requires `Authorization: Bearer <web-secret>`, never a cookie or browser sign-in token.
- The server renders folded state and drives typed contracts.
  New-work spawns fixed argv with the task behind `--`, and machine-run is allow-listed to authored files.
  Answers write only the addressed run's answer files (session id, answer id, and machine target state dir each validated to one path component).
  Merge, prune, and config-set are fixed agent6 subcommands.
- No request carries an ambient credential, so no POST needs a cross-site guard: a cross-origin page cannot add the header without a preflight the server never answers.
  DNS rebinding is not covered (that needs a Host allow-list incompatible with the tailnet name): a rebound page holds no token (storage belongs to one origin), so it reaches the public routes and can try secrets at the login; keeping it out is left to the network layer.
- Request framing is bounded: 1 MiB body cap (413), chunked refused (411), and any unread-body refusal closes the connection.
- The machine write surface (`POST /api/machine/<name>/{poke,stop,steer,approve,answer}`) uses the same guards.
  `poke` writes the instance's signal file (inert JSON the next `tool` reads) and `stop` its stop marker; `steer`, `approve` and `answer` write only the current agent state's per-state dir.
- Finding actions use the same authenticated session lookup and body limits.
  Fix constructs steering text from stored evidence, not a request-supplied instruction.
  Resolve and Reopen validate reasons and duplicate references before writing.
- A live worker is the only journal writer for finding actions.
  `sessions.ipc` publishes whole request and reply files under the private session's `review-actions/`; request IDs cannot escape that directory.
  Without a live worker, the action path acquires the `worker.lock` and rechecks liveness and the review version before appending.
- All agent6 merge paths enforce open grounded block findings in `app.merge.execute_merge()`, independent of browser controls and `review.block_on`.
  The session writer lock excludes dispositions and concurrent session writers until the pinned source commit lands and its merge stamp is recorded.
  Git's conditional target-ref update still refuses a concurrently moved target.
  Resolve-with-reason is the only finding bypass; unreadable evidence refuses with its reason.
- Review comparison and file routes read Git objects by recorded tree ID.
  Paths and sides pass `ReviewLocation` validation; regular-file reads do not follow stored symlinks.
  Finding text and resolution reasons enter the browser as text, not HTML.
- PWA assets are static and the service worker is a no-op passthrough (no Web Push, no VAPID).
  Telemetry requires an operator-started export or the opt-in scrape route ([Telemetry export](cli.md#telemetry-export)); no auto-update, no remote control plane.

### 6. Approvals

- `[sandbox].run_commands` (`ask` default, `allow`, `deny`) gates every command tool; an `ask` prompt shows the argv
- Each `mcp__<server>__<tool>` call prompts with its arguments (`[mcp.servers.<name>].approve`, default `ask`).
- An "allow all" answer covers that server for the run, never the command tools or another server
- A tool name matching no configured server is refused, not prompted.
- The `fetch` off-list host prompt and the sandbox-off gate take no standing answer; both say so, and no front-end shows the button
- `sandbox.isolation = "none"` with auto-approved `run_command` adds a one-time gate: `Continue? [y/N]` interactively, a warning in CI and `machine run`.
- A prompt with no operator to answer it: a headless run refuses to start under `ask` unless `AGENT6_DETACHED_AWAY` is `deny` (auto-deny), `wait` (park it for a front-end) or `approve` (grant every scope).
  An unattended machine auto-denies.
- `agent6 mcp serve` has no operator at all, so the tools that would prompt are not published.
  Under `run_commands = "ask"` or `"deny"` its command tools are absent from `tools/list`, and a client that names one is told which setting withheld it ([the tools](acp.md#as-an-mcp-server)).

### 7. Git

**agent6's own git**

- agent6's own commits, refs and merges go through `vcs/git_ops.py`, its queries through `vcs/git_read.py`, and the web Git and Files tabs' reads through `vcs/git_history.py` and `vcs/git_files.py`.
  Only the `vcs` package's modules may import the private runner, `git_command.py`, enforced by `test_only_the_vcs_package_imports_the_runner`.
    - the package spells no destructive verb at all: `push`, `reset --hard`, `commit --amend`, `rebase`, `filter-branch` / `filter-repo` and any `--force` / `-f` appear nowhere in it, so there is nothing to enable.
      `test_the_vcs_package_never_spells_a_destructive_verb` pins every verb but `filter-repo` in every module of the package, exempting two argv: `push` inside a `git stash push`, and the one `branch -D` below
    - the collectors on the [subprocess allowlist](#12-host-side-subprocess-allowlist) carry the same hardening flags.
      By default, these flags switch off the repository's hooks, fsmonitor, external diff, textconv, commit signing, and filter and merge drivers.
    - `sessions diff` and `ask` read only.
    - `skills install` clones with fixed argv.
- One operator-only exception: `sessions prune --delete-squashed` force-deletes a run branch or chain ref the manifest confirms was squash-merged.
  A branch's or chain ref's line carries the sha to undelete it.
- Host-side git calls use `git_policy.git_environment` to remove configured `api_key_env` names, so credential helpers and content drivers never inherit them.
    - Of the shell's `GIT_*` variables, only `GIT_ASKPASS`, `GIT_AUTHOR_*`, `GIT_COMMITTER_*`, `GIT_EDITOR`, `GIT_SSH`, `GIT_SSH_COMMAND`, `GIT_SSH_VARIANT` and `GIT_TRACE*` pass through; agent6 can still supply explicit settings such as `GIT_INDEX_FILE`.
    - Other PATH, SSH, proxy and credential-helper variables stay.

**Receipt evidence**

- Receipts read the journal and manifest, with no provider call, test execution or environment probe.
- Execution-target stamps persist environment inheritance policy and variable names, not variable values; they record user, HOME, workspace roots and effective controls explicitly.
- Target and lifecycle observations grant no stop authority and do not prove restrictions beyond their recorded enforcement owner and probe state.
- Reversal guidance uses recorded object IDs and checks commit relationships through read-only git operations.
  A missing stamp or object cannot authorize a guessed reversal.
  See [Session receipts](usage.md#session-receipts) for evidence limits.

**A `git` the model runs through `run_command`** is bounded by the sandbox, and its argv is not screened.

- A linked checkout's recorded repository git dir and its admin entry are bound read-only under `full`, including for user-created worktrees, with the same hidden-path masks as other binds ([Sandbox](#2-sandbox)).
  `git status` and `git log` can read, but status cannot save refreshed stat data in the index, so later calls may repeat filesystem checks; commits cannot create their locks.
- `protect_git = "on"` requires `full` `sandbox.isolation` to bind-remount `.git` as read-only; unless `git.control = "model"`, `auto` (the default) binds under `full` and keeps the edit tools' refusal at every level
    - a mount nested under `.git` stays visible and read-only
    - a rewrite fails; `push` has no egress unless `sandbox.network = "host"`
    - a grant that contains the workspace does not lift it ([Sandbox](#2-sandbox))
- Under `partial` `sandbox.isolation`, `protect_git = "auto"` degrades with a warning and `"on"` refuses to run
    - a jailed command there can plant a `filter.<n>.clean` plus a `.gitattributes`; agent6's own auto-commit (a temp-index `git add -A` on the host) blanks every driver the repo defines unless `git.run_repo_drivers` is set
    - Landlock cannot express the exclusion: a directory grant is recursive and stacked rulesets intersect, so denying `.git` means not granting the workspace root either
- `git.control = "agent6"` is enforced where `.git` is read-only to jailed commands; under `sandbox.isolation = "none"` the run warns that a commit the model runs lands on the checkout's branch.
- The protected scope is the project's own `.git` at every isolation level (the in-process edit tools refuse writes under it; [File access](#3-file-access))
    - a nested `.git` (a vendored repo's, a submodule's) is workspace content, writable like any other file

**Repo-controlled host code in a poisoned `.git/config`**

- `core.fsmonitor`, `diff.external` and `diff.<d>.textconv` are always off.
- `core.pager` and `pager.<cmd>` never run: every agent6 git call carries `GIT_PAGER`, set to your `GIT_PAGER`, else your `PAGER`, else `less`.
- `blame.ignoreRevsFile` is honoured for a repository-relative regular file, which blame reads as a private copy in a private work tree, since git opens the configured path itself after any check; the Files view's blame refuses any other value, since git would open the file on the host and quote its first unparsable line.
    - Finding line-origin classification uses the same bounded blame reader and records an unknown reason on refusal.
    - Review refuses a `blame.ignoreRevsFile` hidden by `sandbox.hide_paths` before opening it.
    - Blame refusals name the setting and failed constraint, not the configured value.
    - Line origins remain unknown when the citation or any changed path is hidden, so rename detection cannot read hidden candidates.
    - Line-origin and rule-loading records omit Git subprocess output from failure reasons.
- `.git/hooks/*` run only under `git.run_repo_hooks = true` (default false); `core.hooksPath` points away, so a hook cannot fire on agent6's auto-commit.
- Content drivers (`filter.<n>.clean` / `smudge` / `process`, `merge.<n>.driver`) are off by default (`git.run_repo_drivers`), neutralized per name
    - the clean filter runs on the auto-commit's `git add`, the merge driver on the chain merge's `merge-tree`: a cloned poisoned repo fires one without any model action
    - `true` honors them (the Git-LFS opt-in)

### 8. Secrets and `connect`

`agent6 pack` copies credentials into the workspace at `.agent6/config/secrets.toml`.
A present pack stops every run until it is unpacked or deleted, even on its origin host and home.
`pack` verifies git's ignore decision and refuses tracked `.agent6` paths before writing the snapshot.
File tools, symbol indexing and host-side tree reads deny `.agent6` path components; the `full` jail also masks the workspace and project roots' `.agent6/` directories.

The secrets copy is created through a `0600` descriptor in a `0700` directory.
A workspace sync can expose it to the sync service and the destination host; use only trusted destinations.
A devcontainer that maps the user to another uid can leave a synced secrets file foreign-owned, which the secrets loader refuses.
`agent6 unpack` creates a fresh `0600` secrets file owned by the importing user.
The [workspace transfer flow](usage.md#carry-your-state-with-the-workspace) explains import conflicts and startup refusals.

- Provider keys live in `$XDG_CONFIG_HOME/agent6/secrets.toml`, `0600` and owner-only (refused if group- or other-accessible, or foreign-owned; opened without following a link and judged by the descriptor it opened; rewritten only under its sibling lock), or come from `[providers.<name>].api_key_env` (env wins).
- `config set --secret telemetry.headers.<name> <value>` and `config unset --secret telemetry.headers.<name>` use the same locked, atomic secrets writer for `[telemetry.headers]`.
  They accept no other keys or write targets and never print header values; `config show` lists only the configured names.
  **Command-line exposure:** a value passed to `config set --secret` can remain in shell history and the process list.
- Bound `StoredGrant` records may carry an `account_id`; records without it remain readable.
  Raw id tokens and plan labels are not stored.
- ChatGPT grants are bound to their fixed sign-in profile and provider transport settings like configured grants.
  Only the current grant format is accepted.
  `agent6 connect <name>` replaces an unusable entry after a successful sign-in.
  Logout sends no credentials for records outside the current format, rebound grants or contradictory-account grants and removes them locally only.
- A configured issuer (`[providers.<name>].oauth`) names the only other hosts a sign-in dials: its endpoints are https, or plain http on a literal loopback address, carry no userinfo or fragment, and are never discovered; the stored grant is bound to those settings and to the entry's transport settings (`api_format`, `base_url`, `deployment`, `auth_style`, `extra_query`, `extra_headers`).
    - the grant is sent only while its binding matches the entry's transport and issuer settings; a mismatch refuses before any request, naming `agent6 connect <name>`
    - a refresh or a second exchange runs under the secrets file's lock, follows no redirect, and scrubs the tokens it sent from any error
    - a 401 retries once per call, arming rotation only if the rejected bearer is still current
    - a 403 never rotates
    - `agent6 connect <name>` signs in only at those endpoints: the browser opens `authorization_endpoint`, the code returns on the configured loopback redirect (or is pasted, state-checked either way), tokens come from `token_endpoint`; `--logout` revokes at `revocation_endpoint` when set and removes the grant regardless
- `[providers.<name>].extra_headers` is the one header surface: a literal string rides every API request; a template table renders from a fixed fact vocabulary ([the list](config.md#request-headers)) for the request classes it names, and only a template addressed to `oauth` or `exchange` reaches a credential endpoint.
    - no fact carries a token, a path, a host, a user, an environment value or prompt text; `conversation_id` is an opaque id minted per role and provider in the session's `conversations.json`, kept across a resume and new on a fork; sending it or `request_id` is a correlation the operator opts into
    - [request-header restrictions](config.md#request-headers) apply at config load, and a rendered value is checked before it is sent
    - `provider.request` events carry an attempt's request id and class, never a header value
- Secrets are absent from transcripts, redacted in `config show`, and masked from the jail under `full` `sandbox.isolation` ([Sandbox](#2-sandbox)).
- `config show` and `config get` mask mapping values unless the config model marks the field display-safe (`skills.state`, `review.reviewers`); other mappings expose only their keys, including in JSON output.
- Child processes inherit no provider keys.
    - a jailed command's environment is built from its policy
    - unconfined paths (`sandbox.isolation = "none"`, git subprocesses, MCP servers, token commands) drop every configured `api_key_env` name
    - an `[mcp.servers.*].pass_env` naming a provider's `api_key_env` refuses at config load, so the name cannot be handed over deliberately either
- `agent6 connect` prompts locally (each keystroke masked on a terminal) and writes config and secrets
    - one read-only `GET` of the provider's `/models` endpoint (OpenRouter's auth-gated `/key`) confirms auth (status only; `--no-verify` skips it)
    - it executes nothing a remote returns
- `agent6 connect chatgpt` is a PKCE OAuth sign-in
    - a browser hits the authorize page of OpenAI's fixed OAuth authority; the code returns on `localhost:1455` (or is pasted), state-checked either way
    - token exchange and refreshes `POST` only to the authority's `/oauth/token`
    - the device flow posts JSON to `/api/accounts/deviceauth/usercode` and `/api/accounts/deviceauth/token`, then exchanges the returned code and verifier as a form with `/deviceauth/callback` as its redirect.
    - device requests refuse redirects, validate displayed codes as visible ASCII of at most 64 characters, and scrub both submitted codes from errors.
    - tokens live in `secrets.toml` under the same `0600` and executes-nothing rules
    - agent6 never sends a rating or feedback on a response and has no rating surface.
      The backend may use rated turns for training, and that choice is never made on the operator's behalf
- `agent6 connect claude` stores nothing: agent6 never reads or copies Claude Code's own login (`~/.claude/.credentials.json`, `~/.claude.json`)
    - no default mount or grant holds it.
      A jailed command reads it when the workspace or a path grant (`sandbox.extra_read_paths`, `sandbox.extra_write_paths`) is the home directory or a directory above it, and under `sandbox.isolation = "none"`
    - a `sandbox.hide_paths` entry hides it from commands under `full` `sandbox.isolation`
    - the child's environment carries no `ANTHROPIC_*` or `CLAUDE*` variable from the operator shell, so a shell API key cannot override the subscription login; `CLAUDE_CONFIG_DIR` is the one passthrough
    - `claude auth status --json` is parsed for `loggedIn` only; its body (email, org) is never printed or journaled
    - the binary's initialize handshake carries the account block.
      agent6 keeps the email in memory only, to replace it with `<operator-email>` in the model's returned text, and records none of it

### 9. State and locks

- An in-process `GraphCurator` owns the task graph ([the curator and its locks](architecture.md#the-curator-and-its-locks))
    - every mutation validates against a pydantic schema before writing
- Per-repo state lives at `$XDG_STATE_HOME/agent6/<repo-id>/`, outside the working directory jailed commands run in.
    - every directory under the base is created `0700` whatever the umask (`mkdir_for_real_user`, the one creator), keeping transcripts, memory and run history from other local users.
      A base an older release created at `755` keeps that mode
- The config write lock serializes read-modify-write cycles and enforces nothing
    - publishes are atomic: a torn config is impossible with or without it
    - it fails open (`O_NOFOLLOW` refuses a planted symlink; a stale root-owned lock is ignored): see [config write behavior](config.md#creating-and-inspecting)

### 10. Parallel lanes

`agent6 run --parallel` and a live run's `/parallel` steer directive (see [architecture.md, Parallel runs](architecture.md#parallel-runs)) each spawn subordinate work.

- Every lane is an ordinary run: a detached `agent6 run` on its own clone, its own jail per `sandbox.isolation`, its own `run_commands` policy
- Every spawned lane carries `AGENT6_SUBRUN=1`; both the `--parallel` flag and the coordinator's `lane_spawner` wiring refuse when it is set, so a lane cannot fan out or dispatch again.
- A lane's config carries key references, never secret values
    - the orchestrator writes each lane a `--config` via `materialize()` (the resolved `Config`: provider `base_url`, `api_key_env` names)
    - the lane reads the same `secrets.toml` or provider env var as any other run
- Lane git plumbing (clone, fetch, merge) goes through `git_ops.py` and its private runner; lane spawning goes through `ui/spawn.py` (see the [subprocess allowlist](#12-host-side-subprocess-allowlist)).
- A lane starts from committed state only
    - the fan-out clones HEAD; a coordinator dispatch cuts lanes at the run's chain tip after chain-committing changes
    - `--parallel` refuses an origin with uncommitted tracked changes under `git.dirty_tree = "ask"`

### 11. State machines

`machine run` is a supervisor that makes no model calls.
Each `tool` state is jailed with a network of its own, so one state's `network` is set independently ([Network](#5-network)).
A machine can keep agents on the provider API while one reviewed, fixed-argv `tool` reaches the network.

**An overlay never widens the sandbox**

- The machine's `[config]` and each state's `config` may set `sandbox` keys, each bounded by your own layers alone ([the rule per key](state-machines.md#machine-config-overlay-config))
    - the ceiling is read from one load of your layers (defaults, global, repo, preset, `--config FILE`), never from a config the machine file has merged into
    - a path grant is at or below one of yours, compared by real path, so a symlink out of your tree is refused
    - a state's grants are judged against the workspace, the union of the machine's and states' resolved write grants, and host `/tmp` if any runs off `full` `sandbox.isolation`, not the operator ceiling.
      A grant below such a region, or reached through a link inside one, refuses `machine check` and machine start, since a state writing the region can point it anywhere; the region itself may be granted.
    - a wider value refuses at `machine check` and at machine start, naming the state and the key
    - the other operator-only tables and leaves are rejected at load ([the list](state-machines.md#machine-config-overlay-config))
- One resolved config per state feeds both enforcement sites: a `tool` state's jail, and an `agent` state's subprocess (its in-process file tools and its jailed commands)
    - every startup check a plain run makes (the `hide_paths` check under `partial` `sandbox.isolation`, network support) runs per state and names the state
    - a state the supervisor resolved no config for runs nothing
- A `--config FILE` inside the workspace is read-only in every jail, like the bundle: a state cannot raise its own ceiling
- A grant bounds what a state touches directly: a granted state can copy what it reads to the blackboard, the data dir, the jail's `HOME` or a commit, where a later state reads it
- A `tool` only declares `pass_env`; whether a variable is passed is your `[machine].pass_env`
    - its setup command gets none of them: setup runs the tree's code at the chain tip, which a run state can edit, so a passed secret would reach model-written code

**Bundle confinement**

- Scripts live in a reviewed `scripts/` beside the `.asm.toml`
    - `machine check` verifies every entry and static reference resolves inside the bundle (escaping symlinks rejected)
- Scripts are committed, never fetched or generated at run time
    - the `.asm.toml` and `scripts/` are read-only in every jail during a run
- Front-ends render `machine.notify` as an overlay, and `attach` / TUI call `notify-send` with fixed argv, so a model message is inert data.
- The out-of-band hook `[machine.notify].on_event` runs an operator argv on the host with the minimal `hook_env` env plus `AGENT6_MACHINE_*`.

### Container discovery boundary

`sandbox/container.py::discover` is an internal, read-only discovery boundary.
The separate `prepare` entry point opens the [container transport](#container-transport-boundary); `sandbox.isolation = "container"` activates both.
Its host runtime CLI has the operator's authority, including daemon socket access and any privilege wrapper the operator supplies.
Discovery never adds `sudo -n`; a refused socket query names `container.runtime` and the docker group rather than retrying with more privilege.
The operator must trust the runtime executable, its context and its configuration; a remote daemon's mount paths do not prove a shared host tree.

Only `ps` and `inspect` run, with fixed flags, a workspace label value and validated full hexadecimal container IDs.
No model-chosen runtime flag enters this path.
Each query has a 10-second deadline and a 1 MiB combined stdout/stderr cap; a failed query refuses instead of returning no matches.
Discovery considers at most two runtime executables and two host-root spellings, and queries stopped labels only after no running match.
The subprocess has no shell or stdin, starts outside the checkout, and inherits a curated host environment with runtime connection/context settings but without registered provider keys.
Timeout or output overflow kills the owned client process group and waits at most one more second for its exit; this does not claim control over daemon-side work or privileged descendants that the host user cannot signal.

Inspect JSON and the `devcontainer.metadata` label are untrusted pydantic inputs, not executable configuration.
Validation errors omit their input values, including in exception chains.
The selected user becomes one runtime argument; metadata environment values remain unexpanded and are not applied to a process.
All other metadata fields, including lifecycle commands, are ignored.
Repository definition files are checked only for existence, never read or parsed.
A nonempty manual exec prefix is returned byte-for-byte without invoking it or discovery, and confers no identity, mapping proof or stop authority.
There is no launcher transfer, model-command execution, mount creation, container start or container stop in discovery.

### Container transport boundary

**Host jail bypass is deliberate.** `sandbox.isolation = "container"` runs every command inside the selected container, where `container.prepare` launches the existing Rust `none` branch, not the host jail; the operator chooses it by name, and the start block says so.
The runtime and container configuration are the command boundary: agent6 adds no namespaces, Landlock rules, seccomp filters, mounts or `/dev` nodes there.
The launcher still enforces per-command memory limits, owns job handles, captures output and cleans up its descendants.
It neither starts nor stops the container, and attaching grants no ownership.
The application must resolve or refuse configured restrictions before composing this internal transport.

**Container access includes its host exposures.** Commands can reach writable host mounts, runtime sockets and credentials exposed by the container or its environment.
Sharing the container also shares its network and HOME.
A mount inventory is an observation, not confinement or a complete credential scan.
The host file tools and session records remain outside this transport.
A same-user process or container administrator can tamper with the environment; protocol identity detects accidental replacement, not a hostile runtime.

**Observed failure behaviour** (`tests/sandbox/test_container_transport.py`, `test_container_runtime.py`).
A lost runtime client, a lost agent process or a closed channel ends the launcher and every job it owns; the record reads unknown, never settled.
A stop reaches a foreground command within a few polls; a backgrounded command and a `setsid` escapee end with their job's subreaper, and a sibling session or an unrelated container process lives on.
Output past the retained cap comes back cut with the session alive.
A container replaced under a lease is attached, never owned.

**Installation trusts the selected binary and target utilities.** The selected `AGENT6_JAIL_BIN`, bundled launcher or PATH launcher must be a native x86-64 or aarch64 static Linux ELF without an interpreter or shared-library dependencies.
No runtime download, build, cross-architecture fallback or host execution fallback occurs.
The fixed `/bin/sh` bootstrap requires `uname`, `mktemp`, `head -c` and `chmod`; absent utilities refuse by name.
Only the byte count and architecture are bootstrap arguments, never model argv or interpolated operator paths.
The host sends policy bytes only after the bootstrap acknowledges installation; buffered utility reads cannot consume them.
The selected user creates a fresh private directory under target `/tmp`; exclusive creation refuses an existing launcher or symlink.
The bootstrap uses umask `077` for installation, then restores the target's original mask before executing the launcher and its commands.
Rust opens the installation directory without following symlinks, checks its owner and mode, and compares the installed inode with `/proc/self/exe` before unlinking it.
An installation interrupted before readiness may retain its directory or partial executable; shutdown removes only an empty installation directory, never a tree of command-created files.

**Start and stop commands are the operator's, run on the host.** A container start or stop command runs as the operator with fixed argv, no shell, a 120-second bound and bounded output; agent6 never derives one, adds sudo or runs a repository's lifecycle definition.
The host-side lease records what discovery observed before and after a start; only a run whose start made one container appear may stop it, and only after every attached run released.
A container that ran before the run, an opaque exec prefix, an ambiguous result or a replacement container is attached, never owned.

**Preparation proves shared mapping, not workspace permissions.** Independent random nonces cross the mapping in both directions under the selected user.
The probe directory permits writes across differing UIDs; success does not establish write permission elsewhere in the workspace.
The host creates and opens the probe under a host-owned staging directory with 0700 access permissions before publishing it in the workspace.
Inherited setgid is allowed because it grants no group access.
The host never changes the staging directory's permissions or deletes its existing entries, even if that directory was substituted before opening.
Retained no-follow descriptors anchor nonce reads, writes and cleanup; symlinks and non-regular nonce files refuse.
The host checks the target nonce with `O_PATH | O_NOFOLLOW` before any I/O open, so substituted devices cannot run open handlers on the host.
It then reopens the verified regular inode through `/proc/self/fd`, not the mutable pathname.
Directory traversal needs search permission, not read permission on ancestors.
Probe files contain no secrets and are readable across differing UIDs.
Read-only or mismatched mappings refuse before commands; successful proof is not protection against later remounts or tampering.
The target's inherited HOME and PATH remain target values, and TMPDIR is short and target-local.
Bootstrap scratch state does not overwrite inherited target variables.
`${containerEnv:NAME}` substitutions read only the observed target environment; host and unsupported substitutions refuse by name.
Substituted target values remain literal data.
Non-UTF-8 environment names or values refuse with an error that discloses neither.
Probe validation errors omit environment values from their tracebacks.

The raw JSON channel uses no tty and drains runtime stderr with bounded retention.
Protocol version, request ID and launcher-instance checks bind every response to its prepared session.
A dead runtime client or restarted container loses unconfirmed outcomes; no container file or replacement launcher can settle an old job handle.
The [supervisor](architecture.md#jail-job-ownership) owns command completion; client exit alone proves neither completion nor cleanup.

### 12. Host-side subprocess allowlist

Every child process whose argv depends on model output runs through `run_in_jail`, `spawn_in_jail` or the prepared `JailSession` supervisor channel ([Sandbox](#2-sandbox)).
These are the reviewed command boundary, not exemptions for transport-specific subprocesses.
The exception is `notify-send`: it has no stdin mode, so a machine `notify` message is one data argument after `--`.
A fixed set of modules also shells out directly with `subprocess.run` / `Popen`, each with fixed argv.
`tests/security/test_subprocess_allowlist.py` pins the file list; audit with `rg 'subprocess\.|os\.(system|exec|posix_spawn)' src/agent6/`.

<details markdown="1">
<summary>Allowed host-side subprocesses by module</summary>

- `vcs/git_command.py` and `vcs/git_ops.py`: agent6's own git operations ([Git](#7-git)); the other `vcs` modules run git through `git_command`, except the config query the next line names.
- `vcs/git_policy.py`: shared hardening flags and hook/driver policy; its fixed git config query enumerates repo content drivers.
- `sandbox/detect.py`: probes the host's sandboxing capabilities.
- `sandbox/container.py`: bounded `ps`/`inspect` discovery and the fixed installation bootstrap through operator-owned prefixes ([Container transport boundary](#container-transport-boundary)).
  Model argv travels only through the prepared `JailSession` channel, never through a host subprocess argv.
- `sandbox/jail.py`: the jail launcher, and the plain spawn under `sandbox.isolation = "none"`.
  An MCP server spawns through the same launcher and `JailPolicy` a jailed command gets (`spawn_in_jail`); one the operator opted out takes the `none` `sandbox.isolation` level.
- `sandbox/jail_session.py`: the serving launcher transport at every local isolation level; `sandbox/jail_jobs.py` persists its responses without spawning command processes.
- `sandbox/jail_protocol.py`: process type annotations match the audit; a separate pin forbids process-spawning references outside annotations and calls even inside annotations.
- `providers/token_command.py`: the `[providers.*].token_command` that mints a provider bearer.
- `providers/claude_code.py` and `providers/claude_code_session.py`: the `backend = "claude_code"` provider and its session lifecycle helpers run the Claude Code binary the operator connected with fixed argv from operator configuration, never model output (binary, model, effort, literal flags, the path of a 0600 system-prompt file in a private empty directory).
  Prompts, tool results, and notices travel on stdin, so model- or repo-derived text never becomes an argv element.
  Curated environment, `--tools ""`, `--allowedTools mcp__agent6`, `--setting-sources ""`, `--strict-mcp-config`, `--disable-slash-commands`, `--no-session-persistence`; CLAUDE.md, auto-memory, and auto-compaction off by environment; `system/init` is audited so a tool outside `mcp__agent6__*` or an API-key source refuses the run.
  Unjailed: it needs the operator's login under `$HOME` and its own egress, at the agent process's trust tier.
  Tool-result text and notices stay under Claude Code's 50,000-byte persistence threshold.
  The loop clips tool-result text at 34,000 bytes for this provider, keeping 16,000 for the turn's notices.
  A tool result with its notices above 50,000 bytes is refused.
  `claude auth status --json` runs the same way for the sign-in preflight.
- `sessions/process.py`: `ps -p <pid> -o lstart=` and `ps -p <pid> -o stat=` on hosts without `/proc` (macOS), for process start-time identity and zombie checks, over a pid agent6 recorded.
- `ui/btw.py`: spawns `agent6 ask` detached for `/btw` (every composer).
  Argv is the agent6 exe plus the question the operator typed, with `--` before it.
- `ui/spawn.py`: the shared front-end spawn helper.
  Spawns the agent6 CLI detached for run and machine launches (`machine create` and `machine run`), and captures `sessions merge` / `prune` / `remove` / `review` and `config set` / `unset`.
- `ui/notify.py`: `notify-send` with fixed argv (exe, `--`, two positional data args, no shell) for the device-present machine notification.
- `ui/cli/` helpers:
    - `connect_oauth.py`: the platform's URL opener (`xdg-open`, `open`) for a sign-in URL built from the operator's configuration, with the curated environment; never `$BROWSER`'s command template
    - `plan_watch.py`: `$EDITOR` for plan editing
    - `sessions_cmds.py` and `ask.py`: `git diff` / `log` for the `sessions` / `ask` diff views, with argv from the run manifest the CLI wrote outside the jail
    - `history_cmds.py`: `rg` for history search
    - `live.py`: the fixed-argv `python -m agent6.ui.tui` co-process behind `run --tui`
    - `system_cmds.py`: `cp` / `rm` / `apparmor_parser` via sudo with fixed argv for `agent6 system apparmor`
    - `completions_cmd.py`: runs nothing itself; its one `subprocess.run` is text inside the xonsh completer script it emits, which the operator's xonsh runs at tab time to ask `agent6` for candidates
    - `skills_cmds.py`: `git clone --depth 1 -- <url>` with fixed argv for `agent6 skills install` and `skills update`.
      The URL is operator-supplied and nothing fetched is executed
- `app/finalize.py`: both operator notify hooks (`run_notify_hook`): `[notify].on_complete` at run end and `[machine.notify].on_event` from `machine run` (built by `app/machine/host_preflight.build_machine_notify_hook`).
  Argv from config, env from `hook_env` (a minimal base plus `AGENT6_SESSION_*` or `AGENT6_MACHINE_*`, never the provider keys in the operator environment).
- `app/machine/scriptcheck.py`: ruff and ty with fixed argv, reading generated scripts statically.
  Those scripts execute only via `run_in_jail`.
- `app/machine/agent.py`: spawns each agent state as a fixed-argv `python -m agent6.ui.cli.machine_agent` subprocess whose request travels in a temp file.
- `ui/tui/clipboard.py`: `tmux load-buffer -w -` with fixed argv and the copied transcript text on stdin.
- `ui/tui/conversation.py`: the operator's `$PAGER`, argv from the environment, transcript text on stdin.

</details>

## Skills

- A skill is operator-installed config: install only from trusted sources
    - `skills install <url>` is an operator-initiated CLI fetch; what it installs enters the system prompt and tool results verbatim
- Nothing in a skill runs at install or load
    - its scripts run only through the jailed command path, subject to `run_commands`
- `use_skill` is read-only and path-contained: the skill's own dir through a component-walked descriptor (any symlink hop or `..` inside it refused)
    - a skill directory or `SKILL.md` that is itself a link is followed, and a read that lands under `sandbox.hide_paths` or an agent6 private directory is refused, as the workspace file tools refuse it
    - a skill whose `SKILL.md` resolves under one of those paths is not loaded for a run at all: its name, description and text stay out of the prompt's skill index, the `always` injection and `/name`, and the run logs a warning naming it; `agent6 skills list` still shows it
    - skill dirs are not mounted into the jail
- Repo-local `.claude/skills/` are not discovered; only the installed dir and `[skills].extra_dirs` are scanned

## Prompt-injection tests

[`tests/security/test_prompt_injection.py`](https://github.com/agent6-dev/agent6/blob/master/tests/security/test_prompt_injection.py) drives the dispatcher with the calls a compromised model would make.
Those are path traversal, a swapped symlink between check and open, an unknown tool name, extra fields, and a write outside the workspace.
It asserts the tool surface refuses them whatever the model says; it does not test the model's judgement, and calls no model.
The structural defenses above confine a model that follows an injection.

## Known limitations

- seccomp is required; kernels that block it from unprivileged callers make the jail fail closed.
- Under `full` `sandbox.isolation` in a Docker or Podman container, the jail's `/proc` is empty and `ps` fails.
  Docker's `--security-opt systempaths=unconfined` or Podman's `--security-opt unmask=ALL` lets the jail mount its own `/proc`.
  Under Podman's default seccomp profile, jailed commands also see the container's hostname.
- In a devcontainer the XDG state base is ephemeral (lost on rebuild): mount a volume at the state dir or point `XDG_STATE_HOME` at one
- agent6 installed inside the project it works on (pip into the project's own venv) puts the running agent's code in the jail's writable workspace
    - a jailed command can rewrite it; agent6 runs the rewrite as you, outside the jail
    - install agent6 outside the tree (pipx, `uv tool`); agent6 warns at run entry on this shape
- Claude Code (`backend = "claude_code"`) appends the account email to every system prompt it sends and has no switch for it.
  The model can echo it, and agent6 scrubs it only from returned text, never from tool inputs (an edit carrying it lands as written)
    - managed settings (`/etc/claude-code/managed-settings.json`) still apply inside the child.
      The `system/init` audit refuses the run when the tool list differs from what agent6 offered or `apiKeySource` is not `none`, and checks nothing else.
      A managed hook, or a managed MCP server exposing no tool, passes it
- Side channels: no claim about timing, cache, or speculative side channels.
- Supply chain: pin your install
    - runtime deps `pydantic`, `httpx2`, `argcomplete`, the `tree-sitter` pair, `textual`, `ruff`, `ty`; build dep `hatchling`; jail crates `nix`, `libc`, `landlock`, `seccompiler`, `serde`, `serde_json`

## Media retention

`read_file` sends supported static images and PDFs from the workspace or an explicit read grant to the calling provider.
The read uses `Workspace.resolve_read` and `open_contained`, retaining hidden-path, regular-file and symlink-race checks.
It reads media under the 10 MiB document cap plus one byte, checks image containers without decoding pixels, and recognizes PDFs by their header.

Transcripts, snapshots, checkpoints and shared message objects omit recognized payloads and retain metadata (path, media type, SHA-256 and byte size).
A resume or fork needs a fresh `read_file`; it never reopens a saved path automatically.
Streaming error excerpts that reach 8 KiB on a media request are omitted because clipping can split an echoed payload.

**Pixels and PDF contents are sent upstream unscrubbed.**
A screenshot or PDF can contain credentials even when configuration directories are hidden.
Metadata-only retention and credential scrubbing of headers and text do not redact those contents.

The Claude Code binary runs outside the jail and receives the original attachment bytes through its stream-json pipes.
The adapter sends images as MCP image items and PDFs as embedded resource items, not base64 text tool results.
Its 50,000-byte text guard and 8 MiB line cap remain in force; the line cap includes base64, envelopes and notices on both continuation and replay.
The installed binary's forwarding and storage behavior require a live check; agent6's retention policy covers its own records, not the child or upstream provider.
