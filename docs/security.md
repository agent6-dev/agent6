# Security

agent6 treats the LLM as untrusted.
Each section states what is enforced and what is not.

## Reporting

Report privately through GitHub's [Security tab](https://github.com/agent6-dev/agent6/security/advisories/new).
Include: agent6 version (`agent6 --version`), kernel + distro (`uname -a`, `/etc/os-release`), `agent6 check sandbox` output, and a minimal repro (ideally a failing test under `tests/security/`).

## Threat model

Adversary: a fully malicious worker model, or an honest model prompt-injected by a file in the workspace.

The adversary controls the text of every LLM response and the content of any file the agent reads.
It also chooses the tool calls and their arguments, within the published JSON schema.
Outside its control: the kernel, the agent6 binary, the provider endpoints.

**Holds**

- No writes outside the workspace
    - `sandbox.extra_write_paths` widens it, visible in `agent6 config show`
    - the per-repo memory dir (file tools only) and the jail's persistent `HOME` (below) widen it, visible in `agent6 check sandbox`
- No reads outside the workspace and a read-only system set for installed toolchains
    - `full`: `/usr`, `/bin`, `/sbin`, `/lib`, `/lib64` and `/etc/alternatives` bound read-only into an otherwise empty rootfs, plus a curated `/dev`
    - `partial`: Landlock grants `/usr`, `/bin`, `/sbin`, `/lib`, `/lib64`, `/etc` and the `/dev` nodes `null`, `zero`, `urandom`, `random` and `full`
        - `/proc` and the host's `/dev/shm` are not granted: `ps` and Python `multiprocessing` fail there, and work under `full`
    - `sandbox.extra_read_paths` adds more; `agent6 check sandbox` prints the resolved set
    - a child on the host network (`network = "host"`, or any level but `full`) also reads what name resolution and TLS verification need, where the host has them: `/etc/resolv.conf` and the file it points to, `/etc/hosts`, `/etc/nsswitch.conf`, and the CA store (`/etc/ssl`, `/etc/pki`, `/etc/ca-certificates`).
      File permissions still apply: the root-only private keys beside the certificates stay unreadable to a run that is not root
- `/tmp` is writable at every level
    - `full`: a private tmpfs discarded with the run
    - `partial`: the host's `/tmp`
- `HOME` is a directory of agent6's own, never the operator's
    - `full`: `/tmp/agent6-home`, created inside the private tmpfs and gone with the run; `[sandbox].home = "cache"` swaps in the persistent dir below, bind-mounted read-write at its real path
    - `partial` and `none`: `$XDG_CACHE_HOME/agent6/homes/<repo-id>` (default `~/.cache/agent6/homes/<repo-id>`), one directory per repository, created `0700` by agent6 and, under `partial`, Landlock-granted read-write like the workspace; the rest of the operator's home and every other repository's directory stay ungranted
    - a linked worktree (`run --worktree`, a fork) gets its repository's directory, named by the git dir the session recorded and never by the worktree's own `.git` pointer; a lane's clone and a machine state's clone get it too, named by the first directory below the workspace base directory, the owning repository's id
    - a symlink, another user's directory, a mode open to group or others, or a path inside agent6's private dirs at that location refuses the run.
      The mode is checked on every run (a jailed command can `chmod` its own HOME), and the refusal names `chmod 700`; nothing restores it silently
    - persistence is a cross-run channel inside the jail's world: a poisoned cache or a `~/.gitconfig` alias reaches the next jailed run in the same repository, never another repository's runs and never the operator's own tools
    - a `$XDG_CACHE_HOME/agent6/home` directory is the shared `HOME` of an earlier version: nothing reads or grants it, and it can be deleted
- agent6's own git never pushes, force-pushes, rewrites history, or `reset --hard` ([Git](#7-git))
    - a `git` the model runs through `run_command` is bounded by the sandbox instead: `protect_git` keeps `.git` unwritable under `full`, and push needs egress
- No persistence after the run: no daemon, cron, `.bashrc` write, or setuid binary (one create path stays open, below)
    - the exception is the jail's persistent `HOME` (`partial`, `none`, or `[sandbox].home = "cache"`): a file or binary written there reaches the next jailed run in that repository, and stays executable.
      The operator's own shell, tools, and dotfiles never read it
    - chmod-family syscalls (`fchmodat2` included) deny modes carrying `S_ISUID` / `S_ISGID`; ordinary chmod passes
    - so do the create paths that take a mode: `creat`, `mknod`/`mknodat`, and `open`/`openat` with `O_CREAT` or `O_TMPFILE`.
      `openat2` carries its mode behind a struct pointer, out of seccomp's reach, so a command can plant the bit through it.
      Every jail mount is `nosuid`, and the file is owned by the operator's own uid, so it grants nothing outside the jail either (`--allow-root` below is the exception)
    - every mount carries `nosuid` and `nodev`, except the bound `/dev` nodes (the builtin five: `null`, `zero`, `urandom`, `random`, `full`, plus any `sandbox.extra_device_paths` grant), which `nodev` would make unusable
    - `/tmp` allows exec (toolchain helpers)
    - children write inside the jail's mount namespace (`full`) or the Landlock write grants (`partial`)
    - a process a command leaves behind dies with the run at the latest
        - `full`: the PID namespace takes the tree down at close.
          Its launcher, the namespace's init, sweeps what a stopped background command left outside its process group (a `setsid` daemon reparented onto it).
          The sweep covers what appeared after the command started and before the next still-running one did
        - `partial`: the agent process holds `PR_SET_CHILD_SUBREAPER` and kills every process that appeared during the command
    - a survivor the sweep cannot kill fails the command, and a background `stop` answers with its pids
    - one still running when the run's jail session or a spawned MCP server closes is recorded as a `jail.degraded` event

**Does not hold**

- The agent process's own egress is unbounded
    - agent6 reaches the configured providers (each `[providers.*].base_url` host, plus the fixed ChatGPT OAuth authority for token grants); nothing stops the process reaching elsewhere
    - a jailed command's egress is bounded ([Network](#5-network))
- Under `partial` `sandbox.isolation`, a command can hand work to a user daemon already running (tmux, `systemd --user`): unix sockets have no Landlock hook and stay nameable without a mount namespace
    - the environment carries no address for one.
      A jailed command gets `LANG`, `LC_ALL`, `TERM` and `CI` from the host, plus `HOME`, `PATH`, `PYTHONDONTWRITEBYTECODE` and `UV_NO_SYNC` set by agent6 (a machine tool also `AGENT6_MACHINE_DATA_DIR`), and the `[git.commit]` name and email as `GIT_AUTHOR_*` and `GIT_COMMITTER_*` when the config sets them, so a `git commit` the model runs under `git.control = "model"` carries the operator's identity.
      A jailed MCP server's base drops `DISPLAY`, `WAYLAND_DISPLAY`, `DBUS_SESSION_BUS_ADDRESS`, and `XDG_RUNTIME_DIR` (`curated_env(desktop=False)`)
    - `full` does not expose them

## Defense layers

`agent6 check sandbox` prints a table for the file tools and one for the jailed commands: a root row, a home-directory row and a row per path, a parent before its children.
It also prints a line per MCP server, the network, the approval mode and the secrets posture, then runs the jail probes.

### 1. Trust boundary

The agent's Python process is trusted and runs unconfined at every isolation level; the model is untrusted.

- the process holds the provider keys, writes the per-repo state dir, spawns the jail
- isolation levels differ only in which jail features the launcher enables

The model reaches the machine through two surfaces:

- The in-process tools, whose paths resolve against the file boundary ([File access](#3-file-access)).
- Anything that executes, which runs in the jail ([Sandbox](#2-sandbox)).

The model sees the fixed tool set in `src/agent6/tools/schema.py`.

- structured edits, read-only navigation, `read_session`, fixed-argv verify and metric commands, `finish_session`, `ask_user`, a curator task notepad, `use_skill` ([Skills](#skills)), approval-gated `fetch` ([Network](#5-network)), capability-gated `run_command` with `read_background` / `stop_background`, and `agent6_docs` under `agent6 ask` only
- no `shell`, no `write_file`, no `eval`
- adding a tool needs a security review note ([AGENTS.md](https://github.com/agent6-dev/agent6/blob/master/AGENTS.md))

Under `api_format = "claude_code"` the model runs inside the operator's installed Claude Code binary, unjailed, as the operator, with every built-in tool off (`--tools ""`).
Its only reach into the machine is agent6's tools served over the sdk MCP tunnel, so the dispatcher, the jail, and the approval gates are unchanged.
The binary holds the operator's Claude login exactly as an interactive `claude` does, and agent6 never reads it.

Every line the CLI prints to the operator's terminal, on stdout, stderr or `/dev/tty`, passes one scrubber (`terminal.scrub_terminal_output`).
It drops a control sequence inside text the model influenced (a file name, a commit subject, a summary, a task a plan wrote), so the model cannot write the clipboard (OSC 52), retitle the window or forge a line.

- SGR styling is the allowlist, conceal excepted; an approval prompt drops every sequence from the text under judgment
- the spinners and the interactive composer write their own erase and cursor sequences under the wrapper, the composer with its rows scrubbed
- the ACP and MCP stdio transports write protocol bytes to their peer, not a terminal; the TUI and the web render text as text by construction
- `agent6 exec` is the exception: the operator's own command writes the terminal unscrubbed, so `agent6 exec cat FILE` prints whatever sequences a model-written file holds
    - with stdin and stdout on a terminal the command runs on a pty of its own and agent6 relays the bytes, so it never holds the operator's terminal device.
      The pty is the controlling terminal of the command's session, and the launcher puts the command's process group in its foreground before the command starts.
      A command the pty stops is resumed by the launcher, which first puts its group back in the foreground
    - otherwise it inherits agent6's stdin, stdout and stderr as they are, a terminal among them included: it can change that terminal's modes, which agent6 puts back when the command ends
    - it runs in a session of its own, so an inherited terminal is never its controlling terminal: the kernel refuses it `TIOCSCTTY` on that terminal, and it takes no foreground there

### 2. Sandbox

Every tool call that allows the model to run arbitrary commands (`run_command`, `run_verify_command`, backgrounded commands, and MCP servers) runs in a jail at the effective isolation level.

`agent6 check mcp` and `agent6 mcp connect` start a spawned MCP server under that jail with the repository bound read-only.
`check mcp` applies the run's refusals first and leaves a server it cannot hold that way unstarted: `unconfined = true`, a write grant (`extra_write_paths`, `sandbox.extra_write_paths`, `sandbox.extra_device_paths`), or no jail at all.
`mcp connect` probes under the same jail, and skips the probe only when there is no jail at all, saying so.
It is the operator's own invocation naming the server they are adding.
A diagnostic hands a server the repository to read and writes nothing but the config it was asked to write.

**Modes** (`[sandbox].isolation`)

| Mode | Applies |
|---|---|
| `full` | user namespace (one uid, the operator's own) + mount + PID + IPC + UTS namespaces (hostname `agent6`), a network namespace unless `sandbox.network = "host"`, `pivot_root` rootfs, private `/proc` and `/tmp`, curated `/dev`, Landlock where the kernel has it, seccomp, `NO_NEW_PRIVS`, capability drop, timeout |
| `partial` | Landlock, seccomp, `NO_NEW_PRIVS`, capability drop, timeout. No namespaces, no rootfs |
| `none` | No confinement: no namespaces, Landlock or seccomp. The timeout and `memory_limit_mib` still apply |
| `auto` *(default)* | The strongest of the three this host supports |

Under `full` a kernel without Landlock skips that layer, warned once per run; under `partial` Landlock is the only filesystem boundary.

**Resolution**

Probed per host: `unshare` for namespaces, the Landlock ABI syscall, the seccomp filter's architecture (x86_64 and aarch64).

| Setting | Host | Effective |
|---|---|---|
| `auto` | namespaces | `full` |
| `auto` | no namespaces, Landlock ABI >= 3 | `partial` |
| `auto` | no namespaces, Landlock ABI 1-2 | `partial`, warned: truncation outside write grants is unconfined |
| `auto` | no namespaces, no Landlock | `none`, warned |
| `auto` | non-Linux, or no seccomp filter for the arch | `none`, warned |
| `full` | no namespaces | refused |
| `partial` | Landlock ABI < 3 | refused |
| `full`, `partial` | non-Linux, or no seccomp filter for the arch | refused |
| `none` | any | `none`, warned |

`--disable-sandbox` and `AGENT6_DISABLE_SANDBOX=1` force `none` for one invocation.
Config, flag, and env var are operator-only; the model reaches neither argv nor env.

**Inside the jail**

- Mounts (`full`): cwd and a private `/tmp` writable, system paths read-only, `extra_read_paths` / `extra_write_paths`, a persistent `HOME` under `[sandbox].home = "cache"`, and operator tool dirs at their real paths
    - every mount keeps the path it has outside
    - the workspace is mounted at its resolved path and the launcher starts the command there: under a workspace named through a symlink, `pwd` prints the resolved path
    - a grant is resolved once, when the run starts, and every jail launch of the run binds that resolved directory at the grant's configured path: `/opt/tool -> /opt/tool-1.2` shows the content of `/opt/tool-1.2` at `/opt/tool`, and a link re-pointed during the run changes nothing.
      Under `partial` the Landlock rule goes on the resolved directory
    - a grant that contains the workspace is mounted under it: the workspace stays writable under a read grant
    - a grant the launcher cannot bind fails the launch, and the error names the grant, the step and the OS error (`write grant /srv: remount of submount cache: EPERM`); a read-only mount inside a write grant is such a case
    - a grant on `/` or on `/dev` fails the launch by name: `/` would replace the jail's own root, and `/dev` would cover the jail's device nodes (name a device in `extra_device_paths`)
    - a grant whose path an earlier grant already shows (a file listed twice, a file inside a granted directory) is bound over it.
      Where the earlier grant shows a symlink at that path, nothing is mounted through the link: a read grant leaves the link as it is, and a write grant fails the launch by name
    - a grant that is the home directory or a directory above it lets the run's commands and file tools reach the credentials other tools keep there (`~/.ssh`, `~/.aws`, another agent's login).
      agent6 warns about such a grant at run start and in `agent6 check sandbox`.
      Under `full`, agent6 hides only its own config dir and state base, plus the paths in `sandbox.hide_paths`
    - the paths kept read-only inside the workspace (`.git` under `protect_git`, a machine bundle) are bound after every grant: a grant that contains the workspace does not make them writable
    - a tool dir is mounted only when its identity clears a default-deny.
      Refused: a dir at, inside, or containing an agent6 private dir, and `$HOME` or an ancestor of it
    - a fork's execution or a `run --worktree` run (cwd is its linked worktree) also grants the repository's `.git`: a read-only bind under `full`, a Landlock read+exec rule under `partial`, in both cases whatever `protect_git` says.
      The main checkout's `.git` is read-only only under `full` with `protect_git`, so a worktree's is the more confined of the two.
      The model's prompt says so
    - the granted dir is the one agent6 recorded when it added the worktree (the manifest's `worktree_git_dir`, taken from the repository it ran in).
      The worktree's own `.git` pointer is never the source: a jailed command can rewrite it under `partial`.
      The policy builder refuses when the pointer no longer resolves to the record.
      A linked worktree agent6 did not record gets no grant
    - under `partial`, a `hide_paths` entry inside the recorded dir refuses like any other unmaskable entry
- Masked last, after every bind (`full`): the config dir, the state base, `[sandbox].hide_paths`
    - each is masked under its given name and under the path it resolves to, so an entry spelled through a symlink covers the real directory
    - a hidden path inside the tree a symlinked grant resolves to is masked under the grant's spelled name too, where the jail shows that tree
    - a hidden directory reads as an empty directory and a hidden file as an empty file; a write to either fails with `Read-only file system`
    - a grant at or inside a private dir is refused at config load
    - `partial` cannot mask: a grant containing a private dir warns, an unmaskable `hide_paths` entry refuses
    - a `hide_paths` entry that is the workspace or a directory above it refuses at every level, before anything runs: drop the entry, or work in a directory outside it
    - a `hide_paths` entry hides a path, not content git tracks: a tracked file still reads through git's objects (`git show HEAD:<file>`) and in every clone or worktree a lane, a fork, a `run --worktree` run or a machine state works in
    - `XDG_CONFIG_HOME` and `XDG_STATE_HOME` relocate those two dirs, `XDG_CACHE_HOME` the persistent `HOME`; the mask, the grant validator, and the tool-mount scan all read the relocated path
- `/dev` (`full`): `/dev/null`, `/dev/zero`, `/dev/urandom`, `/dev/random`, `/dev/full`, a private `/dev/shm`; no `/dev/tty`
    - `sandbox.extra_device_paths` binds named `/dev` nodes read-write (GPU compute).
      Each must be a char/block device on the host, or the launch refuses.
      On `partial` the same grant is a Landlock read+write rule on the node
    - creating one is denied at both levels: Landlock handles `MakeChar` / `MakeBlock` and grants them nowhere, and seccomp `EPERM`s the `mknod` pair by device type (below).
      `full` also has the user namespace (no `CAP_MKNOD` in the initial one) and `MS_NODEV` on the `/dev` binds
    - under `partial` with `--allow-root`, Landlock, seccomp and the capability drop deny it
- `/proc` (`full`): fresh and private, empty if that fails; Landlock grants it read without execute
    - the launcher, PID 1 there, hides its `/proc` entry from the command (dumpable 0, capabilities dropped) and runs with an empty environment
- seccomp: a 36-syscall deny-list returning `EPERM`.
  It covers process inspection (`ptrace`, `pidfd_getfd`, `process_vm_readv`/`writev`, `kcmp`), `io_uring_setup`, `userfaultfd`, the whole mount family (`mount`, `umount2`, `pivot_root`, `mount_setattr`, `open_tree`, `move_mount`, `fsopen`, `fsconfig`, `fsmount`, `fspick`), `setns`, `unshare`, `kexec`, `bpf`, `perf_event_open`, the keyring calls (`keyctl`, `add_key`, `request_key`), module loading, `reboot`, swap, and the clock-setting family
    - the list denies unconditionally.
      Argument-conditional rules `EPERM` two more cases on syscalls that stay allowed: a mode carrying `S_ISUID` / `S_ISGID` on `chmod` and the create family (above), and a `mknod` / `mknodat` naming a character or block device (`/dev` above)
    - anything else is allowed; the list itself is the source (`jail/src/main.rs`), and it grows by syscall, never by class
- Capabilities: cleared between fork and exec.
- Timeout: the policy's `timeout_s` (verify and metric gates set it from `[harness].verify_timeout_s`, default 600), then SIGKILL of the process group, rc=124
    - a model's `run_command` is not wall-clock killed: at `[harness].command_checkin_s` it is handed back as a background job ([Commands and environment](#4-commands-and-environment))
- One launcher per run at every isolation level; under `full` its commands share that netns, PID namespace, and private `/tmp`, and under `partial` the host's network and `/tmp`
    - closing the run's channel takes the PID namespace down
    - a launcher that cannot start leaves each command its own
- The policy arrives as JSON on the launcher's stdin, validated against a strict schema; unknown fields are refused.
- `[sandbox].memory_limit_mib` (default 0, off): per-process data-segment cap (`RLIMIT_DATA`) in MiB, inherited by children.

**Verify**

`agent6 check sandbox` prints the isolation this host resolves to and the reason when it is not `full`.
It also lists each directory the jail mounts read-only for operator tools, and why.
It then runs live probes at that isolation.
It exits 1 when a check fails, 0 otherwise.

| Probe | Passes when |
|---|---|
| `landlock_abi` | the kernel reports Landlock ABI >= 1 |
| `jail_true` | `/usr/bin/true` runs in the jail, rc=0 |
| `jail_blocks_network` | a jailed `getent hosts example.com` fails. n/a under `partial`, which has no network namespace |
| `jail_blocks_etc_write` | a jailed write to `/etc/agent6-escape` leaves no file on the host |

### 3. File access

`read_file`, `list_dir`, `outline`, `find_definition`, `find_references`, `apply_edit`, and `apply_patch` run in the agent process and ask no approval.
Every path they take resolves through `Workspace`:

    workspace root
      + [sandbox].extra_read_paths (read) / extra_write_paths (write)
      - [sandbox].hide_paths
      - agent6's own config dir and state base

- The boundary comes from config values
    - a degradation (`auto` falling back to `partial` or `none`, a host with no jail) leaves it unchanged; under `none` isolation it is the only boundary
- A grant is resolved once per run, and the file tools and every jail launch hold that resolved path
    - a grant spelled inside the workspace root or a write grant, or resolving into one, is dropped: a jailed command can re-point it there.
      The run names each dropped grant in one line at its start, and the region still covers what the grant held
    - a grant reached through a link that sits in one of those regions is dropped too, though it ends outside: a command of one run could point the link elsewhere for the next.
      Its line says to grant the target by its own path.
      The same holds for a link in `/tmp` under any `sandbox.isolation` but `full`
    - an MCP server's own `extra_read_paths` and `extra_write_paths` (`[mcp.servers.<name>.sandbox]`) follow the same rule, resolved once when the server is spawned; a dropped one is named at run start under its own key
    - `agent6 check sandbox` lists each grant by its resolved path, with the configured spelling when the two differ, and each dropped grant with the region it sits in
- A denied path is refused with the reason
    - `list_dir` drops the entry and reports `hidden: N`
    - the jail masks the path instead (empty dir, empty file; [Sandbox](#2-sandbox))
- The symbol index skips a hidden file (an indexed one leaks symbol names and line numbers through `find_definition`)
- While `sandbox.protect_git` is on (`auto`, the default, is on except where `git.control = "model"` needs a writable `.git`) the edit tools refuse a write into the project's own `.git`, raw or symlink-resolved, at every isolation level
    - the name matches case-folded on every platform (macOS and Windows open `.GIT/config` as `.git/config`; macOS runs unsandboxed)
    - a `pyvenv.cfg` dir, a `site-packages` ancestor, and an operator protect path (a machine bundle's `.asm.toml` and `scripts/`) refuse the same way, whatever `protect_git` says
- Rewriting an editable-install `.pth` corrupts a venv invisibly (venvs are gitignored), so those writes refuse
    - reads stay allowed; an editable install still imports itself inside the jail (the repo is bound at its real path)
- Inside the per-repo state dir (`$XDG_STATE_HOME/agent6/<repo-id>/`), the edit tools may write `memory/` and nothing else; harness-written operator rulings live outside that grant in `decisions.md`.
- Uncovered: with `isolation = "none"` and `run_commands = "yes"`, a command reads a denied path directly (`run_command` argv is not screened)

**Prompt reads.** The agent process itself reads `AGENTS.md` and the manifests the verify inference uses (`pyproject.toml`, `package.json`, a Makefile and the like), and their text reaches the provider.

- A file is read only when its real path stays inside the workspace root (the git root for `AGENTS.md`), outside `.git` and outside the hidden set; anything else reads as absent.
- A symlink that stays inside the root is followed (`AGENTS.md -> docs/AGENTS.md`).
- The existence probes (a root `verify.sh`, a manifest, loose `test_*.py` files) and the model call's top-level listing leave out a file under the hidden set, by its name or its link target.
  A link out of the root still counts as present.

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
`run_commands = "no"` withholds the verify gate too, and such a run is gateless: its editing turns commit as checkpoints.
Under `ask`, a denied gate is withheld the same way for the rest of the run (no retry loop can discharge a refusal), and the run ends unverified.

The agent works within the environment it is given and cannot expand it:

- `sudo` cannot escalate, even passwordless: `NO_NEW_PRIVS` voids setuid, so jailed `sudo` fails under any `NOPASSWD` rule.
- Package installs fail: `apt` / `dnf` / `apk` need root (blocked), a route to a mirror (the default `network` has none off the box), and `/usr` or `/var` writes (denied)
- Host-installed toolchains compile and run; a networked build step needs `network` loosened.
    - a toolchain that keeps its state under your home needs that directory granted and named, because the jail's `HOME` is its own: for rustup's `cargo`, `sandbox.extra_read_paths = ["/home/you/.rustup"]` and a gate of `env RUSTUP_HOME=/home/you/.rustup cargo test`
- Toolchains, venvs, and deps are installed outside agent6. Access widens through config (`extra_read_paths`, `network`, `[providers.*].base_url`), all visible in `config show`.
- The setup command of an isolated workspace (`[workspace].setup_command`, or an AGENTS.md `## Setup command` block) is a jailed command like any other
    - Machine clones use the operator's configured command, else the AGENTS.md block at the starting commit recorded in `machine.begin`, never a block committed by a run state; read-only agent states run this setup too, under their command policy and jail
    - Machine setup outcomes follow the [state setup and failure rules](state-machines.md#agent)
    - A machine clone needing setup refuses `sandbox.run_commands = "ask"` at preflight rather than asking; use `--auto-approve` or set it to `"yes"` to allow setup
    - it runs through the run's own jail: the isolation, network and grants every command of that run gets, the approval `run_commands = "ask"` asks, and nothing under `"no"`
    - it has no network of its own and no read of the checkout the workspace was cut from: a setup step with a wider network would let a dependency it fetches run with it
    - an AGENTS.md block is repository content, so a repository can state the command; it reaches what a command the model chose would reach
    - the files it leaves untracked stay out of the run's commits
    - under a sandbox with no route out (the default network), the command has to work offline
    - the recipe measured under `full`, for a uv project:
        1. `agent6 config set --repo sandbox.home cache`, so jailed commands in this repository share one persistent `HOME` and its tool cache
        2. warm that cache once with your own networked command, from the checkout: `agent6 --config net.toml exec --new -- uv sync --reinstall --link-mode=copy`, where `net.toml` sets `[sandbox] home = "cache"` and `network = "host"` (the host network brings the resolver and CA files with it)
           The reinstall fetches every locked package; a plain `uv sync` over a checkout whose `.venv` is complete fetches nothing, and step 3 then fails offline.
        3. `agent6 config set --repo workspace.setup_command '["uv", "sync", "--offline", "--link-mode=copy"]'`: a new worktree's `.venv` is then built from the cache with no route out (1.2 s for a two-package project), importing the worktree's own source
           `--link-mode=copy` on both steps: the jail binds the cache `HOME` and the working tree as separate mounts, so uv cannot hardlink between them and warns on every install without it
    - the warmed cache serves every working tree agent6 makes for the repository: a `run --worktree` worktree, a fork's, a `--parallel` lane's clone and a machine state's clone all get the repository's `HOME`
- Running agent6 as root needs `--allow-root` / `AGENT6_ALLOW_ROOT=1` (plus a banner) and weakens the boundary
    - jailed children run as real root (uid 0), even under `full` `sandbox.isolation`: the uid is no boundary, and the level's layers (the table in [Sandbox](#2-sandbox)) are the whole confinement
    - a setuid bit planted through `openat2` (the one create path seccomp cannot filter) then lands on a root-owned file in the workspace.
      That is local root for whoever runs it outside the jail
    - writes outside the workspace and routes off the box stay closed
    - readable files include root-only ones (`/etc/shadow` under `partial`; `full`'s rootfs hides it)
- Under `sudo`, agent6 reads the real user's config and secrets (`SUDO_UID` / `SUDO_USER`) and chowns state-dir writes back
    - it does not drop privileges in-process; confinement comes from the jail

### 5. Network

The agent process's own egress is unbounded.
Everything below bounds a child of it.

A `claude_code` provider's child dials `api.anthropic.com` and `claude.ai` (login refresh) from the agent process; telemetry, crash reporting, and update hosts are disabled by its environment.
While it runs it listens on a same-uid session socket under `/tmp/cc-socks/`, reachable from a jailed command only under `partial`, whose `/tmp` is the host's.

**`fetch`** is the model's only direct egress.

- One https URL, GET, no redirects followed, no credential, text only, 1 MiB, 20 s.
- Hosts on `sandbox.fetch_approved_hosts` are read without asking (empty by default); any other host prompts.
- Nothing resolves before that gate: a DNS query delivers the hostname to whoever runs its authoritative server, so an unapproved URL never reaches a resolver.
- Behind the gate the name resolves, and every address it yields must be global: loopback, the LAN, and `169.254.169.254` are refused, as is a URL naming one literally.
  The request dials the resolved address with the name in SNI and `Host`, so no second lookup can rebind it.
- A `content-encoding` other than identity is refused: the cap counts bytes as they arrive, and a decoded stream would expand past it.
- Hidden when a jailed command already has the host network: `network = "host"`, or any isolation but `full`.

**Which network a jailed child joins.** `sandbox.network` is one of:

- `auto` *(default)*: the run's own network where the host can give one, the host's with a warning where it cannot.
- `session`: the run's own, enforced; refuses where the host cannot give one.
  Its members reach each other and nothing off the box.
- `host`: the machine's network, with the host's resolver settings and CA store readable ([Threat model](#threat-model)).

A run's commands share one launcher, so there is no per-command `none`.
A machine state sets its own value in its `config` table, never wider than this one ([State machines](#11-state-machines)); an MCP server takes its own in its `[sandbox]` block.

The run owns one session network.
A holder process creates it, the run keeps it alive with an open descriptor on `/proc/<holder>/ns/{user,net}`, and every child that asks joins those.
Entering a network namespace needs `CAP_SYS_ADMIN` in the user namespace that owns it, so a joiner enters that user namespace too.
It still gets its own mount, PID, IPC, and UTS namespaces, so two members cannot see or signal each other.

Under `full`, the only level with namespaces:

| jailed command | `auto` *(def)* | `session` | `host` |
|---|---|---|---|
| `run_command` | the run's session network | same | host network |
| a machine `tool` state | its own, alone | its own, alone | host network |

The column is the value the command's config resolves to: your `sandbox.network`, narrowed by the machine's `[config]` or set by the state's own `config`.

Below `full` there is no network namespace: `auto` degrades to the agent process's network with a warning (once per run under `partial`; under `none` the unsandboxed warning covers every `auto` knob), while `session` refuses.

**Refusals** (fail-closed)

| Configuration | When |
|---|---|
| a machine overlay sets `network = host` under your `auto` or `session` | `machine check`, machine start |
| explicit `network = session`, yours or a machine state's | run or machine start under `partial` or `none` |

**MCP servers** take their own value per server (`[mcp.servers.<name>.sandbox].network`, default `auto`):

- `auto`: a network of its own where the host can give one, degrading to the host's with a warning.
- `none`: refuses instead of degrading.
- `session`: joins the run's network, so a browser server reaches the dev server a background command started.
- `host`: the machine's network.
- A server is spawned on stdio or dialled at an operator-set `url`, outbound either way.

**Ingress.** The loop opens no accept-side socket; the task graph is an in-process curator.
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
      Every response carries `Content-Security-Policy: frame-ancestors 'none'`, so such a page cannot frame the signed-in UI either.
      A browser that signed in before the token kept a cookie: the page's response expires it, and the token's derivation changed, so a stale cookie another local server collects opens nothing.
- The server renders folded state and drives typed contracts.
  New-work spawns fixed argv with the task behind `--`, and machine-run is allow-listed to authored files.
  Answers write only the addressed run's answer files (session id, answer id, and machine target state dir each validated to one path component).
  Merge, prune, and config-set are fixed agent6 subcommands.
- No request carries an ambient credential, so no POST needs a cross-site guard: a cross-origin page cannot add the header without a preflight the server never answers.
  DNS rebinding is not covered (that needs a Host allow-list incompatible with the tailnet name): a rebound page holds no token (storage belongs to one origin), so it reaches the public routes and can try secrets at the login; keeping it out is left to the network layer.
- Request framing is bounded: 1 MiB body cap (413), chunked refused (411), and any unread-body refusal closes the connection.
- The machine write surface (`POST /api/machine/<name>/{poke,stop,steer,approve,answer}`) uses the same guards.
  `poke` writes the instance's signal file (inert JSON the next `tool` reads) and `stop` its stop marker; `steer`, `approve` and `answer` write only the current agent state's per-state dir.
- PWA assets are static and the service worker is a no-op passthrough (no Web Push, no VAPID).
  No telemetry, no auto-update, no remote control plane.

### 6. Approvals

- `[sandbox].run_commands` (`ask` default, `yes`, `no`) gates every command tool; an `ask` prompt shows the argv
- Each `mcp__<server>__<tool>` call prompts with its arguments (`[mcp.servers.<name>].approve`, default `ask`).
- An "allow all" answer covers that server for the run, never the command tools or another server
- A tool name matching no configured server is refused, not prompted.
- The `fetch` off-list host prompt and the sandbox-off gate take no standing answer; both say so, and no front-end shows the button
- `isolation = "none"` with auto-approved `run_command` adds a one-time gate: `Continue?
  [y/N]` interactively, a warning in CI and `machine run`.
- A prompt with no operator to answer it: a headless run refuses to start under `ask` unless `AGENT6_DETACHED_AWAY` is `deny` (auto-deny), `wait` (park it for a front-end) or `approve` (grant every scope).
  A question under `deny` or `approve` gets empty answers with a note to decide alone.
  An unattended machine auto-denies.
- `agent6 mcp serve` has no operator at all, so the tools that would prompt are not published.
  Under `run_commands = "ask"` or `"no"` its command tools are absent from `tools/list`, and a client that names one is told which setting withheld it ([the tools](acp.md#as-an-mcp-server)).

### 7. Git

**agent6's own git**

- agent6's own git writes go through `git_ops.py`; only it may import the private runner, `_git_command.py`, enforced by tach and `test_only_git_ops_imports_the_runner`
    - it wraps the safe ops (status, add, commit, diff, branch, checkout)
    - it spells no destructive verb at all: `push`, `reset --hard`, `commit --amend`, `rebase`, `filter-branch` / `filter-repo` and any `--force` / `-f` appear nowhere in it, so there is nothing to enable.
      `test_git_ops_never_spells_a_destructive_verb` pins every verb but `filter-repo`, exempting two argv: `push` inside a `git stash push`, and the one `branch -D` below
    - the collectors on the [subprocess allowlist](#12-host-side-subprocess-allowlist) carry the same hardening flags.
      `sessions diff` and `ask` read only; `review` stages untracked files with `add -N` so they appear in its diff, undoing it with `reset` in a `finally`; `skills install` clones with fixed argv
- One operator-only exception: `sessions prune --delete-squashed` force-deletes a run branch or chain ref the manifest confirms was squash-merged.
  A branch's commit survives in its reflog; a chain ref's line carries the sha to undelete it.
- `_git_command.py` runs git with the configured `api_key_env` names removed from its environment: a credential helper or content driver never inherits one
    - PATH, SSH, proxy, and credential-helper vars stay
    - the read-only collectors inherit the environment untouched (no remote contact; the hardening flags leave no repo-controlled code to receive it)

**A `git` the model runs through `run_command`** is bounded by the sandbox, and its argv is not screened.

- `protect_git = "on"` requires `full` `sandbox.isolation` to bind-remount `.git` as read-only; `auto` (the default) binds under `full` and keeps the edit tools' refusal at every level
    - a mount nested under `.git` stays visible and read-only
    - a rewrite fails; `push` has no egress
    - a grant that contains the workspace does not lift it ([Sandbox](#2-sandbox))
- Under `partial`, `protect_git = "auto"` degrades with a warning and `"on"` refuses to run
    - a jailed command there can plant a `filter.<n>.clean` plus a `.gitattributes`; agent6's own auto-commit (a temp-index `git add -A` on the host) blanks every driver the repo defines unless `git.run_repo_drivers` is set
    - Landlock cannot express the exclusion: a directory grant is recursive and stacked rulesets intersect, so denying `.git` means not granting the workspace root either
- `git.control = "agent6"` is enforced by the sandbox; without it the run warns that a commit the model runs lands on the checkout's branch.
- The protected scope is the project's own `.git` at every isolation level (the in-process edit tools refuse writes under it; [File access](#3-file-access))
    - a nested `.git` (a vendored repo's, a submodule's) is workspace content, writable like any other file

**Repo-controlled host code in a poisoned `.git/config`**

- `core.fsmonitor` and `diff.external` are always off.
- `.git/hooks/*` run only under `git.run_repo_hooks = true` (default false); `core.hooksPath` points away, so a hook cannot fire on agent6's auto-commit.
- Content drivers (`filter.<n>.clean` / `smudge` / `process`, `merge.<n>.driver`) are off by default (`git.run_repo_drivers`), neutralized per name
    - the clean filter runs on the auto-commit's `git add`, the merge driver on the chain merge's `merge-tree`: a cloned poisoned repo fires one without any model action
    - `true` honors them (the Git-LFS opt-in)

### 8. Secrets and `connect`

- Provider keys live in `$XDG_CONFIG_HOME/agent6/secrets.toml`, `0600` and owner-only (refused if group- or other-readable, or foreign-owned), or come from `[providers.<name>].api_key_env` (env wins).
- They are absent from transcripts, redacted in `config show`, and masked from the jail.
  The config dir stays masked even under an explicit grant, and a grant naming it directly is refused at config load.
- `config show` and `config get` mask mapping values unless the config model marks the field display-safe (`skills.state`); other mappings expose only their keys, including in JSON output.
- No child agent6 spawns carries one: a jailed command's environment is built from its policy, and the unconfined paths (`isolation = "none"`, git subprocesses, MCP servers) drop every configured `api_key_env` name.
    - an `[mcp.servers.*].pass_env` naming a provider's `api_key_env` refuses at config load, so the name cannot be handed over deliberately either
- `agent6 connect` prompts locally (`getpass`) and writes config and secrets
    - one read-only `GET` of the provider's `/models` endpoint (OpenRouter's auth-gated `/key`) confirms auth (status only; `--no-verify` skips it)
    - it executes nothing a remote returns
- `agent6 connect chatgpt` is a PKCE OAuth sign-in
    - a browser hits the authorize page of OpenAI's fixed OAuth authority (a constant, not config); the code returns on `localhost:1455` (or is pasted), state-checked either way
    - token exchange and refreshes `POST` only to the authority's `/oauth/token`
    - tokens live in `secrets.toml` under the same `0600` and executes-nothing rules
    - agent6 never sends a rating or feedback on a response and has no rating surface.
      The backend may use rated turns for training, and that choice is never made on the operator's behalf
- `agent6 connect claude` stores nothing: agent6 never reads or copies Claude Code's own login (`~/.claude/.credentials.json`, `~/.claude.json`)
    - no default mount or grant holds it.
      A jailed command reads it when the workspace or a path grant (`sandbox.extra_read_paths`, `sandbox.extra_write_paths`) is the home directory or a directory above it, and under `isolation = "none"`
    - a `sandbox.hide_paths` entry hides it from commands under `full`
    - the child's environment carries no `ANTHROPIC_*` or `CLAUDE*` variable from the operator shell, so a shell API key cannot override the subscription login; `CLAUDE_CONFIG_DIR` is the one passthrough
    - `claude auth status --json` is parsed for `loggedIn` only; its body (email, org) is never printed or journaled
    - the binary's initialize handshake carries the account block.
      agent6 keeps the email in memory only, to replace it with `<operator-email>` in the model's returned text, and records none of it

### 9. State and locks

- An in-process `GraphCurator` owns the task graph
    - every mutation validates against a pydantic schema before writing, under a per-mutation flock on the session dir
    - a write-path fault after the in-memory update reloads from disk before surfacing: a later read never observes a node that was never persisted
- Per-repo state lives at `$XDG_STATE_HOME/agent6/<repo-id>/`, outside the working directory jailed commands run in.
    - every directory under the base is created `0700` whatever the umask (`mkdir_for_real_user`, the one creator), keeping transcripts, memory and run history from other local users.
      A base an older release created at `755` keeps that mode
- The config write lock serializes read-modify-write cycles and enforces nothing
    - publishes are atomic: a torn config is impossible with or without it
    - it fails open (a planted symlink refuses `O_NOFOLLOW`; a stale root-owned lock is ignored); a write proceeding without it is kept, reported "kept as written" (docs/config.md)

### 10. Parallel lanes

`agent6 run --parallel` and a live run's `/parallel` steer directive (see [architecture.md, Parallel runs](architecture.md#parallel-runs)) each spawn subordinate work.

- Every lane is an ordinary run: a detached `agent6 run` on its own clone, its own jail per `sandbox.isolation`, its own `run_commands` policy
    - no sandbox socket is shared across lanes or with the parent
- Every spawned lane carries `AGENT6_SUBRUN=1`; both the `--parallel` flag and the coordinator's `lane_spawner` wiring refuse when it is set, so a lane cannot fan out or dispatch again.
- A lane's config carries key references, never secret values
    - the orchestrator writes each lane a `--config` via `materialize()` (the resolved `Config`: provider `base_url`, `api_key_env` names), never a raw key
    - the lane reads the same `secrets.toml` or provider env var as any other run
- Lane git plumbing (clone, fetch, merge) goes through `git_ops.py` and its private runner; lane spawning goes through `ui/spawn.py` (see the [subprocess allowlist](#12-host-side-subprocess-allowlist)).
- A lane starts from committed state only
    - the fan-out clones HEAD; a coordinator dispatch cuts lanes at the run's chain tip after chain-committing changes
    - `--parallel` refuses an origin with uncommitted tracked changes under `git.dirty_tree = "ask"`

### 11. State machines

`machine run` is a supervisor that makes no network calls.
Each `tool` state is jailed with a network of its own, so one state's `network` is set independently ([Network](#5-network)).
A machine can keep agents on the provider API while one reviewed, fixed-argv `tool` reaches the network.

**An overlay never widens the sandbox**

- The machine's `[config]` and each state's `config` may set `sandbox` keys, each bounded by your own layers alone ([the rule per key](state-machines.md#47-machine-config-overlay-config))
    - the ceiling is read from one load of your layers (defaults, global, repo, preset, `--config FILE`), never from a config the machine file has merged into
    - a path grant is at or below one of yours, compared by real path, so a symlink out of your tree is refused
    - a state's grants are judged against every region the machine writes (your write grants, the workspace, `/tmp` off `full`), as one run's are against its own: a grant below such a region, or reached through a link inside one, refuses `machine check` and machine start, since a state writing the region can point it anywhere; the region itself may be granted
    - a wider value refuses at `machine check` and at machine start, naming the state and the key
    - the other operator-only tables and leaves are rejected at load ([the list](state-machines.md#47-machine-config-overlay-config))
- One resolved config per state feeds both enforcement sites: a `tool` state's jail, and an `agent` state's subprocess (its in-process file tools and its jailed commands)
    - every startup check a plain run makes (the `hide_paths` check under `partial`, network support) runs per state and names the state
    - a state the supervisor resolved no config for runs nothing
- A `--config FILE` inside the workspace is read-only in every jail, like the bundle: a state cannot raise its own ceiling
- A grant bounds what a state touches directly: a granted state can copy what it reads to the blackboard, the data dir, the jail's `HOME` or a commit, where a later state reads it
- A `tool` only declares `pass_env`; whether a variable is passed is your `[machine].pass_env`
    - its setup command gets none of them: setup runs the tree's code at the chain tip, which a run state can edit, so a passed secret would reach model-written code

**Bundle confinement**

- Scripts live in a reviewed `scripts/` beside the `.asm.toml`
    - `machine check` verifies every entry and static reference resolves inside the bundle (escaping symlinks rejected)
- Scripts are operator-authored and committed, never fetched or generated at run time
    - the `.asm.toml` and `scripts/` are read-only in every jail during a run
- Front-ends render `machine.notify` as an overlay, and `attach` / TUI call `notify-send` with fixed argv, so a model message is inert data.
- The out-of-band hook `[machine.notify].on_event` runs an operator argv on the host with the minimal `hook_env` env plus `AGENT6_MACHINE_*`.

### 12. Host-side subprocess allowlist

Everything the model can influence runs through `run_in_jail` ([Sandbox](#2-sandbox)).
A fixed set of modules also shells out directly with `subprocess.run` / `Popen`, each with fixed argv depending only on operator input.
`tests/security/test_subprocess_allowlist.py` pins the file list; audit with `rg 'subprocess\.|os\.(system|exec|posix_spawn)' src/agent6/`.

- `_git_command.py` and `git_ops.py`: agent6's own git operations ([Git](#7-git)).
- `git_policy.py`: shared hardening flags and hook/driver policy; its fixed git config query enumerates repo content drivers.
- `sandbox/detect.py`: probes the host's sandboxing capabilities.
- `sandbox/jail.py`: the jail launcher, and the plain spawn under `isolation = "none"`.
  An MCP server spawns through the same launcher and `JailPolicy` a jailed command gets (`spawn_in_jail`); one the operator opted out takes the `none` level.
- `sandbox/jail_jobs.py`: detached jail commands and the plain background spawn under `isolation = "none"`.
- `sandbox/jail_session.py`: persistent jail launcher sessions and their teardown.
- `providers/token_command.py`: the `[providers.*].token_command` that mints a provider bearer.
- `providers/claude_code.py` and `providers/_claude_code_session.py`: the `api_format = "claude_code"` provider and its session lifecycle helpers run the Claude Code binary the operator connected with fixed argv from operator configuration, never model output (binary, model, effort, literal flags, the path of a 0600 system-prompt file in a private empty directory).
  Prompts, tool results, and notices travel on stdin, so model- or repo-derived text never becomes an argv element.
  Curated environment, `--tools ""`, `--allowedTools mcp__agent6`, `--setting-sources ""`, `--strict-mcp-config`, `--disable-slash-commands`, `--no-session-persistence`; CLAUDE.md, auto-memory, and auto-compaction off by environment; `system/init` is audited so a tool outside `mcp__agent6__*` or an API-key source refuses the run.
  Unjailed: it needs the operator's login under `$HOME` and its own egress, at the agent process's trust tier.
  Tool results stay under Claude Code's 50,000-byte persistence threshold, so no tool output is written under `~/.claude`.
  The loop caps them at 34,000 bytes for this provider, keeping 16,000 for the turn's notices; a wider payload is refused.
  `claude auth status --json` runs the same way for the sign-in preflight.
- `sessions/process.py`: `ps -p <pid> -o lstart=` and `ps -p <pid> -o stat=` on hosts without `/proc` (macOS), for process start-time identity and zombie checks, over a pid agent6 recorded.
- `ui/btw.py`: spawns `agent6 ask` detached for `/btw` (every composer), so the side question keeps provider egress while the run is confined.
  Argv is the agent6 exe plus the question the operator typed, with `--` before it.
- `ui/spawn.py`: the shared front-end spawn helper.
  Spawns the agent6 CLI detached for run and machine launches (`machine create` and `machine run`), and captures `sessions merge` / `prune` / `rm` and `config set` / `unset`.
- `ui/notify.py`: `notify-send` with fixed argv (exe, `--`, two positional data args, no shell) for the device-present machine notification.
- `ui/cli/` helpers:
    - `plan_watch.py` and `_steer.py`: `$EDITOR` for plan and steer editing
    - `review_cmds.py`, `sessions_cmds.py` and `_ask.py`: `git diff` / `log` for the review subcommand and the `sessions` / `ask` diff views, with argv from the run manifest the CLI wrote outside the jail
    - `history_cmds.py`: `rg` for history search
    - `_live.py`: the fixed-argv `python -m agent6.ui.tui` co-process behind `run --tui`
    - `system_cmds.py`: `cp` / `rm` / `apparmor_parser` via sudo with fixed argv for `agent6 system apparmor`
    - `completions_cmd.py`: runs nothing itself; its one `subprocess.run` is text inside the xonsh completer script it emits, which the operator's xonsh runs at tab time to ask `agent6` for candidates
- `app/finalize.py`: both operator notify hooks (`run_notify_hook`): `[notify].on_complete` at run end and `[machine.notify].on_event` from `machine run` (built by `app/machine/_preflight.build_machine_notify_hook`).
  Argv from config, env from `hook_env` (a minimal base plus `AGENT6_SESSION_*` or `AGENT6_MACHINE_*`, never the provider keys in the operator environment).
- `app/machine/_scriptcheck.py`: ruff and ty with fixed argv, reading generated scripts statically.
  Those scripts execute only via `run_in_jail`.
- `app/machine_agent.py`: spawns each agent state as a fixed-argv `python -m agent6.ui.cli.machine_agent` subprocess whose request travels in a temp file.
- `ui/cli/skills_cmds.py`: `git clone --depth 1 -- <url>` with fixed argv for `agent6 skills install`.
  The URL is operator-supplied and nothing fetched is executed.
- `ui/tui/clipboard.py`: `tmux set-buffer -w` with the copied transcript text as one data argument.
- `ui/tui/conversation.py`: the operator's `$PAGER`, argv from the environment, transcript text on stdin.

## Skills

- A skill is operator-installed config: install only from trusted sources
    - `skills install <url>` is an operator-initiated CLI fetch (the `connect` trust class); what it installs enters the system prompt and tool results verbatim
- Nothing in a skill runs at install or load
    - its scripts run only through the jailed command path, subject to `run_commands`
- `use_skill` is read-only and path-contained: the skill's own dir through a component-walked descriptor (any symlink hop or `..` inside it refused)
    - a skill directory or `SKILL.md` that is itself a link is followed, and a read that lands under `sandbox.hide_paths` or an agent6 private directory is refused, as the workspace file tools refuse it
    - a skill whose `SKILL.md` resolves under one of those paths is not loaded for a run at all: its name, description and text stay out of the prompt's skill index, the `always` injection and `/name`, and the run prints a line naming it; `agent6 skills list` still shows it
    - skill dirs are not mounted into the jail; content reaches the model engine-side
- Repo-local `.claude/skills/` are not discovered; only the installed dir and `[skills].extra_dirs` are scanned

## Prompt-injection tests

[`tests/security/test_prompt_injection.py`](https://github.com/agent6-dev/agent6/blob/master/tests/security/test_prompt_injection.py) drives the dispatcher with the calls a compromised model would make.
Those are path traversal, a swapped symlink between check and open, an unknown tool name, extra fields, and a write outside the workspace.
It asserts the tool surface refuses them whatever the model says; it does not test the model's judgement, and calls no model.
It catches prompt regressions; the structural defenses above confine a model that follows an injection.

## Known limitations

- `full` `sandbox.isolation` requires user namespaces and refuses on distros that disable them.
- AppArmor userns (Ubuntu 24.04+) blocks unprivileged userns without a profile
    - `agent6 system apparmor install` adds the profile that `full` `sandbox.isolation` needs, scoped to the launcher
- seccomp is required; kernels that block it from unprivileged callers make the jail fail closed.
- Devcontainers get `partial` `sandbox.isolation`.
  The container bounds filesystem damage, and jailed commands share its network ([Network](#5-network))
    - the XDG state base is ephemeral (lost on rebuild): mount a volume at the state dir or point `XDG_STATE_HOME` at one
- agent6 installed inside the project it works on (pip into the project's own venv) puts the running agent's code in the jail's writable workspace
    - a jailed command can rewrite it; the next tool call runs the rewrite as you, outside the jail
    - install agent6 outside the tree (pipx, `uv tool`); agent6 warns at run entry on this shape
- Claude Code (`api_format = "claude_code"`) appends the account email to every system prompt it sends and has no switch for it.
  The model can echo it, and agent6 scrubs it only from returned text, never from tool inputs (an edit carrying it lands as written)
    - managed settings (`/etc/claude-code/managed-settings.json`) still apply inside the child.
      The `system/init` audit refuses the run when the tool list differs from what agent6 offered or `apiKeySource` is not `none`, and checks nothing else.
      A managed hook, or a managed MCP server exposing no tool, passes it
- Side channels: no claim about timing, cache, or speculative side channels.
- Supply chain: pin your install
    - runtime deps `pydantic`, `httpx2`, `argcomplete`, the `tree-sitter` pair, `textual`, `ruff`, `ty`; build dep `hatchling`; jail crates `nix`, `libc`, `landlock`, `seccompiler`, `serde`, `serde_json`
