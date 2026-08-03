# SECURITY

agent6 treats the LLM as untrusted. Concrete claims below, layer by layer, each
with what it means and where it stops.

## Reporting

Open a GitHub issue prefixed `[security]`. Include: agent6 version
(`agent6 --version`), kernel + distro (`uname -a`, `/etc/os-release`),
`agent6 check sandbox` output, and a minimal repro (ideally a failing test under
`tests/security/`).

## Threat model

Adversary: a fully malicious worker model, or an honest model that has
been prompt-injected by a file in the workspace.

We assume the adversary controls:

- the text of every LLM response
- the choice of tool calls and their arguments (within the published
  JSON schema)
- the content of any file the agent reads during the run

We do NOT assume the adversary controls:

- the kernel
- the agent6 binary
- the provider endpoints

Under that adversary, agent6 aims to hold:

1. **No writes outside the workspace.**
2. **No reads outside the workspace and a read-only system set.**
    - The system set (`/usr /bin /sbin /lib /lib64 /etc /dev /proc /tmp`) exists
      so installed toolchains resolve.
    - `hardened` also exposes `$HOME` + `/run` (Landlock can't carve them out);
      `sandbox.extra_read_paths` adds more.
3. **The agent process's own egress is NOT bounded.** agent6 talks to the
   configured providers; nothing stops the PROCESS reaching elsewhere.
    - There was once an empty netns plus a broker proxying provider calls. It was
      deleted: under `strict` the agent process has no filesystem confinement, so
      code execution there could write `~/.ssh/authorized_keys` or a cron entry
      and exfiltrate on its own schedule. Blocking the socket while leaving that
      open is a partial mitigation that reads as a guarantee, and it cost four
      special cases plus a finding of its own.
    - What IS bounded is what a COMMAND reaches: `sandbox.tool_network`
      (default `auto`; §8). That boundary is per-jailed-child and unchanged.

4. **agent6's own git never pushes, `--force`s, rewrites history, or `reset
   --hard`s** (§5).
    - This does NOT bind a `git` the model runs via `run_command`; that path is
      bounded by the sandbox (`protect_git` keeps `.git` unwritable under
      `strict`; push needs egress).
5. **No persistence after the run:** no daemon, cron, `.bashrc` write, or
   setuid binary.
    - **No setuid/setgid bit can be set.** `chmod`/`fchmod`/`fchmodat`/
      `fchmodat2` are denied by seccomp when the mode carries
      `S_ISUID`/`S_ISGID` (ordinary chmod is untouched). `fchmodat2` is named
      separately because it SUPERSEDED `fchmodat` on Linux 6.6+: a filter
      listing only the older spelling is open on any current kernel, and
      `chmod` never reaches it, so only a direct syscall test finds that.
      The bit would land on the HOST inode and outlive the jail, and under
      `sudo agent6 --allow-root` the uid_map makes the jailed child real root
      -- a setuid-root binary left in the workspace is local root for anyone
      who runs it. Mount `nosuid` does not cover this: it stops the JAIL
      honouring the bit, not the host. Every mount carries `nosuid`
      and `nodev` anyway -- the workspace, the system binds, the protect and
      read-only binds, the writable grants, and the private `/tmp`. Not
      `noexec` on `/tmp`: HOME lives there and toolchains run helpers from it,
      and a child that can already execute from the workspace gains nothing
      from being stopped there.
    - Children can only write inside the jail's mount namespace (strict) or the
      Landlock write grants (hardened).
    - **Nothing a command starts outlives it.** strict's PID namespace takes the
      whole tree down. hardened has none, so the agent makes itself a child
      subreaper: a `setsid` daemon that escapes the launcher's process group
      reparents to the agent, which kills it when the command returns.
    - The bound is the command's own descendants. On hardened a command can
      still hand work to a user daemon that was ALREADY running (a tmux server,
      `systemd --user`): `AF_UNIX` connect has no Landlock hook, and without a
      mount namespace those sockets are nameable. strict does not expose them.

## Defense layers

### 1. Agent-process Landlock (`hardened` only)

Applied at `run`/`resume` start, before any network object. Restricts the Python
process irrevocably, inherited by every child:

| Landlock rule | Allowed |
|---|---|
| FS read+exec | cwd, `$HOME`, `/usr`, `/etc`, `/tmp`, `/bin` `/sbin` `/lib` `/lib64` `/dev`, `/run` + `/proc` when present |
| FS write | cwd, `/tmp`, the `/dev` char devices, `/proc` when present |

- **`strict` skips this layer.**
    - Its per-command namespaces are stronger, and this would
      break the jail's `pivot_root`/`mount` at Landlock ABI ≥ 7.
- **The read+exec set mirrors the jail child's roots.**
    - The launcher opens each from here to grant the child, so a missing one
      (e.g. `/dev` on merged-`/usr`) makes the child's execve fail EACCES.
- **Neither level bounds the AGENT's egress. Both say so rather than pretending.**
    - Landlock filters connects by PORT, not host, so the only rule available
      here was "any host on the provider ports". That stops nothing worth
      stopping -- an exfiltrating agent needs one HTTPS endpoint, and every host
      offers one -- while breaking a legitimate tool on another port. It was
      removed: a security claim nobody can rely on is worse than no claim.
    - Bind/listen is not denied either. Blocking inbound while outbound stays
      open is the same non-claim, and it cost a dev server the model runs its
      listening socket. Nothing it binds outlives the command (§5).

### 2. `agent6-jail` (Rust), for every child command

`apply_edit` is in-process; every `run_verify_command`/`run_command` runs in
`agent6-jail`, as does every `run_background` (same policy, same launcher; the
only difference is that nothing waits on it, so the run's teardown kills it).
Under `strict` it:

- Forks a new user/mount/PID/IPC/UTS/net namespace.
- `pivot_root`s into a minimal bind-mount rootfs on a fresh tmpfs: cwd + private
  `/tmp` writable, system paths read-only, `extra_read_paths` grants read+exec
  at their real paths (`extra_rw` grants writable at theirs). Operator-tool
  dirs join as read+exec `tool_paths` mounts (standard bin dirs that exist, the
  real dirs their symlinks resolve to, uv-managed CPythons), derived by
  `sandbox.jail.operator_tool_paths` -- which never mounts agent6's OWN config,
  state, data or cache dirs however a tool symlink resolves into them, so
  `secrets.toml` and the run history stay outside the jail by construction
  rather than by directory layout. A tool dir whose read-only remount fails is
  detached, and a failed detach refuses the run: best-effort means unreachable,
  never writable. run_command/verify jails and machine tool jails share that one
  computation, and `machine check` probes the same PATH.
- Exposes curated `/dev` (`null zero urandom random full`); omits `/dev/tty`
  (it would let a child write escape sequences to the parent's terminal).
- Mounts a fresh private `/proc`; if that fails, leaves `/proc` empty (never the
  host's, which would leak process info).
    - The launcher is PID 1 of that namespace, so `/proc/1/environ` is readable
      by the jailed command. It is spawned with an EMPTY environment for that
      reason: inheriting agent6's put a provider key supplied via `api_key_env`
      one file read away from the model. The launcher needs none -- its policy
      arrives on stdin and the child's env is set explicitly in it.
- Applies Landlock FS rules (net confinement is the namespace); best-effort:
  a kernel without Landlock skips this layer, warned loudly at run entry.
- Installs a seccomp deny-list: dangerous syscalls (ptrace, mount, setns,
  unshare, kexec, bpf, perf, keyctl, module loading, reboot, clock-set, …)
  return `EPERM`, the rest allowed.
- Sets `NO_NEW_PRIVS`, so the kernel ignores setuid bits (`sudo`/setuid can't
  escalate).
- `execve`s the binary and SIGKILLs the group at the wall-clock timeout.

Notes:

- **The memory cap is operational, not a threat-model control.** A per-process
  `RLIMIT_DATA` (`[sandbox].memory_limit_mb`, default 4096, `0` off; not
  `RLIMIT_AS`, so V8/JVM/ASAN keep working) stops one runaway allocation, nothing
  more.
- **The seccomp layer is a deny-list (defense-in-depth), not a boundary.** It
  enumerates known-dangerous syscalls, so by construction it does not catch
  everything: namespace creation via `clone`/`clone3` (`CLONE_NEWUSER|CLONE_NEWNS`)
  is not blocked, only `unshare` is. Accepted, because a nested namespace grants
  no new access against the host — the whole mount family (`mount`,
  `mount_setattr`, `open_tree`, `move_mount`, `fsopen`/`fsconfig`/`fsmount`/`fspick`,
  `pivot_root`, `umount2`) is denied, Landlock is inherited and irrevocable, and
  `mknod` checks caps against the initial user namespace. Filtering `clone3` is
  declined on purpose: seccomp cannot read its flags (they sit behind a struct
  pointer), and denying it outright would break glibc/Go process spawning (they
  fall back to `clone` only on `ENOSYS`). The real boundaries are the namespaces,
  Landlock, and the mount-syscall denials — not this list.
- **No `capset`.** `strict` maps namespaced-root to your uid; `hardened` keeps the
  caller's caps (none for a normal user).
- **`hardened` drops the namespaces + rootfs;** Landlock, seccomp,
  `NO_NEW_PRIVS`, and the timeout remain.
- The policy arrives as JSON on stdin from `run_in_jail`; the Rust side validates
  it against a strict schema and refuses unknown fields.

### 2a. Environment: sudo, packages, provisioning

The jail is one-way: the agent works within the environment you give it and
can't expand it.

- **`sudo` can't escalate, even passwordless.** `NO_NEW_PRIVS` voids setuid, so
  jailed `sudo` fails regardless of any `NOPASSWD` rule.
- **Package installs are impossible.** `apt`/`dnf`/`apk` need all three of root
  (blocked), mirror network (provider-only egress), and `/usr`/`/var` writes
  (denied).
- **Compiling and running host-installed toolchains works.**
    - Every command tool -- `run_command`, `run_verify_command`,
      `run_background`, `stop_background` -- answers to `run_commands` and runs
      jailed. They just can't install new tools, and a networked build step
      needs `tool_network` loosened.
    - The verify gate is a command like any other: `run_commands = "no"`
      withholds it too, and such a run starts gateless rather than chasing a
      green it can never reach.
- **Provisioning is operator-first.** Install toolchains, venvs, and deps
  yourself before/outside agent6; widen access via config, never sudo
  (`extra_read_paths`, `tool_network`, `[providers.*].base_url`, all in
  `config show`).
- **Running agent6 as root** (`--allow-root` / `AGENT6_ALLOW_ROOT=1`) **weakens
  the boundary.**
    - `strict` maps inside-root to real root, so jailed children run as real root
      under only Landlock + seccomp + `NO_NEW_PRIVS`.
    - Still no writes outside the workspace and no egress beyond providers, but
      the allowed *reads* now include root-only files (`/etc/shadow` under
      `hardened`; `strict`'s rootfs hides them). Run as your normal user.

### 2b. Host-side subprocess allowlist

Everything the model can influence runs through `run_in_jail` (§2). A fixed set
of modules also shells out directly with `subprocess.run`/`Popen`; each has
fixed argv depending only on operator input, never LLM output.
`tests/security/test_subprocess_allowlist.py` pins the file list; audit with
`rg 'subprocess\.|os\.(system|exec|posix_spawn)' src/agent6/`.

- `git_ops.py`: agent6's own git operations (§5).
- `sandbox/detect.py`: probes the host's sandboxing capabilities.
- `sandbox/exec_confined.py`: confines itself, then `execvp`s the argv after
  `--`. For a long-lived child agent6 spawns but does not drive (a configured
  MCP server): the jail launcher captures stdio and owns the process to
  completion, which cannot host a live MCP pipe. Both mechanisms are
  restrict-self-then-exec and inherited, so the server and everything it spawns
  get them: Landlock's domain is irrevocable across `execve`, and
  `[mcp.servers.<name>.sandbox].network = "none"` additionally
  `unshare`s a user + network namespace, which the stdio pipes (file
  descriptors) survive untouched. Either mechanism stands alone: a block naming
  no paths applies no Landlock domain (an empty one would grant nothing, to the
  shim and to whatever it becomes), and a shim asked for neither refuses rather
  than exec unconfined while looking confined. The namespace work runs BEFORE Landlock,
  because it writes `/proc/self/{setgroups,uid_map,gid_map}` (mapping the
  operator's uid straight through, so the server does not become `nobody`) and
  opens a socket to bring `lo` up -- none of which the domain grants. A host
  whose kernel forbids unprivileged user namespaces makes the server REFUSE
  rather than start connected. The paths and the argv are the operator's, from
  config; no LLM input reaches it, and `preexec_fn` -- the obvious alternative
  -- is unsafe in a threaded process, which the MCP client is.
- `sandbox/jail.py`: the jail launcher itself.

- `tools/lsp.py`: the `ty` language server, exe resolved from PATH.
- `tools/mcp_client.py`: operator-configured `[mcp.servers.*]` server commands.
- `providers/token_command.py`: the operator-configured
  `[providers.*].token_command` that mints a provider bearer; argv from config.
- `sessions/ipc.py`: `ps -p <pid> -o lstart=` on hosts without /proc (macOS) for
  the worker.pid start-time identity; fixed argv over a pid agent6 itself
  recorded.
- `ui/cli/_btw.py`: spawns `agent6 ask` detached for `/btw`, so the side
  question keeps provider egress when the run itself is confined. argv is the
  agent6 exe plus the question the OPERATOR typed at the pause menu -- never
  LLM output -- with `--` before it so a question starting with a dash cannot
  read as a flag.
- `ui/spawn.py`: the shared front-end spawn helper; spawns the agent6 CLI
  detached for run/machine launches and captures `sessions merge`/`prune`/
  `config set`; argv is the agent6 exe plus operator-chosen args.
- `ui/notify.py`: fires `notify-send` with fixed argv (exe, `--`
  end-of-options, two positional data args, no shell) for the device-present
  machine notification; the message is inert data, never a command or an
  option.
- `ui/cli/` helpers:
    - `$EDITOR` for plan and steer editing.
    - `git diff/log` for the review subcommand and the `sessions`/`ask` diff views;
      argv from the run manifest the CLI wrote outside the jail.
    - `rg` for history search.
    - The fixed-argv `python -m agent6.ui.tui` co-process behind `run --tui`.
    - `ui/cli/system_cmds.py`: `cp`/`rm`/`apparmor_parser` via sudo with fixed
      argv for `agent6 system apparmor` (operator host setup).
- `app/` helpers:
    - `app/finalize.py`: the operator `[notify].on_complete` hook fired at
      run end; argv from config, env from `hook_env` (a minimal base plus
      `AGENT6_SESSION_*`, never the provider keys in the operator environment).
    - `app/machine/_scriptcheck.py`: ruff/ty with fixed argv to statically read
      generated scripts, which only ever execute via `run_in_jail`.
    - The `machine run` supervisor (`app/machine_agent.py`): spawns each agent
      state as a fixed-argv `python -m agent6.ui.cli.machine_agent` subprocess
      whose request travels in a temp file, never on argv; its operator
      `[machine.notify].on_event` hook (argv from config, fired from
      `app/machine/_preflight.py`) runs on the host with the same minimal
      `hook_env` base plus `AGENT6_MACHINE_*`, mirroring `[notify].on_complete`.
    - `ui/cli/skills_cmds.py`: `git clone --depth 1 -- <url>` with fixed argv
      for `agent6 skills install`; the URL is operator-supplied on the CLI and
      nothing fetched is ever executed.
- `ui/tui/clipboard.py`: fixed-argv `tmux set-buffer -w` with the copied
  transcript text as one inert data argument.
- `ui/tui/conversation.py`: the operator's `$PAGER`, argv from the environment,
  transcript text on stdin.

### 3. Profile selection

You set `sandbox.isolation`; it resolves against the host to the *effective*
isolation level. No silent downgrade: a request the host can't meet is refused, and
`auto` reaches `none` only when the host offers no confinement mechanism at
all (non-Linux, or a Linux kernel with neither userns nor Landlock) -- always
loudly. Capabilities are probed (`unshare` for userns, the Landlock ABI
syscall for hardened), never guessed from the kernel version.

| `sandbox.isolation` | Host | Effective |
|---|---|---|
| `auto` *(default)* | Linux + user namespaces | `strict` |
| `auto` | Linux, no userns, Landlock | `hardened` |
| `auto` | Linux, no userns, no Landlock | `none` (loud warning) |
| `auto` | non-Linux | `none` |
| `strict` | Linux + user namespaces | `strict` |
| `strict` | else | ⛔ refuse |
| `hardened` | Linux + Landlock | `hardened` |
| `hardened` | else | ⛔ refuse (Landlock is hardened's only FS boundary) |
| `none` *(opt-out)* | any | `none` (the environment is the boundary) |

- **strict**: full namespaces + `pivot_root` + Landlock + seccomp + `NO_NEW_PRIVS`.
    - On a kernel without Landlock the jail's in-rootfs Landlock layer is
      skipped (namespaces, read-only binds, and seccomp still confine) with a
      loud once-per-run warning; hardened has no mounts to fall back on, so it
      refuses instead.
- **hardened**: Landlock + seccomp + `NO_NEW_PRIVS`, no namespaces.
    - Works in default-seccomp Docker (the container blocks the inner
      `clone(CLONE_NEW*)`); the container is the blast radius.
    - With no PID namespace, teardown is the agent's job (§5): it holds
      `PR_SET_CHILD_SUBREAPER`, and each command kills every process that
      appeared during it from outside the agent's session. A survivor the
      sweep cannot kill fails the command rather than passing silently.
- **none**: unsandboxed, always with a loud warning.

- **Unsandboxing is explicit and self-authorizing.** `isolation = "none"`,
  `--dangerously-disable-sandbox`, or `AGENT6_DANGEROUSLY_DISABLE_SANDBOX=1`. The
  LLM can't reach argv/env, so setting one is the consent.
- **Sandbox-off + auto-approved `run_command` adds a one-time gate.** For that
  combination only: `Continue? [y/N]` interactively, a warning in CI/`machine
  run`.
- CI should set `strict` to fail loud if the sandbox is weaker than expected.

### 4. Fixed tool surface

- **`fetch` is the model's only egress, and it is narrow.** One https URL, GET, no redirects followed, no credential, text only, 1 MiB. Hosts on `sandbox.fetch_hosts` are read without asking; any other host prompts, and an absent operator is a no. It exists because a jailed command has no network, so it is hidden entirely when `tool_network = "allow"`. A GET can still carry data out in its path -- the allow-list is empty by default for that reason.
- **The LLM only sees the fixed set in `src/agent6/tools/schema.py`.**
    - Structured edits, read-only navigation, fixed-argv verify/metric commands,
      `finish_session`, `ask_user`, a curator task notepad, a cross-run memory
      notepad, and capability-gated `run_command`.
    - No `shell`, no `write_file` (writes go through `apply_edit`, which refuses
      paths outside cwd), no `web_fetch`, no `eval`.
    - Adding a tool needs a security review note ([AGENTS.md](https://github.com/agent6-dev/agent6/blob/master/AGENTS.md)).
- **The memory notepad and notes scratchpad are prompt-injection persistence
  channels.**
    - `add_memory`/`invalidate_memory` (run mode) write fixed markdown under
      `<state-dir>/<repo-id>/memories/` (code picks the path; the model supplies
      only a schema-validated scope + text); active notes join later runs' system
      prompt on the same repo. `write_notes` replaces one fixed file,
      `<state-dir>/<repo-id>/notes.md`, on the same terms: code owns the path,
      the model supplies only a length-bounded string.
    - Mitigated: both are inert data (never executed), the injected blocks are
      size-capped and framed as untrusted, and both stores are operator-auditable
      (`agent6 memory list --all`, `agent6 memory invalidate` keeps the trail;
      notes are one readable markdown file). Neither is ever mounted into the
      jail, so a jailed command cannot reach or rewrite them.
    - Notes are whole-file replace, so a hostile write can erase earlier notes.
      That is the same authority the agent already has over its own context and
      costs nothing outside it; memories, which carry the audit trail, stay
      append-only precisely because they must survive this.
    - Neither weakens a boundary here: sandbox/egress/git policy come from config,
      not prompt content.

### 5. Git invariants

- **agent6's own git refuses the destructive ops, by construction.**
    - `git_ops.py` is the only module through which agent6 invokes git; it wraps
      the safe ops (status, add, commit, diff, branch, checkout) and refuses
      `push`, `reset --hard`, `commit --amend`, `rebase`,
      `filter-branch`/`filter-repo`, `branch -D`/`--force`, and any `--force`/`-f`
      on a destructive verb.
- **A `git` the model runs via `run_command` is bounded by the sandbox, not this
  list, and its argv is NOT screened.**
    - `protect_git` (default on) keeps `.git` unwritable under `strict`, which
      re-binds it read-only, recursively: a mount nested under it (e.g.
      `.git/objects` on its own bind) stays visible and read-only rather than
      shadowed. A rewrite fails and `push` has no egress. It is
      STRICT-ONLY: see below. On `hardened` the default degrades with a warning
      and an explicitly-set `true` refuses to run.
    - **The protected scope is the project's own `.git`**: agent6's operational
      state, the repository it commits to each turn. A nested `.git` (a vendored
      repo's, a submodule's) is content, like any other file in the workspace --
      tracked by the root repo or untracked, with no guarantee offered either
      way. Naming it would close nothing: a planted git config is one
      host-execution vector among many in repo content (`.envrc`, a `Makefile`,
      `conftest.py`).
    - agent6 used to refuse mutating git subcommands (plus the `-c alias.*`
      injection that dodged them) in `run_command` argv. Removed: a blocklist
      enumerates badness, and a model that writes a shell script and runs it
      walks past it, so it bought complexity and a false sense of a boundary
      the jail already owns.
- **git_ops neutralizes repo-controlled host code in a poisoned `.git/config`.**
    - `core.fsmonitor` and `diff.external` are always off; `.git/hooks/*` run only
      under `git.run_repo_hooks = true` (default false; `core.hooksPath` points
      away so a hook can't fire on agent6's auto-commit).
    - Defense in depth on top of `protect_git`: those settings bound what a
      poisoned `.git/config` could do, and `protect_git` stops the model
      writing one in the first place.
    - **`protect_git` is strict-only, and hardened leaves `.git` writable.**
      The threat is real there: a jailed command can plant a
      `filter.<n>.clean` plus a `.gitattributes`, and agent6's own auto-commit
      -- `git add -A` on the HOST, outside the jail -- then runs it, reaching
      `$HOME` and the network.
      Hardened has no mount namespace, so the only tool is Landlock, and
      Landlock cannot express this. Two of its properties close the door
      together: a grant on a directory is RECURSIVE (no "this directory
      only"), and stacked rulesets INTERSECT (an access needs every layer to
      allow it). To deny `.git` some layer must not grant it, which by
      recursion means that layer cannot grant the workspace root either --
      so the root becomes unwritable overall and nothing new can be created
      in it. Measured, both shapes: granting the root allows `.git` too;
      granting only its children denies both. A second layer cannot subtract
      what a first one granted.
      agent6 shipped the children-only carve for a while. Its cost was that
      no NEW top-level entry could be created -- `touch`, `mkdir`, `mkfifo`
      all failed at the workspace root, and coreutils reported "File exists",
      sending operators looking in the wrong place. That is too much to pay
      for a protection the operator can have properly by using `strict`.
      The in-process edit tools (`apply_edit`/`apply_patch`) refuse, on both
      isolation levels, a write into the project's own `.git`, raw or
      symlink-resolved. That guard covers only in-process edits, not jailed
      commands.
- **The edit tools refuse writes into an in-repo venv or `site-packages`.**
    - A `pyvenv.cfg` dir or `site-packages` ancestor: a run rewriting an
      editable-install `.pth` would silently corrupt the venv, invisible in `runs
      diff`/merge since venvs are gitignored. Reads stay allowed.
    - Related limit: an editable install records the host path in its `.pth`,
      absent under the jail's `/workspace`, so a `verify_command` importing the
      project can `ModuleNotFoundError`. Fix with pytest `pythonpath`, a
      `conftest.py`, or a non-editable install.

### 5b. Secrets, `connect`, root

- **Provider keys are `0600`, owner-only, and never leave agent6's process.**
    - In `$XDG_CONFIG_HOME/agent6/secrets.toml` (refused if group/other-readable
      or foreign-owned, like an SSH key), or from `[providers.<name>].api_key_env`
      (env wins). Never in transcripts, never in `config show` (redacted), never
      mounted into the jail.
- **`agent6 connect` never executes remote input.**
    - It only prompts locally (`getpass`) and writes config/secrets. It makes one
      read-only `GET` to the provider's key endpoint to confirm auth (status only;
      `--no-verify` to skip).
    - During a run agent6 opens no listening socket (an MCP server is spawned on
      stdio or DIALLED at an operator-set `url` -- outbound either way, never a
      listener; the web UI is
      private unix socket); the only accept-side socket is opt-in `agent6 web` (§7).
- **Running as root is refused without an explicit opt-in.**
    - `--allow-root` / `AGENT6_ALLOW_ROOT=1` (+ a banner). Under `sudo`, agent6
      reads the *real* user's config/secrets (from `SUDO_UID`/`SUDO_USER`), not
      root's, and chowns state-dir writes back. It doesn't drop privileges
      in-process: the jail, not the uid, is the boundary.

### 6. Curator + state location

- **An in-process `GraphCurator` owns the task graph.**
    - It validates every mutation against a pydantic schema before writing, and
      holds a per-mutation flock on the session dir. A write-path fault after the
      in-memory update reloads from disk before surfacing, so a later read never
      observes a node that was never persisted.
- **The session directory is safe because of its location, not any single writer.**
    - Per-repo state lives at `$XDG_STATE_HOME/agent6/<repo-id>/` (override with
      `[agent6].state_dir`), outside the cwd jailed commands run on.
- **The config write lock is a concurrency optimization, not a boundary.**
    - Publishes are atomic, so a torn config is impossible with or without it;
      the lock only serializes read-modify-write cycles. It FAILS OPEN by
      design (a planted symlink is refused `O_NOFOLLOW`; a stale root-owned
      lock is ignored), so it is never a way to block or redirect a write.
      Without the lock a rollback could erase a concurrent writer's update, so
      the write is kept and the error says "kept as written" (docs/config.md).

### 6b. Parallel lanes (fan-out / coordinator dispatch)

`agent6 run --parallel`, `agent6 sessions compare`, and a live run's `/parallel`
steer directive (§ [architecture.md](architecture.md#parallel-runs-fan-out-and-coordinator-dispatch))
each spawn subordinate work. Nothing here loosens the sandbox:

- **Every lane is an ordinary run.** A lane is a plain detached `agent6 run` on
  its own clone: its own jail per `sandbox.isolation`, its own `run_commands`
  policy. Nothing shares a sandbox
  socket across lanes or with the parent run.
- **Recursion is blocked by an env guard, not policy.** Every spawned lane
  carries `AGENT6_SUBRUN=1`; both the `--parallel` flag and the coordinator's
  `lane_spawner` wiring refuse when it is set, so a lane can never itself fan
  out or dispatch (depth 1 by construction).
- **A lane's config carries key references, never secret values.** The
  orchestrator writes each lane a `--config` file via `materialize()`, a dump
  of the resolved `Config` model (provider `base_url`, `api_key_env` names,
  etc.) -- `Config` never holds a raw API key. The lane's own process reads
  the same `secrets.toml` / provider env var as any other run, same user, same
  host.
- **No new subprocess call site.** `workflows/subrun.py`, `app/parallel.py`,
  and `ui/cli/parallel.py` add no direct `subprocess` use; lane git plumbing
  (clone/fetch/merge) goes through `git_ops.py` and lane spawning goes through
  `ui/spawn.py`, both already on the §2b allowlist. The
  `tests/security/test_subprocess_allowlist.py` pin needed no new entry.
- **Dirty-tree refusal, not auto-stash.** A lane clones committed HEAD only,
  so `--parallel` refuses a dirty origin under `git.require_clean_worktree`
  (the same policy and message shape `agent6 run` uses) rather than carrying
  uncommitted work into a lane it cannot see.

### 7. No agent-owned network surface (except opt-in `agent6 web`)

- **The loop opens no accept-side socket.**
    - Only outbound HTTPS to the provider; the task graph is an in-process
      curator, no socket.
- **`agent6 web` is the one accept-side socket, and only when you start it.**
    - Loopback (`127.0.0.1`) by default, no app auth (run behind `tailscale
      serve`; the tailnet identity is the access control, see [the web UI](web.md)).
    - A non-loopback bind is refused unless opted in: `[web].host` needs
      `[web].allow_non_loopback = true`, `--host` needs `--allow-non-loopback`.
- **The server renders folded state and drives typed contracts only; it executes
  nothing.**
    - New-work spawns fixed argv with the task behind `--`; machine-run is
      allow-listed to authored files; answers write only the addressed run's
      answer files (session id, answer id, machine target state dir each validated to
      one path component); merge/prune/config-set are fixed agent6 subcommands.
- **State-changing POSTs carry a CSRF guard.**
    - Body must be `Content-Type: application/json` (a cross-site `fetch` with it
      triggers a preflight the server never answers) and any `Origin` must match
      `Host`. Holds on loopback and behind `tailscale serve`.
    - It does NOT cover DNS rebinding (that needs a Host allow-list incompatible
      with the tailnet name).
- **Request framing is bounded.** 1 MiB body cap (413), chunked refused (411),
  any unread-body refusal closes the connection.
- **The machine write surface (`POST /api/machine/<name>/{poke,answer,approve,steer}`)
  uses the same guards.** `poke` writes only the instance signal file (inert JSON
  the next `tool` reads); the others write only the current agent state's
  per-state dir. PWA assets are static; the service worker is a no-op passthrough
  (no Web Push/VAPID).
- No telemetry, no auto-update, no remote control plane.

### 8. State-machine egress + script bundles

- **`machine run` is a supervisor that makes no network calls.**
    - Each `tool` state is jailed, so a per-tool `allow_network` sets its netns
      independently.
    - This lets a machine keep agents on the provider API while one reviewed,
      fixed-argv `tool` reaches the network: unlike `run_command` (LLM-chosen
      argv), a `tool` isn't a free exfil channel.

Egress = `tool_network` × per-tool `allow_network`; the
effective isolation level decides what's enforceable. "offline" = no egress.

**Agent egress is unconfined at every isolation level** (claim 3). Only jailed
commands have a network boundary.

**Jailed-command egress** (`run_command`, machine `tool`) by `tool_network`
(cells = `strict`, where a per-child netns makes "offline" real):

| jailed command | `auto` *(def)* | `block` | `only_explicit_states` | `allow` |
|---|---|---|---|---|
| `run_command` | offline | offline | offline | host network |
| `tool`, `allow_network` `auto`(def)/`block` | offline | offline | offline | offline |
| `tool`, `allow_network = allow` | ⛔ refuse | ⛔ refuse | host network | host network |

`auto` is the secure default that runs everywhere (see AGENTS.md "Secure by
default, degrade or refuse"): on `strict` it is `offline` above; on `hardened`
(no netns) it cannot be offline, so a jailed child inherits the agent process's
network and a once-per-run warning says so. `block` is
the ENFORCE form — it refuses on `hardened` rather than run under-confined. (On
`none` nothing is enforced or refused: it is the explicit unsandboxed opt-out
with its own loud warning, below.)

**Refusals** (fail-closed):

| Configuration | When |
|---|---|
| a `tool` sets `allow_network = allow` under `tool_network` `auto`/`block` | machine start |
| `tool_network = only_explicit_states`, or explicit `tool_network = block` | run start, `hardened` ¹ |
| a machine with `tool` states, or a `tool` with `allow_network = block`, under `tool_network` `auto`/`block` | machine start, `hardened` ¹ |

- ⚠ `none` (non-Linux, or explicit opt-out) is unsandboxed: nothing enforced,
  nothing refused, loud warning.

- ¹ per-command isolation needs a netns, so it's `strict`-only. On `hardened` a
  jailed child inherits the agent's Landlock (host-agnostic, per provider port;
  UDP unconfined). The secure default `auto` degrades there with a warning; an
  EXPLICIT enforce (`block`, `only_explicit_states`)
  refuses rather than run silently under-confined.

More fail-closed properties:

- **Operator-gated policy.** `tool_network` is read only from the
  operator's config; a machine's `[config]` overlay is rejected at load if it
  declares `[providers.*]`, `[sandbox.*]`, `[presets.*]`, or `git.run_repo_hooks`.
    - Otherwise a strategy preset or a host `[machine.notify]` argv could splice
      into the resolved config, and `run_repo_hooks` would run repo `.git/hooks`
      on the host on a `mode="run"` commit. A `tool` only *declares*
      `allow_network`; honoring `allow` is the operator's call, and every conflict
      is refused at startup naming the state.
- **Bundle confinement.** Scripts live in a reviewed `scripts/` beside the
  `.asm.toml`; `machine check` verifies every entry and static reference resolves
  inside the bundle (escaping symlinks rejected).
    - Scripts are operator-authored and committed, never fetched/generated at run
      time, and the `.asm.toml` + `scripts/` are RO in every jail during a run, so
      a state can't rewrite its own logic or add an `allow_network` flag.
- **Notifications don't widen the agent's surface.** Front-ends render
  `machine.notify` as an overlay, and `attach`/TUI call `notify-send` with a FIXED
  argv (no shell), so a model message is inert data.
    - The out-of-band hook `[machine.notify].on_event` runs an operator argv on the
      host with a minimal env -- PATH/HOME/locale/desktop-bus plus
      `AGENT6_MACHINE_*` (`hook_env` in `app/finalize.py`), never the full
      environment with its provider keys (mirrors `[notify].on_complete`); a
      `[config]` overlay setting `[machine.notify]` is rejected at load. No Web
      Push/VAPID.

## Skills trust model

- **A skill is config, not repo content: install only from trusted sources.**
    - `agent6 skills install <url>` is an operator-initiated CLI fetch (same trust
      class as `connect`); what it installs enters the system prompt/tool results
      verbatim.
- **Nothing in a skill runs at install or load.**
    - Its scripts run only if the model runs them through the jailed command path,
      subject to `run_commands`.
- **`use_skill` is read-only and path-contained.** Serves the skill's own dir only
  (symlinks/`..` resolved first), never the repo or network. Skill dirs aren't
  mounted into the jail; content reaches the model engine-side.
- **Repo-local `.claude/skills/` are deliberately NOT discovered.** Third-party
  repo content must not enter the prompt; only the installed dir +
  `[skills].extra_dirs` are scanned.

## Prompt-injection resilience

[`tests/security/test_prompt_injection.py`](https://github.com/agent6-dev/agent6/blob/master/tests/security/test_prompt_injection.py)
runs an adversarial corpus through the planner/worker/reviewer prompts and
asserts no exfiltration, no out-of-policy tool calls, and no following embedded
instructions to weaken constraints. It's a smoke test, not a proof: the
structural defenses above are the real mitigation; the corpus catches prompt
regressions.

## Known limitations

- **Landlock TCP rules need Linux ≥ 6.7** (ABI ≥ 4); older kernels leave the agent
  process net-unconfined (children stay net-isolated in `strict` via the empty netns).
- **User namespaces must be enabled;** some distros disable them, and agent6
  refuses `strict` there.
- **AppArmor userns (Ubuntu 24.04+)** blocks unprivileged userns without a profile.
    - agent6 ships one scoped to the launcher (`agent6 system apparmor install`);
      with it, per-command jailing is `strict`, without it `hardened`.

- **seccomp is required;** kernels that block it from unprivileged callers make the
  jail fail closed.
- **Devcontainers get `hardened`;** the container is the FS blast radius, network
  still Landlock-confined when supported. The XDG state base is ephemeral (lost on
  rebuild), so mount a volume at the state dir or set `[agent6].state_dir` to
  persist runs.
- **Side channels:** no claim about timing/cache/speculative side channels; don't
  co-locate agent6 with secrets if Spectre-class attacks are in your model.
- **Supply chain:** pin your install. Runtime deps `pydantic`, `httpx2`,
  `argcomplete`, the `tree-sitter` pair, `textual`, `ruff`, `ty`; build-dep
  `hatchling`; the jail's Rust crates `nix`, `libc`, `landlock`, `seccompiler`,
  `serde`, `serde_json`.
