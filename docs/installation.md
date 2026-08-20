# Installation

## From PyPI

With [uv](https://docs.astral.sh/uv/getting-started/installation/) or [pipx](https://pipx.pypa.io/stable/how-to/install-pipx/):

=== "uv"

    ```sh
    uv tool install agent6
    ```

=== "pipx"

    ```sh
    pipx install agent6
    ```

Both put the `agent6` entry point in `~/.local/bin`.
If that directory is not on your `PATH`:

=== "uv"

    ```sh
    uv tool update-shell
    ```

=== "pipx"

    ```sh
    pipx ensurepath
    ```

Restart the shell afterwards.

## Shell completion

One command installs tab-completion.
It detects the shell you are running, or takes the name of one.
Rerunning it is safe and refreshes the completion.

```sh
agent6 completions          # or: agent6 completions bash|zsh|fish|xonsh
```

Bash and zsh get a marker-guarded source line in their rc file, pointing at a script under the agent6 config dir.
Fish and xonsh get a file in their auto-loaded native locations (`fish/completions`, `xonsh/rc.d`), with no rc edit.
`agent6 completions --print bash` emits the script instead, for `eval` or a dotfiles repo.

## Check the install

```sh
agent6 --version
agent6 check                # sandbox, config, keys, MCP, verify, boundaries
```

`agent6 check sandbox` runs the jail through live probes and reports the isolation level a run will use on your kernel.

## Requirements

- Python 3.12 or newer
- One provider: Anthropic, any OpenAI-compatible endpoint (a local one needs no key), or a ChatGPT subscription
- Linux on x86_64 or aarch64 for the sandbox
- Unprivileged user namespaces for `strict` isolation
- A Rust toolchain to build from source (the PyPI wheels bundle `agent6-jail`)

The jail uses Landlock, seccomp, and user namespaces, and its seccomp filter exists for x86_64 and aarch64.

- On other architectures and on macOS, `isolation = "auto"` resolves to `none`: commands run as ordinary subprocesses behind a startup warning, and an explicit `strict` or `hardened` refuses.
- On Windows, use WSL.
- Unprivileged user namespaces are on by default on Ubuntu, Debian, and most cloud images.
  Ubuntu 24.04+ blocks them with `kernel.apparmor_restrict_unprivileged_userns = 1`: set that sysctl to 0, or install the bundled AppArmor profile with `agent6 system apparmor install` (`agent6 system apparmor remove` reverts it).
- Without user namespaces agent6 falls back to `hardened` isolation, which keeps Landlock, seccomp, and `NO_NEW_PRIVS`.

The [security model](security.md) describes what each isolation level enforces.

## From source

```sh
git clone https://github.com/agent6-dev/agent6
cd agent6
uv sync
uv run agent6 --help
```

`AGENT6_JAIL_BIN=/path/to/agent6-jail` overrides the bundled jail binary.
