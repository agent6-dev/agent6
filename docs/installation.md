# Installation

## Requirements

- **Linux** for the sandbox. The jail uses Landlock, seccomp, and user namespaces, which
  are Linux-only. macOS runs unsandboxed: the default `profile = "auto"` resolves to
  `none`, commands run as ordinary subprocesses behind a startup warning, and an explicit
  `strict` or `hardened` profile is refused. On Windows use WSL; the CLI does not run
  natively there.
- **Kernel 6.7 or newer** for the Landlock network rules. Older kernels fall back to
  filesystem-only Landlock with a warning.
- **Unprivileged user namespaces** for the `strict` profile. They are on by default on
  Ubuntu, Debian, and most cloud images. On Ubuntu 24.04+, where
  `kernel.apparmor_restrict_unprivileged_userns = 1` blocks them, either set that sysctl
  to 0 or install the bundled AppArmor profile with `agent6 system apparmor install`
  (removed again with `agent6 system apparmor remove`). Without user namespaces agent6
  falls back to the `hardened` profile, which is still real isolation.
- **Python 3.12 or newer**, and an API key for at least one provider.
- A **Rust toolchain** only when building from source; the PyPI wheels bundle a prebuilt
  `agent6-jail`.

The [security model](security.md) describes what each profile enforces.

## From PyPI

=== "uv"

    ```sh
    uv tool install agent6
    ```

=== "pipx"

    ```sh
    pipx install agent6
    ```

Both put the `agent6` entry point in `~/.local/bin`. If that is not on your `PATH`, run
`uv tool update-shell` or `pipx ensurepath` and restart the shell.

## From source

```sh
git clone https://github.com/agent6-dev/agent6
cd agent6
uv sync
uv run agent6 --help
```

`AGENT6_JAIL_BIN=/path/to/agent6-jail` overrides the bundled jail binary.

## Shell completion

One command installs tab-completion (it detects the shell you are running,
even one started from another shell, or pass `bash`/`zsh`/`fish`/`xonsh`
explicitly; rerunning is safe and refreshes it):

```sh
agent6 completions
```

Bash/zsh get a marker-guarded source line in their rc file pointing at a
script under the agent6 config dir; fish and xonsh get a file in their
auto-loaded native locations (`fish/completions`, `xonsh/rc.d`), no rc edit.
`agent6 completions --print bash` emits the script instead, for `eval` or a
dotfiles repo. (For bash/zsh/fish the underlying
[argcomplete](https://kislyuk.github.io/argcomplete/)
`register-python-argcomplete` also works when it is on your PATH; the xonsh
completer is agent6's own, driving the same argcomplete protocol.)

## Check the install

```sh
agent6 --version
agent6 check          # sandbox probes, config, and provider keys
```

`agent6 check sandbox` runs the jail through a set of probes and reports which profile a
run will use on your kernel.
