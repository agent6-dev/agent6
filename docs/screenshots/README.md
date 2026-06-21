# Screenshots

[vhs](https://github.com/charmbracelet/vhs) tape scripts that drive the agent6
TUI and capture screenshots — for the README/release notes, and as a way to
**visually exercise the TUI** (navigate the hub, the config page, the menu bar,
the help page) without a human at the keyboard.

We commit the **scripts only**. Generated media (`out/`, `*.gif`, `*.png`) is
gitignored.

## Run

```sh
# install vhs (brew install vhs, or see the vhs README), then:
AGENT6_DEMO_REPO=/path/to/a/git/repo vhs docs/screenshots/tour.tape
# -> docs/screenshots/out/01-hub.png, 02-config.png, …
```

`tour.tape` walks: the hub → the config page → a menu dropdown → settings
search → the keys/help page. Each `Screenshot` line writes one PNG. Tweak the
`Set FontSize/Width/Height/Theme` lines for the look you want.

If a keyboard step doesn't fire on your terminal (`Alt+<letter>` / function keys
are terminal-dependent), substitute the command palette (`Ctrl+P`) or a mouse
step — see the comments in the tape.

## Per-release screenshots (the GitHub Action)

The intended setup: a workflow that, on release (or on demand), installs vhs +
`agent6`, runs the tapes against a small demo repo, and uploads the PNGs as
release assets / refreshes the README images — so the docs always match the
current UI. The action lives in `.github/workflows/`; only the tapes live here.
A sketch:

```yaml
# .github/workflows/screenshots.yml (sketch)
- uses: charmbracelet/vhs-action@v2
  with: { path: docs/screenshots/tour.tape }
- uses: actions/upload-artifact@v4
  with: { name: tui-screenshots, path: docs/screenshots/out/ }
```
