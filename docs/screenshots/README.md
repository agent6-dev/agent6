# Screenshots

The TUI screenshots and tour video for the docs site (agent6.dev) are generated
from **real, recorded runs** with [vhs](https://github.com/charmbracelet/vhs) —
no live LLM calls, no network, no API key. Small, sanitized run fixtures are
committed under `seed/`; vhs replays them through the TUI and captures the images.

We commit the **tape, scripts, and seed fixtures only**. Generated media
(`out/`, `*.png`, `*.webm`) is gitignored and rebuilt on demand.

## Files

- `tour.tape` — vhs script for the PNGs: drives hub -> config -> search -> help ->
  dashboard -> transcript -> logs and captures one PNG per stop.
- `reel.tape` — vhs script for the `tour.webm` video: one TUI session (hub, into a
  run, transcript, log, config, search, help) with NO Screenshots (those stall the
  video clock). Holds are paced by how dense each screen is.
- `keystroke_overlay.py` — overlays animated keystroke toasts on the recording,
  parsing the tape and scaling the keypress timeline to the actual duration.
  Vendored verbatim from the author's picosnitch (relicensed Apache).
- `seed/` — committed, sanitized run fixtures (trimmed `logs.jsonl` + a few
  transcripts). The TUI renders the hub and dashboard entirely from these.
- `seed.py` — installs `seed/` into an isolated `$AGENT6_STATE_HOME` under the
  demo repo's id and writes a demo `config.toml` + `ui.toml` (theme `agent6-dark`).
- `generate.sh` — the orchestrator: seeds a temp home, runs `tour.tape` for the
  PNGs (1920x1080), then records `reel.tape` and overlays keystroke toasts to make
  `tour.webm`. `AGENT6_SCREENSHOTS=placeholder` writes flat placeholders for a
  fast, tool-free docs-only build.
- `build_fixtures.py` — dev tool, NOT run in CI: rebuilds `seed/` from real runs
  under `$XDG_STATE_HOME/agent6/`, trimming token-delta bloat and scrubbing paths.

The reel records at 1280x720 / Framerate 8. The live dashboard redraws ~5x/s, and at
full resolution vhs cannot capture that fast (the dashboard records several times too
fast, which desyncs the single-scale toast overlay). At 1280x720 the capture keeps up,
so the whole session records at real time (~97%) and the toasts stay matched to the
screen. The PNGs are captured separately at 1920x1080.

## Run locally

```sh
# needs vhs + ttyd + ffmpeg + agent6 on PATH
bash docs/screenshots/generate.sh
# -> docs/screenshots/out/01-hub.png, 02-run-dashboard.png, …, tour.webm
```

vhs renders through a headless Chromium; on Ubuntu set
`kernel.apparmor_restrict_unprivileged_userns=0` first if rendering aborts.

## In CI

The [`pages`](../../.github/workflows/pages.yml) workflow runs `generate.sh`
before `mkdocs build` on every release (or manual dispatch), so the published
site's images always match the current UI. The media ships inside the site
artifact — not as release assets, not committed — so it never expires and clicking
an image opens a lightbox instead of forcing a download.
