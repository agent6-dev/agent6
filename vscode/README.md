# agent6: tail run

Minimal VS Code extension that follows an agent6 run's structured event log
(`<state base>/<repo-id>/runs/<run-id>/logs.jsonl`, out of the workspace) in
a VS Code output channel.

It is intentionally tiny:

- One command: `agent6: Tail a run`
- Pick a run; the list is newest first by each run's `logs.jsonl` mtime (the
  CLI's definition of recency). Runs without a `logs.jsonl` yet sort last.
- The extension polls the JSONL file every 500ms and appends new events to
  the `agent6` output channel.
- Read-only. No tree view, no status bar, no settings panel.

## Where runs live

Run state is out of the workspace; a checkout never carries an `.agent6/`
dir. The extension mirrors the CLI's path logic (`src/agent6/paths.py`):

- State base: `$AGENT6_STATE_HOME` if set (it names the base itself), else
  `$XDG_STATE_HOME/agent6`, else `~/.local/state/agent6`.
- Repo id: `<folder>-<first 12 hex of sha256(canonical path)>`, keyed on the
  first workspace folder with symlinks resolved. agent6 keys state off the
  directory it is invoked in, so start runs from the workspace root.
- Runs: `<state base>/<repo-id>/runs/<run-id>/logs.jsonl`.

The global `[agent6].state_dir` config override is not read; set
`AGENT6_STATE_HOME` to the same base if you use one.

## Build

```bash
cd vscode
npm install
npm run compile
```

Then in VS Code: `Developer: Install Extension From Location...` and select
this folder.

## Why so small

The same JSONL stream is consumable by any tail-style tool (`tail -f`, `jq`,
`rg`, etc.). The extension exists so VS Code users don't have to leave the
editor; it deliberately doesn't try to be a dashboard. agent6 is the source
of truth for run state; this is just a viewer.
