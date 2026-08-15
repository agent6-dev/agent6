# agent6: tail run

Minimal VS Code extension that follows an agent6 run's structured event log
(`<state base>/<repo-id>/sessions/runs/<id>/logs.jsonl`, out of the workspace) in
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
dir. The extension asks the installed CLI (`agent6 sessions dir`, run at the
first workspace folder) for the resolved per-repo state dir, so it always
finds exactly the runs the CLI writes -- including under `AGENT6_STATE_HOME`
and the global `[agent6].state_dir` override. Runs are
`<state dir>/sessions/runs/<id>/logs.jsonl`. agent6 keys state off the
directory it is invoked in, so start runs from the workspace root; `agent6`
must be on the PATH VS Code inherits.

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
