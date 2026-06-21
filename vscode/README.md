# agent6: tail run

Minimal VS Code extension that follows an agent6 run's structured event log
(`<state-dir>/<repo-id>/runs/<id>/logs.jsonl`, out of the workspace) in a VS
Code output channel.

It is intentionally tiny:

- One command: `agent6: Tail a run`
- Pick a run from the most-recent-first list of run directories.
- The extension polls the JSONL file every 500ms and appends new events to
  the `agent6` output channel.
- Read-only. No tree view, no status bar, no settings panel.

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
