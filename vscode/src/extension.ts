// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Eric Lesiuta
//
// agent6: tail run
//
// Minimal VS Code extension that tails an agent6 run's structured log
// (`.agent6/runs/<id>/logs.jsonl`) in an output channel. Read-only.
//
// This is intentionally tiny: pick a run dir, follow its logs.jsonl by
// re-reading appended bytes, pretty-print each event. No tree view, no
// status bar, no settings panel.

import * as fs from "fs";
import * as path from "path";
import * as vscode from "vscode";

const CHANNEL_NAME = "agent6";

export function activate(context: vscode.ExtensionContext): void {
  const channel = vscode.window.createOutputChannel(CHANNEL_NAME);
  context.subscriptions.push(channel);

  const tailDisposable = vscode.commands.registerCommand(
    "agent6.tailRun",
    async () => {
      const folders = vscode.workspace.workspaceFolders;
      if (folders === undefined || folders.length === 0) {
        vscode.window.showErrorMessage("agent6: open a workspace first.");
        return;
      }
      const root = folders[0].uri.fsPath;
      const runsDir = path.join(root, ".agent6", "runs");
      if (!fs.existsSync(runsDir)) {
        vscode.window.showErrorMessage(
          `agent6: no runs dir at ${runsDir}. Run 'agent6 run ...' first.`,
        );
        return;
      }
      const runs = fs
        .readdirSync(runsDir, { withFileTypes: true })
        .filter((d) => d.isDirectory())
        .map((d) => d.name)
        .sort()
        .reverse();
      if (runs.length === 0) {
        vscode.window.showErrorMessage("agent6: no runs to tail.");
        return;
      }
      const picked = await vscode.window.showQuickPick(runs, {
        title: "agent6: pick a run to tail (newest first)",
      });
      if (picked === undefined) {
        return;
      }
      const logsPath = path.join(runsDir, picked, "logs.jsonl");
      channel.clear();
      channel.show(true);
      channel.appendLine(`[agent6] tailing ${logsPath}`);
      const tail = new JsonlTail(logsPath, (line) => channel.appendLine(line));
      tail.start();
      context.subscriptions.push({ dispose: () => tail.stop() });
    },
  );
  context.subscriptions.push(tailDisposable);
}

export function deactivate(): void {
  /* nothing */
}

class JsonlTail {
  private offset = 0;
  private timer: NodeJS.Timeout | undefined;
  private leftover = "";

  constructor(
    private readonly file: string,
    private readonly onLine: (formatted: string) => void,
  ) {}

  start(): void {
    this.tick();
    this.timer = setInterval(() => this.tick(), 500);
  }

  stop(): void {
    if (this.timer !== undefined) {
      clearInterval(this.timer);
      this.timer = undefined;
    }
  }

  private tick(): void {
    let stat: fs.Stats;
    try {
      stat = fs.statSync(this.file);
    } catch {
      return; // file not yet created
    }
    if (stat.size < this.offset) {
      // File rotated / truncated.
      this.offset = 0;
      this.leftover = "";
    }
    if (stat.size === this.offset) {
      return;
    }
    const fd = fs.openSync(this.file, "r");
    try {
      const length = stat.size - this.offset;
      const buf = Buffer.alloc(length);
      fs.readSync(fd, buf, 0, length, this.offset);
      this.offset = stat.size;
      const chunk = this.leftover + buf.toString("utf-8");
      const lines = chunk.split("\n");
      this.leftover = lines.pop() ?? "";
      for (const raw of lines) {
        if (raw.trim().length === 0) {
          continue;
        }
        try {
          const ev = JSON.parse(raw) as Record<string, unknown>;
          this.onLine(formatEvent(ev));
        } catch {
          this.onLine(raw);
        }
      }
    } finally {
      fs.closeSync(fd);
    }
  }
}

function formatEvent(ev: Record<string, unknown>): string {
  const ts = typeof ev.ts === "string" ? ev.ts : "";
  const type = typeof ev.type === "string" ? ev.type : "?";
  const rest: Record<string, unknown> = { ...ev };
  delete rest.ts;
  delete rest.type;
  const detail = Object.keys(rest).length === 0 ? "" : ` ${JSON.stringify(rest)}`;
  return `${ts} ${type}${detail}`;
}
