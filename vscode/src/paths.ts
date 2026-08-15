// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Eric Lesiuta
//
// Where agent6 keeps run state: asked of the installed CLI, never mirrored.
//
// Runs live out of the workspace, under the per-repo state dir. The CLI owns
// the id and base-dir algorithms (`agent6 sessions dir` prints the resolved
// state dir as one bare line, honoring AGENT6_STATE_HOME and the global
// `[agent6].state_dir`); a second implementation here drifted once and made
// every run invisible. No `vscode` import, so the module also loads under
// plain node for sanity checks.

import { execFileSync } from "child_process";
import * as path from "path";

/**
 * The per-repo state dir for a workspace root, from `agent6 sessions dir`
 * run at that root. Throws with the CLI's own words when agent6 is not
 * installed or refuses; the command surfaces that message.
 */
export function stateDirFor(workspaceRoot: string): string {
  let out: string;
  try {
    out = execFileSync("agent6", ["sessions", "dir"], {
      cwd: workspaceRoot,
      encoding: "utf-8",
      timeout: 10_000,
    });
  } catch (err) {
    throw new Error(
      `running 'agent6 sessions dir' failed (is agent6 on PATH?): ${String(err)}`,
    );
  }
  const dir = out.trim();
  if (dir.length === 0 || !path.isAbsolute(dir)) {
    throw new Error(`'agent6 sessions dir' printed no usable path: ${JSON.stringify(out)}`);
  }
  return dir;
}

/** The runs bucket for a workspace root: `<state dir>/sessions/runs`. */
export function runsDirFor(workspaceRoot: string): string {
  return path.join(stateDirFor(workspaceRoot), "sessions", "runs");
}
