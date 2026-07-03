// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Eric Lesiuta
//
// Where agent6 keeps run state, mirrored from src/agent6/paths.py.
//
// Runs live out of the workspace, under the per-repo state dir
// `<state base>/<repo-id>/runs/<run-id>/`. Keep this in lockstep with
// paths.state_base and paths.repo_id: the extension is a viewer and must
// find exactly the runs the CLI writes. No `vscode` import, so the module
// also loads under plain node for sanity checks.

import * as crypto from "crypto";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";

/** Leading-tilde expansion like Python's Path.expanduser(); `~user` is left as-is. */
function expandUser(p: string): string {
  if (p === "~") {
    return os.homedir();
  }
  if (p.startsWith("~/")) {
    return path.join(os.homedir(), p.slice(2));
  }
  return p;
}

/**
 * The agent6 state BASE directory, mirroring paths.state_base:
 * $AGENT6_STATE_HOME (names the base itself) > $XDG_STATE_HOME/agent6 >
 * ~/.local/state/agent6. A global `[agent6].state_dir` config override is
 * not read here; set AGENT6_STATE_HOME to the same base if you use one.
 */
export function stateBase(): string {
  const override = process.env.AGENT6_STATE_HOME;
  if (override) {
    return expandUser(override);
  }
  const xdg = process.env.XDG_STATE_HOME;
  if (xdg) {
    return path.join(xdg, "agent6");
  }
  return path.join(os.homedir(), ".local", "state", "agent6");
}

/**
 * Stable per-repo id, mirroring paths.repo_id:
 * `<folder>-<first 12 hex of sha256(canonical path)>`. Python canonicalizes
 * with Path.resolve() (symlinks resolved); realpathSync matches it for an
 * existing directory, path.resolve is the fallback when it does not exist.
 */
export function repoId(repoRoot: string): string {
  let real: string;
  try {
    real = fs.realpathSync(repoRoot);
  } catch {
    real = path.resolve(repoRoot);
  }
  const digest = crypto.createHash("sha256").update(real, "utf-8").digest("hex");
  return `${path.basename(real)}-${digest.slice(0, 12)}`;
}

/** The runs/ directory for a workspace root: `<state base>/<repo-id>/runs`. */
export function runsDirFor(workspaceRoot: string): string {
  return path.join(stateBase(), repoId(workspaceRoot), "runs");
}
