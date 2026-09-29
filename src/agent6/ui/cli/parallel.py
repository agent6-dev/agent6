# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The CLI adapter for `agent6 run --parallel` and the coordinator `/parallel` dispatch.

The pipeline is headless in `agent6.app.parallel`; this module supplies the `LaneRuntime`
it drives (the detached spawn from `ui.spawn`, the reviewer provider and judging spinner
from `_compare`) and the CLI-side preflight and refusals. `run.py` routes here.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable, Sequence

from agent6 import directive, git_ops, paths
from agent6.app import parallel, preflight
from agent6.config import Config, ConfigError
from agent6.models import validate
from agent6.sessions import id
from agent6.ui import spawn as ui_spawn
from agent6.ui.cli import _common, _compare, _interact


def lane_runtime() -> parallel.LaneRuntime:
    """Return the front-end primitives the parallel pipeline drives.

    Injected so `agent6.app` never imports `agent6.ui`. Lane liveness and stop are the
    run-dir bridge, which `agent6.app.parallel` imports directly.
    """

    def spawn(
        argv: list[str],
        cwd: pathlib.Path,
        *,
        before: set[pathlib.Path],
        list_dirs: Callable[[], list[pathlib.Path]],
        env: dict[str, str],
    ) -> tuple[pathlib.Path | None, str]:
        """Spawn a detached agent6 process and locate its session dir.

        Returns:
            The session dir and an error text, from `spawn_and_locate`.
        """
        return ui_spawn.spawn_and_locate(
            [ui_spawn.agent6_exe(), *argv], cwd, before=before, list_dirs=list_dirs, env=env
        )

    return parallel.LaneRuntime(
        spawn=spawn,
        build_provider=_compare._reviewer_provider,
        judging_status=_compare._judging_status,
    )


def _parallel_approval_refusal(cfg: Config) -> str | None:
    """Return why `--parallel` refuses under `ask`, naming the two coherent choices.

    Lanes run detached at the same time, so waiting for an approver would mean attaching
    to each lane in turn; the decision is made once, at launch.

    Returns:
        The refusal, or None when commands are not on `ask`.
    """
    if cfg.sandbox.run_commands != "ask":
        return None
    return (
        "sandbox.run_commands = 'ask' cannot drive parallel lanes: each lane runs"
        " detached, with nobody to answer its prompts.\n"
        "  --auto-approve   approve every command in every lane\n"
        "  --no-commands    withhold commands from every lane\n"
        "  (a hub has no flags: agent6 config set sandbox.run_commands yes|no, --repo"
        " for this checkout)"
    )


def dispatch_parallel(
    cfg: Config,
    task: str,
    spec: str,
    *,
    cwd: pathlib.Path,
    max_usd: float | None = None,
    auto_approve: bool = False,
    pins: Sequence[str] = (),
) -> int:
    """Preflight and route `agent6 run --parallel`.

    Refuses an unenforceable `--max-usd` or a dirty origin (lanes clone committed HEAD
    only), plans the lanes, then hands off to the headless `run_parallel`.

    Args:
        cfg: The effective config.
        task: The task text.
        spec: The `--parallel` lane spec.
        cwd: The origin repo.
        max_usd: The spend ceiling forwarded to every lane.
        auto_approve: `--auto-approve`, forwarded to every lane.
        pins: The pinned session ids.

    Returns:
        The exit code; 2 on a refusal.
    """
    origin = cwd
    origin_state = paths.state_dir(origin)
    for err in (preflight.budget_preflight(cfg), _parallel_approval_refusal(cfg)):
        if err is not None:
            _common.refuse(f"{err}")
            return 2
    try:
        modified = git_ops.modified_paths(origin)
    except git_ops.GitError as exc:
        _common.error(f"{exc}")
        return 2
    if modified and cfg.git.dirty_tree == "ask":
        listed = "\n".join(f"    {p}" for p in modified[:10])
        more = f"\n    ... {len(modified) - 10} more" if len(modified) > 10 else ""
        n = len(modified)
        _common.refuse(
            f"{n} tracked {'file has' if n == 1 else 'files have'} uncommitted"
            f" changes:\n{listed}{more}\n"
            "Lanes clone committed HEAD, so those changes would not reach them. Commit or"
            ' stash them first, or set [git].dirty_tree to "stash" or "include" to fan out'
            " without them."
        )
        return 2

    fanout_id = id.friendly_token()
    try:
        lanes = parallel.build_lane_specs(spec, cfg=cfg, origin=origin, fanout_id=fanout_id)
    except (ConfigError, directive.DirectiveError, parallel.ParallelError) as exc:
        _common.refuse(f"{exc}")
        return 2
    # Before any clone or spawn: a typo refuses when a cache can check it, else warn and proceed.
    verdict = validate.validate_spec_models([ln.route for ln in lanes], cfg)
    if verdict.refused:
        _common.refuse(f"{validate.refusal_message(verdict, directive=False)}")
        return 2
    if verdict.warned:
        _common.warn(f"{validate.warning_message(verdict)}")
    lane_away = _interact.lane_away_mode()
    if lane_away == "deny":
        _common.warn(
            "no terminal to attach from: a lane's questions get empty answers and its"
            " fetch and MCP approvals are denied"
        )
    return parallel.run_parallel(
        task,
        lanes,
        cfg=cfg,
        origin=origin,
        origin_state=origin_state,
        runtime=lane_runtime(),
        max_usd=max_usd,
        fanout_id=fanout_id,
        auto_approve=auto_approve,
        lane_away=lane_away,
        pins=pins,
        spec=spec,
    )
