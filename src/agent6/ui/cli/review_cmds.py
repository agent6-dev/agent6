# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 review`: the freeform review and the adversarial panel.

`save_review` is the one writer of `<state-dir>/reviews/`, shared with `sessions review`.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
import time

from agent6 import budget as agent6_budget
from agent6 import git_ops
from agent6 import paths as agent6_paths
from agent6.app import _setup, finalize, providers
from agent6.config import (
    Config,
    ConfigError,
    layer,
    parse_seat_spec,
)
from agent6.harness import _context, _panel, _reviewer, _toolset, code_review
from agent6.providers import (
    ProviderError,
    TranscriptSink,
)
from agent6.tools import dispatch as tools_dispatch
from agent6.ui.cli import _common


def _collect_review_diff(
    git: str,
    root: pathlib.Path,
    *,
    base: str,
    head: str,
    paths: tuple[str, ...],
) -> subprocess.CompletedProcess[str]:
    """Collect the diff `agent6 review` reviews, leaving the index as it was.

    With a base, a plain read-only `git diff base..head`. Without one, the working tree
    against HEAD including untracked files: git shows those only through intent-to-add
    entries, so the currently untracked paths are registered and `git reset` afterwards, in
    a `finally`; staged and tracked changes are never touched. Every invocation carries
    git_ops' hardening flags plus `--no-ext-diff --no-textconv`, so a poisoned `.git/config`
    (`diff.external`, a textconv, `core.fsmonitor`) cannot run its payload on the host.

    Args:
        git: The git binary.
        root: The repo root.
        base: The base rev, or "" for the working tree.
        head: The head rev.
        paths: The pathspecs, or empty for everything.

    Returns:
        The completed `git diff`, decoded with replacement so a non-UTF-8 file cannot crash it.
    """
    hardening = git_ops.git_hardening_flags(root)
    untracked: list[str] = []
    if not base:
        status = subprocess.run(
            [git, *hardening, "status", "--porcelain", "-z"],
            cwd=root,
            capture_output=True,
            errors="replace",
            check=False,
        )
        untracked = [entry[3:] for entry in status.stdout.split("\0") if entry.startswith("?? ")]
        if untracked:
            subprocess.run([git, *hardening, "add", "-N", "--", *untracked], cwd=root, check=False)
    try:
        rev = f"{base}..{head}" if base else "HEAD"
        # `--end-of-options` and `--` keep a rev that is also a path a rev.
        diff_args = [
            git,
            *hardening,
            "diff",
            *git_ops.DIFF_SHOW_SAFETY_FLAGS,
            "--end-of-options",
            rev,
        ]
        diff_args.extend(["--", *paths])
        # git diff emits raw file bytes; a non-UTF-8 file must not crash the review.
        return subprocess.run(
            diff_args, cwd=root, capture_output=True, errors="replace", check=False
        )
    finally:
        if untracked:
            subprocess.run(
                [git, *hardening, "reset", "-q", "--", *untracked], cwd=root, check=False
            )


def save_review(reviews_dir: pathlib.Path, *, label: str, body: str) -> pathlib.Path:
    """Write one rendered review under the reviews dir, beside the provider transcripts.

    The file is `<utc-stamp>-review.md`: a `# review: <label>` line, then the body. A later
    session working on a module finds its review by searching the directory.

    Args:
        reviews_dir: The `<state-dir>/reviews/` directory.
        label: What was reviewed.
        body: The review text.

    Returns:
        The file's path.
    """
    agent6_paths.mkdir_for_real_user(reviews_dir)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    content = f"# review: {label}\n\n{body.rstrip()}\n".encode()
    n = 1
    while True:
        path = reviews_dir / (f"{stamp}-review.md" if n == 1 else f"{stamp}-{n}-review.md")
        # An exclusive create claims the name, so two reviews in one second never collide.
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            n += 1
            continue
        with os.fdopen(fd, "wb") as fh:
            fh.write(content)
        return path


def _is_checked_out(git: str, root: pathlib.Path, rev: str) -> bool:
    """Return whether the checkout at the root is the rev, with nothing uncommitted on top.

    An explore-tier seat's read-only tools read the checkout, so anything else answers
    `read_file` from a tree the diff does not describe.

    Args:
        git: The git binary.
        root: The repo root.
        rev: The reviewed head.
    """
    checked_out = git_ops.chain_tip(root, "HEAD")
    if checked_out is None or git_ops.chain_tip(root, rev) != checked_out:
        return False
    status = subprocess.run(
        [git, *git_ops.git_hardening_flags(root), "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        capture_output=True,
        errors="replace",
        check=False,
    )
    return status.returncode == 0 and not status.stdout.strip()


def _run_review_panel(
    cfg: Config,
    *,
    git: str,
    root: pathlib.Path,
    base: str,
    head: str,
    label: str,
    diff: str,
    agents_md: str,
    reviewers: int,
    personas: str,
    transcript_sink: TranscriptSink,
    reviews_dir: pathlib.Path,
    budget: agent6_budget.BudgetTracker,
) -> int:
    """Run the grounded adversarial review panel over the diff and print its verdict.

    Read-only. The merged findings are saved under the reviews dir; per-seat status and the
    budget go to stderr.

    Args:
        cfg: The effective config.
        git: The git binary.
        root: The repo root.
        base: The base rev.
        head: The head rev.
        label: What is reviewed, for the saved file.
        diff: The diff text.
        agents_md: The repo's AGENTS.md text.
        reviewers: The number of seats.
        personas: The persona roster.
        transcript_sink: Where the seats' transcripts go.
        reviews_dir: The `<state-dir>/reviews/` directory.
        budget: The invocation's budget.

    Returns:
        The verdict on `agent6 run`'s scale: 0 PASS (clean or non-blocking findings),
        1 INCONCLUSIVE (every seat abstained), 3 budget, 4 BLOCK (a grounded gating
        finding). 2 stays the refusal before any seat ran, so a script tells "blocked"
        from "not reviewed" from "could not start".
    """
    persona_tuple = tuple(p.strip() for p in personas.split(",") if p.strip())
    try:
        seats = providers.build_review_seats(
            cfg,
            transcript_sink=transcript_sink,
            budget=budget,
            n=reviewers,
            personas=persona_tuple,
        )
    except ProviderError as exc:
        _common.error(f"provider init failed: {exc}")
        return 2
    # check_provider_keys reads a configured roster; --personas is the roster only without one.
    pinned = [] if cfg.review.seats else [parse_seat_spec(spec)[1] for spec in persona_tuple]
    err = _setup.check_provider_keys(cfg, extra_providers=pinned)
    if err is not None:
        _common.error(f"{err}")
        return 2
    ctx = _panel.ReviewContext(task=f"code review: {label}", agents_md=agents_md, diff=diff)
    # explore-tier seats need a read-only tool surface over the repo.
    tools = None
    dispatch = None
    if any(s.tier == "explore" for s in seats):
        if base and not _is_checked_out(git, root, head):
            _common.error(
                "review.tier = 'explore' reads the checkout, but the checkout is not"
                f" --head {head!r} (another commit, or uncommitted changes on top): a"
                " seat's read_file would answer from a tree the diff does not describe."
                " Check it out clean, or set review.tier = 'diff'."
            )
            return 2
        disp = tools_dispatch.ToolDispatcher(root=root, config=cfg)
        tools, dispatch = _toolset.build_readonly_review_tools(disp)
    print(
        f"[agent6] review panel: {len(seats)} seats"
        f" ({', '.join(s.persona for s in seats)}) | decision={cfg.review.decision}"
        f" | tier={cfg.review.tier} | range={label}",
        file=sys.stderr,
    )
    try:
        result = _reviewer.run_panel(
            seats,
            ctx,
            decision=cfg.review.decision,
            quorum=cfg.review.quorum,
            panel_id="cli",
            concurrency=len(seats),  # one-shot CLI: run all seats in parallel
            tools=tools,
            dispatch=dispatch,
        )
    except agent6_budget.BudgetExceededError as exc:
        print(f"BUDGET EXCEEDED: {exc}", file=sys.stderr)
        return 3
    # One all-abstain owner, shared with the in-loop panel: nothing reviewed is never a pass.
    inconclusive = _panel.panel_is_inconclusive(result)
    if inconclusive:
        verdict, rc = f"INCONCLUSIVE ({_panel.inconclusive_note(result)})", 1
    elif result.blocked:
        verdict, rc = "BLOCK", finalize.EXIT_VERIFY_FAILED
    elif result.merged_findings:
        verdict, rc = "PASS (with findings)", 0
    else:
        verdict, rc = "PASS", 0
    body = _panel.render_findings(result.merged_findings)
    stdout = f"VERDICT: {verdict}\n" + (f"{body}\n" if body else "")
    print(stdout, end="", flush=True)
    print(
        f"[agent6] review saved: {save_review(reviews_dir, label=label, body=stdout)}",
        file=sys.stderr,
    )
    transcript_sink.record(
        url="agent6://review-panel/result",
        request_headers={},
        request_body={"range": label},
        response_status=200,
        response_body={"stdout": stdout},
        seat="review:panel",
    )
    print(
        f"\nper-seat ({result.n_block} blocking model(s), {result.n_abstain} abstained):",
        file=sys.stderr,
    )
    for v in result.per_seat:
        status = f"abstain: {v.error}" if v.error else f"{v.verdict} ({len(v.findings)} findings)"
        print(f"  - {v.seat} [{v.model}]: {status}", file=sys.stderr)
    print(budget.format_summary(), file=sys.stderr)
    return rc


def _reviewed_diff(
    base: str, head: str, paths: tuple[str, ...]
) -> tuple[str, pathlib.Path, str, str] | int:
    """Resolve the git binary, the repo root, the diff a review reads and its label.

    Args:
        base: The base rev, or "" for the working tree.
        head: The head rev.
        paths: The pathspecs.

    Returns:
        `(git, root, diff, label)`, else the exit code: 2 when git is missing or failed,
        0 when there is nothing to review (said on stderr, naming the range).
    """
    root = pathlib.Path.cwd()
    git = shutil.which("git")
    if git is None:
        _common.error("git not found on PATH.")
        return 2
    label = ("working tree vs HEAD" if not base else f"{base}..{head}") + (
        f" -- {' '.join(paths)}" if paths else ""
    )
    diff_proc = _collect_review_diff(git, root, base=base, head=head, paths=paths)
    if diff_proc.returncode != 0:
        _common.error(f"git diff failed: {diff_proc.stderr.strip()}")
        return 2
    diff = diff_proc.stdout
    if not diff.strip():
        print(f"(no diff to review: {label})", file=sys.stderr)
        return 0
    return git, root, diff, label


def _reviewer_config(config_path: pathlib.Path | None, model: str) -> Config:
    """Return the effective config with `--model` applied to the reviewer route.

    Args:
        config_path: The `--config` file, if any.
        model: The `[provider/]model` value; "" leaves the route as configured.

    Raises:
        ConfigError: The value names no configured provider or no model.
    """
    cfg = layer.load_effective(pathlib.Path.cwd(), config_path).config
    if not model:
        return cfg
    return cfg.with_model_route("reviewer", cfg.model_route("reviewer", model))


def _cmd_review(  # noqa: PLR0911
    config_path: pathlib.Path | None,
    *,
    base: str,
    head: str,
    paths: tuple[str, ...],
    model: str = "",
    reviewers: int = 0,
    personas: str = "",
) -> int:
    """Print a code review of a diff to stdout.

    Read-only; no jail. With `reviewers` at 1 or more, the grounded adversarial panel runs
    instead of the single freeform review.

    Args:
        config_path: The `--config` file, if any.
        base: The base rev, or "" for the working tree.
        head: The head rev.
        paths: The pathspecs.
        model: The `[provider/]model` value applied to the reviewer route.
        reviewers: The number of panel seats; 0 for the freeform review.
        personas: The panel's persona roster.

    Returns:
        The exit code: 0 reviewed or PASS, 2 refused, 3 budget, and the panel's 1 or 4.
    """
    if not base and head not in ("", "HEAD"):
        _common.error(
            "--head requires --base; without --base, review uses the working tree vs HEAD."
        )
        return 2
    if personas.strip() and reviewers < 1:
        print(
            "note: --personas ignored (no --reviewers N; this is the single freeform review).",
            file=sys.stderr,
        )
    try:
        cfg = _reviewer_config(config_path, model)
    except ConfigError as exc:
        _common.error(str(exc))
        return 2
    if personas.strip() and reviewers >= 1 and cfg.review.seats:
        print("note: --personas ignored ([review].seats names the roster).", file=sys.stderr)

    reviewed = _reviewed_diff(base, head, paths)
    if isinstance(reviewed, int):
        return reviewed
    git, root, diff, label = reviewed

    if reviewers < 1:
        cfg.require_runnable("reviewer")
        err = _setup.check_provider_keys(cfg)
        if err is not None:
            _common.error(f"{err}")
            return 2

    log_proc = subprocess.run(
        [
            git,
            *git_ops.git_hardening_flags(root),
            "log",
            "-n",
            "10",
            "--oneline",
            "--end-of-options",
            head or "HEAD",  # a ref that is also a path stays a ref
            "--",
        ],
        cwd=root,
        capture_output=True,
        errors="replace",
        check=False,
    )
    recent_log = log_proc.stdout if log_proc.returncode == 0 else ""

    agents_md = _context.agents_md_text(root)

    # The reviewer route; the budget is per invocation, a one-shot.
    budget = _setup.budget_tracker(cfg)
    layout_root = agent6_paths.state_dir(root) / "reviews"
    transcript_sink = TranscriptSink(layout_root)

    if reviewers >= 1:
        return _run_review_panel(
            cfg,
            git=git,
            root=root,
            base=base,
            head=head,
            label=label,
            diff=diff,
            agents_md=agents_md,
            reviewers=reviewers,
            personas=personas,
            transcript_sink=transcript_sink,
            reviews_dir=layout_root,
            budget=budget,
        )

    try:
        reviewer = providers.build_role_provider(
            cfg,
            "reviewer",
            transcript_sink=transcript_sink,
            budget=budget,
        )
    except ProviderError as exc:
        _common.error(f"provider init failed: {exc}")
        return 2

    print(f"[agent6] reviewing: {label}", file=sys.stderr)
    try:
        text = code_review.code_review(
            reviewer,
            diff=diff,
            agents_md=agents_md,
            recent_log=recent_log,
        )
    except code_review.CodeReviewError as exc:
        print(f"REVIEW FAILED: {exc}", file=sys.stderr)
        return 2
    except agent6_budget.BudgetExceededError as exc:
        print(f"BUDGET EXCEEDED: {exc}", file=sys.stderr)
        return 3

    print(text, flush=True)
    print(
        f"[agent6] review saved: {save_review(layout_root, label=label, body=text)}",
        file=sys.stderr,
    )
    print(budget.format_summary(), file=sys.stderr)
    return 0
