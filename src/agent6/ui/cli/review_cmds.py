# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 review`: the freeform review and the adversarial panel.

`save_review` is the one writer of `<state-dir>/reviews/`, shared with `sessions review`.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import shutil
import subprocess
import sys
import time
import uuid

import pydantic

from agent6 import budget as agent6_budget
from agent6 import errors, event_log, review_records, terminal
from agent6 import paths as agent6_paths
from agent6.app import _setup, finalize, providers
from agent6.config import (
    Config,
    ConfigError,
    ReviewScope,
    layer,
)
from agent6.harness import _context, _panel, _review_history, _reviewer, _toolset, code_review
from agent6.providers import (
    ProviderError,
    TranscriptSink,
)
from agent6.tools import dispatch as tools_dispatch
from agent6.ui.cli import _common
from agent6.vcs import git_policy, git_read, review
from agent6.viewmodel import review as viewmodel_review


def _reviewed_head(base: str, head: str) -> str:
    """Return the rev the review reads as its head: a range's right endpoint, else `head`.

    Args:
        base: The base rev, or an explicit `A..B` range; `A..` ends at HEAD, as git reads it.
        head: The `--head` rev.

    Returns:
        The reviewed head.
    """
    return (base.rsplit("..", 1)[-1] or "HEAD") if ".." in base else head


def _review_range(base: str, head: str) -> str:
    """Use the merge base for ref pairs and preserve an explicit range.

    Args:
        base: The base rev, an explicit `a..b` range, or "" for the working tree.
        head: The head rev; unused when `base` is a range or "".

    Returns:
        The git diff revision expression, or HEAD for a working-tree review.
    """
    if not base:
        return "HEAD"
    if ".." in base:
        return base
    return f"{base}...{head}"


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

    A repo-scope seat's read-only tools read the checkout, so anything else answers
    `read_file` from a tree the diff does not describe.

    Args:
        git: The git binary.
        root: The repo root.
        rev: The reviewed head.

    Returns:
        True when HEAD is the rev's commit and no tracked file has changes.
    """
    checked_out = git_read.chain_tip(root, "HEAD")
    if checked_out is None or git_read.chain_tip(root, rev) != checked_out:
        return False
    status = subprocess.run(
        [
            git,
            *git_policy.git_hardening_flags(root),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
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
    rules: review_records.ReviewRules,
    target: review_records.ReviewTarget,
    journal: event_log.EventSink,
    panel_id: str,
    reviewers: int,
    focus: str,
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
        root: The review's working directory.
        base: The base rev.
        head: The head rev.
        label: What is reviewed, for the saved file.
        diff: The diff text.
        rules: The complete base-tree rule set.
        target: The immutable comparison.
        journal: The saved review journal.
        panel_id: The review identity.
        reviewers: The number of seats.
        focus: The panel's comma-separated focus list.
        transcript_sink: Where the seats' transcripts go.
        reviews_dir: The `<state-dir>/reviews/` directory.
        budget: The invocation's budget.

    Returns:
        The verdict on `agent6 run`'s scale: 0 PASS (clean or non-blocking findings),
        1 INCONCLUSIVE (every seat abstained), 3 budget, 4 BLOCK (a grounded gating
        finding). 2 stays the refusal before any seat ran, so a script tells "blocked"
        from "not reviewed" from "could not start".
    """
    focus_tuple = tuple(f.strip() for f in focus.split(",") if f.strip())
    try:
        seats = providers.build_review_seats(
            cfg,
            transcript_sink=transcript_sink,
            budget=budget,
            n=reviewers,
            focus=focus_tuple,
        )
    except ProviderError as exc:
        _common.error(f"provider init failed: {exc}")
        return 2
    err = _setup.check_provider_keys(cfg)
    if err is not None:
        _common.error(f"{err}")
        return 2
    ctx = _panel.ReviewContext(task=f"code review: {label}", rules=rules, diff=diff)
    # repo-scope seats need a read-only tool surface over the repo.
    tools = None
    dispatch = None
    if any(s.scope == "repo" for s in seats):
        reviewed = _reviewed_head(base, head)
        if base and not _is_checked_out(git, root, reviewed):
            _common.error(
                "review.scope = 'repo' reads the checkout, but the checkout is not the"
                f" reviewed head {reviewed!r} (another commit, or uncommitted changes on top):"
                " a seat's read_file would answer from a tree the diff does not describe."
                " Check it out clean, or set review.scope = 'diff'."
            )
            return 2
        disp = tools_dispatch.ToolDispatcher(root=root, config=cfg)
        tools, dispatch = _toolset.build_readonly_review_tools(disp)
    print(
        f"[agent6] review panel: {len(seats)} seats"
        f" ({', '.join(s.focus for s in seats)}) | block_on={cfg.review.block_on}"
        f" | scope={cfg.review.scope} | range={label}",
        file=sys.stderr,
    )
    _start_review(journal, panel_id, target, rules, seats)
    try:
        result = _reviewer.run_panel(
            seats,
            ctx,
            block_on=cfg.review.block_on,
            min_blocking_models=cfg.review.min_blocking_models,
            panel_id=panel_id,
            repo_max_turns=cfg.review.repo_max_turns,
            concurrency=len(seats),  # one-shot CLI: run all seats in parallel
            tools=tools,
            dispatch=dispatch,
        )
    except agent6_budget.BudgetExceededError as exc:
        journal.emit(
            "loop.review.skipped", panel_id=panel_id, execution_id=panel_id, reason=str(exc)
        )
        print(str(exc), file=sys.stderr)  # the reason names the ledger that ran out
        return 3
    for verdict_record in result.per_seat:
        journal.emit(
            "loop.review.seat",
            panel_id=panel_id,
            execution_id=panel_id,
            **pydantic.TypeAdapter(review_records.ReviewVerdict).dump_python(
                verdict_record, mode="json"
            ),
        )
    record = review_records.ReviewPanel(
        panel_id=panel_id,
        execution_id=panel_id,
        turn=0,
        trigger="standalone",
        target=target,
        findings=_review_history.observations(
            panel_id,
            result.per_seat,
            {},
            root=git_read.toplevel(root) or root,
            target=target,
            hide=cfg.sandbox.hide_paths,
        ),
        injected_finding_ids=(),
        block_on=result.block_on,
        raw_blocked=result.blocked,
        blocked=result.blocked,
        disarmed=False,
        n_block=result.n_block,
        n_abstain=result.n_abstain,
        completion="inconclusive" if _panel.panel_is_inconclusive(result) else "complete",
    )
    journal.emit("loop.review.panel", **record.model_dump(mode="json"))
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
        print(f"  - {v.focus} [{v.model_route}]: {status}", file=sys.stderr)
    budget.settle()
    print(budget.format_summary(), file=sys.stderr)
    return rc


def _reviewed_diff(
    base: str, head: str, paths: tuple[str, ...], hide: tuple[str, ...] = ()
) -> tuple[str, pathlib.Path, str, str, review_records.ReviewTarget] | int:
    """Resolve the git binary, working directory, reviewed diff and label.

    Args:
        base: The base rev, or "" for the working tree.
        head: The head rev.
        paths: The pathspecs.
        hide: `[sandbox].hide_paths`, excluded from the review.

    Returns:
        `(git, root, diff, label, target)`, else the exit code: 2 when git is missing or failed,
        0 when there is nothing to review (said on stderr, naming the range).
    """
    root = pathlib.Path.cwd()
    git = shutil.which("git")
    if git is None:
        _common.error("git not found on PATH.")
        return 2
    rev = _review_range(base, head)
    label = ("working tree vs HEAD" if not base else rev) + (
        f" -- {' '.join(terminal.printable_name(path) for path in paths)}" if paths else ""
    )
    try:
        compared = review.comparison(root, base=base, head=head, paths=paths, hide=hide)
    except (errors.GitError, OSError) as exc:
        _common.error(f"git diff failed: {terminal.printable_name(str(exc))}")
        return 2
    if compared.target.hidden_paths:
        print("note: review excludes hidden paths (sandbox.hide_paths).", file=sys.stderr)
    if compared.dirty_submodules:
        names = ", ".join(terminal.printable_name(name) for name in compared.dirty_submodules)
        print(f"note: submodule working-tree contents are not reviewed: {names}", file=sys.stderr)
    if not compared.diff.strip():
        print(f"(no diff to review: {label})", file=sys.stderr)
        return 0
    if base:
        label += f" (from {compared.target.base_commit})"
    return git, root, compared.diff, label, compared.target


def _start_review(
    journal: event_log.EventSink,
    panel_id: str,
    target: review_records.ReviewTarget,
    rules: review_records.ReviewRules,
    seats: list[_reviewer.ReviewSeat],
) -> None:
    fields = dict(
        panel_id=panel_id,
        execution_id=panel_id,
        turn=0,
        trigger="standalone",
        target=pydantic.TypeAdapter(review_records.ReviewTarget).dump_python(target, mode="json"),
        rule_sources=[dataclasses.asdict(source) for source in rules.sources],
        rules_complete=rules.complete,
        rules_reason=rules.reason,
        seats=[
            dict(
                seat_index=i,
                focus=s.focus,
                model_route=s.model,
                scope=s.scope,
                temperature=s.temperature,
            )
            for i, s in enumerate(seats)
        ],
    )
    journal.emit("loop.review.start", **fields)
    print(viewmodel_review.rules_notice(fields), file=sys.stderr)


def _reviewer_config(
    config_paths: tuple[pathlib.Path, ...], model: str, scope: ReviewScope | None = None
) -> Config:
    """Return the effective config with `--model` applied to the reviewer route.

    Args:
        config_paths: The `--config` files in command-line order, if any.
        model: The `[provider/]model` value; "" leaves the route as configured.
        scope: The panel reading scope; None leaves it as configured.

    Returns:
        The current directory's effective config, its reviewer route replaced when `model`
        is given.

    Raises:
        ConfigError: The value names no configured provider or no model.
    """
    cfg = layer.load_effective(pathlib.Path.cwd(), config_paths).config
    if model:
        cfg = cfg.with_model_route("reviewer", cfg.model_route("reviewer", model))
    if scope is not None:
        data = cfg.model_dump(mode="python")
        data["review"]["scope"] = scope
        cfg = Config.model_validate(data)
    return cfg


def _cmd_review(  # noqa: PLR0911
    config_paths: tuple[pathlib.Path, ...],
    *,
    base: str,
    head: str,
    paths: tuple[str, ...],
    model: str = "",
    reviewers: int = 0,
    focus: str = "",
    scope: ReviewScope | None = None,
) -> int:
    """Print a code review of a diff to stdout.

    Read-only; no jail. With `reviewers` at 1 or more, the grounded adversarial panel runs
    instead of the single freeform review.

    Args:
        config_paths: The `--config` files in command-line order, if any.
        base: The base rev, or "" for the working tree.
        head: The head rev.
        paths: The pathspecs.
        model: The `[provider/]model` value applied to the reviewer route.
        reviewers: The number of panel seats; 0 for the freeform review.
        focus: The panel's comma-separated focus list.
        scope: The panel's reading scope; None keeps the config's.

    Returns:
        The exit code: 0 reviewed or PASS, 2 refused, 3 budget, and the panel's 1 or 4.
    """
    if ".." in head:
        _common.error("--head must be a ref; use --base a..b for an explicit range.")
        return 2
    if not base and head not in ("", "HEAD"):
        _common.error(
            "--head requires --base; without --base, review uses the working tree vs HEAD."
        )
        return 2
    if focus.strip() and reviewers < 1:
        print(
            "note: --focus ignored (no --reviewers N; this is the single freeform review).",
            file=sys.stderr,
        )
    try:
        cfg = _reviewer_config(config_paths, model, scope)
    except ConfigError as exc:
        _common.error(str(exc))
        return 2
    if focus.strip() and reviewers >= 1 and cfg.review.reviewers:
        print("note: --focus ignored ([review].reviewers names the roster).", file=sys.stderr)

    reviewed = _reviewed_diff(base, head, paths, cfg.sandbox.hide_paths)
    if isinstance(reviewed, int):
        return reviewed
    git, root, diff, label, target = reviewed

    if reviewers < 1:
        cfg.require_runnable("reviewer")
        err = _setup.check_provider_keys(cfg)
        if err is not None:
            _common.error(f"{err}")
            return 2

    log_proc = subprocess.run(
        [
            git,
            *git_policy.git_hardening_flags(root),
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

    rules = _context.review_rules(root, target.base_tree, cfg.sandbox.hide_paths)

    # The reviewer route; the budget is per invocation, a one-shot.
    budget = _setup.budget_tracker(cfg)
    layout_root = agent6_paths.state_dir(root) / "reviews"
    transcript_sink = TranscriptSink(layout_root)
    panel_id = uuid.uuid4().hex
    journal = event_log.EventSink(layout_root / f"{panel_id}.jsonl")
    if not rules.complete:
        _start_review(journal, panel_id, target, rules, [])
        journal.emit(
            "loop.review.skipped", panel_id=panel_id, execution_id=panel_id, reason=rules.reason
        )
        _common.error(terminal.printable_name(rules.reason or "review rules incomplete"))
        return 2

    print(f"[agent6] reviewing: {label}", file=sys.stderr)
    if reviewers >= 1:
        return _run_review_panel(
            cfg,
            git=git,
            root=root,
            base=base,
            head=head,
            label=label,
            diff=diff,
            rules=rules,
            target=target,
            journal=journal,
            panel_id=panel_id,
            reviewers=reviewers,
            focus=focus,
            transcript_sink=transcript_sink,
            reviews_dir=layout_root,
            budget=budget,
        )

    return _freeform_review(
        cfg,
        diff=diff,
        rules=rules,
        target=target,
        label=label,
        recent_log=recent_log,
        transcript_sink=transcript_sink,
        budget=budget,
        journal=journal,
        panel_id=panel_id,
        layout_root=layout_root,
    )


def _freeform_review(
    cfg: Config,
    *,
    diff: str,
    rules: review_records.ReviewRules,
    target: review_records.ReviewTarget,
    label: str,
    recent_log: str,
    transcript_sink: TranscriptSink,
    budget: agent6_budget.BudgetTracker,
    journal: event_log.EventSink,
    panel_id: str,
    layout_root: pathlib.Path,
) -> int:
    """Run and save the freeform review with its rule-source journal.

    Args:
        cfg: The effective config.
        diff: The reviewed diff.
        rules: The complete base-tree rule set.
        target: The immutable comparison.
        label: The saved review's label.
        recent_log: Recent commit summaries for context.
        transcript_sink: Where the provider transcript goes.
        budget: The invocation's budget.
        journal: The saved review journal.
        panel_id: The review identity.
        layout_root: The reviews directory.

    Returns:
        Zero on success, 2 on refusal, or 3 when the budget is spent.
    """
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

    _start_review(journal, panel_id, target, rules, [])
    try:
        text = code_review.review_diff(
            reviewer,
            diff=diff,
            rules=rules,
            recent_log=recent_log,
        )
    except code_review.CodeReviewError as exc:
        journal.emit(
            "loop.review.skipped", panel_id=panel_id, execution_id=panel_id, reason=str(exc)
        )
        print(f"REVIEW FAILED: {exc}", file=sys.stderr)
        return 2
    except agent6_budget.BudgetExceededError as exc:
        journal.emit(
            "loop.review.skipped", panel_id=panel_id, execution_id=panel_id, reason=str(exc)
        )
        print(str(exc), file=sys.stderr)  # the reason names the ledger that ran out
        return 3

    journal.emit("loop.review.freeform", panel_id=panel_id, execution_id=panel_id)
    print(text, flush=True)
    print(
        f"[agent6] review saved: {save_review(layout_root, label=label, body=text)}",
        file=sys.stderr,
    )
    budget.settle()
    print(budget.format_summary(), file=sys.stderr)
    return 0
