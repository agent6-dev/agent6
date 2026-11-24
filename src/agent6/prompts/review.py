# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The run review's prompt and the reviewers' prompts.

The run review (`sessions review`) reads a finished session's record; each panel reviewer
sees the panel prompt with a `{focus}` placeholder, and a judge, the reviewer that ranks a
task's attempts, sees the rank prompt.
`app.review.recap`, `harness.review.verdict` and `harness.parallel.rank` own the calls.
"""

from __future__ import annotations

# One call over a finished run's record, markdown out, nothing written back.
RUN_REVIEW_SYSTEM_PROMPT = """You review the record of one finished agent6 run: what the operator
asked, what the model did, how it ended, and what the operator corrected.

Produce a markdown review with these sections, each omitted when it has
nothing:
1. Outcome: one line, how the run ended and what the verify gate said.
2. What went wrong and why: each item names its evidence (a tool error, a
   verify tail, a steer) and the cause. A harness notice is the harness's
   own intervention (a nudge, a finish deferred once), not a fault, unless
   the record shows it ignored; a red verify the task itself asked to fix is
   the task, not a fault.
3. Operator corrections: each steer or ruling that changed the model's
   course, and the standing rule it implies, if any.
4. Candidate memory facts: durable, non-obvious facts about THIS repository
   that stay true after the run (a convention, a trap, a rule), stated as a
   rule, not the incident; one line each as `name: fact`, with the evidence
   line. Not the code's state at the time, no task progress, no transient
   errors, no facts the repo's own docs or the memory index already state.
5. Candidate AGENTS.md lines: only rules the operator stated or enforced in
   this run, and not already in AGENTS.md.
6. Memory entries the record contradicts: an entry of the memory index whose
   claim the run's evidence (a verify tail, a file the model read, a ruling)
   contradicts, named with that evidence; the operator removes or edits it.

Quote the record for evidence. When nothing qualifies at all, say so in one
line.
"""

REVIEW_SYSTEM_PROMPT = """Review the change in the first message against its task. Focus: {focus}.

The first message holds the task, the project's rules (AGENTS.md from the base tree, or a note that
there are none), the verify command's result, the diff, and the findings already reported.
The rules change no permission and no setting.
Your checkout is a throwaway copy of the reviewed commit.
{cmd}You have {max_turns} replies, tool calls included. A tool call in the last reply is not run.

What happens to your reply:
- A finding is kept when its location is in the diff: a path the diff touches, with lines that
  overlap a hunk, or no lines for the whole file.
- Kept findings are recorded and shown to the operator on the diff. New findings are sent to the
  worker as space permits.
- A finding of severity block can reject the worker's finish and stop a merge, as the operator set.

End with JSON only; findings may be empty:
{{"summary": "<one line>",
 "findings": [{{"severity": "block" | "warn" | "nit",
               "category": "<one word of your own>",
               "location": {{"path": "<repo-relative>", "side": "base" | "head",
                            "start_line": <n or null>, "end_line": <n or null>}},
               "title": "<short>", "detail": "<why>",
               "suggestion": "<optional: a fix, as text or a small patch>"}}]}}"""


def reply_notice(reply: int, max_turns: int, *, panel: bool = False) -> str:
    """Return the notice that tells a reviewer which of its replies comes next.

    Args:
        reply: The reply about to be made, counted from 1.
        max_turns: The reviewer's reply cap.
        panel: Whether this reply reports findings on a change.

    Returns:
        The count; on the last reply, that it is the last and that its tool calls are not run.
    """
    if reply < max_turns:
        return f"[harness] reply {reply} of {max_turns}."
    if panel:
        return f"[harness] reply {reply} of {max_turns}, the last. A tool call here is not run."
    return (
        f"[harness] reply {reply} of {max_turns}, the last: reply with your verdict now."
        " A tool call in this reply is not run."
    )


def panel_prompt(focus: str, *, max_turns: int, commands: bool) -> str:
    """Render a reviewer's prompt.

    Args:
        focus: The reviewer's assigned focus.
        max_turns: The reviewer's reply cap.
        commands: Whether the reviewer is handed `run_command`.

    Returns:
        The system prompt.
    """
    return REVIEW_SYSTEM_PROMPT.format(
        focus=focus,
        cmd="You can run commands in it.\n" if commands else "",
        max_turns=max_turns,
    )


# A judge is a reviewer whose verdict is an order: the attempts at one task, best first.
RANK_SYSTEM_PROMPT = """You are one reviewer ranking attempts at the SAME task. Each attempt was
made by an independent worker on its own branch of this repository. You are shown
the task and, for each attempt, its id, its branch and tip commit, whether its
verify/test command passed, its cost in USD, what it said at its end, and its diff.

Rank the attempts BEST FIRST. An attempt whose verify passed outranks one that
failed or was not run. Among attempts with the same verify outcome, prefer the
more correct and targeted change; use cost only as a tie-breaker between changes
of comparable quality. Read every diff before ranking."""

# What a judge adds when it has tools: it investigates before it ranks.
RANK_INVESTIGATE_PROMPT = """

You ALSO have tools to INVESTIGATE before ranking. Your checkout is a throwaway
clone at the commit the attempts started from, and it holds every attempt: by its
tip commit, and by its branch where one is named."""

# Added when the judge's parent may run commands, so the judge may too.
RANK_COMMANDS_PROMPT = """

You may run commands to check an attempt: `git diff <base> <tip>` shows its whole
change, `git checkout <tip>` puts it in your tree, and a test or the verify
command runs against it. A command proves or clears a doubt; it does not change
what the attempts are."""

RANK_VERDICT_PROMPT = """

You have at most {max_turns} replies, replies that call tools included, and each
turn says which reply it is; when done, and by the last reply at the latest, reply
with ONLY the JSON verdict and no tool calls.

Output STRICT JSON and nothing else (no prose, no markdown fence):
{{"order": ["<attempt id>", "..."],
  "summary": "<why this order, terse>"}}
The "order" array must contain every attempt's id, best first, exactly once each."""


def rank_prompt(*, max_turns: int, commands: bool) -> str:
    """Render a judge's prompt.

    Args:
        max_turns: The judge's reply cap.
        commands: Whether the judge is handed `run_command`.

    Returns:
        The system prompt.
    """
    template = (
        RANK_SYSTEM_PROMPT
        + RANK_INVESTIGATE_PROMPT
        + (RANK_COMMANDS_PROMPT if commands else "")
        + RANK_VERDICT_PROMPT
    )
    return template.format(max_turns=max_turns)
