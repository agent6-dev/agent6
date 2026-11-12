from __future__ import annotations

import json
import os
import pathlib
import resource
import tempfile
import time

import pydantic
from tests import child_env, review_traces

from agent6 import budget
from agent6.app import role_providers
from agent6.harness.review import gate as review_gate
from agent6.harness.review import panel as review_panel

# The tools the measured reviewer is handed: the ones the recorded repo reviewers had.
READING_TOOLS = ("find_definition", "find_references", "list_dir", "outline", "read_file")


def main() -> None:
    """Measure the registered scripted corpus through installed read-only lanes."""
    output = pathlib.Path(__file__).with_name("review-lanes-rerun.jsonl")
    with tempfile.TemporaryDirectory(prefix="review-lanes-") as temporary:
        root = pathlib.Path(temporary)
        for name in ("CONFIG", "STATE", "CACHE"):
            os.environ[f"XDG_{name}_HOME"] = str(root / name.lower())
        os.environ["AGENT6_ALLOW_ROOT"] = "1"
        repo = root / "repo"
        child_env.init_repo(repo)
        os.chdir(repo)
        (repo / "README.md").write_text("base\nchanged\n")
        diff = child_env.git(repo, "diff")
        with output.open("w") as stream:
            for rep in range(3):
                for case in review_traces.CASES:
                    sid = f"measure-{rep}-{case}"
                    directory = child_env.create_record(repo, sid)
                    with review_traces.server(case) as (url, calls):
                        cfg = child_env.config(url)
                        meter = budget.BudgetTracker(
                            max_usd=10, max_tokens_fallback=10000, max_plan_points=20
                        )
                        executor = child_env.dispatcher(repo, directory, cfg, meter)
                        target = executor.chain.review_target()
                        assert target is not None
                        target = executor.chain.retain_review_target(target, sid)
                        before = resource.getrusage(resource.RUSAGE_CHILDREN)
                        start, cpu = time.perf_counter(), time.process_time()
                        result = review_gate.run_panel(
                            role_providers.resolve_reviewers(cfg),
                            review_panel.ReviewContext(task="Review changes", diff=diff),
                            target=target,
                            executor=executor,
                            block_on="any",
                            min_blocking_models=2,
                            panel_id="measure",
                            max_turns=3,
                            tools=READING_TOOLS,
                        )
                        after = resource.getrusage(resource.RUSAGE_CHILDREN)
                        verdict = result.per_reviewer[0]
                        row = {
                            "case": case,
                            "rep": rep,
                            "elapsed_s": time.perf_counter() - start,
                            "cpu_s": time.process_time() - cpu,
                            "child_cpu_s": after.ru_utime
                            + after.ru_stime
                            - before.ru_utime
                            - before.ru_stime,
                            "lane_processes": 1,
                            "calls": len(calls),
                            "usd": meter.estimate_usd()[0],
                            "verdict": pydantic.TypeAdapter(type(verdict)).dump_python(
                                verdict, mode="json"
                            ),
                        }
                        stream.write(json.dumps(row) + "\n")
                        stream.flush()
                        print(case, rep, row["elapsed_s"], flush=True)


if __name__ == "__main__":
    main()
