from __future__ import annotations

import json
import os
import pathlib
import resource
import tempfile
import time

import pydantic
from tests import review_traces, subordinate_env

from agent6 import budget
from agent6.app import role_providers
from agent6.harness import review_gate, review_panel


def main() -> None:
    """Measure the registered scripted corpus through installed read-only lanes."""
    output = pathlib.Path(__file__).with_name("review-lanes-new.jsonl")
    with tempfile.TemporaryDirectory(prefix="review-lanes-") as temporary:
        root = pathlib.Path(temporary)
        for name in ("CONFIG", "STATE", "CACHE"):
            os.environ[f"XDG_{name}_HOME"] = str(root / name.lower())
        os.environ["AGENT6_ALLOW_ROOT"] = "1"
        repo = root / "repo"
        subordinate_env.init_repo(repo)
        os.chdir(repo)
        (repo / "README.md").write_text("base\nchanged\n")
        diff = subordinate_env.git(repo, "diff")
        with output.open("w") as stream:
            for rep in range(3):
                for scope in ("diff", "repo"):
                    for case in review_traces.CASES:
                        sid = f"measure-{rep}-{scope}-{case}"
                        directory = subordinate_env.create_record(repo, sid)
                        with review_traces.server(case, scope) as (url, calls):
                            cfg = subordinate_env.config(url)
                            cfg = cfg.model_copy(
                                update={"review": cfg.review.model_copy(update={"scope": scope})}
                            )
                            meter = budget.BudgetTracker(
                                max_usd=10, max_tokens_fallback=10000, max_plan_points=20
                            )
                            executor = subordinate_env.dispatcher(repo, directory, cfg, meter)
                            target = executor.chain.review_target()
                            assert target is not None
                            target = executor.chain.retain_review_target(target, sid)
                            before = resource.getrusage(resource.RUSAGE_CHILDREN)
                            start, cpu = time.perf_counter(), time.process_time()
                            result = review_gate.run_panel(
                                role_providers.resolve_review_seats(cfg),
                                review_panel.ReviewContext(task="Review changes", diff=diff),
                                target=target,
                                executor=executor,
                                block_on="any",
                                min_blocking_models=2,
                                panel_id="measure",
                                repo_max_turns=3,
                            )
                            after = resource.getrusage(resource.RUSAGE_CHILDREN)
                            verdict = result.per_seat[0]
                            row = {
                                "scope": scope,
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
                            print(scope, case, rep, row["elapsed_s"], flush=True)


if __name__ == "__main__":
    main()
