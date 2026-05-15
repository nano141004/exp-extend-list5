"""Single-process experiment runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import time

import pandas as pd

from .config import ExperimentConfig, ExperimentJob
from .evaluator import build_grouping_evaluator, make_args
from .results import output_is_complete, save_results_snapshot, table2_value, write_summary_tables
from .runtime import ListT5Runtime


def iter_experiment_jobs(config: ExperimentConfig) -> list[ExperimentJob]:
    jobs: list[ExperimentJob] = []
    for dataset_name in config.datasets:
        for strategy in config.strategies:
            strategy_seeds = config.seeds if strategy == "random" else (config.seeds[0],)
            for seed in strategy_seeds:
                jobs.append(ExperimentJob(dataset=dataset_name, strategy=strategy, seed=seed))
    return jobs


def run_single(
    job: ExperimentJob,
    config: ExperimentConfig,
    runtime: ListT5Runtime,
    reuse_existing: bool | None = None,
) -> dict[str, Any]:
    """Run one dataset/strategy/seed experiment."""

    effective_reuse = config.reuse_existing if reuse_existing is None else reuse_existing
    args = make_args(job, config, runtime)
    output_path = Path(args.output_path)
    start = time.time()

    print(
        f"[job start] dataset={job.dataset} strategy={job.strategy} "
        f"seed={job.seed} output={output_path}",
        flush=True,
    )

    if effective_reuse and output_is_complete(output_path, args.input_path, runtime):
        print(f"[reuse] Found complete output at {output_path}", flush=True)
        ndcg10, metric_text = runtime.run_rerank_eval(str(output_path))
        mode = "reused_output"
    else:
        evaluator_cls = build_grouping_evaluator(runtime)
        evaluator = evaluator_cls(args)
        ndcg10, metric_text = evaluator.run_tournament_sort()
        mode = "new_inference"

    elapsed = time.time() - start
    table2 = table2_value(job.dataset)
    row = {
        "dataset": job.dataset,
        "strategy": job.strategy,
        "seed": job.seed,
        "max_queries": config.max_queries,
        "ndcg@10": float(ndcg10),
        "table2_listt5_base_r2": table2,
        "delta_vs_table2": None
        if table2 is None or config.max_queries is not None
        else float(ndcg10) - table2,
        "seconds": elapsed,
        "mode": mode,
        "output_path": str(output_path),
    }
    print(
        f"[job done] dataset={job.dataset} strategy={job.strategy} "
        f"seed={job.seed} ndcg@10={float(ndcg10):.6f} seconds={elapsed:.1f}",
        flush=True,
    )
    return row


def run_grid_single(config: ExperimentConfig, runtime: ListT5Runtime) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    jobs = iter_experiment_jobs(config)
    print(f"[grid] mode=single total_jobs={len(jobs)}", flush=True)

    for job_idx, job in enumerate(jobs, start=1):
        print(
            f"\n=== job {job_idx}/{len(jobs)} | {job.dataset} | "
            f"{job.strategy} | seed={job.seed} ===",
            flush=True,
        )
        rows.append(run_single(job, config, runtime))
        save_results_snapshot(rows, config, runtime)

    results = save_results_snapshot(rows, config, runtime)
    write_summary_tables(results, config, runtime)
    return results
