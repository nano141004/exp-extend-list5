"""Two-GPU subprocess runner for independent experiment jobs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
from typing import Any

import pandas as pd

from .config import ExperimentConfig, ExperimentJob
from .evaluator import make_args
from .results import output_is_complete, save_results_snapshot, write_summary_tables
from .runner import iter_experiment_jobs, run_single
from .runtime import ListT5Runtime


def write_worker_config(
    job: ExperimentJob,
    config: ExperimentConfig,
    runtime: ListT5Runtime,
) -> tuple[Path, Path]:
    args = make_args(job, config, runtime)
    output_dir = config.output_dir(runtime.project_root)
    config_dir = output_dir / "parallel_job_configs"
    result_dir = output_dir / "parallel_job_results"
    config_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{job.key}__{config.subset_tag()}"
    config_path = config_dir / f"{stem}.json"
    result_path = result_dir / f"{stem}.json"

    payload = {
        "config": config.with_project_root(runtime.project_root).to_dict(),
        "job": job.__dict__,
        "result_path": str(result_path),
        "args_output_path": args.output_path,
        "args_input_path": args.input_path,
    }
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return config_path, result_path


def run_subprocess_job_on_gpu(
    job: ExperimentJob,
    gpu_id: str,
    config: ExperimentConfig,
    runtime: ListT5Runtime,
) -> dict[str, Any]:
    config_path, result_path = write_worker_config(job, config, runtime)
    args = make_args(job, config, runtime)
    output_path = Path(args.output_path)

    if config.reuse_existing and output_is_complete(output_path, args.input_path, runtime):
        print(f"[gpu {gpu_id}] [reuse] {job.dataset} | {job.strategy} | seed={job.seed}", flush=True)
        return run_single(job, config, runtime, reuse_existing=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    command = [
        sys.executable,
        "-m",
        "grouping_method_experiment.cli",
        "worker",
        "--config",
        str(config_path),
    ]

    print(f"[gpu {gpu_id}] launching {job.dataset} | {job.strategy} | seed={job.seed}", flush=True)
    proc = subprocess.Popen(
        command,
        cwd=str(runtime.project_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"[gpu {gpu_id}] {line}", end="", flush=True)

    return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"Worker failed on gpu {gpu_id} with return code {return_code}: {job}")

    return json.loads(result_path.read_text(encoding="utf-8"))


def run_grid_parallel_2gpu(config: ExperimentConfig, runtime: ListT5Runtime) -> pd.DataFrame:
    jobs = iter_experiment_jobs(config)
    job_queue: queue.Queue[ExperimentJob] = queue.Queue()
    for job in jobs:
        job_queue.put(job)

    rows: list[dict[str, Any]] = []
    errors: list[tuple[ExperimentJob, str]] = []
    lock = threading.Lock()
    print(f"[grid] mode=parallel_2gpu total_jobs={len(jobs)} gpu_ids={config.gpu_ids}", flush=True)

    def gpu_worker(gpu_id: str) -> None:
        while True:
            try:
                job = job_queue.get_nowait()
            except queue.Empty:
                return
            try:
                row = run_subprocess_job_on_gpu(job, gpu_id, config, runtime)
                with lock:
                    rows.append(row)
                    save_results_snapshot(rows, config, runtime)
                    print(f"[grid] completed {len(rows)}/{len(jobs)} jobs", flush=True)
                    print(pd.DataFrame([row]).to_string(index=False), flush=True)
            except Exception as exc:
                with lock:
                    errors.append((job, repr(exc)))
                    print(f"[grid error] gpu={gpu_id} job={job} error={exc!r}", flush=True)
            finally:
                job_queue.task_done()

    threads = [threading.Thread(target=gpu_worker, args=(gpu_id,), daemon=True) for gpu_id in config.gpu_ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if errors:
        raise RuntimeError(f"{len(errors)} parallel jobs failed. First error: {errors[0]}")

    results = save_results_snapshot(rows, config, runtime)
    results = results.sort_values(["dataset", "strategy", "seed"]).reset_index(drop=True)
    write_summary_tables(results, config, runtime)
    return results
