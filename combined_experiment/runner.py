"""Execution layer for ListT5 combined experiments."""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from .config import ExperimentConfig, ExperimentJob
from .constants import TABLE2_LISTT5_BASE_TOP100
from .data import dataset_path, output_is_complete, read_jsonl_count
from .evaluator import make_combined_evaluator_class
from .jobs import build_jobs
from .paths import resolve_listt5_code_root
from .results import (
    load_existing_live_results,
    save_final_summaries,
    save_results_snapshot,
    write_row_json,
)
from .runtime import ListT5Runtime, prepare_runtime


def subset_tag(config: ExperimentConfig) -> str:
    return "allq" if config.max_queries is None else f"maxq{config.max_queries}"


def output_path_for_job(config: ExperimentConfig, job: ExperimentJob) -> Path:
    return (
        config.resolved_output_dir()
        / job.experiment_kind
        / job.strategy
        / f"topk{job.topk}"
        / f"seed{job.seed}"
        / subset_tag(config)
        / f"{job.dataset}_output.jsonl"
    )


def make_args(config: ExperimentConfig, job: ExperimentJob, runtime: ListT5Runtime) -> SimpleNamespace:
    input_path = dataset_path(job.dataset, config, runtime)
    output_path = output_path_for_job(config, job)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if job.dataset not in runtime.beir_length_mapping:
        raise ValueError(f"No BEIR_LENGTH_MAPPING entry for dataset: {job.dataset}")

    return SimpleNamespace(
        firststage_result_key="bm25_results",
        docid_key="docid",
        pid_key="pid",
        qrels_key="qrels",
        score_key="bm25_score",
        question_text_key="q_text",
        text_key="text",
        title_key="title",
        model_path=config.model_path,
        topk=job.topk,
        max_input_length=config.max_input_length,
        padding="max_length",
        listwise_k=config.listwise_k,
        rerank_topk=config.rerank_topk,
        out_k=config.out_k,
        dummy_number=21,
        verbose=False,
        seed=job.seed,
        bsize=config.batch_size,
        input_path=str(input_path),
        output_path=str(output_path),
        measure_flops=False,
        skip_no_candidate=False,
        skip_issubset=False,
        max_gen_length=config.listwise_k + 2,
        grouping_strategy=job.strategy,
        experiment_kind=job.experiment_kind,
        print_every_forwards=config.print_every_forwards,
    )


def clear_cuda_memory(runtime: ListT5Runtime, label: str = "") -> None:
    gc.collect()
    torch = runtime.torch
    if not torch.cuda.is_available():
        return

    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    free = free_bytes / (1024**3)
    total = total_bytes / (1024**3)
    prefix = f"[cuda cleanup {label}]" if label else "[cuda cleanup]"
    print(
        f"{prefix} allocated={allocated:.2f}GB reserved={reserved:.2f}GB free={free:.2f}/{total:.2f}GB",
        flush=True,
    )


def create_runtime(config: ExperimentConfig) -> ListT5Runtime:
    code_root = resolve_listt5_code_root(config.project_root, config.listt5_code_root)
    runtime = prepare_runtime(code_root)
    print(f"Project root: {config.project_root}", flush=True)
    print(f"ListT5 code root: {runtime.code_root}", flush=True)
    print(f"Data dir: {config.resolved_data_dir()}", flush=True)
    print(f"Output dir: {config.resolved_output_dir()}", flush=True)
    print(f"CUDA available: {runtime.torch.cuda.is_available()}", flush=True)
    if runtime.torch.cuda.is_available():
        print(f"CUDA device: {runtime.torch.cuda.get_device_name(0)}", flush=True)
    return runtime


def run_single(config: ExperimentConfig, job: ExperimentJob, runtime: ListT5Runtime) -> dict:
    args = make_args(config, job, runtime)
    started = time.time()
    mode = "new_inference"
    num_forward = None

    print("", flush=True)
    print(
        (
            f"=== {job.experiment_kind} | {job.dataset} | {job.strategy} "
            f"| topk={job.topk} | seed={job.seed} | max_queries={config.max_queries} ==="
        ),
        flush=True,
    )
    print(f"Model path: {args.model_path}", flush=True)
    print(f"Input: {args.input_path}", flush=True)
    print(f"Output: {args.output_path}", flush=True)
    print(f"Batch size: {args.bsize}", flush=True)
    print(f"Max input length: {args.max_input_length}", flush=True)

    clear_cuda_memory(runtime, "before job")
    evaluator = None
    try:
        if config.reuse_existing and output_is_complete(args.output_path, args.input_path, runtime):
            print("[reuse] Complete output found. Recomputing metrics only.", flush=True)
            ndcg10, metric_text = runtime.run_rerank_eval(args.output_path)
            mode = "reused_output"
        else:
            evaluator_cls = make_combined_evaluator_class(runtime, config.print_every_forwards)
            evaluator = evaluator_cls(args)
            ndcg10, metric_text = evaluator.run_tournament_sort()
            num_forward = evaluator.num_forward
    except runtime.torch.cuda.OutOfMemoryError:
        print(
            "[oom] CUDA out of memory. Freeing model/cache before raising. "
            "Try --batch-size 5 or --max-input-length 512 if this repeats.",
            flush=True,
        )
        raise
    finally:
        if evaluator is not None:
            try:
                evaluator.model.to("cpu")
            except Exception:
                pass
            del evaluator
        clear_cuda_memory(runtime, "after job")

    seconds = time.time() - started
    num_queries = read_jsonl_count(args.input_path, runtime)
    table2 = TABLE2_LISTT5_BASE_TOP100.get(job.dataset)
    row = {
        "experiment_kind": job.experiment_kind,
        "dataset": job.dataset,
        "strategy": job.strategy,
        "topk": job.topk,
        "seed": job.seed,
        "max_queries": config.max_queries,
        "ndcg@10": float(ndcg10),
        "table2_listt5_base_top100": table2,
        "delta_vs_table2_top100": None if table2 is None else float(ndcg10) - table2,
        "seconds": seconds,
        "seconds_per_query": seconds / num_queries if num_queries else None,
        "num_queries": num_queries,
        "num_forward": num_forward,
        "mode": mode,
        "output_path": str(args.output_path),
    }
    print(f"[done] {row}", flush=True)
    return row


def run_grid_single(config: ExperimentConfig) -> list[dict]:
    runtime = create_runtime(config)
    rows = load_existing_live_results(config.resolved_output_dir())
    jobs = build_jobs(config)
    print(f"[grid] total_jobs={len(jobs)} datasets={config.datasets}", flush=True)

    for job_index, job in enumerate(jobs, start=1):
        print("", flush=True)
        print(f"[grid] job {job_index}/{len(jobs)} -> {job}", flush=True)
        row = run_single(config, job, runtime)
        rows.append(row)
        save_results_snapshot(rows, config.resolved_output_dir())

    save_final_summaries(rows, config.resolved_output_dir())
    return rows


def run_grid_parallel(config: ExperimentConfig, run_py: Path) -> list[dict]:
    jobs = build_jobs(config)
    rows = load_existing_live_results(config.resolved_output_dir())
    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[parallel] total_jobs={len(jobs)} gpu_ids={config.gpu_ids}. "
        "Each process runs one independent dataset/strategy/top-k job.",
        flush=True,
    )

    pending = list(enumerate(jobs, start=1))
    running: list[tuple[subprocess.Popen, Path, ExperimentJob, str, int]] = []

    with tempfile.TemporaryDirectory(prefix="listt5_combined_rows_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        while pending or running:
            while pending and len(running) < len(config.gpu_ids):
                job_index, job = pending.pop(0)
                busy_gpus = {item[3] for item in running}
                free_gpus = [gpu for gpu in config.gpu_ids if gpu not in busy_gpus]
                gpu_id = free_gpus[0] if free_gpus else config.gpu_ids[len(running) % len(config.gpu_ids)]
                row_path = temp_dir_path / f"row_{job_index}.json"
                cmd = [
                    sys.executable,
                    str(run_py),
                    "run-one",
                    "--project-root",
                    str(config.project_root),
                    "--dataset",
                    job.dataset,
                    "--strategy",
                    job.strategy,
                    "--topk",
                    str(job.topk),
                    "--experiment-kind",
                    job.experiment_kind,
                    "--seed",
                    str(job.seed),
                    "--row-output",
                    str(row_path),
                ] + config_to_child_args(config)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                print(f"[parallel] launch job {job_index}/{len(jobs)} gpu={gpu_id} -> {job}", flush=True)
                proc = subprocess.Popen(cmd, env=env)
                running.append((proc, row_path, job, gpu_id, job_index))

            time.sleep(5)
            still_running = []
            for proc, row_path, job, gpu_id, job_index in running:
                code = proc.poll()
                if code is None:
                    still_running.append((proc, row_path, job, gpu_id, job_index))
                    continue
                if code != 0:
                    for other_proc, _, _, _, _ in running:
                        if other_proc.poll() is None:
                            other_proc.terminate()
                    raise RuntimeError(f"Parallel job failed with exit code {code}: {job}")
                row = json.loads(row_path.read_text(encoding="utf-8"))
                rows.append(row)
                print(f"[parallel] finished job {job_index}/{len(jobs)} gpu={gpu_id} -> {job}", flush=True)
                save_results_snapshot(rows, output_dir)
            running = still_running

    save_final_summaries(rows, output_dir)
    return rows


def config_to_child_args(config: ExperimentConfig) -> list[str]:
    args = [
        "--model-path",
        config.model_path,
        "--hf-dataset-repo",
        config.hf_dataset_repo,
        "--listwise-k",
        str(config.listwise_k),
        "--out-k",
        str(config.out_k),
        "--rerank-topk",
        str(config.rerank_topk),
        "--batch-size",
        str(config.batch_size),
        "--max-input-length",
        str(config.max_input_length),
        "--print-every-forwards",
        str(config.print_every_forwards),
        "--output-dir",
        str(config.resolved_output_dir()),
        "--data-dir",
        str(config.resolved_data_dir()),
    ]
    if config.max_queries is None:
        args += ["--max-queries", "none"]
    else:
        args += ["--max-queries", str(config.max_queries)]
    if config.listt5_code_root is not None:
        args += ["--listt5-code-root", str(config.listt5_code_root)]
    if not config.reuse_existing:
        args.append("--no-reuse-existing")
    return args


def run_one_to_json(config: ExperimentConfig, job: ExperimentJob, row_output: Path) -> dict:
    runtime = create_runtime(config)
    row = run_single(config, job, runtime)
    write_row_json(row, row_output)
    return row
