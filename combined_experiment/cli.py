"""Command-line interface for the combined ListT5 experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import ExperimentConfig, ExperimentJob
from .constants import (
    DEFAULT_DATASETS,
    DEFAULT_GROUPING_METHODS,
    DEFAULT_GROUPING_TOPK,
    DEFAULT_TOPK_METHODS,
    DEFAULT_TOPK_VALUES,
    TABLE2_LISTT5_BASE_TOP100,
)
from .grouping import validate_grouping_methods
from .jobs import build_jobs
from .paths import resolve_project_root
from .runner import run_grid_parallel, run_grid_single, run_one_to_json


def parse_optional_int(value: str) -> int | None:
    if value.lower() in {"none", "null", "all", ""}:
        return None
    return int(value)


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--listt5-code-root", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-path", default="Soyoung97/ListT5-base")
    parser.add_argument("--hf-dataset-repo", default="Soyoung97/beir-eval-bm25-top100")
    parser.add_argument("--max-queries", type=parse_optional_int, default=50)
    parser.add_argument("--listwise-k", type=int, default=5)
    parser.add_argument("--out-k", type=int, default=2)
    parser.add_argument("--rerank-topk", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--print-every-forwards", type=int, default=20)
    parser.add_argument("--reuse-existing", action=argparse.BooleanOptionalAction, default=True)


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    project_root = resolve_project_root(args.project_root)
    config = ExperimentConfig(
        project_root=project_root,
        listt5_code_root=args.listt5_code_root,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        hf_dataset_repo=args.hf_dataset_repo,
        datasets=getattr(args, "datasets", list(DEFAULT_DATASETS)),
        max_queries=args.max_queries,
        grouping_methods=getattr(args, "grouping_methods", list(DEFAULT_GROUPING_METHODS)),
        grouping_topk=getattr(args, "grouping_topk", DEFAULT_GROUPING_TOPK),
        enable_grouping_experiment=not getattr(args, "disable_grouping", False),
        topk_methods=getattr(args, "topk_methods", list(DEFAULT_TOPK_METHODS)),
        topk_values=getattr(args, "topk_values", list(DEFAULT_TOPK_VALUES)),
        enable_topk_experiment=not getattr(args, "disable_topk", False),
        listwise_k=args.listwise_k,
        out_k=args.out_k,
        rerank_topk=args.rerank_topk,
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        seeds=getattr(args, "seeds", [0]),
        print_every_forwards=args.print_every_forwards,
        gpu_mode=getattr(args, "gpu_mode", "single"),
        gpu_ids=getattr(args, "gpu_ids", ["0", "1"]),
        reuse_existing=args.reuse_existing,
    )
    validate_grouping_methods(config.grouping_methods)
    validate_grouping_methods(config.topk_methods)
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the combined ListT5 grouping-method and BM25 top-k experiment."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the configured experiment grid.")
    add_shared_args(run_parser)
    run_parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    run_parser.add_argument("--grouping-methods", nargs="+", default=list(DEFAULT_GROUPING_METHODS))
    run_parser.add_argument("--grouping-topk", type=int, default=DEFAULT_GROUPING_TOPK)
    run_parser.add_argument("--topk-methods", nargs="+", default=list(DEFAULT_TOPK_METHODS))
    run_parser.add_argument("--topk-values", nargs="+", type=int, default=list(DEFAULT_TOPK_VALUES))
    run_parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    run_parser.add_argument("--disable-grouping", action="store_true")
    run_parser.add_argument("--disable-topk", action="store_true")
    run_parser.add_argument("--gpu-mode", choices=["single", "parallel_2gpu"], default="single")
    run_parser.add_argument("--gpu-ids", nargs="+", default=["0", "1"])

    plan_parser = subparsers.add_parser("show-plan", help="Print the jobs that would run.")
    add_shared_args(plan_parser)
    plan_parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    plan_parser.add_argument("--grouping-methods", nargs="+", default=list(DEFAULT_GROUPING_METHODS))
    plan_parser.add_argument("--grouping-topk", type=int, default=DEFAULT_GROUPING_TOPK)
    plan_parser.add_argument("--topk-methods", nargs="+", default=list(DEFAULT_TOPK_METHODS))
    plan_parser.add_argument("--topk-values", nargs="+", type=int, default=list(DEFAULT_TOPK_VALUES))
    plan_parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    plan_parser.add_argument("--disable-grouping", action="store_true")
    plan_parser.add_argument("--disable-topk", action="store_true")
    plan_parser.add_argument("--gpu-mode", choices=["single", "parallel_2gpu"], default="single")
    plan_parser.add_argument("--gpu-ids", nargs="+", default=["0", "1"])

    one_parser = subparsers.add_parser("run-one", help=argparse.SUPPRESS)
    add_shared_args(one_parser)
    one_parser.add_argument("--dataset", required=True)
    one_parser.add_argument("--strategy", required=True)
    one_parser.add_argument("--topk", type=int, required=True)
    one_parser.add_argument("--experiment-kind", choices=["grouping", "topk"], required=True)
    one_parser.add_argument("--seed", type=int, default=0)
    one_parser.add_argument("--row-output", type=Path, required=True)

    dataset_parser = subparsers.add_parser("show-datasets", help="Show datasets with Table 2 values.")
    dataset_parser.add_argument("--project-root", type=Path, default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "show-datasets":
        for dataset, ndcg in sorted(TABLE2_LISTT5_BASE_TOP100.items()):
            print(f"{dataset:16s} table2_listt5_base_top100={ndcg:.3f}")
        return 0

    if args.command == "run-one":
        config = config_from_args(args)
        validate_grouping_methods([args.strategy])
        job = ExperimentJob(
            experiment_kind=args.experiment_kind,
            dataset=args.dataset,
            strategy=args.strategy,
            topk=args.topk,
            seed=args.seed,
        )
        run_one_to_json(config, job, args.row_output)
        return 0

    config = config_from_args(args)

    if args.command == "show-plan":
        jobs = build_jobs(config)
        print(f"datasets: {config.datasets}")
        print(f"grouping experiment: methods={config.grouping_methods}, topk={config.grouping_topk}")
        print(f"top-k sweep: methods={config.topk_methods}, topk_values={config.topk_values}")
        print(f"max_queries: {config.max_queries}")
        print(f"total_jobs: {len(jobs)}")
        for index, job in enumerate(jobs, start=1):
            print(f"{index:02d}. {job}")
        return 0

    if args.command == "run":
        if config.gpu_mode == "parallel_2gpu":
            run_py = Path(__file__).resolve().parents[1] / "run.py"
            run_grid_parallel(config, run_py)
        else:
            run_grid_single(config)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
