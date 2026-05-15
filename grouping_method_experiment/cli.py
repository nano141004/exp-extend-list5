"""Command-line interface for the grouping-method experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import ExperimentConfig, ExperimentJob
from .constants import TABLE2_LISTT5_BASE_TOP100
from .runtime import prepare_runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ListT5 grouping-method experiments.")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the experiment grid.")
    run_parser.add_argument("--project-root", default=None)
    run_parser.add_argument("--model-path", default="Soyoung97/ListT5-base")
    run_parser.add_argument("--hf-dataset-repo", default="Soyoung97/beir-eval-bm25-top100")
    run_parser.add_argument("--datasets", nargs="+", default=["trec-covid", "nfcorpus", "fiqa", "scifact", "arguana"])
    run_parser.add_argument("--strategies", nargs="+", default=["sequential", "score_balanced", "random"])
    run_parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    run_parser.add_argument("--max-queries", type=int, default=None)
    run_parser.add_argument("--listwise-k", type=int, default=5)
    run_parser.add_argument("--out-k", type=int, default=2)
    run_parser.add_argument("--topk", type=int, default=100)
    run_parser.add_argument("--rerank-topk", type=int, default=10)
    run_parser.add_argument("--batch-size", type=int, default=20)
    run_parser.add_argument("--gpu-mode", choices=["single", "parallel_2gpu"], default="single")
    run_parser.add_argument("--gpu-ids", nargs="+", default=["0", "1"])
    run_parser.add_argument("--print-every-forwards", type=int, default=20)
    run_parser.add_argument("--no-reuse-existing", action="store_true")
    run_parser.add_argument("--data-dir-name", default="data/beir-eval-bm25-top100")
    run_parser.add_argument("--output-dir-name", default="outputs/grouping_method_experiment")

    subparsers.add_parser("show-datasets", help="Print datasets with Table 2 reference values.")

    worker_parser = subparsers.add_parser("worker", help=argparse.SUPPRESS)
    worker_parser.add_argument("--config", required=True)

    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        project_root=args.project_root,
        model_path=args.model_path,
        hf_dataset_repo=args.hf_dataset_repo,
        max_queries=args.max_queries,
        datasets=tuple(args.datasets),
        strategies=tuple(args.strategies),
        seeds=tuple(args.seeds),
        listwise_k=args.listwise_k,
        out_k=args.out_k,
        topk=args.topk,
        rerank_topk=args.rerank_topk,
        batch_size=args.batch_size,
        gpu_mode=args.gpu_mode,
        gpu_ids=tuple(args.gpu_ids),
        print_every_forwards=args.print_every_forwards,
        reuse_existing=not args.no_reuse_existing,
        data_dir_name=args.data_dir_name,
        output_dir_name=args.output_dir_name,
    )


def run_command(args: argparse.Namespace) -> None:
    from .parallel import run_grid_parallel_2gpu
    from .runner import run_grid_single

    config = config_from_args(args)
    runtime = prepare_runtime(config.project_root)
    config = config.with_project_root(runtime.project_root)

    print(f"Project root: {runtime.project_root}", flush=True)
    print(f"ListT5 root: {runtime.listt5_root}", flush=True)
    print(f"Data dir: {config.data_dir(runtime.project_root)}", flush=True)
    print(f"Output dir: {config.output_dir(runtime.project_root)}", flush=True)

    if config.gpu_mode == "single":
        results = run_grid_single(config, runtime)
    elif config.gpu_mode == "parallel_2gpu":
        results = run_grid_parallel_2gpu(config, runtime)
    else:
        raise ValueError(f"Unknown gpu_mode: {config.gpu_mode}")

    print(results.to_string(index=False), flush=True)


def worker_command(args: argparse.Namespace) -> None:
    from .runner import run_single

    payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config = ExperimentConfig.from_dict(payload["config"])
    job = ExperimentJob(**payload["job"])
    runtime = prepare_runtime(config.project_root)
    config = config.with_project_root(runtime.project_root)
    row = run_single(job, config, runtime, reuse_existing=config.reuse_existing)
    result_path = Path(payload["result_path"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(row, indent=2), encoding="utf-8")


def show_datasets() -> None:
    for dataset, score in sorted(TABLE2_LISTT5_BASE_TOP100.items()):
        print(f"{dataset}\t{score}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        args = parser.parse_args(["run"])

    if args.command == "run":
        run_command(args)
    elif args.command == "worker":
        worker_command(args)
    elif args.command == "show-datasets":
        show_datasets()
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
