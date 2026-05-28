"""Configuration objects for the combined experiment CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .constants import (
    DEFAULT_DATASETS,
    DEFAULT_GROUPING_METHODS,
    DEFAULT_GROUPING_TOPK,
    DEFAULT_TOPK_METHODS,
    DEFAULT_TOPK_VALUES,
)


@dataclass(frozen=True)
class ExperimentConfig:
    project_root: Path
    listt5_code_root: Path | None = None
    data_dir: Path | None = None
    output_dir: Path | None = None

    model_path: str = "Soyoung97/ListT5-base"
    hf_dataset_repo: str = "Soyoung97/beir-eval-bm25-top100"

    datasets: list[str] = field(default_factory=lambda: list(DEFAULT_DATASETS))
    max_queries: int | None = 50

    grouping_methods: list[str] = field(default_factory=lambda: list(DEFAULT_GROUPING_METHODS))
    grouping_topk: int = DEFAULT_GROUPING_TOPK
    enable_grouping_experiment: bool = True

    topk_methods: list[str] = field(default_factory=lambda: list(DEFAULT_TOPK_METHODS))
    topk_values: list[int] = field(default_factory=lambda: list(DEFAULT_TOPK_VALUES))
    enable_topk_experiment: bool = True

    listwise_k: int = 5
    out_k: int = 2
    rerank_topk: int = 10
    batch_size: int = 20
    max_input_length: int = 512
    seeds: list[int] = field(default_factory=lambda: [0])
    print_every_forwards: int = 20

    gpu_mode: str = "single"
    gpu_ids: list[str] = field(default_factory=lambda: ["0", "1"])
    reuse_existing: bool = True

    def resolved_data_dir(self) -> Path:
        return self.data_dir or (self.project_root / "data" / "beir-eval-bm25-top100")

    def resolved_output_dir(self) -> Path:
        return self.output_dir or (self.project_root / "outputs" / "combined_grouping_topk")


@dataclass(frozen=True)
class ExperimentJob:
    experiment_kind: str
    dataset: str
    strategy: str
    topk: int
    seed: int = 0
