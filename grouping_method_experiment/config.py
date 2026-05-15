"""Configuration objects for the grouping-method experiment."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .constants import DEFAULT_DATASETS, DEFAULT_SEEDS, DEFAULT_STRATEGIES


@dataclass(frozen=True)
class ExperimentJob:
    """One independent dataset/strategy/seed run."""

    dataset: str
    strategy: str
    seed: int = 0

    @property
    def key(self) -> str:
        return f"{self.dataset}__{self.strategy}__seed{self.seed}"


@dataclass(frozen=True)
class ExperimentConfig:
    """Stable knobs shared by all experiment runs."""

    project_root: str | None = None
    model_path: str = "Soyoung97/ListT5-base"
    hf_dataset_repo: str = "Soyoung97/beir-eval-bm25-top100"
    max_queries: int | None = None
    datasets: tuple[str, ...] = DEFAULT_DATASETS
    strategies: tuple[str, ...] = DEFAULT_STRATEGIES
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    listwise_k: int = 5
    out_k: int = 2
    topk: int = 100
    rerank_topk: int = 10
    batch_size: int = 20
    gpu_mode: str = "single"
    gpu_ids: tuple[str, ...] = ("0", "1")
    print_every_forwards: int = 20
    reuse_existing: bool = True
    data_dir_name: str = "data/beir-eval-bm25-top100"
    output_dir_name: str = "outputs/grouping_method_experiment"

    def with_project_root(self, project_root: Path) -> "ExperimentConfig":
        return replace(self, project_root=str(project_root))

    def data_dir(self, project_root: Path) -> Path:
        return project_root / self.data_dir_name

    def output_dir(self, project_root: Path) -> Path:
        return project_root / self.output_dir_name

    def subset_tag(self) -> str:
        return "full" if self.max_queries is None else f"first{self.max_queries}"

    def to_dict(self) -> dict[str, Any]:
        data = self.__dict__.copy()
        data["datasets"] = list(self.datasets)
        data["strategies"] = list(self.strategies)
        data["seeds"] = list(self.seeds)
        data["gpu_ids"] = list(self.gpu_ids)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        normalized = dict(data)
        for key in ("datasets", "strategies", "gpu_ids"):
            if key in normalized and normalized[key] is not None:
                normalized[key] = tuple(normalized[key])
        if "seeds" in normalized and normalized["seeds"] is not None:
            normalized["seeds"] = tuple(int(seed) for seed in normalized["seeds"])
        return cls(**normalized)
