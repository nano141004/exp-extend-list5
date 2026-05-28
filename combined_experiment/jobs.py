"""Experiment job construction."""

from __future__ import annotations

from .config import ExperimentConfig, ExperimentJob


def build_jobs(config: ExperimentConfig) -> list[ExperimentJob]:
    jobs: list[ExperimentJob] = []

    for dataset in config.datasets:
        if config.enable_grouping_experiment:
            for strategy in config.grouping_methods:
                for seed in config.seeds:
                    jobs.append(
                        ExperimentJob(
                            experiment_kind="grouping",
                            dataset=dataset,
                            strategy=strategy,
                            topk=config.grouping_topk,
                            seed=seed,
                        )
                    )

        if config.enable_topk_experiment:
            for strategy in config.topk_methods:
                for topk in config.topk_values:
                    for seed in config.seeds:
                        jobs.append(
                            ExperimentJob(
                                experiment_kind="topk",
                                dataset=dataset,
                                strategy=strategy,
                                topk=topk,
                                seed=seed,
                            )
                        )

    return jobs
