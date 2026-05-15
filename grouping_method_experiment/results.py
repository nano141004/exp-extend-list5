"""Result persistence and summary table generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import ExperimentConfig
from .constants import TABLE2_LISTT5_BASE_TOP100
from .runtime import ListT5Runtime


def result_paths(config: ExperimentConfig, runtime: ListT5Runtime) -> dict[str, Path]:
    output_dir = config.output_dir(runtime.project_root)
    return {
        "live_csv": output_dir / "results_live.csv",
        "live_txt": output_dir / "results_live.txt",
        "baseline_csv": output_dir / "baseline_check.csv",
        "baseline_txt": output_dir / "baseline_check.txt",
        "comparison_csv": output_dir / "grouping_comparison.csv",
        "comparison_txt": output_dir / "grouping_comparison.txt",
    }


def output_is_complete(output_path: Path, input_path: str, runtime: ListT5Runtime) -> bool:
    if not output_path.exists():
        return False
    try:
        return len(runtime.read_jsonl(str(output_path))) == len(runtime.read_jsonl(input_path))
    except Exception:
        return False


def save_results_snapshot(
    rows: list[dict[str, Any]],
    config: ExperimentConfig,
    runtime: ListT5Runtime,
) -> pd.DataFrame:
    """Persist the current experiment table after every completed approach."""

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    key_cols = ["dataset", "strategy", "seed", "max_queries"]
    df = df.drop_duplicates(subset=key_cols, keep="last")
    df = df.sort_values(key_cols).reset_index(drop=True)

    paths = result_paths(config, runtime)
    paths["live_csv"].parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths["live_csv"], index=False)
    paths["live_txt"].write_text(df.to_string(index=False) + "\n", encoding="utf-8")

    print(f"[saved] live results -> {paths['live_csv']}", flush=True)
    print(f"[saved] live results -> {paths['live_txt']}", flush=True)
    return df


def write_summary_tables(
    results: pd.DataFrame,
    config: ExperimentConfig,
    runtime: ListT5Runtime,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write baseline and grouping comparison tables."""

    paths = result_paths(config, runtime)
    paths["baseline_csv"].parent.mkdir(parents=True, exist_ok=True)

    baseline_cols = ["dataset", "ndcg@10", "table2_listt5_base_r2", "delta_vs_table2", "output_path"]
    baseline_check = (
        results[results["strategy"] == "sequential"][baseline_cols]
        .sort_values("dataset")
        .reset_index(drop=True)
    )
    baseline_check.to_csv(paths["baseline_csv"], index=False)
    paths["baseline_txt"].write_text(baseline_check.to_string(index=False) + "\n", encoding="utf-8")

    comparison = results.pivot_table(
        index="dataset",
        columns="strategy",
        values="ndcg@10",
        aggfunc="mean",
    )
    if "sequential" in comparison.columns:
        for col in list(comparison.columns):
            comparison[f"{col}_minus_sequential"] = comparison[col] - comparison["sequential"]

    comparison_table = comparison.reset_index()
    comparison_table.to_csv(paths["comparison_csv"], index=False)
    paths["comparison_txt"].write_text(comparison_table.to_string(index=False) + "\n", encoding="utf-8")

    print(f"[saved] baseline summary -> {paths['baseline_csv']}", flush=True)
    print(f"[saved] baseline summary -> {paths['baseline_txt']}", flush=True)
    print(f"[saved] grouping comparison -> {paths['comparison_csv']}", flush=True)
    print(f"[saved] grouping comparison -> {paths['comparison_txt']}", flush=True)
    return baseline_check, comparison_table


def table2_value(dataset: str) -> float | None:
    return TABLE2_LISTT5_BASE_TOP100.get(dataset)
