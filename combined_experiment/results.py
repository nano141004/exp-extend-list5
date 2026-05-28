"""Result persistence and summary views."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _string_key(row: dict, columns: list[str]) -> tuple[str, ...]:
    return tuple("" if row.get(column) is None else str(row.get(column)) for column in columns)


def _coerce_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _ordered_rows(rows: list[dict], key_cols: list[str]) -> list[dict]:
    deduped: dict[tuple[str, ...], dict] = {}
    for row in rows:
        deduped[_string_key(row, key_cols)] = row
    return sorted(deduped.values(), key=lambda row: _string_key(row, key_cols))


def _fieldnames(rows: list[dict]) -> list[str]:
    preferred = [
        "experiment_kind",
        "dataset",
        "strategy",
        "topk",
        "seed",
        "max_queries",
        "ndcg@10",
        "table2_listt5_base_top100",
        "delta_vs_table2_top100",
        "seconds",
        "seconds_per_query",
        "num_queries",
        "num_forward",
        "mode",
        "output_path",
        "sequential_top100_ndcg@10",
        "sequential_top100_seconds_per_query",
        "delta_vs_sequential_top100",
        "speedup_vs_sequential_top100",
        "table2_note",
    ]
    present = {key for row in rows for key in row}
    return [key for key in preferred if key in present] + sorted(present - set(preferred))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = _fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_text_table(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = _fieldnames(rows)
    widths = {
        field: max(len(field), *(len("" if row.get(field) is None else str(row.get(field))) for row in rows))
        for field in fields
    }
    lines = [" ".join(field.ljust(widths[field]) for field in fields)]
    lines.append(" ".join("-" * widths[field] for field in fields))
    for row in rows:
        lines.append(
            " ".join(("" if row.get(field) is None else str(row.get(field))).ljust(widths[field]) for field in fields)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_existing_live_results(output_dir: Path) -> list[dict]:
    path = output_dir / "results_live.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save_results_snapshot(rows: list[dict], output_dir: Path) -> list[dict]:
    if not rows:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    key_cols = ["experiment_kind", "dataset", "strategy", "topk", "seed", "max_queries"]
    ordered = _ordered_rows(rows, key_cols)

    live_csv = output_dir / "results_live.csv"
    live_txt = output_dir / "results_live.txt"
    _write_csv(live_csv, ordered)
    _write_text_table(live_txt, ordered)

    print(f"[saved] live csv -> {live_csv}", flush=True)
    print(f"[saved] live txt -> {live_txt}", flush=True)
    return ordered


def save_final_summaries(rows: list[dict], output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = save_results_snapshot(rows, output_dir)
    if not summary:
        return summary

    baseline_scores: dict[str, list[float]] = defaultdict(list)
    baseline_times: dict[str, list[float]] = defaultdict(list)
    for row in summary:
        topk = _coerce_int(row.get("topk"))
        if row.get("strategy") == "sequential" and topk == 100:
            ndcg = _coerce_number(row.get("ndcg@10"))
            seconds_per_query = _coerce_number(row.get("seconds_per_query"))
            if ndcg is not None:
                baseline_scores[str(row.get("dataset"))].append(ndcg)
            if seconds_per_query is not None:
                baseline_times[str(row.get("dataset"))].append(seconds_per_query)

    enriched = []
    for row in summary:
        new_row = dict(row)
        dataset = str(row.get("dataset"))
        baseline_ndcg = _mean(baseline_scores.get(dataset, []))
        baseline_time = _mean(baseline_times.get(dataset, []))
        ndcg = _coerce_number(row.get("ndcg@10"))
        seconds_per_query = _coerce_number(row.get("seconds_per_query"))

        new_row["sequential_top100_ndcg@10"] = baseline_ndcg
        new_row["sequential_top100_seconds_per_query"] = baseline_time
        new_row["delta_vs_sequential_top100"] = (
            None if baseline_ndcg is None or ndcg is None else ndcg - baseline_ndcg
        )
        new_row["speedup_vs_sequential_top100"] = (
            None
            if baseline_time is None or seconds_per_query in (None, 0)
            else baseline_time / seconds_per_query
        )
        new_row["table2_note"] = (
            "Table 2 is full-query top-100; this run is subset-based unless max_queries is blank."
        )
        enriched.append(new_row)

    enriched = _ordered_rows(enriched, ["dataset", "experiment_kind", "topk", "strategy", "seed"])

    summary_csv = output_dir / "combined_summary.csv"
    summary_txt = output_dir / "combined_summary.txt"
    _write_csv(summary_csv, enriched)
    _write_text_table(summary_txt, enriched)
    print(f"[saved] summary csv -> {summary_csv}", flush=True)
    print(f"[saved] summary txt -> {summary_txt}", flush=True)

    _write_grouping_view(output_dir / "grouping_view.csv", enriched)
    _write_topk_view(output_dir / "topk_view.csv", enriched)
    return enriched


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _write_grouping_view(path: Path, rows: list[dict]) -> None:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    strategies = set()
    datasets = set()
    for row in rows:
        if row.get("experiment_kind") != "grouping":
            continue
        ndcg = _coerce_number(row.get("ndcg@10"))
        if ndcg is None:
            continue
        dataset = str(row.get("dataset"))
        strategy = str(row.get("strategy"))
        datasets.add(dataset)
        strategies.add(strategy)
        values[(dataset, strategy)].append(ndcg)

    ordered_strategies = sorted(strategies)
    view_rows = []
    for dataset in sorted(datasets):
        row = {"dataset": dataset}
        for strategy in ordered_strategies:
            row[strategy] = _mean(values.get((dataset, strategy), []))
        if {"sequential", "score_balanced"}.issubset(ordered_strategies):
            seq = row.get("sequential")
            balanced = row.get("score_balanced")
            row["score_balanced_minus_sequential"] = (
                None if seq is None or balanced is None else balanced - seq
            )
        view_rows.append(row)

    _write_csv(path, view_rows)
    print(f"[saved] grouping view -> {path}", flush=True)


def _write_topk_view(path: Path, rows: list[dict]) -> None:
    values: dict[tuple[str, int], list[float]] = defaultdict(list)
    datasets = set()
    topks = set()
    for row in rows:
        if row.get("experiment_kind") not in {"grouping", "topk"}:
            continue
        if row.get("strategy") != "sequential":
            continue
        topk = _coerce_int(row.get("topk"))
        ndcg = _coerce_number(row.get("ndcg@10"))
        if topk is None or ndcg is None:
            continue
        dataset = str(row.get("dataset"))
        datasets.add(dataset)
        topks.add(topk)
        values[(dataset, topk)].append(ndcg)

    ordered_topks = sorted(topks)
    view_rows = []
    for dataset in sorted(datasets):
        row = {"dataset": dataset}
        for topk in ordered_topks:
            row[str(topk)] = _mean(values.get((dataset, topk), []))
        view_rows.append(row)

    _write_csv(path, view_rows)
    print(f"[saved] topk view -> {path}", flush=True)


def write_row_json(row: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
