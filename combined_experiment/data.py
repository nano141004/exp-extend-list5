"""Dataset download, subset, and JSONL helpers."""

from __future__ import annotations

import json
from pathlib import Path

from .config import ExperimentConfig
from .runtime import ListT5Runtime


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def read_jsonl_count(path: str | Path, runtime: ListT5Runtime) -> int:
    return len(runtime.read_jsonl(str(path)))


def dataset_path(dataset_name: str, config: ExperimentConfig, runtime: ListT5Runtime) -> Path:
    data_dir = config.resolved_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    full_path = data_dir / f"{dataset_name}.jsonl"
    bundled_path = runtime.code_root / f"{dataset_name}.jsonl"

    if not full_path.exists():
        if bundled_path.exists():
            full_path = bundled_path
        else:
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(
                repo_id=config.hf_dataset_repo,
                filename=f"{dataset_name}.jsonl",
                repo_type="dataset",
                local_dir=str(data_dir),
            )
            full_path = Path(downloaded)

    if config.max_queries is None:
        return full_path

    subset_tag = f"maxq{config.max_queries}"
    subset_path = data_dir / subset_tag / f"{dataset_name}.jsonl"
    if not subset_path.exists():
        rows = runtime.read_jsonl(str(full_path))[: config.max_queries]
        write_jsonl(subset_path, rows)
        print(f"[data] wrote {len(rows)} rows -> {subset_path}", flush=True)
    else:
        print(f"[data] reuse subset -> {subset_path}", flush=True)
    return subset_path


def output_is_complete(output_path: str | Path, input_path: str | Path, runtime: ListT5Runtime) -> bool:
    output_path = Path(output_path)
    if not output_path.exists():
        return False
    try:
        return read_jsonl_count(output_path, runtime) == read_jsonl_count(input_path, runtime)
    except Exception:
        return False
