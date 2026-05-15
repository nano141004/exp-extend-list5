"""Dataset resolution and JSONL helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .runtime import ListT5Runtime


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    import jsonlines

    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, "w") as writer:
        writer.write_all(rows)


def dataset_path(dataset_name: str, config: ExperimentConfig, runtime: ListT5Runtime) -> Path:
    """Return a local JSONL path for a BEIR BM25 top-100 dataset."""

    data_dir = config.data_dir(runtime.project_root)
    data_dir.mkdir(parents=True, exist_ok=True)

    local_copy = data_dir / f"{dataset_name}.jsonl"
    bundled_copy = runtime.listt5_root / f"{dataset_name}.jsonl"

    if local_copy.exists():
        base_path = local_copy
    elif bundled_copy.exists():
        base_path = bundled_copy
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("Install huggingface_hub to download BEIR JSONL files.") from exc

        downloaded = hf_hub_download(
            repo_id=config.hf_dataset_repo,
            filename=f"{dataset_name}.jsonl",
            repo_type="dataset",
            local_dir=str(data_dir),
            local_dir_use_symlinks=False,
        )
        base_path = Path(downloaded)

    if config.max_queries is None:
        return base_path

    subset_path = data_dir / f"{dataset_name}.first{config.max_queries}.jsonl"
    if not subset_path.exists():
        rows = runtime.read_jsonl(str(base_path))[: config.max_queries]
        write_jsonl(subset_path, rows)
    return subset_path
