"""Path resolution for local and Kaggle runs."""

from __future__ import annotations

import subprocess
from pathlib import Path


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_project_root(project_root: str | Path | None = None) -> Path:
    if project_root is None:
        return default_project_root()
    return Path(project_root).expanduser().resolve()


def resolve_listt5_code_root(
    project_root: Path,
    listt5_code_root: str | Path | None = None,
    clone_if_missing: bool = True,
) -> Path:
    if listt5_code_root is not None:
        root = Path(listt5_code_root).expanduser().resolve()
        if not (root / "run_listt5.py").exists():
            raise FileNotFoundError(f"run_listt5.py not found in --listt5-code-root: {root}")
        return root

    candidates = [
        project_root / "ListT5",
        project_root,
    ]
    for candidate in candidates:
        if (candidate / "run_listt5.py").exists():
            return candidate.resolve()

    target = project_root / "ListT5"
    if not clone_if_missing:
        raise FileNotFoundError(f"Could not find run_listt5.py under {project_root}")

    print(f"[setup] cloning official ListT5 code into {target}", flush=True)
    subprocess.run(
        ["git", "clone", "https://github.com/soyoung97/ListT5.git", str(target)],
        check=True,
    )
    return target.resolve()
