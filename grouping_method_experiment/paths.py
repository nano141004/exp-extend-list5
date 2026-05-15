"""Path discovery helpers."""

from __future__ import annotations

from pathlib import Path


def resolve_project_root(start: str | Path | None = None) -> Path:
    """Find the ListT5 repository root used by this experiment.

    The experiment root is the directory that contains the official code folder
    `ListT5/run_listt5.py`. In this workspace that is `Project/ListT5`.
    """

    start_path = Path(start).expanduser().resolve() if start else Path.cwd().resolve()
    candidates: list[Path] = [start_path, *start_path.parents]

    # Common when running from the parent `Project` directory.
    candidates.extend([start_path / "ListT5", start_path / "Project" / "ListT5"])

    for candidate in candidates:
        if (candidate / "ListT5" / "run_listt5.py").exists():
            return candidate
        if candidate.name == "ListT5" and (candidate / "run_listt5.py").exists():
            return candidate.parent

    raise FileNotFoundError(
        "Could not locate the ListT5 experiment root. Run from Project/ListT5 "
        "or pass --project-root explicitly."
    )


def resolve_listt5_code_root(project_root: Path) -> Path:
    """Return the directory that contains run_listt5.py."""

    nested = project_root / "ListT5"
    if (nested / "run_listt5.py").exists():
        return nested
    if (project_root / "run_listt5.py").exists():
        return project_root
    raise FileNotFoundError(f"Could not find run_listt5.py under {project_root}")
