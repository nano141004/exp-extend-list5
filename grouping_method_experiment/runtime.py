"""Runtime preparation for importing the original ListT5 code safely."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib
import sys
from types import ModuleType
from typing import Any

from .paths import resolve_listt5_code_root, resolve_project_root


@dataclass(frozen=True)
class ListT5Runtime:
    project_root: Path
    listt5_root: Path
    torch: ModuleType
    fid_module: ModuleType
    ListT5Evaluator: type
    read_jsonl: Any
    run_rerank_eval: Any
    BEIR_LENGTH_MAPPING: dict[str, int]


def prepare_runtime(project_root: str | Path | None = None) -> ListT5Runtime:
    """Resolve paths, import ListT5 modules, and patch version compatibility."""

    resolved_project_root = resolve_project_root(project_root)
    listt5_root = resolve_listt5_code_root(resolved_project_root)
    if str(listt5_root) not in sys.path:
        sys.path.insert(0, str(listt5_root))

    torch = importlib.import_module("torch")
    fid_module = importlib.import_module("FiDT5")
    run_listt5 = importlib.import_module("run_listt5")
    beir_eval = importlib.import_module("beir_eval")
    beir_length_mapping = importlib.import_module("beir_length_mapping")

    _patch_transformers_compat(torch, fid_module)

    return ListT5Runtime(
        project_root=resolved_project_root,
        listt5_root=listt5_root,
        torch=torch,
        fid_module=fid_module,
        ListT5Evaluator=run_listt5.ListT5Evaluator,
        read_jsonl=run_listt5.read_jsonl,
        run_rerank_eval=beir_eval.run_rerank_eval,
        BEIR_LENGTH_MAPPING=beir_length_mapping.BEIR_LENGTH_MAPPING,
    )


def _patch_transformers_compat(torch: ModuleType, fid_module: ModuleType) -> None:
    """Patch the original ListT5 classes for newer transformers versions."""

    if not hasattr(fid_module.EncoderWrapper, "embed_tokens"):
        fid_module.EncoderWrapper.embed_tokens = property(lambda self: self.encoder.embed_tokens)

    def checkpoint_wrapper_forward_compat(self, *args, **kwargs):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                lambda *inner_args: self.module(*inner_args, **kwargs),
                *args,
            )
        return self.module(*args, **kwargs)

    fid_module.CheckpointWrapper.forward = checkpoint_wrapper_forward_compat
