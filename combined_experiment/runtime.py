"""Runtime imports and compatibility patches for the original ListT5 code."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ListT5Runtime:
    code_root: Path
    torch: object
    fid_module: object
    ListT5Evaluator: type
    read_jsonl: object
    run_rerank_eval: object
    beir_length_mapping: dict


def prepare_runtime(code_root: Path) -> ListT5Runtime:
    code_root = code_root.resolve()
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))

    import torch
    import FiDT5 as fid_module
    from beir_eval import run_rerank_eval
    from beir_length_mapping import BEIR_LENGTH_MAPPING
    from run_listt5 import ListT5Evaluator, read_jsonl

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

    return ListT5Runtime(
        code_root=code_root,
        torch=torch,
        fid_module=fid_module,
        ListT5Evaluator=ListT5Evaluator,
        read_jsonl=read_jsonl,
        run_rerank_eval=run_rerank_eval,
        beir_length_mapping=BEIR_LENGTH_MAPPING,
    )
