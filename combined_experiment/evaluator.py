"""Thin subclass around the official ListT5 evaluator."""

from __future__ import annotations

import time

from .grouping import GROUPING_POLICIES
from .runtime import ListT5Runtime


def make_combined_evaluator_class(runtime: ListT5Runtime, default_print_every: int):
    class CombinedListT5Evaluator(runtime.ListT5Evaluator):
        def load_model(self):
            start = time.time()
            print("Loading model..", flush=True)
            print(f"Loading fid model from {self.args.model_path}", flush=True)
            model = runtime.fid_module.FiDT5.from_pretrained(
                self.args.model_path,
                use_safetensors=False,
            ).to("cuda")
            model.eval()
            print(f"Done! took {time.time() - start:.2f} seconds", flush=True)
            return model

        def group2chunks(self, l, n=5):
            strategy = getattr(self.args, "grouping_strategy", "sequential")
            seed = getattr(self.args, "seed", 0)
            if strategy not in GROUPING_POLICIES:
                raise ValueError(f"Unknown grouping strategy: {strategy}")
            yield from GROUPING_POLICIES[strategy](l, n, seed=seed)

        def run_inference(self, input_tensors):
            with runtime.torch.inference_mode():
                output = self.model.generate(
                    **input_tensors,
                    max_length=self.args.max_gen_length,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            self.num_forward += 1
            print_every = getattr(self.args, "print_every_forwards", default_print_every)
            if print_every and self.num_forward % print_every == 0:
                print(f"[progress] forward_calls={self.num_forward}", flush=True)
            return output

    return CombinedListT5Evaluator
