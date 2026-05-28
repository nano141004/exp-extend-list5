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

        def get_leftover_idx(self, exclude, k, full_list):
            """Same logic as the base class but without the spammy prints."""
            out = []
            i = 0
            exclude = list(set(exclude + self.global_exclude))
            allow_exclude = set(full_list) - set(exclude) == set()
            while len(out) != k:
                if i == len(full_list):
                    i = 0
                if allow_exclude or (full_list[i] not in exclude):
                    out.append(full_list[i])
                i += 1
            return out

        def get_out_k(self, question, full_ctxs, index, use_cache=True, k=-1):
            """Fix: resolve k before the all-same early return.

            The original code does ``return index[:k]`` while k is still -1,
            which evaluates to ``index[:-1]`` (N-1 items) instead of the
            intended ``index[:out_k]`` (2 items).  With score_balanced
            grouping this causes saved_index to never shrink, leading to
            infinite recursion in run_one_loop.
            """
            if k == -1:
                k = self.args.out_k
            if len(set(index)) == 1:
                return index[:k]
            index.sort()
            if use_cache and self.best_cache.get(tuple(set(index))) is not None:
                return self.best_cache.get(tuple(set(index)))[-k:]
            ctxs = [full_ctxs[x] for x in index]
            full_input_texts = self.make_listwise_text(question, ctxs)
            input_tensors = self.make_input_tensors(full_input_texts)
            output = self.run_inference(input_tensors)
            out_k_rel_index = self.get_rel_index(output, k=k)[0]
            try:
                out_k_def_index = [index[x - 1] for x in out_k_rel_index]
            except IndexError:
                out_k_def_index = index[-k:]
            self.best_cache[tuple(set(index))] = out_k_def_index
            return out_k_def_index

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
