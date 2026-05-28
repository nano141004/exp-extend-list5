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

        def _sequential_chunks(self, values, n):
            values = list(values)
            return [values[i : i + n] for i in range(0, len(values), n)]

        def _finalize_loop_fallback(self, question, topk_ctxs, full_list_idx):
            unique_idx = self.remove_duplicates(list(full_list_idx))
            if len(unique_idx) == 0:
                return full_list_idx[0]
            if len(unique_idx) == 1:
                return unique_idx[0]
            if len(unique_idx) < self.args.listwise_k:
                other_index = self.get_leftover_idx(
                    unique_idx,
                    self.args.listwise_k - len(unique_idx),
                    list(range(len(topk_ctxs))),
                )
                return self.get_out_k(question, topk_ctxs, unique_idx + other_index)[-1]
            return self.get_out_k(question, topk_ctxs, unique_idx[: self.args.listwise_k])[-1]

        def run_one_loop(
            self,
            question,
            topk_ctxs,
            full_list_idx,
            _depth=0,
            _seen=None,
            _force_sequential=False,
        ):
            if _seen is None:
                _seen = set()

            state = (tuple(full_list_idx), bool(_force_sequential))
            if state in _seen or _depth >= 25:
                if not _force_sequential:
                    print(
                        "[recursion guard] switching recursive aggregation to sequential grouping",
                        flush=True,
                    )
                    return self.run_one_loop(
                        question,
                        topk_ctxs,
                        self.remove_duplicates(list(full_list_idx)),
                        _depth=0,
                        _seen=set(),
                        _force_sequential=True,
                    )
                print("[recursion guard] using final duplicate-safe fallback", flush=True)
                return self._finalize_loop_fallback(question, topk_ctxs, full_list_idx)
            _seen.add(state)

            saved_index = []
            if (self.args.out_k * 2) > self.args.listwise_k:
                full_list_idx = self.remove_duplicates(full_list_idx)

            if _force_sequential:
                grouped_list_idxs = self._sequential_chunks(full_list_idx, self.args.listwise_k)
            else:
                grouped_list_idxs = list(self.group2chunks(full_list_idx, n=self.args.listwise_k))

            for cut_list in grouped_list_idxs:
                if len(cut_list) < self.args.listwise_k:
                    other_index = self.get_leftover_idx(
                        cut_list,
                        self.args.listwise_k - len(cut_list),
                        full_list_idx,
                    )
                    saved_index += self.get_out_k(question, topk_ctxs, cut_list + other_index)
                else:
                    if len(set(cut_list)) == 1:
                        saved_index.append(cut_list[0])
                    else:
                        saved_index += self.get_out_k(question, topk_ctxs, cut_list)

            if len(saved_index) < self.args.listwise_k:
                other_index = self.get_leftover_idx(
                    saved_index,
                    self.args.listwise_k - len(saved_index),
                    full_list_idx,
                )
                full_index = saved_index + other_index
                topk_out = self.get_out_k(question, topk_ctxs, full_index)
                return topk_out[-1]
            if len(saved_index) > self.args.listwise_k:
                next_index = saved_index
                if _force_sequential:
                    next_index = self.remove_duplicates(list(saved_index))
                return self.run_one_loop(
                    question,
                    topk_ctxs,
                    next_index,
                    _depth=_depth + 1,
                    _seen=_seen,
                    _force_sequential=_force_sequential,
                )
            if len(saved_index) == 1:
                return saved_index[0]
            return self.get_out_k(question, topk_ctxs, saved_index)[-1]

        def run_batchwise_caching(self, batch_holder):
            """Precompute first-round cache using the active grouping strategy."""
            try:
                from tqdm import tqdm
            except ImportError:
                tqdm = lambda values: values

            initial_groups = list(self.group2chunks(list(range(self.args.topk)), self.args.listwise_k))
            print(
                f"[cache] strategy={getattr(self.args, 'grouping_strategy', 'sequential')} "
                f"precomputing {len(initial_groups)} first-round groups",
                flush=True,
            )

            for group in tqdm(initial_groups):
                # get_out_k sorts candidate ids before inference, so the batched
                # precompute must use the same canonical order and cache key.
                cand_def_ids = tuple(sorted(group))
                if len(cand_def_ids) != self.args.listwise_k:
                    continue

                questions = [x["question"] for x in batch_holder]
                topk_ctxs = [[x["topk_ctxs"][idx] for idx in cand_def_ids] for x in batch_holder]

                full_input_texts_batchwise = [
                    self.make_listwise_text(q, c) for q, c in zip(questions, topk_ctxs)
                ]
                if (
                    not full_input_texts_batchwise
                    or len(full_input_texts_batchwise[0]) != self.args.listwise_k
                ):
                    continue

                raw_tensors_batchwise = [
                    self.tok(
                        texts,
                        padding=self.args.padding,
                        return_tensors="pt",
                        max_length=self.args.max_input_length,
                        truncation=True,
                    )
                    for texts in full_input_texts_batchwise
                ]
                batch_inputids = runtime.torch.stack(
                    [x["input_ids"] for x in raw_tensors_batchwise]
                ).to("cuda")
                batch_attnmasks = runtime.torch.stack(
                    [x["attention_mask"] for x in raw_tensors_batchwise]
                ).to("cuda")

                output = self.run_inference(
                    {"input_ids": batch_inputids, "attention_mask": batch_attnmasks}
                )
                del batch_inputids
                del batch_attnmasks

                batch_best_rel_ids = [[x - 1 for x in topk] for topk in self.get_rel_index(output)]
                batch_best_def_ids = [[cand_def_ids[x] for x in y] for y in batch_best_rel_ids]

                for i in range(len(batch_holder)):
                    batch_holder[i]["best_cache"][tuple(set(cand_def_ids))] = batch_best_def_ids[i]

            return batch_holder

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
