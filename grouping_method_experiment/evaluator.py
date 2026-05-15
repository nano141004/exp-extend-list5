"""ListT5 evaluator integration for grouping-method experiments."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import time

from .config import ExperimentConfig, ExperimentJob
from .data import dataset_path
from .grouping import group_items
from .runtime import ListT5Runtime


def build_grouping_evaluator(runtime: ListT5Runtime):
    """Create a subclass of the official evaluator with grouping override."""

    class GroupingListT5Evaluator(runtime.ListT5Evaluator):
        def load_model(self):
            start = time.time()
            print("Loading model..", flush=True)
            print(f"Loading fid model from {self.args.model_path}", flush=True)
            model = runtime.fid_module.FiDT5.from_pretrained(
                self.args.model_path,
                use_safetensors=False,
            ).to("cuda")
            model.eval()
            print(f"Done! took {time.time() - start} second", flush=True)
            return model

        def group2chunks(self, l, n=5):
            strategy = getattr(self.args, "grouping_strategy", "sequential")
            seed = getattr(self.args, "seed", 0)
            yield from group_items(strategy, l, n, seed=seed)

        def run_inference(self, input_tensors):
            output = self.model.generate(
                **input_tensors,
                max_length=self.args.max_gen_length,
                return_dict_in_generate=True,
                output_scores=True,
            )
            self.num_forward += 1
            print_every = getattr(self.args, "print_every_forwards", 20)
            if print_every and self.num_forward % print_every == 0:
                print(f"[progress] forward_calls={self.num_forward}", flush=True)
            return output

    return GroupingListT5Evaluator


def make_args(job: ExperimentJob, config: ExperimentConfig, runtime: ListT5Runtime) -> SimpleNamespace:
    """Build the args namespace expected by the original ListT5 evaluator."""

    input_path = dataset_path(job.dataset, config, runtime)
    output_path = (
        config.output_dir(runtime.project_root)
        / job.strategy
        / f"seed{job.seed}"
        / config.subset_tag()
        / f"{job.dataset}_output.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    max_input_length = runtime.BEIR_LENGTH_MAPPING.get(job.dataset)
    if max_input_length is None:
        raise ValueError(
            f"No max input length for dataset '{job.dataset}'. Add it to "
            "BEIR_LENGTH_MAPPING or choose a known BEIR dataset."
        )

    return SimpleNamespace(
        firststage_result_key="bm25_results",
        docid_key="docid",
        pid_key="pid",
        qrels_key="qrels",
        score_key="bm25_score",
        question_text_key="q_text",
        text_key="text",
        title_key="title",
        model_path=config.model_path,
        topk=config.topk,
        max_input_length=max_input_length,
        padding="max_length",
        listwise_k=config.listwise_k,
        rerank_topk=config.rerank_topk,
        out_k=config.out_k,
        dummy_number=21,
        verbose=False,
        seed=job.seed,
        bsize=config.batch_size,
        input_path=str(input_path),
        output_path=str(output_path),
        measure_flops=False,
        skip_no_candidate=False,
        skip_issubset=False,
        max_gen_length=config.listwise_k + 2,
        grouping_strategy=job.strategy,
        print_every_forwards=config.print_every_forwards,
    )


def output_path_for(job: ExperimentJob, config: ExperimentConfig, runtime: ListT5Runtime) -> Path:
    return Path(make_args(job, config, runtime).output_path)
