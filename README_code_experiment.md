# ListT5 Grouping Method Experiment Code

This is the Python codebase version of the notebook:

```text
notebook/listt5_grouping_method_experiment.ipynb
```

The goal is the same as the notebook: run a small inference-only experiment that
changes only the ListT5 tournament grouping method and keeps the rest of the
official ListT5 pipeline unchanged.

## 1. What This Code Tests

The original ListT5 tournament sort uses sequential grouping:

```text
[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], ...
```

This code compares that official grouping with alternative grouping strategies:

```text
sequential      official contiguous grouping
score_balanced  round-robin grouping by first-stage BM25 rank
random          seeded random grouping baseline
```

Everything else stays fixed:

```text
model           Soyoung97/ListT5-base
input           BEIR BM25 top-100 JSONL
listwise_k      5
out_k           2
topk            100
rerank_topk     10
metric          NDCG@10 through the original beir_eval.py
training        none
```

## 2. Folder Structure

```text
Project/ListT5/
  run_grouping_experiment.py
  README_code_experiment.md

  grouping_method_experiment/
    __init__.py
    cli.py
    config.py
    constants.py
    data.py
    evaluator.py
    grouping.py
    parallel.py
    paths.py
    results.py
    runner.py
    runtime.py

  ListT5/
    run_listt5.py
    FiDT5.py
    beir_eval.py
    beir_length_mapping.py
```

The experiment code is separated from the original ListT5 code. It imports the
official evaluator and subclasses it instead of editing `run_listt5.py`.

## 3. Module Responsibilities

`config.py`

Defines `ExperimentConfig` and `ExperimentJob`. These hold all experiment knobs,
such as datasets, strategies, seeds, GPU mode, output directory, and ListT5
parameters.

`constants.py`

Stores default datasets/strategies and the paper Table 2 ListT5-base BM25
top-100 NDCG@10 values.

`paths.py`

Finds the experiment root and the original ListT5 code root. This lets the code
work when launched from `Project/ListT5`, the parent `Project` folder, or with
`--project-root`.

`runtime.py`

Imports the original ListT5 modules only at runtime and applies compatibility
patches for newer Kaggle/transformers environments:

```text
EncoderWrapper.embed_tokens
CheckpointWrapper.forward
```

`grouping.py`

Contains the grouping policies:

```text
sequential_groups
score_balanced_groups
random_groups
```

This is the core experimental variable.

`data.py`

Resolves dataset JSONL files. It checks:

```text
data/beir-eval-bm25-top100/{dataset}.jsonl
ListT5/{dataset}.jsonl
Soyoung97/beir-eval-bm25-top100 on Hugging Face
```

It also creates `.firstN.jsonl` subset files when `--max-queries` is used.

`evaluator.py`

Builds `GroupingListT5Evaluator`, a subclass of the official
`ListT5Evaluator`. The subclass overrides only:

```text
load_model       uses use_safetensors=False
group2chunks     swaps the grouping strategy
run_inference    adds plain print progress
```

`runner.py`

Runs the single-GPU grid and one-job execution path.

`parallel.py`

Runs independent jobs in subprocesses for two-GPU Kaggle sessions. Each
subprocess gets a GPU through `CUDA_VISIBLE_DEVICES`.

`results.py`

Writes live results and final summary tables.

`cli.py`

Defines the command-line interface.

## 4. Install Requirements

On Kaggle, run:

```bash
pip install -q pandas jsonlines sentencepiece huggingface_hub beir
```

Do not pin an old `transformers` version on Python 3.12 Kaggle kernels. The
code uses the installed `transformers` and patches the ListT5 compatibility
issues at runtime.

## 5. Basic Commands

Run from:

```text
Project/ListT5
```

Show available Table 2 dataset names:

```bash
python run_grouping_experiment.py show-datasets
```

Run one full Table 2 sanity check on TREC-COVID. Omit `--max-queries` for the
full run:

```bash
python run_grouping_experiment.py run \
  --datasets trec-covid \
  --strategies sequential
```

Run the first grouping comparison:

```bash
python run_grouping_experiment.py run \
  --datasets trec-covid \
  --strategies sequential score_balanced
```

Run a quick smoke test:

```bash
python run_grouping_experiment.py run \
  --datasets trec-covid \
  --strategies sequential score_balanced \
  --max-queries 5
```

Subset runs are for debugging only. Do not compare subset metrics to Table 2.

## 6. Two-GPU Kaggle Mode

For two T4 GPUs:

```bash
python run_grouping_experiment.py run \
  --datasets trec-covid nfcorpus fiqa scifact arguana \
  --strategies sequential score_balanced random \
  --gpu-mode parallel_2gpu \
  --gpu-ids 0 1
```

This does not split one query across two GPUs. It runs independent jobs in
parallel:

```text
GPU 0 -> one dataset/strategy/seed job
GPU 1 -> another dataset/strategy/seed job
```

If two-GPU mode fails, rerun with:

```bash
--gpu-mode single
```

The completed output files are reused automatically, so already finished jobs
do not need to run inference again.

## 7. Kaggle Notebook Usage

Use this after uploading this modified `Project/ListT5` folder to Kaggle, or
after cloning a fork that already contains `grouping_method_experiment/` and
`run_grouping_experiment.py`.

```python
%cd /kaggle/working/ListT5
!pip install -q pandas jsonlines sentencepiece huggingface_hub beir
```

Then run:

```python
!python run_grouping_experiment.py run --datasets trec-covid --strategies sequential score_balanced
```

For two GPUs:

```python
!python run_grouping_experiment.py run \
  --datasets trec-covid nfcorpus fiqa scifact arguana \
  --strategies sequential score_balanced random \
  --gpu-mode parallel_2gpu \
  --gpu-ids 0 1
```

If the code is copied somewhere else, pass:

```bash
--project-root /kaggle/working/ListT5
```

## 8. Output Files

The default output directory is:

```text
outputs/grouping_method_experiment/
```

Each approach writes reranked output JSONL here:

```text
outputs/grouping_method_experiment/{strategy}/seed{seed}/{subset_tag}/{dataset}_output.jsonl
```

Live result snapshots are saved after every completed approach:

```text
outputs/grouping_method_experiment/results_live.csv
outputs/grouping_method_experiment/results_live.txt
```

Final summary files:

```text
outputs/grouping_method_experiment/baseline_check.csv
outputs/grouping_method_experiment/baseline_check.txt
outputs/grouping_method_experiment/grouping_comparison.csv
outputs/grouping_method_experiment/grouping_comparison.txt
```

`baseline_check` is for reproducing Table 2 with `sequential`.

`grouping_comparison` is for comparing grouping methods directly. The most
important columns are:

```text
score_balanced_minus_sequential
random_minus_sequential
```

## 9. Caching and Reuse

There are two layers of reuse.

The first layer is the original ListT5 in-run cache:

```python
self.best_cache[tuple(set(index))]
```

This is still used because the experiment keeps `run_tournament_sort`,
`run_one_loop`, `get_out_k`, and `run_batchwise_caching` from the official
evaluator.

The second layer is output-level reuse. Before running inference, the code
checks whether the output JSONL already exists and has the same row count as the
input JSONL. If yes, it skips model inference and reruns only BEIR evaluation.

Disable output reuse with:

```bash
--no-reuse-existing
```

## 10. Reporting Interpretation

Use this order:

1. Run `sequential` fully on at least one dataset.
2. Check `baseline_check.csv`; the value should be close to Table 2.
3. Run `score_balanced` on the same dataset.
4. Report `score_balanced_minus_sequential`.
5. If reporting `random`, use multiple seeds:

```bash
--strategies random --seeds 0 1 2
```

The clean claim is:

```text
Only the grouping phase changed. Model, data, tournament sort, caching, and metric evaluation stayed fixed.
```

## 11. Common Errors

`ModuleNotFoundError: jsonlines`

Run:

```bash
pip install -q jsonlines sentencepiece huggingface_hub beir
```

`Failed building wheel for tokenizers`

Avoid pinning old `transformers` versions on Kaggle Python 3.12.

`EncoderWrapper has no attribute embed_tokens`

The code patches this in `runtime.py`. Restart the kernel/process and rerun.

`CheckpointWrapper.forward() takes 4 positional arguments but 7 were given`

The code patches this in `runtime.py`. Restart the kernel/process and rerun.

`KeyError: 'event_id'`

Usually a Kaggle/Jupyter progress display issue. The code prints normal progress
lines, so check the log output for `[progress]`, `[job start]`, and `[job done]`.
