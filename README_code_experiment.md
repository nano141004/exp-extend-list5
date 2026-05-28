# ListT5 Combined Experiment Code

This is the Python codebase version of:

```text
notebook/listt5_combined_grouping_topk_experiment.ipynb
```

It keeps the original ListT5 reranking pipeline intact and exposes only the two
experiment knobs from the notebook:

1. grouping method at BM25 top-100
2. BM25 candidate budget, also called top-k play

No training is done. The code imports the original `ListT5/` implementation and
subclasses the official evaluator only to swap grouping, set top-k, add progress
prints, and keep compatibility with newer Kaggle/Transformers environments.

## Default Experiment

The default CLI matches the current combined notebook:

```text
datasets          nfcorpus, scifact, arguana, scidocs, fiqa
max_queries       50
model             Soyoung97/ListT5-base
max_input_length  512 for every dataset
batch_size        20
listwise_k        5
out_k             2
rerank_topk       10
```

Default jobs per dataset:

```text
grouping experiment:
  sequential      topk=100
  score_balanced  topk=100

top-k sweep:
  sequential      topk=40
  sequential      topk=60
  sequential      topk=80
```

The top-k sweep applies only to the methods named by `--topk-methods`. By
default that is `sequential`. To later test score-balanced at smaller BM25
candidate budgets, pass:

```bash
python run.py run --topk-methods sequential score_balanced
```

## Folder Structure

```text
Project/ListT5/
  run.py
  README_code_experiment.md

  combined_experiment/
    __init__.py
    cli.py
    config.py
    constants.py
    data.py
    evaluator.py
    grouping.py
    jobs.py
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

The entry point is `run.py`.

## Environment Setup

This experiment is inference-only, so use the lightweight experiment
requirements:

```text
requirements_experiment.txt
```

Do not use `ListT5/requirements.txt` unless you are reproducing the original
training environment. It pins older packages and includes training-only
dependencies that are unnecessary for this code.

### Option A: uv, Lightweight Local Setup

This is the cleanest local setup when `uv` is available:

```bash
cd Project/ListT5
uv venv .venv
.venv\Scripts\activate
uv pip install -r requirements_experiment.txt
python run.py show-plan
```

On Linux/Kaggle-style shells:

```bash
cd Project/ListT5
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements_experiment.txt
python run.py show-plan
```

### Option B: Standard venv + pip

Use this if `uv` is not installed:

```bash
cd Project/ListT5
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements_experiment.txt
python run.py show-plan
```

On Linux/Kaggle-style shells:

```bash
cd Project/ListT5
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements_experiment.txt
python run.py show-plan
```

### Option C: No venv

This is acceptable on Kaggle or a disposable environment:

```bash
cd Project/ListT5
python -m pip install -r requirements_experiment.txt
python run.py show-plan
```

If `torch` and `transformers` are already installed by the runtime, keep them.
Do not downgrade them just for this experiment.

## Install on Kaggle

Run this in the notebook/session before launching the CLI:

```bash
pip install -q -r requirements_experiment.txt
```

Do not pin old `transformers` on Python 3.12 Kaggle kernels. This code uses the
installed version and patches the ListT5 compatibility issues at runtime.

## Basic Commands

Run from:

```text
Project/ListT5
```

Show the default plan without running inference:

```bash
python run.py show-plan
```

Run the default combined experiment:

```bash
python run.py run
```

Run only two datasets:

```bash
python run.py run --datasets nfcorpus scifact
```

Run only the grouping comparison:

```bash
python run.py run --disable-topk
```

Run only the top-k sweep:

```bash
python run.py run --disable-grouping
```

Use full query sets instead of the first 50 queries:

```bash
python run.py run --max-queries none
```

Subset runs are useful for sanity checks and runtime control. Do not treat their
Table 2 deltas as exact reproduction results because Table 2 is full-query,
BM25 top-100.

## Custom Grouping and Top-K

Grouping methods currently available:

```text
sequential       official contiguous grouping
score_balanced   round-robin over BM25 rank positions
random           seeded random grouping baseline
```

Choose grouping methods for the top-100 grouping experiment:

```bash
python run.py run \
  --grouping-methods sequential score_balanced random \
  --seeds 0 1 2
```

Choose the BM25 top-k values for the candidate-budget sweep:

```bash
python run.py run --topk-values 20 40 60 80
```

Choose which grouping method the top-k sweep applies to:

```bash
python run.py run \
  --topk-methods sequential score_balanced \
  --topk-values 40 60 80
```

This means:

```text
top-k sweep = every method in --topk-methods crossed with every value in --topk-values
```

The grouping experiment remains controlled separately by:

```text
--grouping-methods
--grouping-topk
```

## Two-GPU Kaggle Mode

Single GPU is the default:

```bash
python run.py run --gpu-mode single
```

For two T4 GPUs:

```bash
python run.py run --gpu-mode parallel_2gpu --gpu-ids 0 1
```

This does not split one query across GPUs. It runs independent jobs in separate
processes:

```text
GPU 0 -> one dataset / strategy / top-k job
GPU 1 -> another dataset / strategy / top-k job
```

If two-GPU mode fails, rerun with `--gpu-mode single`. Completed JSONL outputs
are reused automatically.

## Output Files

Default output directory:

```text
outputs/combined_grouping_topk/
```

Reranked output JSONL:

```text
outputs/combined_grouping_topk/{experiment_kind}/{strategy}/topk{topk}/seed{seed}/{subset}/{dataset}_output.jsonl
```

Live snapshots after every completed job:

```text
outputs/combined_grouping_topk/results_live.csv
outputs/combined_grouping_topk/results_live.txt
```

Final summaries:

```text
outputs/combined_grouping_topk/combined_summary.csv
outputs/combined_grouping_topk/combined_summary.txt
outputs/combined_grouping_topk/grouping_view.csv
outputs/combined_grouping_topk/topk_view.csv
```

## Caching and Reuse

The original ListT5 in-run cache is still used because the code keeps the
official tournament methods:

```python
self.best_cache[tuple(set(index))]
```

The code also fixes the first-round precompute cache for custom grouping.
Official ListT5 batches and caches the first round before the tournament starts,
but the original implementation hardcodes sequential groups. This package
precomputes:

```python
list(self.group2chunks(list(range(topk)), listwise_k))
```

That means `score_balanced` caches groups such as `{0,20,40,60,80}` before
`run_one_loop()` asks for them, instead of falling back to slow online forwards.

The evaluator also fixes a small original ListT5 edge case in `get_out_k`.
When all candidate indices are the same, the original method returned
`index[:k]` before resolving `k=-1` to `out_k`, which can return `index[:-1]`
and prevent recursive aggregation from shrinking on duplicate-heavy NFCorpus
queries.

The code also has output-level reuse. If the output JSONL already exists and has
the same row count as the input JSONL, inference is skipped and only BEIR metric
evaluation is recomputed.

Disable output reuse with:

```bash
python run.py run --no-reuse-existing
```

For the same one-job cache speed check as the v2 notebook, use a fresh output
directory:

```bash
python run.py run \
  --datasets nfcorpus \
  --disable-topk \
  --grouping-methods score_balanced \
  --output-dir outputs/combined_grouping_topk_v2_cli \
  --no-reuse-existing
```

## Common Kaggle Commands

```bash
cd /kaggle/working/ListT5
pip install -q -r requirements_experiment.txt
python run.py show-plan
python run.py run --datasets nfcorpus scifact
```

If the original ListT5 code is somewhere else:

```bash
python run.py run --listt5-code-root /kaggle/working/ListT5/ListT5
```

If this repository is copied somewhere else:

```bash
python run.py run --project-root /kaggle/working/ListT5
```

## Interpretation

For the default setup, the clean report is:

```text
We keep model, ListT5 tournament reranking, BEIR evaluation, and inference
settings fixed. We vary either the grouping policy at BM25 top-100 or the BM25
candidate budget for selected grouping policies.
```

`score_balanced` is deterministic. The `--seeds` argument matters mainly for
`random`, or for keeping output paths explicit when you intentionally repeat
runs.
