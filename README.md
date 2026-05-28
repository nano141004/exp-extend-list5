# ListT5 Research Extension

This repository contains an Information Retrieval course project extending the
inference procedure of ListT5. The project studies whether candidate grouping
and BM25 candidate budget affect ListT5 tournament reranking quality and
inference cost.

## Group Members

- Mariano Gerardus Senduk (2206814236)
- Narendra Dzulqarnain (2206081881)

## Credit

This project builds on the original ListT5 work:

> Soyoung Yoon, Eunbi Choi, Jiyeon Kim, Hyeongu Yun, Yireun Kim, and Seung-won Hwang. 2024. **ListT5: Listwise Reranking with Fusion-in-Decoder Improves Zero-shot Retrieval**. ACL 2024.

- Original paper: <https://aclanthology.org/2024.acl-long.125/>
- DOI: <https://doi.org/10.18653/v1/2024.acl-long.125>
- Official source code: <https://github.com/soyoung97/ListT5>

The ListT5 model, checkpoints, and base implementation are credited to the
original authors. This project contributes an inference-only experiment around
candidate budget, grouping strategy, and tournament-cache behavior.

## Project Summary

ListT5 is a listwise reranker for information retrieval. A first-stage retriever
such as BM25 retrieves candidate documents, then ListT5 reranks those candidates
by processing small lists of query-document inputs with a Fusion-in-Decoder
T5 model.

Because ListT5 ranks only a small candidate group per forward call, the original
implementation uses tournament sorting for larger candidate sets such as BM25
top-100. Candidates are split into groups of five, ListT5 selects candidates
from each group, and the tournament continues until the final top-ranked
documents are produced.

This project keeps the pretrained ListT5 model and evaluation pipeline fixed.
The experiment changes only inference-time behavior:

- BM25 candidate budget: sequential ListT5 with top-40, top-60, top-80, and top-100.
- Grouping method: official sequential grouping versus score-balanced grouping at top-100.
- Cache behavior: original sequential first-round cache versus strategy-aware first-round cache.

No model training or fine-tuning is performed in this repository.

## Repository Structure

```text
Project/ListT5/
  README.md
  README_code_experiment.md
  requirements_experiment.txt
  run.py

  combined_experiment/
    Python CLI/package version of the experiment.

  notebook/
    listt5_combined_grouping_topk_experiment.ipynb
    listt5_combined_grouping_topk_experiment_v2.ipynb
    progress/

  experiment-results/
    final_compiled-results.md
    raw/

  tex/
    LaTeX report source.

  IR_Project-paper.pdf
    Compiled paper/report PDF.

  ListT5/
    Original ListT5 source code used by the experiment.
```

For detailed environment setup, CLI arguments, output files, and implementation
notes, see [README_code_experiment.md](README_code_experiment.md).

## Main Experiment

The current experiment uses five BEIR datasets:

- NFCorpus
- SciFact
- ArguAna
- SCIDOCS
- FiQA-2018

All compiled runs use the first 50 queries of each dataset. This makes the
experiment practical on Kaggle GPUs and should be interpreted as an early
controlled experiment, not a full BEIR benchmark reproduction.

The main metrics are:

- `NDCG@10` for reranking effectiveness.
- seconds per query for wall-clock inference cost.
- forward calls, counted as the number of ListT5 `model.generate(...)` calls.

The final compiled result table is available at:

```text
experiment-results/final_compiled-results.md
```

Raw outputs used for the compilation are stored under:

```text
experiment-results/raw/
```

## Key Implementation Notes

The original ListT5 tournament code precomputes first-round cache entries for
sequential groups such as:

```text
[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], ...
```

For score-balanced grouping, the first round instead uses groups such as:

```text
[0, 20, 40, 60, 80], [1, 21, 41, 61, 81], ...
```

The v2 implementation aligns the first-round cache with the active grouping
strategy, reducing unnecessary online forward calls for score-balanced grouping.
The code also fixes a duplicate-index edge case in the original `get_out_k`
logic where `k=-1` was used before being resolved to `out_k`.

## Running the Code

Install the lightweight inference dependencies:

```bash
pip install -r requirements_experiment.txt
```

Show the default experiment plan:

```bash
python run.py show-plan
```

Run the default experiment:

```bash
python run.py run
```

Run a one-job score-balanced cache check:

```bash
python run.py run \
  --datasets nfcorpus \
  --disable-topk \
  --grouping-methods score_balanced \
  --output-dir outputs/combined_grouping_topk_v2_cli \
  --no-reuse-existing
```

See [README_code_experiment.md](README_code_experiment.md) for more setup and
execution details, including Kaggle usage and two-GPU mode.

## Paper and Report

The report source is in:

```text
tex/
```

The compiled report PDF is:

```text
IR_Project-paper.pdf
```

The report explains the motivation, method, experiment setup, results, and
limitations in the format required for the assignment.

## Administration

This repository is the final/current project workspace. Earlier progress
submission materials were intentionally removed from this tree to keep the
repository focused on the final experiment, report, and results.

For the historical progress submission state, see:

```text
https://github.com/nano141004/exp-extend-list5/tree/cfb310495a02a98d4202f855e5cd4ca004ae0d26
```
