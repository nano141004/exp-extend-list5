# Progress Report: Score-Balanced Tournament Grouping for ListT5

## 1. Current Objective

This progress report describes the current implementation and preliminary evaluation of a ListT5 inference-time extension: comparing different grouping strategies inside tournament-based listwise reranking.

The current comparison covers:

1. **Sequential grouping**: an original-like ListT5 baseline where BM25 top-100 candidates are grouped sequentially.
2. **Score-balanced grouping**: the proposed extension, where candidates are distributed across groups based on their BM25 ranks.

The experiments are still small-scale and intended to validate the pipeline and provide early analysis, not final full evaluation results.

## 2. Main Notebook

The main notebook is:

```text
notebook_progress.ipynb
```

The notebook compares the two grouping methods on two datasets:

```text
trec-covid.jsonl
scifact.jsonl
```

## 3. Implemented Components

### 3.1 Model and Dataset Setup

The notebook uses the pretrained checkpoint:

```text
Soyoung97/ListT5-base
```

The BM25 top-100 candidate files are loaded from:

```text
Soyoung97/beir-eval-bm25-top100
```

The datasets evaluated so far are:

```text
trec-covid
scifact
```

The main configuration is:

```text
TOPK = 100
LISTWISE_K = 5
OUT_K = 2
RERANK_TOPK = 10
MAX_INPUT_LENGTH = 512
```

This follows the original ListT5 evaluation setup for BEIR-style reranking, especially `listwise_k=5`, `out_k=2`, and BM25 top-100 candidates.

### 3.2 Tournament Reranking

The tournament reranking implementation was adjusted to be closer to the official `run_listt5.py` source code.

The original-like sequential flow is:

1. Take the BM25 top-100 candidates.
2. Split the candidates into groups of 5.
3. Run ListT5 on each group.
4. Keep the top `out_k=2` candidates from each group.
5. Recursively rerank the surviving candidates through the same tournament procedure.
6. Once the top-1 document is selected, repeat the process until the final top-10 ranking is produced.

### 3.3 Sequential Grouping

Sequential grouping follows the original ListT5-style grouping pattern:

```text
G1: rank 1-5
G2: rank 6-10
G3: rank 11-15
...
```

In 0-based indexing:

```text
G1: 0,1,2,3,4
G2: 5,6,7,8,9
...
```

This method is used as the baseline because it matches the default tournament construction in the original ListT5 code.

### 3.4 Score-Balanced Grouping

Score-balanced grouping distributes candidates according to their BM25 ranks.

For top-100 candidates and group size 5, there are 20 groups. The pattern is:

```text
G1: 1,21,41,61,81
G2: 2,22,42,62,82
G3: 3,23,43,63,83
...
```

In 0-based indexing:

```text
G1: 0,20,40,60,80
G2: 1,21,41,61,81
...
```

The motivation is to avoid placing all high-ranked BM25 candidates in the same early group, where strong candidates may eliminate each other too early in the tournament.

### 3.5 Small-Scale Evaluation

The current notebook uses:

```text
MAX_QUERIES_PER_DATASET = 10
```

This means only the first 10 queries from each dataset are evaluated. This was intentionally kept small so the experiment can be run on Kaggle GPU within a reasonable time.

The reported metrics are:

- BM25 mean nDCG@10
- ListT5 mean nDCG@10
- mean calls per query
- total forward calls
- runtime per method

## 4. Preliminary Results

Summary from `listt5_seq_vs_score_balanced_2datasets_summary.csv`:

| Dataset | Strategy | Queries | BM25 nDCG@10 | ListT5 nDCG@10 | Mean Calls/Query | Total Forwards | Time (sec) |
|---|---:|---:|---:|---:|---:|---:|---:|
| trec-covid | sequential | 10 | 0.5078 | 0.7456 | 95.0 | 950 | 325.32 |
| trec-covid | score_balanced | 10 | 0.5078 | 0.7471 | 334.6 | 3346 | 1143.91 |
| scifact | sequential | 10 | 0.9500 | 0.9631 | 98.5 | 985 | 338.00 |
| scifact | score_balanced | 10 | 0.9500 | 0.9631 | 334.8 | 3348 | 1147.59 |

## 5. Initial Observations

### 5.1 Effectiveness

On the first 10 queries:

- TREC-COVID:
  - Sequential: 0.7456 nDCG@10
  - Score-balanced: 0.7471 nDCG@10
  - Score-balanced is slightly higher, but the difference is very small.

- SciFact:
  - Sequential: 0.9631 nDCG@10
  - Score-balanced: 0.9631 nDCG@10
  - Both methods produce the same score on this small subset.

Since the number of evaluated queries is still small, these results should not be treated as final conclusions.

### 5.2 Efficiency

Score-balanced grouping is substantially more expensive in the current implementation:

- TREC-COVID:
  - Sequential: 95.0 calls/query
  - Score-balanced: 334.6 calls/query

- SciFact:
  - Sequential: 98.5 calls/query
  - Score-balanced: 334.8 calls/query

The main reason is that score-balanced grouping changes group composition more often across tournament rounds, which reduces cache reuse compared to sequential grouping.

### 5.3 Trade-Off

The preliminary results suggest that score-balanced grouping can produce comparable effectiveness to sequential grouping, but with significantly higher inference cost in the current implementation.

This means the next analysis should focus on:

1. Whether the effectiveness improvement is consistent over more queries.
2. Whether the additional inference cost is justified.
3. Whether score-balanced grouping can be optimized to improve cache reuse.

## 6. Current Implementation Status

Completed:

- Loaded pretrained ListT5-base on Kaggle.
- Added compatibility patches for Kaggle Python 3.12 and recent Transformers versions.
- Loaded BEIR BM25 top-100 candidate files.
- Implemented tournament reranking.
- Implemented sequential grouping.
- Implemented score-balanced grouping.
- Evaluated two datasets on a small query subset.
- Exported summary and per-query outputs to CSV.

Not completed yet:

- Full evaluation over all queries.
- Random grouping baseline.
- Statistical analysis of whether nDCG differences are significant.

## 7. Next Steps

1. Run more queries, for example 25 queries or the full dataset.
2. Compare calls/query and runtime for all grouping strategies.
3. Analyze whether score-balanced grouping provides enough effectiveness gain to justify its additional inference cost.

