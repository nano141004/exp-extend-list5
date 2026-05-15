# ListT5 Codebase Walkthrough

**Paper:** [ListT5: Listwise Reranking with Fusion-in-Decoder Improves Zero-shot Retrieval](https://arxiv.org/abs/2402.15838) (ACL 2024)

---

## Table of Contents
1. [Overview & End-to-End Flow](#1-overview--end-to-end-flow)
2. [Core Architecture: FiDT5](#2-core-architecture-fidt5)
3. [Inference: Tournament Sort (run_listt5.py)](#3-inference-tournament-sort-run_listt5py)
4. [Inference: Sliding Window (sliding_window_eval.py)](#4-inference-sliding-window-sliding_window_evalpy)
5. [Baseline Models: MonoT5 & RankT5 (run_monot5_rankt5.py)](#5-baseline-models-monot5--rankt5-run_monot5_rankt5py)
6. [Minimal Example (test.py)](#6-minimal-example-testpy)
7. [BEIR Evaluation (beir_eval.py)](#7-beir-evaluation-beir_evalpy)
8. [Input Length Mapping (beir_length_mapping.py)](#8-input-length-mapping-beir_length_mappingpy)
9. [Training Pipeline (train_code/)](#9-training-pipeline-traincode)
10. [Positional Bias Experiments](#10-positional-bias-experiments)
11. [Data Format](#11-data-format)
12. [Dependency Graph](#12-dependency-graph)

---

## 1. Overview & End-to-End Flow

**What ListT5 does:** Given a query and a set of candidate passages (e.g., from BM25 first-stage retrieval), ListT5 reranks them by encoding multiple passages together and generating a relevance-ordered list of indices in a single forward pass — a listwise approach.

**High-level pipeline:**

```
BM25 Retrieval → Candidate Passages (top-K)
                      ↓
        Format each passage as:
   "Query: ..., Index: i, Context: ..."
                      ↓
        FiDT5 Encoder: encode N passages in parallel
        (Fusion-in-Decoder: [B*N, L] → separate → concat)
                      ↓
        FiDT5 Decoder: generates index sequence
        (e.g., "3 1 4 2 5" meaning passage 3 is most relevant)
                      ↓
        Tournament Sort (run_listt5.py) or Sliding Window (sliding_window_eval.py)
        → Full ranking of all top-K passages
                      ↓
        BEIR Evaluation → NDCG@10, etc.
```

### End-to-End File Relationships

```
                    ┌──────────────┐
                    │  FiDT5.py    │ ← Core model class (Fusion-in-Decoder)
                    └──────┬───────┘
                           │
           ┌───────────────┼───────────────────────┐
           │               │                       │
           ▼               ▼                       ▼
   ┌───────────────┐ ┌──────────────┐ ┌──────────────────────┐
   │ run_listt5.py │ │ test.py      │ │ sliding_window_eval.py│
   │ (Tournament   │ │ (Minimal     │ │ (Sliding Window      │
   │  Sort Eval)   │ │  Example)    │ │  Evaluation)          │
   └───────┬───────┘ └──────────────┘ └──────────┬───────────┘
           │                                     │
           │        ┌───────────────────┐        │
           └────────┤   beir_eval.py   ├────────┘
                    │  (NDCG/Recall/   │
                    │   MRR/MAP Calc)  │
                    └──────────────────┘
                     
  ┌──────────────────────┐
  │ train_code/train.py  │──→ train_code/models/fid_gr_modules.py
  │ (Training orchestr.) │    (FiDGRDataset + FiDGRModel)
  └──────────────────────┘    └──→ models/shared_modules.py
                                     └──→ train_code/models/FiDT5.py (training variant)
```

---

## 2. Core Architecture: FiDT5

**File:** `FiDT5.py` (root) and `train_code/models/FiDT5.py` (training variant with `encoder_output_k`)

### Class Hierarchy

```
T5ForConditionalGeneration (HuggingFace)
  └── FiDT5  (wraps encoder with EncoderWrapper)
        └── forward()
        └── generate()
        └── wrap_encoder()
        └── load_t5()
```

### Fusion-in-Decoder (FiD) Mechanism

Unlike standard pointwise rerankers (encode one query-passage pair at a time), FiD processes **N passages simultaneously**:

1. **Input**: `[B, N, L]` tensor (B=batch, N=#passages, L=passage length)
2. **EncoderWrapper** reshapes to `[B*N, L]` and runs the T5 encoder on each passage independently
3. The encoder outputs `[B*N, L, D]` are **concatenated back** to `[B, N*L, D]` — fusing all passage representations into a single sequence
4. **Decoder** attends to this fused representation to generate the output sequence (the relevance ordering)

```python
# EncoderWrapper.forward() (FiDT5.py:144-162)
bsz, total_length = input_ids.shape
passage_length = total_length // self.n_passages

# Separate passages: (B*N, L)
input_ids = input_ids.view(bsz * self.n_passages, passage_length)

# Run T5 encoder on each passage independently
outputs = self.encoder(input_ids, attention_mask, ...)

# Fuse: concat all passage representations back
last_hidden_state = outputs[0].view(bsz, self.n_passages * passage_length, -1)
```

### Key Difference Between Root and Training FiDT5

| Feature | Root `FiDT5.py` | Training `train_code/models/FiDT5.py` |
|---|---|---|
| `encoder_output_k` | Not used | Truncates encoder output to `k` tokens per passage |
| Purpose | Inference | Training (reduces memory) |

### Helper Classes

- **`EncoderWrapper`**: Wraps the T5 encoder, handles the B*N → B reshape
- **`CheckpointWrapper`**: Enables gradient checkpointing for memory-efficient training
- **`cross_attention_forward()`**: Custom cross-attention that can save attention scores for retriever distillation
- **`Retriever`**: Separate BERT-based retriever model (used for potential future training)

---

## 3. Inference: Tournament Sort (run_listt5.py)

**File:** `run_listt5.py`

### Purpose

Rerank the top-K passages (e.g., 100) using ListT5's listwise scoring via a **tournament sort** algorithm. Since ListT5 can only compare `listwise_k` (e.g., 5) passages at a time, we need multiple rounds.

### Class: `ListT5Evaluator`

#### Data Flow

```
Input JSONL (per instance):
  {
    "qid": int,
    "q_text": "query",
    "qrels": {"pid": relevance_score, ...},
    "bm25_results": [
      {"pid": ..., "text": ..., "title": ..., "bm25_score": ...},
      ... (top-K)
    ]
  }
       │
       ▼
  make_listwise_text(question, ctxs)
       │  Produces: ["Query: {q}, Index: 1, Context: {text}",
       │             "Query: {q}, Index: 2, Context: {text}", ...]
       ▼
  make_input_tensors(texts) → tokenize → {input_ids: [1, N, L], attention_mask: [1, N, L]}
       │
       ▼
  FiDT5.generate() → output sequences (e.g., "3 1 4 2 5")
       │
       ▼
  get_rel_index(output) → [3, 1, 4, 2, 5] (relevance ordering, last = most relevant)
```

#### Tournament Sort Algorithm (`run_tournament_sort`)

For each query with top-K passages:

```
full_list = [0, 1, 2, ..., K-1]  (indices of candidates)

for each round until we have top-N reranked:
  1. Chunk the list into groups of size listwise_k (e.g., 5)
  2. For each chunk:
     - If the chunk has < listwise_k items, pad with leftovers
     - Run ListT5 → get top out_k (e.g., 2) most relevant indices from this chunk
  3. Collect all winners (out_k per chunk)
  4. If winners > listwise_k → recurse (step 1 again)
  5. If winners == listwise_k → final sort → pick top-1
  6. Remove the top-1 from the candidate pool (global_exclude)
  7. Repeat until top-N are found
```

**Caching (`run_batchwise_caching`):** Before the tournament sort, all `listwise_k`-sized chunks are pre-computed in batches and cached in `best_cache`. This avoids redundant forward passes during the recursive tournament.

**Skip optimizations:**
- `--skip_no_candidate`: Skip queries with no relevant (gold) passages in top-K
- `--skip_issubset`: Stop reranking once all gold passages are already in the top-N

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--listwise_k` | 5 | Number of passages ListT5 compares at once |
| `--out_k` | 2 | Number of "winners" selected per group |
| `--topk` | 100 | Initial candidate pool size |
| `--rerank_topk` | 10 | How many top positions to rerank |
| `--bsize` | 20 | Batch size for caching |
| `--max_input_length` | auto | Based on BEIR_LENGTH_MAPPING |

---

## 4. Inference: Sliding Window (sliding_window_eval.py)

**File:** `sliding_window_eval.py`

### Purpose

An alternative to tournament sort that uses a **sliding window** approach (corresponds to Figure 4 in the paper). Instead of recursive tournament rounds, it slides a window of size `listwise_k` across the candidate list, sorting local order at each step.

### Class: `Runner`

#### Algorithm (`run_one_loop`)

```
Initialize idx = [0, 1, 2, ..., K-1]

for start_idx from (K - listwise_k) down to 0, stepping by -stride:
    window = idx[start_idx : start_idx + listwise_k]
    ordered = ListT5(question, passages[window])  # returns ranked indices
    idx[start_idx : start_idx + listwise_k] = ordered

# Final pass: sort the first listwise_k elements
idx[0:listwise_k] = ListT5(question, passages[idx[0:listwise_k]])
```

**Stride** controls overlap between windows (default=2). A smaller stride gives more refinement but more forward passes.

#### Batch-wise Execution

`run_one_loop_batchwise` processes multiple queries in parallel, iterating over the sliding window for all queries simultaneously.

### Multi-Mode Support

The `--sub_mode` flag supports different model types:
- `predefined_5sort`: ListT5 with FiD
- `monot5` / `rankt5`: Baseline models (also supported here for comparison)
- `duot5`: DuoT5 pairwise comparison

---

## 5. Baseline Models (run_monot5_rankt5.py)

**File:** `run_monot5_rankt5.py`

### Purpose

Run pointwise reranking baselines (MonoT5, RankT5) on the same BEIR datasets for comparison.

### Class: `MonoRankT5Runner`

#### MonoT5

For each (query, passage) pair:
```
Input: "Query: {q} Document: {d} Relevant:"
Output: logits for "true" (1176) vs "false" (6136) tokens
Score: log_softmax(true_score)
```

#### RankT5

```
Input: "Query: {q} Document: {d}"
Output: relevance score via <extra_id_10> (token 32089) logit
Score: logit value
```

Both models process passages in batches and produce a score per passage. Passages are then sorted by score.

---

## 6. Minimal Example (test.py)

**File:** `test.py`

A self-contained 23-line script demonstrating the core idea:

```python
model = FiDT5.from_pretrained('Soyoung97/ListT5-base')
texts = [
  "Query: When did Thomas Edison invent the light bulb?, Index: 1, Context: ...",
  "Query: When did Thomas Edison invent the light bulb?, Index: 2, Context: ...",
  ...
]
tok = T5Tokenizer.from_pretrained('t5-base')
raw = tok(texts, return_tensors='pt', padding='max_length', max_length=128)
input_tensors = {'input_ids': raw['input_ids'].unsqueeze(0),
                 'attention_mask': raw['attention_mask'].unsqueeze(0)}
output = model.generate(**input_tensors, max_length=7, ...)
output_text = tok.batch_decode(output.sequences, skip_special_tokens=True)
# Output: ['3 1 4 2 5'] → passage 5 is most relevant
```

---

## 7. BEIR Evaluation (beir_eval.py)

**File:** `beir_eval.py`

### Purpose

Evaluate reranking results using BEIR metrics (NDCG, Recall, MRR, MAP, Precision).

### Key Functions

| Function | Purpose |
|----------|---------|
| `convert_to_result_format()` | Transforms internal JSONL format to `{qid: {pid: score}}` dict |
| `make_corpus()` | Builds corpus dict from results |
| `do_evaluation()` | Wraps BEIR's `EvaluateRetrieval.evaluate()` for all metrics |
| `run_rerank_eval()` | Main entry point: loads data, computes metrics |
| `format_res_for_print()` | Formats NDCG/Recall/MRR/MAP/Precision at k=1,5,10,20,50,100 |

### Standalone Usage

```bash
python beir_eval.py --path results.jsonl --mode ours
```

The `--mode` argument can be:
- `ours`: Use pre-computed scores from the JSONL
- `castorini/monot5-base-msmarco-10k`: Use BEIR's built-in reranker

---

## 8. Input Length Mapping (beir_length_mapping.py)

**File:** `beir_length_mapping.py`

Maps dataset names to their max input lengths:

| Length | Datasets |
|--------|----------|
| 256 | msmarco, dl19, dl20, nq, hotpotqa, signal, quora, fever, climate-fever, dbpedia-entity |
| 512 | trec-covid, nfcorpus, bioasq, fiqa, cqadupstack, scidocs, scifact |
| 1024 | news, robust04, arguana, touche |

---

## 9. Training Pipeline (train_code/)

### Directory Structure

```
train_code/
├── train.py              # Main training script (PyTorch Lightning)
├── models/
│   ├── FiDT5.py          # Training-variant FiDT5 (with encoder_output_k)
│   ├── fid_gr_modules.py # FiDGRModel + FiDGRDataset (Lightning Module + Dataset)
│   └── shared_modules.py # SharedDataset + SharedModel (base classes)
└── utils/
    ├── format.py         # Data formatting utilities
    ├── io_modules.py     # File I/O (jsonl, json, tsv, pickle)
    ├── similarity.py     # Cosine similarity module
    └── eval_official/    # Official KILT evaluation scripts
```

### Training Flow

```
train.py
  │
  ├── Creates FiDGRModel (LightningModule)
  │     ├── Loads T5 → wraps in FiDT5 (select_model_by_mode)
  │     ├── Configures optimizer (Adafactor) + LR scheduler
  │     └── Creates FiDGRDataset for train/val
  │
  ├── Sets up Trainer (PyTorch Lightning)
  │     ├── DeepSpeed stage 2 (or DDP)
  │     ├── ModelCheckpoint every eval_steps
  │     └── WandbLogger (optional)
  │
  └── trainer.fit(model)
```

#### FiDGRDataset (`fid_gr_modules.py:41-108`)

The dataset processes training data where each instance has:
- `query`: The search query
- `ret`: List of `listwise_k` passages (exactly k)
- `pos_idx`: 1-based index of the relevant passage

**Data augmentation** (within `convert_listwise_to_features`):
1. Identify the positive passage by `pos_idx`
2. Replace it at a random position in the list
3. Boost its BM25 score to ensure it ranks first
4. Ground truth = argsorted BM25 scores (with boosted positive)
5. Format with `"Query: ..., Index: i, Context: ..."`
6. Target = space-separated ranking (e.g., `"3 4 2 1 5"`)

#### FiDGRModel (`fid_gr_modules.py:110-281`)

| Method | Purpose |
|--------|---------|
| `forward()` | Calls FiDT5 with source_ids + labels |
| `training_step()` | Computes cross-entropy loss |
| `validation_step()` | Computes validation loss |
| `_test_step()` | Generates with beam search |

#### SharedModel (`shared_modules.py:53-259`)

Base class providing:
- `select_model_by_mode()`: Loads T5 → wraps in FiDT5
- `configure_optimizers()`: Adafactor + linear/constant LR schedule
- `on_save_checkpoint()`: Saves model + tokenizer at checkpoints

#### Training Command (from `orig_README.md`)

```bash
# ListT5-base (4x A6000, ~8-10 hours)
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py --name EXP_NAME --do_train \
  --learning_rate 1e-04 --base_model t5-base --train_batch_size 32 \
  --eval_batch_size 32 --max_input_length 230 --max_output_length 8 \
  --train-files marco-train-coco-top1000-5-20perq.jsonl \
  --eval-files marco-dev-coco-top1000-5-500.jsonl \
  --lr_scheduler linear --gradient_accumulation_steps 2 \
  --eval_steps 2000 --num_train_epochs 10 --listwise_k 5
```

---

## 10. Positional Bias Experiments

**Directory:** `positional_bias_experiments/`

### Files

| File | Purpose |
|------|---------|
| `measure_permutation_consistency.py` | Measures ListT5 consistency across passage permutations |
| `measure_per_consist_gpt.py` | Same but using GPT as the reranker |
| `measure_duot5_permutation_consistency.py` | Permutation consistency for DuoT5 |
| `fiqa_input.jsonl` / `trec-covid_input.jsonl` | Input data for experiments |
| `gpt_out_fiqa.jsonl` | GPT output on fiqa |

These experiments test whether ListT5's ranking is invariant to the input order of passages (a known problem with listwise approaches). See Appendix of the paper for details.

---

## 11. Data Format

### Input JSONL (for inference/evaluation)

```json
{
  "qid": 44,
  "q_text": "How much impact do masks have on preventing the spread of the COVID-19?",
  "qrels": {"pid1": 1, "pid2": 0, ...},
  "bm25_results": [
    {"text": "...", "title": "...", "bm25_score": 8.89, "pid": "abc123"},
    ...
  ]
}
```

### Training Data (from `orig_README.md`)

```json
{
  "query": "when did yorktown begin",
  "pos_idx": "1",
  "ret": [
    {"text": "The American Revolution began in 1775...", "bm25_score": 5.2},
    {"text": "Prior to the battle General Cornwallis...", "bm25_score": 3.1},
    ...
  ]
}
```

---

## 12. Dependency Graph

```
                                 ┌──────────────────────┐
                                 │   beir_length_mapping │
                                 │   .py                 │
                                 └──────────┬───────────┘
                                            │ used by
         ┌───────────────────────────────────┼─────────────────────────┐
         │                                   │                         │
         ▼                                   ▼                         ▼
  ┌──────────────┐                  ┌──────────────────┐   ┌──────────────────────┐
  │ run_listt5.py │                  │ run_monot5_rank  │   │ sliding_window_eval  │
  │              │                  │ t5.py            │   │ .py                  │
  └──────┬───────┘                  └────────┬─────────┘   └──────────┬───────────┘
         │                                   │                        │
         │           ┌──────────┐            │                        │
         ├───────────┤ FiDT5.py ├────────────┤────────────────────────┤
         │           └──────────┘            │                        │
         │                                   │                        │
         ▼                                   ▼                        ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │                            beir_eval.py                               │
  │  (convert_to_result_format → make_corpus → do_evaluation → NDCG@10)   │
  └────────────────────────────────────────────────────────────────────────┘


                    ┌──────────────────────────────┐
                    │        train_code/train.py   │
                    │  (PyTorch Lightning Trainer) │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────┴───────────────┐
                    │  models/fid_gr_modules.py    │
                    │  (FiDGRModel + FiDGRDataset) │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────┴───────────────┐
                    │  models/shared_modules.py    │
                    │  (SharedModel base)          │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────┴───────────────┐
                    │  models/FiDT5.py             │
                    │  (Training variant)          │
                    └──────────────────────────────┘
```

### How the Pieces Connect

| Step | Scripts | What Happens |
|------|---------|-------------|
| **Train** | `train_code/train.py` → `fid_gr_modules.py` → `shared_modules.py` → `FiDT5.py` | Trains ListT5 model. Saves checkpoints |
| **Load** | `FiDT5.py` (root or training) | Wraps T5 into FiD architecture |
| **Infer (Tournament Sort)** | `run_listt5.py` | Uses `FiDT5.generate()`, tournament sort to rerank top-K |
| **Infer (Sliding Window)** | `sliding_window_eval.py` | Uses `FiDT5.generate()`, sliding window to rerank |
| **Infer (Baseline)** | `run_monot5_rankt5.py` | Uses `T5ForConditionalGeneration`, pointwise scoring |
| **Evaluate** | `beir_eval.py` | Computes NDCG, Recall, MRR, MAP, Precision |
| **Length Config** | `beir_length_mapping.py` | Auto-sets `max_input_length` per dataset |
| **Minimal Demo** | `test.py` | Quick test of a single listwise sort |

---

## Model Checkpoints (HuggingFace)

| Model | HF Repo |
|-------|---------|
| ListT5-base | `Soyoung97/ListT5-base` |
| ListT5-3B | `Soyoung97/ListT5-3b` |
| ListT5-base (re-run on A6000) | `Soyoung97/ListT5-base-A6000` |
| RankT5-base | `Soyoung97/RankT5-base` |
| RankT5-large | `Soyoung97/RankT5-large` |
| RankT5-3B | `Soyoung97/RankT5-3b` |

---

## Environment

```yaml
# From listt5_conda_env.yaml
torch: 2.1.0
transformers: 4.33.3  # Critical version — mismatches cause encoder errors
python: 3.10.13
deepspeed: 0.12.2
pytorch-lightning: 2.0.9
```
