# ListT5 Research Extension
## Group Member
- Mariano Gerardus Senduk (2206814236)
- Narendra Dzulqarnain (2206081881)
## Credit

This project is an extension of the original ListT5 work by Yoon et al.:

> Soyoung Yoon, Eunbi Choi, Jiyeon Kim, Hyeongu Yun, Yireun Kim, and Seung-won Hwang. 2024. **ListT5: Listwise Reranking with Fusion-in-Decoder Improves Zero-shot Retrieval**. ACL 2024.

- Original paper: <https://aclanthology.org/2024.acl-long.125/>
- DOI: <https://doi.org/10.18653/v1/2024.acl-long.125>
- Official source code: <https://github.com/soyoung97/ListT5>

The ListT5 model, method, checkpoint references, and base implementation come from the original authors. Our contribution in this project is limited to studying and modifying the **candidate grouping strategy** used during tournament-based inference.

## Topic Overview

This project is based on the paper **ListT5: Listwise Reranking with Fusion-in-Decoder Improves Zero-shot Retrieval**. The paper focuses on reranking for information retrieval. In a typical retrieval pipeline, a first-stage retriever such as BM25 retrieves a set of candidate documents, and a reranker reorders those candidates so that the most relevant documents appear at the top.

ListT5 is a T5-based **listwise reranker**. Instead of scoring one document independently like pointwise reranking, or comparing two documents like pairwise reranking, ListT5 considers multiple candidate passages at the same time. The model then generates an ordered sequence of document identifiers as the ranking output.

## Summary of the Original Method

ListT5 uses a **Fusion-in-Decoder (FiD)** architecture. Each candidate document is combined with the query and a document index, then encoded separately. The decoder fuses the encoded representations and outputs the document indices in ranked order.

A simplified input-output format is:

```text
Input 1: Query + Index 1 + Document 1
Input 2: Query + Index 2 + Document 2
Input 3: Query + Index 3 + Document 3
Input 4: Query + Index 4 + Document 4
Input 5: Query + Index 5 + Document 5
Output: 2 5 1 4 3
```

Since ListT5 only reranks a small number of candidates per model call, the paper uses **m-ary tournament sorting** to handle larger candidate sets such as top-100 results. Candidates are divided into smaller groups, ListT5 ranks each group, and selected candidates continue to later rounds until the final ranking is produced.

## Research Gap

The original paper mainly discusses the ListT5 model, the FiD architecture, and the tournament-based inference framework. However, the effect of **candidate grouping strategy** inside the tournament is not explored in detail.

This is an important gap because the way candidates are grouped may affect the final reranking result. For example, if sequential grouping is used, the first group may contain ranks 1-5, the second group may contain ranks 6-10, and so on. This means strong candidates can meet and eliminate each other too early, while lower-ranked candidates may face easier competition in early rounds.

Therefore, we want to see whether candidate grouping influences:

- **Effectiveness**, measured by ranking quality such as nDCG@10.
- **Efficiency**, measured by the number of ListT5 calls and inference time.

## Proposed Extension

For this progress report, our extension is to study how different **grouping strategies** affect ListT5 reranking performance. Our goal is not to propose a new trained model. Instead, we focus on modifying and analyzing the inference strategy used in tournament reranking.

The planned comparison includes:

- **Sequential grouping:** candidates are grouped based on their original first-stage ranking order.
- **Random grouping:** candidates are assigned randomly to groups, with several random seeds for fair comparison.
- **Score-balanced grouping:** candidates from different first-stage rank ranges are distributed across groups.

The main research question is:

```text
How does the grouping strategy in ListT5 tournament reranking affect reranking effectiveness and inference cost?
```

## Current Experiment Plan

The experiment will use a pretrained ListT5 checkpoint and modify only the grouping logic during inference.

Planned setup:

- **Model:** pretrained ListT5-base.
- **Candidates:** top-k BM25 candidates when available.
- **Datasets:** TREC-COVID, SciFact, and NFCorpus. FiQA may be added if time allows.
- **Main metric:** nDCG@10.
- **Efficiency metrics:** number of ListT5 calls per query and inference time per query.

Planned baselines and variants:

- BM25 without neural reranking.
- ListT5 with sequential tournament grouping.
- ListT5 with random grouping.
- ListT5 with score-balanced grouping.

## Grouping Strategy Illustration

For top-20 candidates with group size 5, sequential grouping forms groups as follows:

```text
G1: 1, 2, 3, 4, 5
G2: 6, 7, 8, 9, 10
G3: 11, 12, 13, 14, 15
G4: 16, 17, 18, 19, 20
```

Score-balanced grouping distributes candidates from different rank ranges:

```text
G1: 1, 5, 9, 13, 17
G2: 2, 6, 10, 14, 18
G3: 3, 7, 11, 15, 19
G4: 4, 8, 12, 16, 20
```

This does not assume that score-balanced grouping is always better. It is used as one strategy to test whether candidate grouping affects tournament reranking behavior.

## Pseudocode

```python
def make_groups(candidates, group_size, strategy, seed=None):
    # candidates are sorted by first-stage retrieval score
    items = list(candidates)

    if strategy == "sequential":
        return [
            items[i:i + group_size]
            for i in range(0, len(items), group_size)
        ]

    if strategy == "random":
        rng = random.Random(seed)
        rng.shuffle(items)
        return [
            items[i:i + group_size]
            for i in range(0, len(items), group_size)
        ]

    if strategy == "score_balanced":
        num_groups = math.ceil(len(items) / group_size)
        groups = [[] for _ in range(num_groups)]

        for i, candidate in enumerate(items):
            group_id = i % num_groups
            groups[group_id].append(candidate)

        return groups
```

## Expected Contribution

The expected contribution of this extension is an empirical analysis of candidate grouping in ListT5 tournament reranking. The result may show whether grouping strategy affects ranking quality or inference efficiency.

Even if the new grouping strategies do not always improve nDCG@10, the experiment can still provide useful findings about the sensitivity of ListT5 tournament inference to candidate grouping.

## Next Steps

The next steps are:

1. Set up the ListT5 inference environment and run the minimal example.
2. Prepare the selected BEIR datasets and BM25 candidate files.
3. Implement sequential, random, and score-balanced grouping.
4. Run the main evaluation using nDCG@10.
5. Compare effectiveness and inference cost across grouping strategies.
