"""Constants used by the grouping-method experiment."""

DEFAULT_DATASETS = ("trec-covid", "nfcorpus", "fiqa", "scifact", "arguana")
DEFAULT_STRATEGIES = ("sequential", "score_balanced", "random")
DEFAULT_SEEDS = (0,)

# Paper Table 2, BM25 top-100, ListT5-base (r=2), NDCG@10.
TABLE2_LISTT5_BASE_TOP100 = {
    "trec-covid": 0.783,
    "nfcorpus": 0.356,
    "bioasq": 0.564,
    "nq": 0.531,
    "hotpotqa": 0.726,
    "fiqa": 0.396,
    "signal": 0.335,
    "news": 0.485,
    "robust04": 0.521,
    "arguana": 0.489,
    "touche": 0.334,
    "cqadupstack": 0.388,
    "quora": 0.864,
    "dbpedia-entity": 0.437,
    "scidocs": 0.176,
    "fever": 0.798,
    "climate-fever": 0.240,
    "scifact": 0.741,
}
