"""Defaults copied from the combined notebook."""

DEFAULT_DATASETS = [
    "nfcorpus",
    "scifact",
    "arguana",
    "scidocs",
    "fiqa",
]

DEFAULT_GROUPING_METHODS = ["sequential", "score_balanced"]
DEFAULT_GROUPING_TOPK = 100

DEFAULT_TOPK_METHODS = ["sequential"]
DEFAULT_TOPK_VALUES = [40, 60, 80]

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
