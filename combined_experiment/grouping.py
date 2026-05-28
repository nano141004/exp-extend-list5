"""Grouping policies used by the experiment."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable


def sequential_groups(items: Iterable[int], group_size: int, seed: int = 0) -> list[list[int]]:
    values = list(items)
    return [values[i : i + group_size] for i in range(0, len(values), group_size)]


def score_balanced_groups(items: Iterable[int], group_size: int, seed: int = 0) -> list[list[int]]:
    values = sorted(list(items))
    if len(values) <= group_size:
        return [values]

    n_groups = math.ceil(len(values) / group_size)
    groups = [[] for _ in range(n_groups)]
    for index, item in enumerate(values):
        groups[index % n_groups].append(item)
    return [group for group in groups if group]


def random_groups(items: Iterable[int], group_size: int, seed: int = 0) -> list[list[int]]:
    values = list(items)
    random.Random(seed).shuffle(values)
    return [values[i : i + group_size] for i in range(0, len(values), group_size)]


GROUPING_POLICIES = {
    "sequential": sequential_groups,
    "score_balanced": score_balanced_groups,
    "random": random_groups,
}


def validate_grouping_methods(methods: Iterable[str]) -> None:
    unknown = sorted(set(methods) - set(GROUPING_POLICIES))
    if unknown:
        valid = ", ".join(sorted(GROUPING_POLICIES))
        raise ValueError(f"Unknown grouping method(s): {unknown}. Valid methods: {valid}")
