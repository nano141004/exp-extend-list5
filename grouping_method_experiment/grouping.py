"""Candidate grouping policies."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable


def sequential_groups(items: Iterable[int], group_size: int, seed: int = 0) -> list[list[int]]:
    """Official ListT5 grouping: contiguous chunks."""

    values = list(items)
    return [values[i : i + group_size] for i in range(0, len(values), group_size)]


def score_balanced_groups(items: Iterable[int], group_size: int, seed: int = 0) -> list[list[int]]:
    """Round-robin candidates by first-stage rank.

    Candidate index 0 is the first-stage rank 1 document. Sorting reconstructs
    first-stage rank order, then round-robin assignment spreads strong and weak
    initial candidates across groups.
    """

    values = list(items)
    if len(values) <= group_size:
        return [values]

    ranked = sorted(values)
    n_groups = math.ceil(len(ranked) / group_size)
    groups = [[] for _ in range(n_groups)]
    for i, item in enumerate(ranked):
        groups[i % n_groups].append(item)
    return [group for group in groups if group]


def random_groups(items: Iterable[int], group_size: int, seed: int = 0) -> list[list[int]]:
    """Seeded random grouping baseline."""

    values = list(items)
    rng = random.Random(seed)
    rng.shuffle(values)
    return sequential_groups(values, group_size, seed=seed)


GROUPING_POLICIES = {
    "sequential": sequential_groups,
    "score_balanced": score_balanced_groups,
    "random": random_groups,
}


def group_items(strategy: str, items: Iterable[int], group_size: int, seed: int = 0) -> list[list[int]]:
    try:
        group_fn = GROUPING_POLICIES[strategy]
    except KeyError as exc:
        valid = ", ".join(sorted(GROUPING_POLICIES))
        raise ValueError(f"Unknown grouping strategy '{strategy}'. Valid strategies: {valid}") from exc
    return group_fn(items, group_size, seed)
