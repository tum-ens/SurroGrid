"""Workload partitioning for the parallel Step 2 generators."""

from __future__ import annotations

import heapq

import pandas as pd


def partition_df_by_cpu(df: pd.DataFrame, n_cpus: int, count_column: str) -> list[pd.DataFrame]:
    """Split one-row-per-building data into balanced, non-empty subsets.

    Uses the longest-processing-time greedy heuristic: buildings are sorted by
    descending ``count_column`` (a proxy for the computational load, e.g. the
    number of households) and each one goes to the currently lightest of
    ``n_cpus`` bins. Buildings are never split and empty bins are dropped.

    Args:
        df: One row per building.
        n_cpus: Maximum number of subsets.
        count_column: Positive load weight per building.

    Returns:
        Subsets of ``df`` (same columns and index labels), none of them empty.

    Raises:
        ValueError: If a building has a non-positive load weight. Such a row
            would otherwise be dropped silently while the serial path keeps it.
    """
    loads = df[count_column]
    if (loads <= 0).any():
        raise ValueError(
            f"Parallel partitioning requires a positive {count_column!r} for every row."
        )
    building_list = list(loads.items())
    building_list.sort(key=lambda item: item[1], reverse=True)

    heap: list[tuple[int, int]] = [(0, bin_id) for bin_id in range(n_cpus)]
    heapq.heapify(heap)
    bins_indices: list[list] = [[] for _ in range(n_cpus)]
    for idx, n_count in building_list:
        current_load, bin_id = heapq.heappop(heap)
        bins_indices[bin_id].append(idx)
        heapq.heappush(heap, (current_load + n_count, bin_id))

    return [df.loc[indices].copy() for indices in bins_indices if indices]
