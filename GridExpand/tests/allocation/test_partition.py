"""LPT partitioning used by the parallel TEASER heat path."""

from __future__ import annotations

import pandas as pd
import pytest

from gridexpand.allocation.functions.partition import partition_df_by_cpu


def test_partition_balances_and_keeps_every_row():
    df = pd.DataFrame({"households": [8, 1, 1, 3, 2, 2, 5]}, index=[10, 11, 12, 13, 14, 15, 16])
    subsets = partition_df_by_cpu(df, 3, "households")
    assert sorted(i for s in subsets for i in s.index) == sorted(df.index)
    loads = sorted(int(s["households"].sum()) for s in subsets)
    assert loads == [7, 7, 8]
    # The heaviest building goes to the first bin, as in the original heuristic.
    assert 10 in subsets[0].index


def test_partition_drops_empty_bins():
    df = pd.DataFrame({"households": [2, 1]})
    subsets = partition_df_by_cpu(df, 4, "households")
    assert len(subsets) == 2
    assert all(not subset.empty for subset in subsets)


def test_partition_rejects_zero_load():
    """B7: a zero-household row was silently dropped from the parallel path."""
    df = pd.DataFrame({"households": [2, 0, 1]})
    with pytest.raises(ValueError, match="positive"):
        partition_df_by_cpu(df, 2, "households")
