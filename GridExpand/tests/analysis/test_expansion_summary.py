"""``gridexpand expansion summary``: CLI dispatch and printing (the indicator SQL runs in test_expansion_database.py)."""

from __future__ import annotations

import pandas as pd
import pytest

from gridexpand.analysis.expansion import grid_expansion
from gridexpand.analysis.expansion.summary import print_indicators


def test_summary_is_a_subcommand_of_expansion(capsys):
    with pytest.raises(SystemExit) as exit_info:
        grid_expansion.main(["summary", "--help"])
    assert exit_info.value.code == 0
    assert "--by-plz" in capsys.readouterr().out


def test_total_row_prints_one_indicator_per_line(capsys):
    print_indicators(pd.DataFrame([{"analysis_key": "k", "grids": 2, "total_cost_eur": 170250.0}]))
    assert capsys.readouterr().out.splitlines() == ["analysis_key: k", "grids: 2", "total_cost_eur: 170,250.00"]
