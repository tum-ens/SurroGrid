"""Cable planning limit and station replacement of the staged rules (db/sql/0010_line_limit_station_replacement.sql)."""

from __future__ import annotations

import re

import pytest

from gridexpand.analysis.expansion import staged
from gridexpand.paths import SQL_DIR

SQL = (SQL_DIR / "0010_line_limit_station_replacement.sql").read_text(encoding="utf-8")


def _params(assumption, **overrides):
    return staged.StagedParameters.from_assumption({**assumption, **overrides})


def _route(p100, installed=0.27):
    return staged.RouteInput(key="r", length_km=0.1, existing_cables=1, installed_capacity_ka=installed, p100_ka=p100)


def test_defaults_keep_the_behaviour_of_0007(staged_assumption):
    params = _params(staged_assumption)
    assert params.line_planning_limit == 1.0 and params.station_replacement_eur == 0.0
    assert staged.select_route(_route(0.25), 2, params).measure == "none"
    assert staged.station_decision(630, 900, 2, params).exchange_cost_eur == 15000


def test_line_planning_limit(staged_assumption):
    params = _params(staged_assumption, line_planning_limit=0.6)
    # 0.2 kA on 0.27 kA is 74 %: above 60 %; one added NAYY 150 gives 0.6 x 0.54 = 0.324 kA
    result = staged.select_route(_route(0.2), 2, params)
    assert result.measure == "local" and (result.n150, result.n185, result.n240) == (1, 0, 0)
    # 0.16 kA is 59 %: within the limit
    assert staged.select_route(_route(0.16), 2, params).measure == "none"
    # the cap: 0.6 x (0.27 + 3 x 0.357) = 0.8046 kA
    unresolved = staged.select_route(_route(0.9), 2, params)
    assert unresolved.measure == "unresolved"
    assert unresolved.missing_capacity_ka == pytest.approx(0.9 - 0.6 * (0.27 + 3 * 0.357))


def test_line_planning_limit_must_be_a_share(staged_assumption):
    with pytest.raises(ValueError, match="line_planning_limit"):
        _params(staged_assumption, line_planning_limit=1.2)


def test_station_replacement_from_1000_kva(staged_assumption):
    params = _params(staged_assumption, station_replacement_from_kva=1000.0, station_replacement_eur=55000.0)
    small = staged.station_decision(400, 500, 2, params)
    assert (small.required_kva, small.exchange_cost_eur) == (630, 12000)
    large = staged.station_decision(630, 900, 2, params)
    assert (large.required_kva, large.exchange_cost_eur) == (1000, 15000 + 55000)
    assert large.cost_basis == "exchange_to_1000kva_station_replaced"
    over = staged.station_decision(630, 1100, 1, params)
    assert (over.measure, over.exchange_cost_eur) == ("over_limit", 15000 + 55000)


def test_migration_rows_document_their_parameters():
    keys = re.findall(r"'assumption_key', '(de_lv_staged_2026_\w+)'", SQL)
    assert keys == ["de_lv_staged_2026_line60", "de_lv_staged_2026_station_replacement",
                    "de_lv_staged_2026_station_replacement_line60"]
    names = "line_planning_limit|station_replacement_from_kva|station_replacement_eur"
    for block in SQL.split("INSERT INTO surrogrid.expansion_cost_assumption\nSELECT")[1:]:
        set_values = {k: float(v) for k, v in re.findall(rf"'({names})', ([\d.]+)", block)}
        documented = {k: float(v) for k, v in re.findall(r'"(\w+)": \{"value": ([\d.]+)', block)}
        assert set_values and all(documented[name] == value for name, value in set_values.items())
