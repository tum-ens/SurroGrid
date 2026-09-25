"""Installed building assets from urbs capacities (``powerflow.assets``)."""

from __future__ import annotations

import pandas as pd

from gridexpand.powerflow.assets import COLUMNS, installed_assets


def series(rows, names):
    return pd.Series([value for *_, value in rows],
                     index=pd.MultiIndex.from_tuples([tuple(key) for *key, _ in rows], names=names))


def test_installed_assets_per_bus_and_technology():
    cap_pro = series([
        (2025, "10", "Rooftop PV_optimized_DEMO_1", 1.0),       # older support timeframe: ignored
        (2026, "10", "Rooftop PV_optimized_DEMO_1", 7.5),
        (2026, "11", "Rooftop PV_optimized_DEMO_2", 0.0),       # not built
        (2026, "10", "heatpump_air", 3.2),
        (2026, "10", "heatpump_booster", 1e-9),                 # solver noise
        (2026, "10", "charging_station0", 11.0),
        (2026, "10", "charging_station1", 11.0),
        (2026, "10", "import", 2000.0),
        (2026, "main", "Rooftop PV_heuristic_X", 5.0),          # not a bus
    ], ["stf", "sit", "pro"])
    cap_sto_c = series([
        (2026, "10", "battery_private", "electricity", 6.0),
        (2026, "10", "heat_storage", "heat", 1.2),
        (2026, "10", "mobility_storage0", "mobility0", 40.0),
        (2026, "10", "mobility_storage1", "mobility1", 60.0),
    ], ["stf", "sit", "sto", "com"])
    cap_sto_p = cap_sto_c / 2

    assets = installed_assets(cap_pro, cap_sto_c, cap_sto_p)

    assert list(assets.columns) == COLUMNS
    rows = {(r.technology, r.building_objectid): r for r in assets.itertuples()}
    assert set(rows) == {("pv", "DEMO_1"), ("heat_pump", ""), ("battery", ""), ("heat_storage", ""), ("ev", "")}
    assert rows["pv", "DEMO_1"].power_kw == 7.5 and rows["pv", "DEMO_1"].bus == 10
    assert (rows["battery", ""].energy_kwh, rows["battery", ""].power_kw) == (6.0, 3.0)
    assert (rows["ev", ""].units, rows["ev", ""].power_kw, rows["ev", ""].energy_kwh) == (2, 22.0, 100.0)


def test_no_storages_and_no_capacity():
    cap_pro = series([(2026, "5", "import", 2000.0)], ["stf", "sit", "pro"])
    assert installed_assets(cap_pro).empty
