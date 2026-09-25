"""A tiny synthetic urbs input (2 buildings, 6 hours) and its solver LP file.

Used by test_lp_golden.py to pin the LP file that Step 3 hands to the solver: the
heuristic case is a degenerate LP whose returned vertex depends on the order of
rows, columns and terms, so model-building refactors must keep this file
byte-identical. The builder only uses functions that also existed before the
2026-09 cleanup, so the golden hash was computed with the former code.
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pyomo.opt import ProblemFormat

HOURS = 6
NAN = float("nan")


def write_input(path: Path) -> Path:
    t = pd.RangeIndex(HOURS, name="t")
    hour = np.arange(HOURS, dtype=float)
    commodity = pd.DataFrame(
        [
            (1, "electricity", "Demand", NAN), (1, "electricity_import", "Buy", 1.0),
            (1, "electricity_feed_in", "Sell", 1.0), (1, "solar_a", "SupIm", NAN),
            (1, "common_heat", "Stock", NAN), (1, "space_heat", "Demand", NAN),
            (2, "electricity", "Demand", NAN), (2, "electricity_import", "Buy", 1.0),
            (2, "electricity_feed_in", "Sell", 1.0), (2, "common_heat", "Stock", NAN),
            (2, "space_heat", "Demand", NAN),
        ],
        columns=["Site", "Commodity", "Type", "price"],
    )
    process_columns = ["Site", "Process", "inst-cap", "cap-up", "inv-cost-fix", "inv-cost",
                       "fix-cost", "var-cost", "wacc", "depreciation"]
    process = pd.DataFrame(
        [
            (1, "import", 2000.0, 2000.0, NAN, 0.0, 0.0, 0.0, 0.07, 30.0),
            (1, "feed_in", 2000.0, 2000.0, NAN, 0.0, 0.0, 0.0, 0.07, 30.0),
            (1, "Rooftop PV_a", 0.0, 10.0, 6565.0, 533.7, 0.0, 0.0, 0.022, 15.0),
            (1, "heatpump_air", 0.0, 6.0, 6600.0, 750.0, 0.0, 0.0, 0.0216, 20.0),
            (1, "Heat_dummy_space", 50.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.07, 1.0),
            (2, "import", 2000.0, 2000.0, NAN, 0.0, 0.0, 0.0, 0.07, 30.0),
            (2, "feed_in", 2000.0, 2000.0, NAN, 0.0, 0.0, 0.0, 0.07, 30.0),
            (2, "heatpump_air", 3.0, 3.0, NAN, 0.0, 0.0, 0.0, 0.0216, 20.0),
            (2, "Heat_dummy_space", 50.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.07, 1.0),
        ],
        columns=process_columns,
    )
    process_commodity = pd.DataFrame(
        [
            ("import", "electricity_import", "In", 1), ("import", "electricity", "Out", 1),
            ("feed_in", "electricity", "In", 1), ("feed_in", "electricity_feed_in", "Out", 1),
            ("Rooftop PV_a", "solar_a", "In", 1), ("Rooftop PV_a", "electricity", "Out", 1),
            ("heatpump_air", "electricity", "In", 1), ("heatpump_air", "common_heat", "Out", 1),
            ("Heat_dummy_space", "common_heat", "In", 1), ("Heat_dummy_space", "space_heat", "Out", 1),
        ],
        columns=["Process", "Commodity", "Direction", "ratio"],
    )
    storage = pd.DataFrame(
        [
            (1, "battery_private", "electricity", 0.0, 8.0, 0.0, 4.0, 0.961, 1.0, 0.0, 2.0,
             0.0, 300.0, 0.0, 0.0, 0.001, 0.022, 15.0, NAN, NAN),
            (2, "heat_storage", "space_heat", 0.0, 2.0, 0.0, 18.0, 0.932, 1.0, 0.0, 0.1163,
             0.0, 58.0, 0.0, 0.0, 0.001, 0.0216, 20.0, "heatpump_air", 0.325),
        ],
        columns=["Site", "Storage", "Commodity", "inst-cap-c", "cap-up-c", "inst-cap-p", "cap-up-p",
                 "eff-in", "eff-out", "discharge", "ep-ratio", "inv-cost-p", "inv-cost-c",
                 "fix-cost-p", "fix-cost-c", "var-cost-p", "wacc", "depreciation",
                 "linked-process", "max-energy-per-process-capacity"],
    )
    demand = pd.DataFrame(
        {
            (1, "electricity"): 0.5 + 0.1 * hour, (1, "space_heat"): 2.0 - 0.2 * hour,
            (2, "electricity"): 0.3 + 0.05 * hour, (2, "space_heat"): 1.5 + 0.1 * hour,
        },
        index=t,
    )
    supim = pd.DataFrame({(1, "solar_a"): [0.0, 0.1, 0.5, 0.7, 0.3, 0.0]}, index=t)
    supim.columns.names = ["Site", "Commodity"]
    eff_factor = pd.DataFrame({(1, "heatpump_air"): 3.0 + 0.1 * hour, (2, "heatpump_air"): 2.8 + 0.1 * hour}, index=t)
    buy_sell_price = pd.DataFrame({"electricity_import": 0.398, "electricity_feed_in": 0.08}, index=t)
    weather = pd.DataFrame({("ambient", "Tamb"): -2.0 + hour, ("ambient", "Irradiation"): 50.0 * hour}, index=t)
    with warnings.catch_warnings(), pd.HDFStore(path, "w") as store:
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        for key, frame in {
            "commodity": commodity, "process": process, "process_commodity": process_commodity,
            "storage": storage, "demand": demand, "supim": supim, "eff_factor": eff_factor,
            "buy_sell_price": buy_sell_price, "weather": weather,
        }.items():
            store[f"urbs_in/{key}"] = frame
    return path


def lp_sha256(tmp_dir: Path, n_clusters: int = 1) -> list[str]:
    """Build the cluster models of the fixture and return the SHA-256 of each LP file."""
    from gridexpand.optimization.urbs.features.typeperiod import select_predefined_timesteps
    from gridexpand.optimization.urbs.identify import get_parallel_building_clusters, identify_mode
    from gridexpand.optimization.urbs.input import get_cluster_data, read_input_h5
    from gridexpand.optimization.urbs.model import create_model
    from gridexpand.optimization.urbs.scenarios import insert_scenario

    data = read_input_h5(write_input(Path(tmp_dir) / "fixture.h5"))
    settings = {"tsam": False, "hoursPerPeriod": 168, "dt": 1, "timesteps": range(0, HOURS + 1)}
    data = insert_scenario(data, settings)
    identify_mode(data)
    data, _ = select_predefined_timesteps(data, settings["timesteps"])
    digests = []
    for i, cluster in enumerate(get_parallel_building_clusters(data, n_clusters)):
        model = create_model(get_cluster_data(data, cluster), settings)
        lp = Path(tmp_dir) / f"cluster_{i}.lp"
        model.write(str(lp), format=ProblemFormat.cpxlp, io_options={})
        digests.append(hashlib.sha256(lp.read_bytes()).hexdigest())
    return digests
