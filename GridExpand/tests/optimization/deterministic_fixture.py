"""A small Step 2 input whose optimum is unique, for comparing the urbs and PyPSA optimizers.

With the real inputs the heuristic LP has many optimal solutions: surplus PV is
worth nothing at a 0 EUR/kWh feed-in tariff (feed-in, heating rod and storage
losses are all free), a constant import price makes every charging hour equally
good, and the content of a lossless storage can be shifted by a constant. Every
solver (and every interface to it) then returns another of these optima, so only
the objective can be compared. This input removes the ties:

- import price, feed-in tariff and COP differ irregularly from hour to hour (no
  near-ties between neighbouring hours), and 0 < tariff < price;
- every storage has a self-discharge and a variable cost of its own, large enough that
  shifting charging between two storages or neighbouring hours is never almost free;
- the heat pump beats the heating rod (COP > 1) and capacities make both work.

``tests/optimization/test_pypsa_equivalence.py`` certifies the uniqueness on the
PyPSA model before comparing hourly results with urbs.

Three buildings, 72 hours; variants:

- ``heuristic``: fixed capacities (LP), legacy mobility buffer;
- ``optimized``: capacity expansion with fixed investment costs (MILP);
- ``ev_sessions``: fixed capacities with dedicated EV charging sessions (LP).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HOURS = 72
NAN = float("nan")
VARIANTS = ("heuristic", "optimized", "ev_sessions")
SITES = (1, 2, 3)
# home stays (model hours t = 1..72) of the vehicles: evenings and nights
HOME = [(1, 7), (17, 31), (41, 55), (65, 72)]


def _hours() -> np.ndarray:
    return np.arange(1, HOURS + 1, dtype=float)


def _series(values: np.ndarray) -> pd.Series:
    return pd.Series(values, index=pd.RangeIndex(HOURS, name="t"))


def _jitter(h: np.ndarray, alpha: float) -> np.ndarray:
    """Irregular values in [0, 1) (Weyl sequence): neighbouring hours never nearly tie."""
    return (h * alpha) % 1.0


def _profiles() -> dict[str, np.ndarray]:
    h = _hours()
    day = 2 * np.pi * (h % 24) / 24
    sun = np.clip(np.sin(np.pi * ((h % 24) - 6) / 12), 0.0, None)  # 6 h .. 18 h
    return {
        "price": 0.30 + 0.06 * np.sin(day + 1.0) + 0.00041 * h + 0.012 * _jitter(h, 0.6180339887),
        "tariff": 0.07 + 0.015 * np.sin(day + 2.5) + 0.00023 * h + 0.009 * _jitter(h, 0.4142135624),
        "pv_1": 0.82 * sun * (1.0 - 0.1 * np.cos(h / 7.0)),
        "pv_2": 0.74 * sun * (1.0 - 0.05 * np.sin(h / 5.0)),
        "cop": 2.6 + 0.6 * np.sin(day - 1.2) + 0.08 * _jitter(h, 0.7320508076),
        "elec_1": 0.45 + 0.25 * (1 + np.sin(day - 2.0)) + 0.003 * h,
        "elec_2": 0.35 + 0.20 * (1 + np.cos(day)) + 0.002 * h,
        "elec_3": 0.55 + 0.30 * (1 + np.sin(day + 0.5)),
        "space_1": 2.2 + 1.2 * np.cos(day) + 0.01 * h,
        "space_2": 1.8 + 1.0 * np.cos(day + 0.3),
        "water_1": 0.25 + 0.5 * np.exp(-((h % 24) - 7) ** 2 / 2) + 0.4 * np.exp(-((h % 24) - 20) ** 2 / 3),
        "water_2": 0.2 + 0.4 * np.exp(-((h % 24) - 8) ** 2 / 2),
    }


def _availability() -> np.ndarray:
    h = _hours()
    return np.array([any(a <= t <= b for a, b in HOME) for t in h], dtype=float)


def _process_rows(variant: str) -> list[tuple]:
    grow = variant == "optimized"
    rows = []
    for site in SITES:
        rows += [
            (site, "import", 2000.0, 2000.0, NAN, 0.0, 0.0, 0.0, 0.07, 30.0),
            (site, "feed_in", 2000.0, 2000.0, NAN, 0.0, 0.0, 0.0, 0.07, 30.0),
        ]
    for site, pv in ((1, 6.0), (2, 4.5)):
        rows.append((site, f"Rooftop PV_{site}", 0.0 if grow else pv, 10.0 if grow else pv,
                     6565.0 if grow else 0.0, 533.7 if grow else 0.0, 0.0, 0.0, 0.022, 15.0))
    for site, hp, rod in ((1, 1.5, 4.0), (2, 1.2, 3.0)):
        rows += [
            (site, "heatpump_air", 0.0 if grow else hp, 6.0 if grow else hp,
             6600.0 if grow else 0.0, 750.0 if grow else 0.0, 0.0, 0.0, 0.0216, 20.0),
            (site, "heatpump_booster", 0.0 if grow else rod, 8.0 if grow else rod,
             100.0 if grow else 0.0, 83.3 if grow else 0.0, 0.0, 0.0, 0.0216, 20.0),
            (site, "Heat_dummy_space", 50.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.07, 1.0),
            (site, "Heat_dummy_water", 50.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.07, 1.0),
        ]
    rows += [(1, "charging_station0", 11.0, 11.0, NAN, 0.0, 0.0, 0.0, 0.07, 1.0),
             (3, "charging_station0", 11.0, 11.0, NAN, 0.0, 0.0, 0.0, 0.07, 1.0),
             (3, "charging_station1", 7.4, 7.4, NAN, 0.0, 0.0, 0.0, 0.07, 1.0)]
    return rows


def _storage_rows(variant: str) -> list[tuple]:
    grow = variant == "optimized"
    rows = [
        # site, storage, commodity, inst-c, up-c, inst-p, up-p, eff-in, eff-out, discharge, ep,
        # inv-p, inv-c, fix-p, fix-c, var, wacc, depreciation, linked-process, ratio
        (1, "battery_private", "electricity", 0.0 if grow else 8.0, 12.0 if grow else 8.0,
         0.0 if grow else 4.0, 6.0 if grow else 4.0, 0.961, 0.98, 0.002, 2.0,
         0.0, 300.0 if grow else 0.0, 0.0, 0.0, 0.004, 0.022, 15.0, NAN, NAN),
        (3, "battery_private", "electricity", 0.0 if grow else 5.0, 10.0 if grow else 5.0,
         0.0 if grow else 2.5, 5.0 if grow else 2.5, 0.955, 0.97, 0.003, 2.0,
         0.0, 310.0 if grow else 0.0, 0.0, 0.0, 0.0045, 0.022, 15.0, NAN, NAN),
    ]
    for site, energy, ratio in ((1, 0.45, 0.325), (2, 0.3, 0.30)):
        power = energy / 0.1163
        rows.append((site, "heat_storage", "space_heat", 0.0 if grow else energy, 2.0 if grow else energy,
                     0.0 if grow else power, 17.2 if grow else power, 0.932, 1.0, 0.01 + 0.002 * site, 0.1163,
                     0.0, 58.0 if grow else 0.0, 0.0, 0.0, 0.012 + 0.001 * site, 0.0216, 20.0,
                     "heatpump_air", ratio))
    if variant != "ev_sessions":
        for site, index, capacity in ((1, 0, 40.0), (3, 0, 55.0), (3, 1, 30.0)):
            rows.append((site, f"mobility_storage{index}", f"mobility{index}", capacity, capacity,
                         capacity, capacity, 0.95, 1.0, 0.001 + 0.0005 * index, NAN,
                         0.0, 0.0, 0.0, 0.0, 0.003 + 0.0005 * site, 0.07, 20.0, NAN, NAN))
    return rows


def _sessions(availability: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One session per home stay; half-connected first and last hours."""
    sessions, hours = [], []
    for site, process, kw, energies in ((1, "charging_station0", 11.0, (6.0, 14.5, 13.0, 5.5)),
                                        (3, "charging_station0", 11.0, (8.0, 17.0, 16.5, 7.0)),
                                        (3, "charging_station1", 7.4, (4.0, 9.5, 11.0, 3.5))):
        for sequence, ((first, last), energy) in enumerate(zip(HOME, energies)):
            session_id = f"{site}:{process[-1]}:{sequence}"
            sessions.append({"session_id": session_id, "site": site, "process": process, "energy_kwh": energy,
                             "charger_kw": kw})
            for order, t in enumerate(range(first, last + 1)):
                fraction = 0.5 if t in (first, last) and first != 1 and last != HOURS else 1.0
                hours.append({"session_id": session_id, "t": t, "order": order, "available_fraction": fraction})
    assert availability.sum() > 0
    return pd.DataFrame(sessions), pd.DataFrame(hours)


def write_input(path: Path, variant: str = "heuristic") -> Path:
    """Write the ``urbs_in/*`` tables of ``variant`` to ``path`` and return it."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}")
    p = _profiles()
    availability = _availability()
    commodity = []
    for site in SITES:
        commodity += [(site, "electricity", "Demand", NAN), (site, "electricity_import", "Buy", 1.0),
                      (site, "electricity_feed_in", "Sell", 1.0)]
    for site in (1, 2):
        commodity += [(site, f"solar_{site}", "SupIm", NAN), (site, "common_heat", "Stock", NAN),
                      (site, "space_heat", "Demand", NAN), (site, "water_heat", "Demand", NAN)]
    if variant != "ev_sessions":
        commodity += [(1, "mobility0", "Demand", NAN), (3, "mobility0", "Demand", NAN),
                      (3, "mobility1", "Demand", NAN)]
    commodity = pd.DataFrame(commodity, columns=["Site", "Commodity", "Type", "price"])
    process = pd.DataFrame(_process_rows(variant), columns=[
        "Site", "Process", "inst-cap", "cap-up", "inv-cost-fix", "inv-cost", "fix-cost", "var-cost",
        "wacc", "depreciation"])
    ratios = [("import", "electricity_import", "In", 1), ("import", "electricity", "Out", 1),
              ("feed_in", "electricity", "In", 1), ("feed_in", "electricity_feed_in", "Out", 1),
              ("heatpump_air", "electricity", "In", 1), ("heatpump_air", "common_heat", "Out", 1),
              ("heatpump_booster", "electricity", "In", 1), ("heatpump_booster", "common_heat", "Out", 1),
              ("Heat_dummy_space", "common_heat", "In", 1), ("Heat_dummy_space", "space_heat", "Out", 1),
              ("Heat_dummy_water", "common_heat", "In", 1), ("Heat_dummy_water", "water_heat", "Out", 1),
              ("charging_station0", "electricity", "In", 1), ("charging_station1", "electricity", "In", 1)]
    for site in (1, 2):
        ratios += [(f"Rooftop PV_{site}", f"solar_{site}", "In", 1), (f"Rooftop PV_{site}", "electricity", "Out", 1)]
    if variant != "ev_sessions":
        ratios += [("charging_station0", "mobility0", "Out", 1), ("charging_station1", "mobility1", "Out", 1)]
    process_commodity = pd.DataFrame(ratios, columns=["Process", "Commodity", "Direction", "ratio"])
    storage = pd.DataFrame(_storage_rows(variant), columns=[
        "Site", "Storage", "Commodity", "inst-cap-c", "cap-up-c", "inst-cap-p", "cap-up-p",
        "eff-in", "eff-out", "discharge", "ep-ratio", "inv-cost-p", "inv-cost-c",
        "fix-cost-p", "fix-cost-c", "var-cost-p", "wacc", "depreciation",
        "linked-process", "max-energy-per-process-capacity"])

    demand = {(1, "electricity"): p["elec_1"], (2, "electricity"): p["elec_2"], (3, "electricity"): p["elec_3"],
              (1, "space_heat"): p["space_1"], (2, "space_heat"): p["space_2"],
              (1, "water_heat"): p["water_1"], (2, "water_heat"): p["water_2"]}
    eff_factor = {(1, "heatpump_air"): p["cop"], (2, "heatpump_air"): p["cop"] * 0.97}
    driving = (1.0 - availability) * (1.1 + 0.2 * np.sin(_hours()))
    if variant == "ev_sessions":
        # the paired pipeline keeps the connected fraction as the chargers' eff_factor (no effect)
        eff_factor.update({(1, "charging_station0"): availability, (3, "charging_station0"): availability,
                           (3, "charging_station1"): availability})
    else:
        demand.update({(1, "mobility0"): driving, (3, "mobility0"): driving * 1.3, (3, "mobility1"): driving * 0.7})
        eff_factor.update({(1, "charging_station0"): availability, (3, "charging_station0"): availability,
                           (3, "charging_station1"): availability})
    frames = {
        "commodity": commodity, "process": process, "process_commodity": process_commodity, "storage": storage,
        "demand": pd.DataFrame({k: _series(v) for k, v in demand.items()}),
        "supim": pd.DataFrame({(1, "solar_1"): _series(p["pv_1"]), (2, "solar_2"): _series(p["pv_2"])}),
        "eff_factor": pd.DataFrame({k: _series(v) for k, v in eff_factor.items()}),
        "buy_sell_price": pd.DataFrame({"electricity_import": _series(p["price"]),
                                        "electricity_feed_in": _series(p["tariff"])}),
        "weather": pd.DataFrame({("ambient", "Tamb"): _series(np.cos(_hours())),
                                 ("ambient", "Irradiation"): _series(100.0 * p["pv_1"])}),
    }
    for key in ("demand", "supim", "eff_factor"):
        frames[key].columns = pd.MultiIndex.from_tuples(frames[key].columns, names=["Site", "Commodity"])
    if variant == "ev_sessions":
        frames["ev_sessions"], frames["ev_session_hours"] = _sessions(availability)
    with warnings.catch_warnings(), pd.HDFStore(path, "w") as store:
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        for key, frame in frames.items():
            store[f"urbs_in/{key}"] = frame
    return path


def read_prepared(path: Path):
    """The urbs input of ``path`` prepared as Step 3 does it: ``(data, mode, settings)``."""
    from gridexpand.optimization.urbs.features.typeperiod import select_predefined_timesteps
    from gridexpand.optimization.urbs.identify import identify_mode
    from gridexpand.optimization.urbs.input import read_input_h5
    from gridexpand.optimization.urbs.scenarios import insert_scenario

    data = read_input_h5(path)
    settings = {"tsam": False, "hoursPerPeriod": 168, "dt": 1, "timesteps": range(0, HOURS + 1)}
    data = insert_scenario(data, settings)
    mode = identify_mode(data)
    data, _ = select_predefined_timesteps(data, settings["timesteps"])
    return data, mode, settings
