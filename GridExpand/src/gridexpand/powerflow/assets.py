"""Building assets of a Step 3 result: the capacities a post-case power flow simulates.

urbs names one PV process per building (``Rooftop PV_<method>_<objectid>``), one
``mobility_storage<i>`` and ``charging_station<i>`` per vehicle, and one heat pump,
heating rod, heat storage and battery per site (= pandapower bus). The rows feed
``surrogrid.powerflow_asset`` (map layer of the GridExpand UI).
"""

from __future__ import annotations

import re

import pandas as pd

COLUMNS = ["bus", "technology", "building_objectid", "units", "power_kw", "energy_kwh"]
# Capacities below 1 W / 1 Wh are "not built" (the optimisation reports zeros and solver noise).
MIN_CAPACITY = 1e-3
PV_PROCESS = re.compile(r"^Rooftop PV_[a-z]+_(?P<objectid>.+)$")
EV_STORAGE = re.compile(r"^mobility_storage\d+$")
EV_CHARGER = re.compile(r"^charging_station\d+$")
PROCESS_TECHNOLOGY = {"heatpump_air": "heat_pump", "heatpump_booster": "heating_rod"}
STORAGE_TECHNOLOGY = {"battery_private": "battery", "heat_storage": "heat_storage"}


def _last_year(capacity: pd.Series, name: str) -> pd.DataFrame:
    """Capacities of the last support timeframe as a frame with ``sit``, ``name``, ``value``."""
    if capacity is None or capacity.empty:
        return pd.DataFrame(columns=["bus", name, "value"])
    frame = capacity.rename("value").reset_index()
    frame = frame[frame["stf"] == frame["stf"].max()]
    frame = frame[pd.to_numeric(frame["sit"], errors="coerce").notna()]
    return frame.assign(bus=frame["sit"].astype(int))[["bus", name, "value"]]


def installed_assets(cap_pro: pd.Series, cap_sto_c: pd.Series | None = None,
                     cap_sto_p: pd.Series | None = None) -> pd.DataFrame:
    """Installed PV, battery, heat pump, heating rod, heat storage and EVs per bus.

    Args:
        cap_pro: urbs ``cap_pro`` (index ``stf, sit, pro``), kW.
        cap_sto_c: urbs ``cap_sto_c`` (index ``stf, sit, sto, com``), kWh.
        cap_sto_p: urbs ``cap_sto_p`` (same index), kW.

    Returns:
        One row per bus and technology (PV: per building) with :data:`COLUMNS`; ``units``
        counts the vehicles of an ``ev`` row, whose ``power_kw`` is the sum of their
        charger capacities and ``energy_kwh`` the sum of their battery capacities.
    """
    processes = _last_year(cap_pro, "pro")
    energy = _last_year(cap_sto_c, "sto")
    power = _last_year(cap_sto_p, "sto").rename(columns={"value": "power"})
    storages = energy.merge(power, on=["bus", "sto"], how="left")
    rows: list[dict] = []

    for row in processes.itertuples(index=False):
        if row.value < MIN_CAPACITY:
            continue
        pv = PV_PROCESS.match(str(row.pro))
        if pv:
            rows.append({"bus": row.bus, "technology": "pv", "building_objectid": pv.group("objectid"),
                         "units": 1, "power_kw": float(row.value), "energy_kwh": None})
        elif row.pro in PROCESS_TECHNOLOGY:
            rows.append({"bus": row.bus, "technology": PROCESS_TECHNOLOGY[row.pro], "building_objectid": "",
                         "units": 1, "power_kw": float(row.value), "energy_kwh": None})
    for row in storages.itertuples(index=False):
        if str(row.sto).startswith("heat_storage_") and row.value >= MIN_CAPACITY:
            rows.append({"bus":row.bus,"technology":"heat_storage","building_objectid":str(row.sto)[len("heat_storage_"):],"units":1,"power_kw":None if pd.isna(row.power) else float(row.power),"energy_kwh":float(row.value)})
        elif row.sto in STORAGE_TECHNOLOGY and row.value >= MIN_CAPACITY:
            rows.append({"bus": row.bus, "technology": STORAGE_TECHNOLOGY[row.sto], "building_objectid": "",
                         "units": 1, "power_kw": None if pd.isna(row.power) else float(row.power),
                         "energy_kwh": float(row.value)})

    vehicles = storages[storages["sto"].astype(str).str.match(EV_STORAGE) & (storages["value"] >= MIN_CAPACITY)]
    chargers = processes[processes["pro"].astype(str).str.match(EV_CHARGER) & (processes["value"] >= MIN_CAPACITY)]
    charger_kw = chargers.groupby("bus")["value"].sum()
    for bus, group in vehicles.groupby("bus"):
        rows.append({"bus": int(bus), "technology": "ev", "building_objectid": "", "units": len(group),
                     "power_kw": float(charger_kw.get(bus, 0.0)) or None, "energy_kwh": float(group["value"].sum())})

    frame = pd.DataFrame(rows, columns=COLUMNS)
    return frame.sort_values(["bus", "technology", "building_objectid"], kind="mergesort").reset_index(drop=True)
