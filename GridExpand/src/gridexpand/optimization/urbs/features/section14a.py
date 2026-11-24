"""Section 14a EnWG control on the HEMS optimization (Lee, 2026, Section 3.6).

The controlled load of the flexible case is an outcome of the optimization, so
the control needs a first run without control (``ref_path``, a Step 3 result of
the same input). From it:

- Congested hours are the hours in which the import of the first run, divided by
  ``PF_ELC``, exceeds the transformer rating.
- Controllable devices are heat-pump systems (heat pump and auxiliary heater of
  a site carry one setpoint, Ziffer 2.4.1.b, 2.4.2), charging stations and
  private batteries while charging (Ziffer 2.4.1.d), each with more than 4.2 kW.
- In a congested hour every device gets the setpoint
  ``cap - ratio * (cap - p_min)`` with one common ratio, the smallest that
  clears the overload of the first run (Ziffer 4.2, 4.3, 4.4.a); ``p_min`` is
  4.2 kW, or 0.4 x connection power for heat pumps above 11 kW (Ziffer 4.5.1).
- Outside congested hours each device may draw its first-run draw plus a share
  of the headroom below the rating, in proportion to its free capacity
  (Ziffer 4.3, staged release).

The second run limits the electricity input of these devices to the setpoint
(``tau <= dt * cap * share``; for batteries the storage input). The capacities
of the controlled devices are fixed to the first run, so a device cannot grow to
escape its setpoint. A last-resort heat source and charger with a high price are
added at controlled sites to keep the model feasible; their throughput is the
unserved heat and mobility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyomo.environ as pyomo

P14A_KW = 4.2
LARGE_HP_KW = 11.0
LARGE_HP_SHARE = 0.4
DEFICIT_PRICE = 1000.0      # EUR/kWh, far above any real source
BACKUP_CAPACITY_KW = 2000.0
TOL = 1e-9


def _smallest_ratio(over, segments):
    """Smallest common ratio that removes ``over`` (kVA); ``(ratio, cleared)``."""
    if not segments:
        return 0.0, False
    segments = sorted(segments)
    slope = offset = 0.0
    for k, (breakpoint, step) in enumerate(segments):
        nxt = segments[k + 1][0] if k + 1 < len(segments) else 1.0
        slope += step
        offset += step * breakpoint
        if slope * nxt - offset >= over - TOL:
            return min(1.0, max(breakpoint, (over + offset) / slope)), True
    return 1.0, False


def _hourly(frame, mask, steps):
    """Sum of the selected rows of a ``t``-indexed result series, one value per step."""
    return frame[mask].groupby("t").val.sum().reindex(steps, fill_value=0.0).to_numpy(dtype=float)


def _pin(table, rows, columns, value):
    for column in columns:
        if table[column].dtype.kind in "iu":
            table[column] = table[column].astype(float)
    table.loc[rows, columns] = value


def _add_backup(data, stf, site, name, deficit, product):
    """Last-resort source: a Buy commodity ``deficit`` feeding process ``name`` into ``product``."""
    commodity, process, ratios = data["commodity"], data["process"], data["process_commodity"]
    commodity.loc[(stf, site, deficit, "Buy"), "price"] = 1.0
    row = process.loc[(stf, site, "import")].copy()
    row[["inst-cap", "cap-up"]] = BACKUP_CAPACITY_KW
    for column in ("inv-cost-fix", "inv-cost", "fix-cost", "var-cost"):
        if column in row.index:
            row[column] = 0.0
    process.loc[(stf, site, name)] = row
    ratios.loc[(stf, name, deficit, "In"), "ratio"] = 1.0
    ratios.loc[(stf, name, product, "Out"), "ratio"] = 1.0
    data["buy_sell_price"][deficit] = DEFICIT_PRICE


def _output(data, stf, name):
    """The commodity a process produces (its non-electricity output)."""
    ratios = data["process_commodity"]
    rows = [key for key in ratios.index
            if key[0] == stf and key[1] == name and key[3] == "Out" and key[2] != "electricity"]
    if len(rows) != 1:
        raise ValueError(f"Section 14a backup: cannot identify the output commodity of {name!r}.")
    return rows[0][2]


def apply_setpoints(data, ref_path, rating_kva, pf_elc):
    """Add the Section 14a setpoints from a first run to the input ``data``.

    Args:
        data: prepared urbs input (after the temporal method).
        ref_path: Step 3 result of the same input without control.
        rating_kva: transformer rating.
        pf_elc: power factor of the import.

    Returns:
        ``(data, diagnostics)``; ``data`` gains ``p14a_process`` and ``p14a_storage``
        (share of the capacity per ``(stf, t)`` and ``(site, process|storage)``).
    """
    with pd.HDFStore(ref_path, "r") as store:
        cap_pro = store["urbs_out/MILP/cap_pro"]
        e_in = store["urbs_out/MILP/e_pro_in"].rename("val").reset_index()
        cap_sto = store["urbs_out/MILP/cap_sto_p"] if "/urbs_out/MILP/cap_sto_p" in store else None
        e_sto = (store["urbs_out/MILP/e_sto_in"].rename("val").reset_index()
                 if "/urbs_out/MILP/e_sto_in" in store else None)

    process, storage = data["process"], data["storage"]
    stf = process.index.get_level_values(0)[0]
    steps = np.sort(data["demand"].index.get_level_values("t").unique())
    steps = steps[steps > 0]
    limit = float(rating_kva)
    load = _hourly(e_in, e_in.pro == "import", steps) / pf_elc
    congested = load > limit + TOL

    def first_cap(site, name):
        key = (stf, site, name)
        return float(cap_pro.get(key, 0.0))

    devices, heat_sites, charge_points = [], [], []
    for (_, site, name) in process.index:
        if name == "heatpump_air":
            units = [u for u in ("heatpump_air", "heatpump_booster") if (stf, site, u) in process.index]
            caps = {u: first_cap(site, u) for u in units}
            cap = sum(caps.values())
            if cap <= P14A_KW:
                continue
            for unit, value in caps.items():
                _pin(process, (stf, site, unit), ["inst-cap", "cap-up"], value)
            draw = _hourly(e_in, (e_in.sit == site) & e_in.pro.isin(units) & (e_in.com == "electricity"), steps)
            devices.append(dict(kind="pro", cols=[(site, u) for u in units], cap=cap,
                                p_min=LARGE_HP_SHARE * cap if cap > LARGE_HP_KW else P14A_KW, draw=draw))
            heat_sites.append(site)
        elif str(name).startswith("charging_station"):
            cap = first_cap(site, name)
            if cap <= P14A_KW:
                continue
            _pin(process, (stf, site, name), ["inst-cap", "cap-up"], cap)
            draw = _hourly(e_in, (e_in.sit == site) & (e_in.pro == name) & (e_in.com == "electricity"), steps)
            devices.append(dict(kind="pro", cols=[(site, name)], cap=cap, p_min=P14A_KW, draw=draw))
            charge_points.append((site, str(name)[len("charging_station"):]))
    if cap_sto is not None and e_sto is not None:
        for (_, site, name, com), cap in cap_sto.items():
            if name != "battery_private" or com != "electricity" or cap <= P14A_KW:
                continue
            if (stf, site, name, com) not in storage.index:
                continue
            _pin(storage, (stf, site, name, com), ["inst-cap-p", "cap-up-p"], float(cap))
            draw = _hourly(e_sto, (e_sto.sit == site) & (e_sto.sto == name) & (e_sto.com == com), steps)
            devices.append(dict(kind="sto", cols=[(site, name)], cap=float(cap), p_min=P14A_KW, draw=draw))

    ratio = np.zeros(len(steps))
    n_not_cleared = 0
    for i in np.flatnonzero(congested):
        segments = []
        for d in devices:
            span = d["cap"] - d["p_min"]
            if span > TOL and d["draw"][i] > d["p_min"]:
                segments.append(((d["cap"] - d["draw"][i]) / span, span / pf_elc))
        ratio[i], cleared = _smallest_ratio(load[i] - limit, segments)
        n_not_cleared += int(not cleared)

    # staged release: first-run draw plus a share of the headroom, in proportion to the free capacity
    spare = np.maximum(0.0, (limit - load) * pf_elc)
    room = sum((np.maximum(0.0, d["cap"] - d["draw"]) for d in devices), np.zeros(len(steps)))
    release = np.minimum(np.divide(spare, room, out=np.zeros(len(steps)), where=room > TOL), 1.0)

    index = pd.MultiIndex.from_product([[stf], steps], names=["support_timeframe", "t"])
    shares = {"pro": {}, "sto": {}}
    for d in devices:
        setpoint = d["cap"] - ratio * (d["cap"] - d["p_min"])
        recovery = d["draw"] + release * np.maximum(0.0, d["cap"] - d["draw"])
        allowed = np.where(congested, setpoint, np.minimum(recovery, d["cap"]))
        for column in d["cols"]:
            shares[d["kind"]][column] = np.clip(allowed / d["cap"], 0.0, 1.0)
    for kind, key in (("pro", "p14a_process"), ("sto", "p14a_storage")):
        frame = pd.DataFrame(shares[kind], index=index)
        if not frame.empty:
            frame.columns = pd.MultiIndex.from_tuples(frame.columns)
        data[key] = frame

    if heat_sites:
        heat = _output(data, stf, "heatpump_air")
        for site in sorted(set(heat_sites)):
            _add_backup(data, stf, site, "heat_backup", "heat_deficit", heat)
    for site, number in sorted(set(charge_points)):
        name = f"charging_station{number}"
        _add_backup(data, stf, site, f"mobility_backup{number}", "mobility_deficit", _output(data, stf, name))
    for key in ("commodity", "process", "process_commodity"):
        data[key] = data[key].sort_index()

    diagnostics = {
        "rating_kva": limit,
        "peak_first_run_kva": float(load.max()) if len(load) else 0.0,
        "congested_hours": int(congested.sum()),
        "hours_not_cleared": n_not_cleared,
        "mean_ratio_congested": float(ratio[congested].mean()) if congested.any() else 0.0,
        "controlled_heat_pumps": len(heat_sites),
        "controlled_charge_points": len(charge_points),
        "controlled_batteries": sum(1 for d in devices if d["kind"] == "sto"),
    }
    return data, diagnostics


def add_section14a(m, data):
    """Setpoint constraints of the controlled devices of this cluster (no-op without setpoints)."""
    process_shares = data.get("p14a_process")
    if process_shares is not None and not process_shares.empty:
        shares = process_shares.to_dict()
        m.p14a_pro_tuples = pyomo.Set(
            within=m.stf * m.sit * m.pro,
            initialize=[(stf, sit, pro) for (stf, sit, pro) in m.pro_tuples if (sit, pro) in shares],
            doc="Processes under Section 14a control")
        m.res_p14a_process = pyomo.Constraint(
            m.tm, m.p14a_pro_tuples,
            rule=lambda m, tm, stf, sit, pro: m.tau_pro[tm, stf, sit, pro]
            <= m.dt * m.cap_pro[stf, sit, pro] * shares[(sit, pro)][(stf, tm)],
            doc="tau_pro <= dt * cap_pro * Section 14a share")
    storage_shares = data.get("p14a_storage")
    if storage_shares is not None and not storage_shares.empty and m.mode["sto"]:
        shares_sto = storage_shares.to_dict()
        m.p14a_sto_tuples = pyomo.Set(
            within=m.stf * m.sit * m.sto * m.com,
            initialize=[(stf, sit, sto, com) for (stf, sit, sto, com) in m.sto_tuples if (sit, sto) in shares_sto],
            doc="Storages under Section 14a control while charging")
        m.res_p14a_storage = pyomo.Constraint(
            m.tm, m.p14a_sto_tuples,
            rule=lambda m, tm, stf, sit, sto, com: m.e_sto_in[tm, stf, sit, sto, com]
            <= m.dt * m.cap_sto_p[stf, sit, sto, com] * shares_sto[(sit, sto)][(stf, tm)],
            doc="e_sto_in <= dt * cap_sto_p * Section 14a share")
    return m
