"""The PyPSA model of the buildings of one building group (Step 3, PyPSA optimizer).

PyPSA calls its model container a ``pypsa.Network``. Here it holds independent
building energy systems, not a power grid: a PyPSA *bus* is one energy balance of
one building (named ``<site>|<carrier>``), there are no lines, and no component
connects two buildings (the LV grid is modelled in Step 4 only).

One building, with the assets of the study's scenarios (fixed capacities in the
heuristic cases, sized capacities where the optimized case invests):

    bus <site>|electricity        Load: household electricity demand
        Generator   grid import       marginal cost = import price(t)
        Generator   feed-in           p <= 0, marginal cost = feed-in tariff(t) (revenue)
        Generator   rooftop PV        p = capacity * supim(t): must-take, no curtailment
        StorageUnit battery           energy = ep-ratio * power
        Link        heat pump         -> common_heat, efficiency = COP(t)
        Link        heating rod       -> common_heat, efficiency 1
        Link        charging station  -> mobility<i>, efficiency = availability(t)   (legacy EVs)
        Generator   EV charger        p = -charging within the sessions              (EV sessions)
    bus <site>|common_heat
        Link        -> space_heat, -> water_heat (urbs' Heat_dummy pass-throughs)
    bus <site>|space_heat         Load; StorageUnit heat storage (energy <= ratio * heat-pump capacity)
    bus <site>|water_heat         Load
    bus <site>|mobility<i>        Load: driving; StorageUnit car battery        (legacy EVs)

The components are read from the urbs input tables and reproduce the urbs model
of ``gridexpand.optimization.urbs`` (``tests/optimization/test_pypsa_equivalence.py``).
The urbs parts PyPSA has no attribute for are in ``constraints.py``. Only what the
scenarios use is mapped; anything else raises ``NotImplementedError``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gridexpand.optimization.urbs.features.modelhelper import invcost_factor

EPS = 1e-9
HOURS_PER_YEAR = 8760.0

# ModelParts.processes['kind']
IMPORT, FEED_IN, PV, LINK, EV_CHARGER = "import", "feed_in", "pv", "link", "ev_charger"


@dataclass
class ModelParts:
    """What the constraints, the results and the cost breakdown need besides the PyPSA model.

    Attributes:
        stf: The support timeframe (index level ``stf`` of all urbs results).
        snapshots: Modelled timesteps ``t = 1..T``.
        weight: urbs' annual weight ``8760 / T`` (1 for the full year).
        bsp: Buy/sell prices are modelled (``Revenue`` and ``Purchase`` cost types).
        processes: One row per urbs process (index: component name): site, process,
            kind, component, inst, up, ext, inv, inv_fix, annuity, capital_cost.
        prices: Price series per import and feed-in process.
        storages: One row per urbs storage (index: component name): site, storage,
            commodity, ext, inst_c, inst_p, max_hours, inv_p, inv_c, var, annuity.
        linked: (storage, process, energy per process capacity) triples.
        ev_label: Session code per (snapshot, charger) or NaN; None without sessions.
        ev_energy: Required energy per session code.
    """

    stf: object
    snapshots: pd.Index
    weight: float
    bsp: bool
    processes: pd.DataFrame
    prices: dict[str, pd.Series]
    storages: pd.DataFrame
    linked: list[tuple[str, str, float]]
    ev_label: pd.DataFrame | None
    ev_energy: np.ndarray | None


def component_name(site, name) -> str:
    return f"{site}|{name}"


def unsupported(message: str):
    return NotImplementedError(f"PyPSA optimizer: {message}")


def _modelled(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows of the modelled timesteps ``t >= 1`` (drops the urbs initialization row)."""
    return frame[frame.index.get_level_values("t") > 0]


def _series(frame: pd.DataFrame, column, snapshots: pd.Index) -> pd.Series:
    return pd.Series(_modelled(frame)[column].to_numpy(dtype=float), index=snapshots)


def build_building_model(data: dict, mode: dict):
    """Return ``(network, parts)``: the PyPSA model of the buildings in ``data``.

    Args:
        data: urbs input dict of one building group (``urbs.input.get_cluster_data``),
            restricted to the modelled timesteps.
        mode: urbs features of the input (``urbs.identify.identify_mode``).
    """
    import pypsa

    if mode.get("tsam") or mode.get("tdy"):
        raise unsupported("full-year chronological inputs only; TSAM type periods are not implemented.")
    t = data["demand"].index.get_level_values("t")
    snapshots = pd.Index(sorted(int(v) for v in set(t) if v > 0), name="snapshot")
    weight = HOURS_PER_YEAR / len(snapshots)
    stf = data["process"].index.get_level_values(0)[0]

    commodity = data["commodity"].reset_index()
    com_type = {(r.Site, r.Commodity): r.Type for r in commodity.itertuples(index=False)}
    com_price = {(r.Site, r.Commodity): r.price for r in commodity.itertuples(index=False)}
    r_in: dict[str, dict] = {}
    r_out: dict[str, dict] = {}
    for r in data["process_commodity"].reset_index().itertuples(index=False):
        (r_in if r.Direction == "In" else r_out).setdefault(r.Process, {})[r.Commodity] = float(r.ratio)

    network = pypsa.Network()
    network.set_snapshots(snapshots)
    network.snapshot_weightings.loc[:, "objective"] = weight

    # one bus per energy balance of a building, and the demands
    balances = [key for key, typ in com_type.items() if typ not in ("SupIm", "Buy", "Sell")]
    bus = {key: component_name(*key) for key in balances}
    network.add("Carrier", sorted({key[1] for key in balances}))
    network.add("Bus", [bus[key] for key in balances], carrier=[key[1] for key in balances])
    demand = data["demand"]
    loads = [key for key in balances if com_type[key] == "Demand" and key in demand.columns]
    network.add(
        "Load", [f"{bus[key]}|demand" for key in loads], bus=[bus[key] for key in loads],
        p_set=pd.DataFrame({f"{bus[key]}|demand": _series(demand, key, snapshots) for key in loads}),
    )

    fractions = _session_fractions(data, snapshots) if mode.get("evs") else {}
    rows, generators, links, prices = [], [], [], {}
    for r in data["process"].reset_index().to_dict("records"):
        site, process = r["Site"], r["Process"]
        name = component_name(site, process)
        if float(r["fix-cost"]) != 0 or float(r["var-cost"]) != 0:
            raise unsupported(f"process {process!r} has fixed or variable operating costs.")
        inputs, outputs = r_in.get(process, {}), r_out.get(process, {})
        if len(inputs) != 1 or len(outputs) > 1 or any(abs(v - 1.0) > EPS for v in inputs.values()):
            raise unsupported(f"process {process!r} needs one input with ratio 1 and at most one output.")
        (cin,) = inputs
        row = {"name": name, "site": site, "process": process,
               "inst": float(r["inst-cap"]), "up": float(r["cap-up"]),
               "inv": float(r["inv-cost"]), "inv_fix": float(np.nan_to_num(r["inv-cost-fix"])),
               "annuity": float(invcost_factor(r["depreciation"], r["wacc"]))}
        row["ext"] = row["up"] > row["inst"] + EPS
        if row["ext"] and row["inst"] > EPS:
            raise unsupported(f"process {process!r} is sized on top of an existing capacity.")
        # a sized component starts at 0; a fixed one has p_nom = its capacity
        spec = {"name": name, "p_nom": 0.0 if row["ext"] else row["up"], "p_nom_extendable": row["ext"],
                "p_nom_max": row["up"] if row["ext"] else np.inf}
        if not outputs:
            if not mode.get("evs") or row["ext"]:
                raise unsupported(f"process {process!r} without output (only fixed EV-session chargers have none).")
            # urbs bounds a charger only in the hours of its sessions (none: unbounded)
            fraction = fractions.get(name)
            row.update(kind=EV_CHARGER, component="Generator")
            generators.append({**spec, "bus": bus[(site, cin)], "marginal_cost": 0.0,
                               "p_min_pu": -1.0 if fraction is None else -fraction, "p_max_pu": 0.0})
        else:
            ((cout, ratio_out),) = outputs.items()
            tin, tout = com_type.get((site, cin)), com_type.get((site, cout))
            if tin == "Buy" or tout == "Sell":
                if abs(ratio_out - 1.0) > EPS:
                    raise unsupported(f"grid process {process!r} with output ratio {ratio_out}.")
                commodity_name = cin if tin == "Buy" else cout
                price = _price_series(data, site, commodity_name, com_price, snapshots)
                prices[name] = price
                if tin == "Buy":  # import: 0 <= p <= capacity
                    kind, bus_name, p_min, p_max = IMPORT, bus[(site, cout)], 0.0, 1.0
                else:  # feed-in: -capacity <= p <= 0, so marginal cost * p is the revenue
                    kind, bus_name, p_min, p_max = FEED_IN, bus[(site, cin)], -1.0, 0.0
                row.update(kind=kind, component="Generator")
                generators.append({**spec, "bus": bus_name, "marginal_cost": price,
                                   "p_min_pu": p_min, "p_max_pu": p_max})
            elif tin == "SupIm":
                profile = _series(data["supim"], (site, cin), snapshots) * ratio_out
                row.update(kind=PV, component="Generator")
                generators.append({**spec, "bus": bus[(site, cout)], "marginal_cost": 0.0,
                                   "p_min_pu": profile, "p_max_pu": profile})
            else:
                efficiency = ratio_out
                if (site, process) in data["eff_factor"].columns:  # COP or availability
                    efficiency = _series(data["eff_factor"], (site, process), snapshots) * ratio_out
                row.update(kind=LINK, component="Link")
                links.append({**spec, "bus0": bus[(site, cin)], "bus1": bus[(site, cout)],
                              "efficiency": efficiency})
        rows.append(row)

    processes = pd.DataFrame(rows).set_index("name")
    processes["capital_cost"] = np.where(processes["ext"], processes["inv"] * processes["annuity"], 0.0)
    _add_components(network, "Generator", generators, processes["capital_cost"])
    _add_components(network, "Link", links, processes["capital_cost"])
    storages, units = _storages(data, bus)
    _add_components(network, "StorageUnit", units, None)

    ev_label, ev_energy = _session_labels(data, snapshots) if fractions else (None, None)
    parts = ModelParts(
        stf=stf, snapshots=snapshots, weight=weight, bsp=bool(mode.get("bsp")),
        processes=processes, prices=prices, storages=storages,
        linked=_linked_storages(data, storages, processes), ev_label=ev_label, ev_energy=ev_energy,
    )
    return network, parts


def _price_series(data, site, commodity, com_price, snapshots) -> pd.Series:
    prices = data["buy_sell_price"]
    if commodity not in prices.columns:
        raise unsupported(f"no buy/sell price column {commodity!r}.")
    return _series(prices, commodity, snapshots) * float(com_price[(site, commodity)])


_TABLES = {"Generator": ("generators", "generators_t"), "Link": ("links", "links_t"),
           "StorageUnit": ("storage_units", "storage_units_t")}


def _add_components(network, component: str, items: list[dict], capital_cost: pd.Series | None) -> None:
    """Add ``items``; an attribute given as a series for some items becomes a time series."""
    if not items:
        return
    names = [item["name"] for item in items]
    static, mixed = {}, {}
    for key in items[0]:
        if key != "name":
            values = [item[key] for item in items]
            (mixed if any(isinstance(value, pd.Series) for value in values) else static)[key] = values
    if capital_cost is not None:
        static["capital_cost"] = capital_cost.loc[names].tolist()
    network.add(component, names, **static)
    table, dynamic = (getattr(network, attr) for attr in _TABLES[component])
    for key, values in mixed.items():
        scalars = {n: v for n, v in zip(names, values) if not isinstance(v, pd.Series)}
        if scalars:
            table.loc[list(scalars), key] = list(scalars.values())
        frame = pd.DataFrame({n: v for n, v in zip(names, values) if isinstance(v, pd.Series)})
        dynamic[key] = frame if dynamic[key].empty else pd.concat([dynamic[key], frame], axis=1)


def _storages(data: dict, bus: dict) -> tuple[pd.DataFrame, list[dict]]:
    """One StorageUnit per urbs storage: fixed, or sized with a fixed energy/power ratio."""
    columns = ["site", "storage", "commodity", "ext", "inst_c", "inst_p", "max_hours", "inv_p", "inv_c",
               "var", "annuity"]
    table = data["storage"].dropna(axis=0, how="all")
    rows, units = [], []
    for r in table.reset_index().to_dict("records"):
        site, storage = r["Site"], r["Storage"]
        name = component_name(site, storage)
        inst_c, up_c, inst_p, up_p = (float(r[k]) for k in ("inst-cap-c", "cap-up-c", "inst-cap-p", "cap-up-p"))
        ep = float(r["ep-ratio"]) if pd.notna(r.get("ep-ratio")) and r["ep-ratio"] > 0 else None
        ext = up_c > inst_c + EPS or up_p > inst_p + EPS
        if float(r["fix-cost-p"]) != 0 or float(r["fix-cost-c"]) != 0:
            raise unsupported(f"storage {storage!r} has fixed operating costs.")
        if ext and (ep is None or inst_c > EPS or inst_p > EPS):
            raise unsupported(f"storage {storage!r}: sizing needs an ep-ratio and no existing capacity.")
        if not ext and inst_p <= EPS:
            raise unsupported(f"storage {storage!r} at site {site!r} has no power.")
        if not ext and ep is not None and abs(inst_c - ep * inst_p) > 1e-6 * max(1.0, inst_c):
            raise ValueError(f"Storage {storage!r} at site {site!r}: its capacities violate its ep-ratio {ep}.")
        max_hours = ep if ext else inst_c / inst_p
        annuity = float(invcost_factor(r["depreciation"], r["wacc"]))
        rows.append({"name": name, "site": site, "storage": storage, "commodity": r["Commodity"], "ext": ext,
                     "inst_c": inst_c, "inst_p": inst_p, "max_hours": max_hours, "inv_p": float(r["inv-cost-p"]),
                     "inv_c": float(r["inv-cost-c"]), "var": float(r["var-cost-p"]), "annuity": annuity})
        units.append({
            "name": name, "bus": bus[(site, r["Commodity"])], "p_nom": 0.0 if ext else inst_p,
            "p_nom_extendable": ext, "p_nom_max": min(up_p, up_c / ep) if ext else np.inf,
            "max_hours": max_hours,
            "capital_cost": (float(r["inv-cost-p"]) + max_hours * float(r["inv-cost-c"])) * annuity if ext else 0.0,
            "efficiency_store": float(r["eff-in"]), "efficiency_dispatch": float(r["eff-out"]),
            "standing_loss": float(r["discharge"]), "marginal_cost": float(r["var-cost-p"]),
            "cyclic_state_of_charge": True,
        })
    frame = pd.DataFrame(rows, columns=["name", *columns]).set_index("name")
    frame["ext"] = frame["ext"].astype(bool)  # also for buildings without any storage
    return frame, units


def _linked_storages(data: dict, storages: pd.DataFrame, processes: pd.DataFrame) -> list[tuple[str, str, float]]:
    """Storages whose energy is bounded by a process capacity (heat storage vs heat pump)."""
    table = data["storage"].dropna(axis=0, how="all")
    if "linked-process" not in table.columns:
        return []
    linked = []
    for r in table.reset_index().to_dict("records"):
        process, ratio = r.get("linked-process"), r.get("max-energy-per-process-capacity")
        if pd.isna(process) and pd.isna(ratio):
            continue
        if pd.isna(process) or pd.isna(ratio) or float(ratio) <= 0:
            raise ValueError("Storage capacity linkage requires linked-process and a positive "
                             "max-energy-per-process-capacity.")
        storage, target = component_name(r["Site"], r["Storage"]), component_name(r["Site"], str(process))
        if target not in processes.index:
            raise ValueError(f"Storage {storage!r} references missing process {target!r}.")
        if storages.at[storage, "ext"] != processes.at[target, "ext"]:
            raise unsupported(f"storage {storage!r} and its linked process must both be fixed or both sized.")
        linked.append((storage, target, float(ratio)))
    return linked


def _session_hours(data: dict) -> pd.DataFrame:
    sessions = data["ev_sessions"][["session_id", "site", "process"]]
    hours = data["ev_session_hours"].merge(sessions, on="session_id", how="left", validate="many_to_one")
    if hours["site"].isna().any():
        raise ValueError("EV session hours reference unknown sessions.")
    hours["name"] = [component_name(site, process) for site, process in zip(hours["site"], hours["process"])]
    if hours.duplicated(["name", "t"]).any():
        raise ValueError("Two EV sessions of one vehicle claim the same model hour.")
    return hours


def _session_fractions(data: dict, snapshots: pd.Index) -> dict[str, pd.Series]:
    """Connected fraction per charger (component name) and snapshot; 0 outside its sessions."""
    if data.get("ev_sessions") is None or data["ev_sessions"].empty:
        return {}
    fractions = {}
    for name, group in _session_hours(data).groupby("name", sort=False):
        fraction = pd.Series(0.0, index=snapshots)
        fraction.loc[group["t"].to_numpy()] = group["available_fraction"].to_numpy(dtype=float)
        fractions[name] = fraction
    return fractions


def _session_labels(data: dict, snapshots: pd.Index) -> tuple[pd.DataFrame, np.ndarray]:
    """Session code per (snapshot, charger), for one vectorised energy constraint."""
    sessions = data["ev_sessions"].reset_index(drop=True)
    codes = pd.Series(np.arange(len(sessions)), index=sessions["session_id"].astype(str))
    hours = _session_hours(data)
    missing = sorted(set(codes.index) - set(hours["session_id"].astype(str)))
    if missing:
        raise ValueError(f"EV sessions without admissible hours: {missing[:5]}")
    chargers = list(dict.fromkeys(hours["name"]))
    values = np.full((len(snapshots), len(chargers)), np.nan)
    values[snapshots.get_indexer(hours["t"].to_numpy()), pd.Index(chargers).get_indexer(hours["name"])] = (
        codes.loc[hours["session_id"].astype(str)].to_numpy(dtype=float)
    )
    return pd.DataFrame(values, index=snapshots, columns=chargers), sessions["energy_kwh"].to_numpy(dtype=float)
