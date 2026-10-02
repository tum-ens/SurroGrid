"""The urbs model parts that PyPSA has no attribute for, as linopy constraints.

Called after ``network.optimize.create_model()``; ``results.py`` reads the added
``<component>-build`` variables.
"""

from __future__ import annotations

import linopy
import numpy as np
import pandas as pd
import xarray as xr

from .building_model import ModelParts

BUILD = "{}-build"  # binary: the process is built (urbs ``pro_cap_expands``)


def add_urbs_constraints(network, parts: ModelParts) -> None:
    """Add fixed investment costs, storage input costs, EV sessions and linked storages."""
    _fixed_investment_costs(network, parts)
    _storage_input_costs(network, parts)
    _ev_sessions(network, parts)
    _linked_storages(network, parts)
    _thermal(network, parts)


def _fixed_investment_costs(network, parts: ModelParts) -> None:
    """urbs: ``capacity <= build * cap_up`` and ``Invest += build * inv_cost_fix * annuity``."""
    m = network.model
    fixed = parts.processes[parts.processes["ext"] & (parts.processes["inv_fix"] > 0)]
    for component, rows in fixed.groupby("component", sort=True):
        names = pd.Index(rows.index, name="name")
        build = m.add_variables(binary=True, coords=[names], name=BUILD.format(component))
        m.add_constraints(m[f"{component}-p_nom"].sel(name=names) - rows["up"].rename_axis("name") * build <= 0,
                          name=f"{component}-build-upper")
        cost = (rows["inv_fix"] * rows["annuity"]).rename_axis("name")
        m.objective = m.objective.expression + (cost * build).sum()


def _storage_input_costs(network, parts: ModelParts) -> None:
    """urbs charges ``var-cost-p`` on storage input and output; PyPSA only on output."""
    storages = parts.storages[parts.storages["var"] != 0]
    if storages.empty:
        return
    m = network.model
    cost = (storages["var"] * parts.weight).rename_axis("name").to_xarray()
    m.objective = m.objective.expression + (m["StorageUnit-p_store"].sel(name=storages.index.tolist()) * cost).sum()


def _ev_sessions(network, parts: ModelParts) -> None:
    """Dedicated EV sessions: charging summed over a session's hours equals its energy.

    The terms are gathered per session (sessions x longest session). A groupby over
    all (snapshot, charger) cells would pad every session to the number of cells
    outside any session, which grows with the square of the vehicles of a building
    (16 GB for 52 vehicles); the constraints are the same.
    """
    if parts.ev_label is None:
        return
    m = network.model
    label = parts.ev_label
    codes = label.to_numpy()
    rows, cols = np.nonzero(~np.isnan(codes))
    session = codes[rows, cols].astype(np.int64)
    counts = np.bincount(session, minlength=len(parts.ev_energy))
    if len(counts) != len(parts.ev_energy) or (counts == 0).any():
        raise ValueError("Every EV session needs at least one admissible hour.")
    order = np.argsort(session, kind="stable")  # hours of a session in snapshot order
    session, rows, cols = session[order], rows[order], cols[order]
    term = np.arange(len(session)) - np.repeat(np.cumsum(counts) - counts, counts)
    labels = (m["Generator-p"].labels.sel(snapshot=label.index, name=label.columns.tolist())
              .transpose("snapshot", "name").to_numpy())
    variables = np.full((len(counts), counts.max()), -1, dtype=labels.dtype)
    coeffs = np.zeros((len(counts), counts.max()))
    variables[session, term] = labels[rows, cols]
    coeffs[session, term] = -1.0  # charging = -p of the charger generator
    coords = {"session": np.arange(len(counts))}
    per_session = linopy.LinearExpression(
        xr.Dataset({"vars": (("session", "_term"), variables), "coeffs": (("session", "_term"), coeffs)},
                   coords=coords),
        m,
    )
    energy = xr.DataArray(parts.ev_energy, coords=coords, dims="session")
    m.add_constraints(per_session == energy, name="EV-session-energy")


def _linked_storages(network, parts: ModelParts) -> None:
    """Storage energy <= ratio * capacity of the linked process (heat storage vs heat pump).

    Both fixed (heuristic cases): checked here. Both sized (optimized case): a constraint.
    """
    if not parts.linked:
        return
    frame = pd.DataFrame(parts.linked, columns=["storage", "process", "ratio"])
    frame["sized"] = parts.storages.loc[frame["storage"], "ext"].to_numpy()
    fixed = frame[~frame["sized"]]
    energy = parts.storages.loc[fixed["storage"], "inst_c"].to_numpy()
    capacity = parts.processes.loc[fixed["process"], "up"].to_numpy()
    violated = energy > fixed["ratio"].to_numpy() * capacity + 1e-6
    if violated.any():
        raise ValueError(f"Fixed storages exceed their linked process capacity: {fixed['storage'][violated].tolist()}")
    sized = frame[frame["sized"]]
    if sized.empty:
        return
    m = network.model
    for component, group in sized.groupby(parts.processes.loc[sized["process"], "component"].to_numpy()):
        index = pd.Index(group["storage"], name="linked")
        along = {"coords": {"linked": index}, "dims": "linked"}
        power = m["StorageUnit-p_nom"].sel(name=group["storage"].tolist()).rename(name="linked").assign_coords(linked=index)
        capacity = m[f"{component}-p_nom"].sel(name=group["process"].tolist()).rename(name="linked").assign_coords(linked=index)
        max_hours = xr.DataArray(parts.storages.loc[group["storage"], "max_hours"].to_numpy(), **along)
        ratio = xr.DataArray(group["ratio"].to_numpy(), **along)
        m.add_constraints(max_hours * power - ratio * capacity <= 0, name=f"linked-storage-{component}")


def _thermal(network, parts):
    """1R1C states use the same implicit-Euler interval balance as Pyomo."""
    if parts.thermal_parameters.empty:
        return
    m = network.model
    params = parts.thermal_parameters.set_index("building_objectid")
    ids = pd.Index(params.index.astype(str), name="building")
    snapshots = parts.snapshots
    def series(field):
        frame = parts.thermal_timeseries.xs(field, level=1, axis=1).xs(parts.stf, level=0)
        return xr.DataArray(frame.loc[snapshots, ids].to_numpy(), coords={"snapshot": snapshots, "building": ids}, dims=("snapshot", "building"))
    def coefficient(field):
        return xr.DataArray(params.loc[ids, field].to_numpy(), coords={"building": ids}, dims="building")
    temp = m.add_variables(lower=series("minimum_temperature_c"), upper=series("upper_temperature_c"), name="Building-temperature")
    heat = -m["Generator-p"].sel(name=[f"thermal|{bid}" for bid in ids]).rename(name="building").assign_coords(building=ids)
    h, c = coefficient("conductance_kw_per_k"), coefficient("capacitance_kwh_per_k")
    outside = series("outside_temperature_c")
    gains = series("internal_gains_kw") + series("solar_gains_kw")
    # Inputs are hourly. Unsupported resolutions are rejected by the Step-3 runner.
    m.add_constraints((c+h)*temp.isel(snapshot=0) - heat.isel(snapshot=0)
                      == c*coefficient("initial_temperature_c") + h*outside.isel(snapshot=0) + gains.isel(snapshot=0), name="Building-first-balance")
    if len(snapshots) > 1:
        current = temp.isel(snapshot=slice(1,None))
        previous = temp.isel(snapshot=slice(None,-1)).assign_coords(snapshot=snapshots[1:])
        m.add_constraints((c+h)*current - c*previous - heat.isel(snapshot=slice(1,None))
                          == (h*outside+gains).isel(snapshot=slice(1,None)), name="Building-balance")
    m.add_constraints(temp.isel(snapshot=-1) == coefficient("terminal_temperature_c"), name="Building-terminal-temperature")
    for row in parts.thermal_parameters.itertuples():
        name = f"{row.Site}|heat_storage_{row.building_objectid}"
        if name in parts.storages.index:
            route = f"{row.Site}|HP_buffer_{row.building_objectid}"
            cop = network.links_t.efficiency[route].to_xarray()
            m.add_constraints(m["Link-p"].sel(name=route)*cop == m["StorageUnit-p_store"].sel(name=name),name=f"Building-tank-charge-{row.building_objectid}")
