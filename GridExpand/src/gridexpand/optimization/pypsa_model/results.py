"""urbs-compatible result entities of a solved PyPSA building model.

The PyPSA optimizer writes these keys to ``urbs_out/MILP/`` with the index names,
index order and units of the urbs entities of the same name:

- consumed by Step 4 and the runners: ``tau_pro`` (t, stf, sit, pro), ``cap_pro``
  (stf, sit, pro), ``cap_sto_c`` and ``cap_sto_p`` (stf, sit, sto, com);
- for analyses: ``e_sto_in``, ``e_sto_out`` (t, stf, sit, sto, com),
  ``e_sto_con`` (also at the initialization step t = 0, which equals the last
  step because storages are cyclic) and ``costs`` (cost_type).

All other urbs entities (``e_pro_in``/``e_pro_out``, ``e_co_buy``/``e_co_sell``,
the ``*_new`` capacities, ``pro_cap_expands``, ``dt``, ``weight``) follow from
these and the inputs in ``urbs_out/reduced_data``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .building_model import EV_CHARGER, FEED_IN, IMPORT, ModelParts
from .constraints import BUILD

RESULT_KEYS = ("tau_pro", "cap_pro", "cap_sto_c", "cap_sto_p", "e_sto_in", "e_sto_out", "e_sto_con", "costs")


def _solution(model, name: str) -> pd.DataFrame | pd.Series | None:
    return model.variables[name].solution.to_pandas() if name in model.variables else None


def process_throughput(network, parts: ModelParts) -> pd.DataFrame:
    """urbs ``tau_pro`` per snapshot (rows) and process (columns, in input order).

    Feed-in and EV chargers have p <= 0 in PyPSA; their throughput is -p.
    """
    frames = []
    for component in ("Generator", "Link"):
        solved = _solution(network.model, f"{component}-p")
        if solved is not None:
            rows = parts.processes[parts.processes["component"] == component]
            sign = pd.Series(np.where(rows["kind"].isin([FEED_IN, EV_CHARGER]), -1.0, 1.0), index=rows.index)
            frames.append(solved[rows.index] * sign)
    tau = pd.concat(frames, axis=1)[parts.processes.index]
    tau.index = parts.snapshots
    return tau


def process_capacity(network, parts: ModelParts) -> pd.Series:
    """Process capacity (urbs ``cap_pro``): fixed or solved."""
    capacity = parts.processes["up"].copy()
    for component in ("Generator", "Link"):
        solved = _solution(network.model, f"{component}-p_nom")
        if solved is not None:
            capacity.loc[solved.index] = solved.to_numpy()
    return capacity


def storage_results(network, parts: ModelParts):
    """(power, energy, input, output, content) of every storage."""
    power = parts.storages["inst_p"].copy()
    solved = _solution(network.model, "StorageUnit-p_nom")
    if solved is not None:
        power.loc[solved.index] = solved.to_numpy()
    energy = power * parts.storages["max_hours"]
    flows = [_solution(network.model, f"StorageUnit-{name}") for name in ("p_store", "p_dispatch", "state_of_charge")]
    flows = [pd.DataFrame(index=parts.snapshots, columns=parts.storages.index, dtype=float) if f is None
             else f[parts.storages.index] for f in flows]
    return power, energy, *flows


def cost_breakdown(network, parts: ModelParts, tau: pd.DataFrame, capacity: pd.Series,
                   sto_power: pd.Series, sto_energy: pd.Series, sto_in: pd.DataFrame,
                   sto_out: pd.DataFrame) -> pd.Series:
    """urbs ``costs`` (EUR/a) by cost type, with urbs' definitions.

    Sized assets start at zero capacity and there are no fixed operating costs
    (both checked in ``building_model.py``), so ``Fixed`` is 0.
    """
    pro, sto, weight = parts.processes, parts.storages, parts.weight
    ext = pro["ext"]
    invest = float((capacity[ext] * pro.loc[ext, "inv"] * pro.loc[ext, "annuity"]).sum())
    for component in ("Generator", "Link"):
        build = _solution(network.model, BUILD.format(component))
        if build is not None:
            invest += float((build * pro.loc[build.index, "inv_fix"] * pro.loc[build.index, "annuity"]).sum())
    sized = sto["ext"]
    invest += float(((sto_power[sized] * sto.loc[sized, "inv_p"] + sto_energy[sized] * sto.loc[sized, "inv_c"])
                     * sto.loc[sized, "annuity"]).sum())
    variable = float(((sto_in.sum() + sto_out.sum()) * sto["var"]).sum() * weight)
    costs = {"Invest": invest, "Fixed": 0.0, "Variable": variable}
    if parts.bsp:
        flows = {name: float((tau[name] * price).sum() * weight) for name, price in parts.prices.items()}
        costs["Revenue"] = -sum(v for name, v in flows.items() if pro.at[name, "kind"] == FEED_IN)
        costs["Purchase"] = sum(v for name, v in flows.items() if pro.at[name, "kind"] == IMPORT)
    return pd.Series(costs, name="costs").rename_axis("cost_type")


def result_entities(network, parts: ModelParts) -> dict[str, pd.Series]:
    """The ``RESULT_KEYS`` entities of one solved model as urbs-formatted Series."""
    stf, pro, sto = parts.stf, parts.processes, parts.storages
    tau = process_throughput(network, parts)
    capacity = process_capacity(network, parts)
    sto_power, sto_energy, sto_in, sto_out, sto_con = storage_results(network, parts)
    costs = cost_breakdown(network, parts, tau, capacity, sto_power, sto_energy, sto_in, sto_out)

    def hourly(frame: pd.DataFrame, keys: list[np.ndarray], names: list[str], steps: np.ndarray, name: str) -> pd.Series:
        n = frame.shape[1]
        arrays = [np.repeat(steps, n), np.full(len(steps) * n, stf)] + [np.tile(k, len(steps)) for k in keys]
        return pd.Series(frame.to_numpy(dtype=float).ravel(),
                         index=pd.MultiIndex.from_arrays(arrays, names=["t", "stf", *names]), name=name)

    steps = parts.snapshots.to_numpy()
    pro_keys = [pro["site"].to_numpy(), pro["process"].to_numpy()]
    sto_keys = [sto["site"].to_numpy(), sto["storage"].to_numpy(), sto["commodity"].to_numpy()]
    sto_index = pd.MultiIndex.from_arrays([np.full(len(sto), stf), *sto_keys], names=["stf", "sit", "sto", "com"])
    content = pd.concat([sto_con.iloc[[-1]], sto_con])  # t = 0 equals the last step (cyclic)
    results = {
        "tau_pro": hourly(tau, pro_keys, ["sit", "pro"], steps, "tau_pro"),
        "cap_pro": pd.Series(capacity.to_numpy(dtype=float), name="cap_pro", index=pd.MultiIndex.from_arrays(
            [np.full(len(pro), stf), *pro_keys], names=["stf", "sit", "pro"])),
        "cap_sto_c": pd.Series(sto_energy.to_numpy(dtype=float), index=sto_index, name="cap_sto_c"),
        "cap_sto_p": pd.Series(sto_power.to_numpy(dtype=float), index=sto_index, name="cap_sto_p"),
        "e_sto_in": hourly(sto_in, sto_keys, ["sit", "sto", "com"], steps, "e_sto_in"),
        "e_sto_out": hourly(sto_out, sto_keys, ["sit", "sto", "com"], steps, "e_sto_out"),
        "e_sto_con": hourly(content, sto_keys, ["sit", "sto", "com"], np.concatenate([[0], steps]), "e_sto_con"),
        "costs": costs,
    }
    if not parts.thermal_parameters.empty:
        params = parts.thermal_parameters
        ids = params.building_objectid.astype(str).tolist()
        temperature = _solution(network.model, "Building-temperature")[ids]
        initial = pd.DataFrame([params.initial_temperature_c.to_numpy()], columns=ids)
        heat = -_solution(network.model, "Generator-p")[[f"thermal|{bid}" for bid in ids]]
        keys = [params.Site.to_numpy(), np.asarray(ids)]
        results["building_temperature"] = hourly(pd.concat([initial,temperature]), keys, ["sit", "building"], np.r_[0,steps], "building_temperature")
        results["building_heat"] = hourly(heat, keys, ["sit", "building"], steps, "building_heat")
        temperature.index = parts.snapshots
        initial.index = [0]
        all_temperature = pd.concat([initial,temperature])
        thermal = parts.thermal_timeseries.xs(parts.stf,level=0)
        gains = thermal.xs("internal_gains_kw",axis=1,level=1).loc[steps,ids] + thermal.xs("solar_gains_kw",axis=1,level=1).loc[steps,ids]
        outside = thermal.xs("outside_temperature_c",axis=1,level=1).loc[steps,ids]
        h = pd.Series(params.conductance_kw_per_k.to_numpy(),index=ids)
        c = pd.Series(params.capacitance_kwh_per_k.to_numpy(),index=ids)
        loss = (temperature-outside)*h
        residual = all_temperature.diff().loc[steps]*c-heat.set_axis(ids,axis=1)-gains+loss
        for name,frame in (("building_gains",gains),("building_loss",loss),("building_energy_residual",residual)):
            results[name] = hourly(frame,keys,["sit","building"],steps,name)
    return results
