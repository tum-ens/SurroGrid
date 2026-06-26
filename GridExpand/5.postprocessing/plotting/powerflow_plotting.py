"""Postprocessing plotting utilities for GridExpand power-flow results.

This module reads Step 4 power-flow results from HDF5 files or the SurroGrid
database and uses pandapower/Plotly/Matplotlib visualizations to draw:
- bus voltage magnitudes as a single-grid heatmap
- line loadings as a single-grid heatmap
- voltage deviation and transformer import distribution plots across one grid,
  one AGS/PLZ subset, or all matching DB results

Example:
    cd GridExpand/5.postprocessing
    uv run python plotting/powerflow_plotting.py ../4.powerflow/Output/900_80803_2_-1.h5 --stage pre --timestep 0 --on-map
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pandapower as pp
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pandapower.plotting import plotly as pp_plotly
from pandapower.plotting import create_generic_coordinates
from plotly.subplots import make_subplots
from sqlalchemy import text


GRIDEXPAND_DIR = Path(__file__).resolve().parents[2]
if str(GRIDEXPAND_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDEXPAND_DIR))

from database import SurroGridDatabase


def _read_net(h5_path: Path) -> pp.pandapowerNet:
    with h5py.File(h5_path, "r") as handle:
        net_json = handle["raw_data/net"][()]
    return pp.from_json_string(net_json)


def _read_results(h5_path: Path, stage: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    vm_key = f"/pwrflw/output/{stage}/vm"
    line_key = f"/pwrflw/output/{stage}/line_loads"

    with pd.HDFStore(h5_path, mode="r") as store:
        keys = set(store.keys())

    missing = [key for key in (vm_key, line_key) if key not in keys]
    if missing:
        raise KeyError(
            f"Missing result keys {missing} in {h5_path}. "
            "Check whether this file contains the requested stage."
        )

    vm_df = pd.read_hdf(h5_path, key=vm_key)
    line_df = pd.read_hdf(h5_path, key=line_key)
    return vm_df, line_df


def _extract_bus_vm(vm_df: pd.DataFrame, timestep: int) -> pd.Series:
    if timestep < 0 or timestep >= len(vm_df):
        raise IndexError(f"timestep {timestep} is outside [0, {len(vm_df) - 1}].")

    bus_vm = vm_df.iloc[timestep].astype(float)
    bus_vm.index = pd.Index([int(bus) for bus in bus_vm.index], name="bus")
    return bus_vm


def _extract_line_current_ka(line_df: pd.DataFrame, timestep: int) -> pd.Series:
    if timestep < 0 or timestep >= len(line_df):
        raise IndexError(f"timestep {timestep} is outside [0, {len(line_df) - 1}].")

    row = line_df.iloc[timestep]
    if not isinstance(row.index, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns for line_loads with (line, metric).")

    i_from_ka = row.xs("i_from_ka", level=1).astype(float)
    i_from_ka.index = pd.Index([int(line) for line in i_from_ka.index], name="line")

    return i_from_ka


def _line_loading_from_current(
    net: pp.pandapowerNet, i_from_ka: pd.Series
) -> pd.Series:

    max_i_ka = net.line["max_i_ka"].reindex(i_from_ka.index).astype(float).replace(0.0, np.nan)
    loading_percent = (i_from_ka / max_i_ka) * 100.0
    return loading_percent


def _set_results_on_net(
    net: pp.pandapowerNet,
    bus_vm: pd.Series,
    i_from_ka: pd.Series,
    line_loading_percent: pd.Series,
) -> None:
    net.res_bus = pd.DataFrame(index=net.bus.index)
    net.res_bus["vm_pu"] = np.nan
    net.res_bus["va_degree"] = 0.0
    net.res_bus.loc[bus_vm.index, "vm_pu"] = bus_vm.values

    net.res_line = pd.DataFrame(index=net.line.index)
    net.res_line["i_from_ka"] = np.nan
    net.res_line["i_to_ka"] = np.nan
    net.res_line["loading_percent"] = np.nan
    net.res_line.loc[i_from_ka.index, "i_from_ka"] = i_from_ka.values
    net.res_line.loc[i_from_ka.index, "i_to_ka"] = i_from_ka.values
    net.res_line.loc[line_loading_percent.index, "loading_percent"] = line_loading_percent.values

    net.res_trafo = pd.DataFrame(index=net.trafo.index)
    net.res_trafo["loading_percent"] = 0.0
    net.res_trafo["i_hv_ka"] = 0.0
    net.res_trafo["i_lv_ka"] = 0.0


def _select_bus_subset(net: pp.pandapowerNet, show_household_buses: bool) -> list[int]:
    buses = [int(bus) for bus in net.bus.index]
    if show_household_buses:
        return buses

    # In this workflow, household points are represented by load buses.
    if "load" not in net or net.load.empty or "bus" not in net.load.columns:
        return buses

    household_buses = set(net.load["bus"].dropna().astype(int).tolist())
    filtered = [bus for bus in buses if bus not in household_buses]
    return filtered or buses


def _ensure_geodata(net: pp.pandapowerNet, on_map: bool) -> bool:
    has_bus_geo = "geo" in net.bus.columns and net.bus["geo"].notna().any()
    has_line_geo = "geo" in net.line.columns and net.line["geo"].notna().any()

    if not (has_bus_geo or has_line_geo):
        create_generic_coordinates(net, respect_switches=True)
        return False
    return on_map


def _normalize_cmap_name(cmap: str) -> str:
    # pandapower trace helpers use matplotlib colormap names.
    if cmap == "Jet":
        return "jet"
    return cmap


def _match_heatmap_colorbars(fig) -> None:
    colorbar_x = {
        "Bus Voltage [pu]": 1.02,
        "Line Loading [%]": 1.20,
    }

    for trace in fig.data:
        marker = getattr(trace, "marker", None)
        colorbar = getattr(marker, "colorbar", None)
        title = getattr(colorbar, "title", None)
        title_text = getattr(title, "text", None)
        if title_text not in colorbar_x:
            continue

        marker.colorbar.update(
            title={"text": title_text, "side": "right"},
            x=colorbar_x[title_text],
            y=0.5,
            yanchor="middle",
            len=0.9,
            thickness=10,
        )


def _bus_hover_info(net: pp.pandapowerNet, buses: list[int]) -> pd.Series:
    hover_text = []
    for bus in buses:
        name = net.bus.at[bus, "name"] if "name" in net.bus.columns else f"Bus {bus}"
        vm_pu = net.res_bus.at[bus, "vm_pu"]
        hover_text.append(f"{name}<br>Voltage: {vm_pu:.4f} p.u.")
    return pd.Series(hover_text, index=buses)


def _line_hover_info(net: pp.pandapowerNet) -> pd.Series:
    hover_text = []
    for line in net.line.index:
        name = net.line.at[line, "name"] if "name" in net.line.columns else f"Line {line}"
        loading_percent = net.res_line.at[line, "loading_percent"]
        hover_text.append(f"{name}<br>Loading: {loading_percent:.2f}%")
    return pd.Series(hover_text, index=net.line.index)


def _draw_plotly_heatmap(
    net: pp.pandapowerNet,
    on_map: bool,
    map_style: str,
    cmap: str,
    climits_volt: tuple[float, float],
    climits_load: tuple[float, float],
    show_household_buses: bool,
):
    cmap = _normalize_cmap_name(cmap)
    on_map = _ensure_geodata(net, on_map)
    use_line_geo = "geo" in net.line.columns and net.line["geo"].notna().all()

    bus_subset = _select_bus_subset(net, show_household_buses)

    bus_traces = pp_plotly.create_bus_trace(
        net,
        buses=bus_subset,
        cmap=cmap,
        cmap_vals=net.res_bus.loc[bus_subset, "vm_pu"].values,
        infofunc=_bus_hover_info(net, bus_subset),
        cbar_title="Bus Voltage [pu]",
        cmin=climits_volt[0],
        cmax=climits_volt[1],
        cpos=1.02,
        size=8,
        trace_name="bus voltage",
    )

    line_traces = pp_plotly.create_line_trace(
        net,
        use_line_geo=use_line_geo,
        cmap=cmap,
        cmap_vals=net.res_line["loading_percent"].values,
        cbar_title="Line Loading [%]",
        infofunc=_line_hover_info(net),
        cmin=climits_load[0],
        cmax=climits_load[1],
        cpos=1.20,
        show_colorbar=True,
        width=2,
        trace_name="line loading",
    )

    trafo_traces = pp_plotly.create_trafo_trace(
        net,
        color="green",
        width=3,
        trace_name="transformers",
    )

    ext_grid_trace = pp_plotly.create_bus_trace(
        net,
        buses=net.ext_grid.bus.tolist(),
        color="grey",
        size=12,
        patch_type="square",
        trace_name="external grid",
    )

    fig = pp_plotly.draw_traces(
        line_traces + trafo_traces + ext_grid_trace + bus_traces,
        on_map=on_map,
        map_style=map_style,
        showlegend=True,
        filename=None,
        auto_open=False,
    )

    fig.update_layout(
        legend={
            "orientation": "h",
            "x": 0.0,
            "xanchor": "left",
            "y": 1.02,
            "yanchor": "bottom",
            "bgcolor": "rgba(255,255,255,0.75)",
        },
        margin={"l": 10, "r": 320, "t": 70, "b": 10},
    )

    fig_json = fig.to_plotly_json()
    colorbar_titles = {
        ((trace.get("marker") or {}).get("colorbar") or {}).get("title", {}).get("text")
        for trace in fig_json.get("data", [])
    }

    if "Line Loading [%]" not in colorbar_titles:
        fig.add_trace(
            {
                "type": "scatter",
                "x": [None, None],
                "y": [None, None],
                "mode": "markers",
                "marker": {
                    "size": 0.01,
                    "color": [climits_load[0], climits_load[1]],
                    "colorscale": cmap,
                    "cmin": climits_load[0],
                    "cmax": climits_load[1],
                    "showscale": True,
                    "colorbar": {
                        "title": {"text": "Line Loading [%]", "side": "right"},
                        "x": 1.20,
                        "y": 0.5,
                        "yanchor": "middle",
                        "len": 0.9,
                        "thickness": 10,
                    },
                },
                "hoverinfo": "skip",
                "showlegend": False,
            }
        )

    _match_heatmap_colorbars(fig)
    return fig


def plot_powerflow_heatmap(
    h5_path: Path,
    stage: str,
    timestep: int,
    on_map: bool,
    map_style: str,
    cmap: str,
    climits_volt: tuple[float, float],
    climits_load: tuple[float, float],
    show_household_buses: bool = False,
    show: bool = True,
):
    net = _read_net(h5_path)
    vm_df, line_df = _read_results(h5_path, stage)

    bus_vm = _extract_bus_vm(vm_df, timestep)
    i_from_ka = _extract_line_current_ka(line_df, timestep)
    line_loading_percent = _line_loading_from_current(net, i_from_ka)
    _set_results_on_net(net, bus_vm, i_from_ka, line_loading_percent)

    try:
        fig = _draw_plotly_heatmap(
            net=net,
            on_map=on_map,
            map_style=map_style,
            cmap=cmap,
            climits_volt=climits_volt,
            climits_load=climits_load,
            show_household_buses=show_household_buses,
        )
    except ImportError as exc:
        raise ImportError(
            "pandapower plotly backend is unavailable. Run `uv sync` in "
            "GridExpand/5.postprocessing to install plotly."
        ) from exc
    if show:
        fig.show()
    return fig


def _resolve_db_grid(
    db: SurroGridDatabase,
    input_id: str,
    plz: int | None,
    kcid: int | None,
    bcid: int | None,
    candidate_index: int,
    min_buildings: int,
) -> dict:
    return db.resolve_grid_identifier(
        input_id,
        plz=plz,
        kcid=kcid,
        bcid=bcid,
        candidate_index=candidate_index,
        min_buildings=min_buildings,
    )


def _resolve_powerflow_run(
    db: SurroGridDatabase,
    grid_ref: dict,
    run_name: str,
    scenario_id: int | None = None,
) -> dict:
    query = text(
        """
        SELECT pr.powerflow_run_id, pr.run_name, pr.pre_only, pr.scenario_id, sc.scenario_key, pr.updated_at
        FROM surrogrid.grid_case gc
        JOIN surrogrid.powerflow_run pr
          ON pr.grid_case_id = gc.grid_case_id
        JOIN surrogrid.scenario sc
          ON sc.scenario_id = pr.scenario_id
        WHERE gc.ags = :ags
          AND gc.plz = :plz
          AND gc.kcid = :kcid
          AND gc.bcid = :bcid
          AND gc.pylovo_grid_result_id = :grid_result_id
          AND pr.run_name = :run_name
          AND (:scenario_id IS NULL OR pr.scenario_id = :scenario_id)
        ORDER BY pr.updated_at DESC, pr.powerflow_run_id DESC
        LIMIT 1
        """
    )
    with db.engine.connect() as conn:
        row = conn.execute(
            query,
            {
                "ags": grid_ref["ags"],
                "plz": grid_ref["plz"],
                "kcid": grid_ref["kcid"],
                "bcid": grid_ref["bcid"],
                "grid_result_id": grid_ref["grid_result_id"],
                "run_name": run_name,
                "scenario_id": scenario_id,
            },
        ).mappings().first()

    if row is None:
        raise ValueError(
            f"No DB power-flow run named {run_name!r} found for "
            f"scenario_id={scenario_id!r}, PLZ={grid_ref['plz']}, "
            f"KCID={grid_ref['kcid']}, BCID={grid_ref['bcid']}."
        )
    return dict(row)



def db_powerflow_timestep_bounds(
    input_id: str,
    stage: str,
    run_name: str = "baseline_static_full_powerflow",
    scenario_id: int | None = None,
    plz: int | None = None,
    kcid: int | None = None,
    bcid: int | None = None,
    candidate_index: int = 0,
    min_buildings: int = 5,
) -> dict:
    db = SurroGridDatabase()
    grid_ref = _resolve_db_grid(db, input_id, plz, kcid, bcid, candidate_index, min_buildings)
    run = _resolve_powerflow_run(db, grid_ref, run_name, scenario_id)

    query = text(
        """
        SELECT MIN(t_index) AS min_timestep,
               MAX(t_index) AS max_timestep,
               COUNT(DISTINCT t_index) AS n_timesteps
        FROM surrogrid.powerflow_bus_voltage
        WHERE powerflow_run_id = :run_id
          AND stage = :stage
        """
    )
    with db.engine.connect() as conn:
        row = conn.execute(
            query,
            {"run_id": int(run["powerflow_run_id"]), "stage": stage},
        ).mappings().one()

    if row["n_timesteps"] == 0:
        raise ValueError(
            f"No DB power-flow voltage results found for run "
            f"{run['powerflow_run_id']}, stage {stage!r}."
        )

    return {
        "grid_ref": grid_ref,
        "run": run,
        "min_timestep": int(row["min_timestep"]),
        "max_timestep": int(row["max_timestep"]),
        "n_timesteps": int(row["n_timesteps"]),
    }


def _read_db_timestep_results(
    db: SurroGridDatabase,
    powerflow_run_id: int,
    stage: str,
    timestep: int,
) -> tuple[pd.Series, pd.Series]:
    bus_query = text(
        """
        SELECT bus, vm_pu
        FROM surrogrid.powerflow_bus_voltage
        WHERE powerflow_run_id = :run_id
          AND stage = :stage
          AND t_index = :timestep
        ORDER BY bus
        """
    )
    line_query = text(
        """
        SELECT line, i_from_ka
        FROM surrogrid.powerflow_line_result
        WHERE powerflow_run_id = :run_id
          AND stage = :stage
          AND t_index = :timestep
        ORDER BY line
        """
    )
    params = {"run_id": powerflow_run_id, "stage": stage, "timestep": timestep}
    with db.engine.connect() as conn:
        bus_df = pd.read_sql_query(bus_query, conn, params=params)
        line_df = pd.read_sql_query(line_query, conn, params=params)

    if bus_df.empty or line_df.empty:
        raise ValueError(
            f"No DB power-flow results found for run {powerflow_run_id}, "
            f"stage {stage!r}, timestep {timestep}."
        )

    bus_vm = bus_df.set_index("bus")["vm_pu"].astype(float)
    bus_vm.index = pd.Index([int(bus) for bus in bus_vm.index], name="bus")
    i_from_ka = line_df.set_index("line")["i_from_ka"].astype(float)
    i_from_ka.index = pd.Index([int(line) for line in i_from_ka.index], name="line")
    return bus_vm, i_from_ka


def plot_powerflow_heatmap_db(
    input_id: str,
    stage: str,
    timestep: int,
    on_map: bool,
    map_style: str,
    cmap: str,
    climits_volt: tuple[float, float],
    climits_load: tuple[float, float],
    show_household_buses: bool = False,
    show: bool = True,
    run_name: str = "baseline_static_full_powerflow",
    scenario_id: int | None = None,
    plz: int | None = None,
    kcid: int | None = None,
    bcid: int | None = None,
    candidate_index: int = 0,
    min_buildings: int = 5,
):
    """Plot one DB-backed grid and stage at one timestep.

    This helper intentionally takes one concrete ``input_id``. Population
    selection happens before calling it, for example by ranking
    ``grid_loading_stress_summary`` rows and passing the selected grid/stage.
    """
    db = SurroGridDatabase()
    grid_ref = _resolve_db_grid(db, input_id, plz, kcid, bcid, candidate_index, min_buildings)
    run = _resolve_powerflow_run(db, grid_ref, run_name, scenario_id)
    net = db.read_pandapower_grid(grid_ref)

    bus_vm, i_from_ka = _read_db_timestep_results(
        db,
        int(run["powerflow_run_id"]),
        stage,
        timestep,
    )
    line_loading_percent = _line_loading_from_current(net, i_from_ka)
    _set_results_on_net(net, bus_vm, i_from_ka, line_loading_percent)

    try:
        fig = _draw_plotly_heatmap(
            net=net,
            on_map=on_map,
            map_style=map_style,
            cmap=cmap,
            climits_volt=climits_volt,
            climits_load=climits_load,
            show_household_buses=show_household_buses,
        )
    except ImportError as exc:
        raise ImportError(
            "pandapower plotly backend is unavailable. Run `uv sync` in "
            "GridExpand/5.postprocessing to install plotly."
        ) from exc
    if show:
        fig.show()
    return fig



def _grid_label_from_row(row: pd.Series) -> str:
    ags = str(int(row["ags"])).zfill(8)
    return f"{ags}-{int(row['plz'])}_{int(row['kcid'])}_{int(row['bcid'])}"


def powerflow_headline_summary_db(
    input_id: str | None = None,
    run_name: str = "baseline_static_pre_powerflow",
    stage: str = "pre",
    scenario_id: int | None = None,
    ags: str | int | None = None,
    plz: int | None = None,
    kcid: int | None = None,
    bcid: int | None = None,
    candidate_index: int = 0,
    min_buildings: int = 5,
) -> pd.DataFrame:
    """Read compact DB-backed headline power-flow metrics for comparison plots."""
    db = SurroGridDatabase()
    run_id = None
    if input_id is not None:
        grid_ref = _resolve_db_grid(db, input_id, plz, kcid, bcid, candidate_index, min_buildings)
        run = _resolve_powerflow_run(db, grid_ref, run_name, scenario_id)
        run_id = int(run["powerflow_run_id"])

    query = text(
        """
        SELECT pr.powerflow_run_id,
               pr.run_name,
               pr.scenario_id,
               sc.scenario_key,
               gc.ags,
               gc.plz,
               gc.kcid,
               gc.bcid,
               gc.pylovo_grid_result_id,
               pfs.stage,
               pfs.n_timesteps,
               pfs.n_voltage_buses,
               pfs.n_cables,
               pfs.transformer_s_rated_mva,
               pfs.trafo_loading_p50_time_percent,
               pfs.trafo_loading_p90_time_percent,
               pfs.trafo_loading_p95_time_percent,
               pfs.trafo_loading_p99_time_percent,
               pfs.trafo_loading_max_time_percent,
               pfs.trafo_loading_hours_above_100,
               pfs.cable_loading_p95_asset_percent,
               pfs.cable_hours_above_100_p95_asset,
               pfs.voltage_p05_load_bus_hour_pu,
               pfs.voltage_hours_below_0_90_p95_asset
        FROM surrogrid.powerflow_summary pfs
        JOIN surrogrid.powerflow_run pr USING (powerflow_run_id)
        JOIN surrogrid.scenario sc USING (scenario_id)
        JOIN surrogrid.grid_case gc USING (grid_case_id)
        WHERE pr.run_name = :run_name
          AND pfs.stage = :stage
          AND (:run_id IS NULL OR pr.powerflow_run_id = :run_id)
          AND (:scenario_id IS NULL OR pr.scenario_id = :scenario_id)
          AND (:ags IS NULL OR gc.ags = :ags)
          AND (:filter_plz IS NULL OR gc.plz = :filter_plz)
          AND (:filter_kcid IS NULL OR gc.kcid = :filter_kcid)
          AND (:filter_bcid IS NULL OR gc.bcid = :filter_bcid)
        ORDER BY gc.ags, gc.plz, gc.kcid, gc.bcid, pr.powerflow_run_id, pfs.stage
        """
    )
    with db.engine.connect() as conn:
        summary = pd.read_sql_query(
            query,
            conn,
            params={
                "run_name": run_name,
                "stage": stage,
                "run_id": run_id,
                "scenario_id": scenario_id,
                "ags": _normalize_optional_ags(ags),
                "filter_plz": plz if input_id is None else None,
                "filter_kcid": kcid if input_id is None else None,
                "filter_bcid": bcid if input_id is None else None,
            },
        )

    if summary.empty:
        raise ValueError(f"No compact DB power-flow summary found for run name {run_name!r}.")

    summary["grid"] = summary.apply(_grid_label_from_row, axis=1)
    return summary[
        [
            "grid",
            "powerflow_run_id",
            "run_name",
            "scenario_id",
            "scenario_key",
            "stage",
            "ags",
            "plz",
            "kcid",
            "bcid",
            "pylovo_grid_result_id",
            "n_timesteps",
            "n_voltage_buses",
            "n_cables",
            "transformer_s_rated_mva",
            "trafo_loading_p50_time_percent",
            "trafo_loading_p90_time_percent",
            "trafo_loading_p95_time_percent",
            "trafo_loading_p99_time_percent",
            "trafo_loading_max_time_percent",
            "trafo_loading_hours_above_100",
            "cable_loading_p95_asset_percent",
            "cable_hours_above_100_p95_asset",
            "voltage_p05_load_bus_hour_pu",
            "voltage_hours_below_0_90_p95_asset",
        ]
    ].reset_index(drop=True)


def powerflow_tail_duration_data_db(
    input_id: str | None = None,
    run_name: str = "baseline_static_pre_powerflow",
    stage: str = "pre",
    scenario_id: int | None = None,
    ags: str | int | None = None,
    plz: int | None = None,
    kcid: int | None = None,
    bcid: int | None = None,
    candidate_index: int = 0,
    min_buildings: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load grid summary, p99/p01 tail values, and percentile profiles together."""
    kwargs = {
        "input_id": input_id,
        "run_name": run_name,
        "stage": stage,
        "scenario_id": scenario_id,
        "ags": ags,
        "plz": plz,
        "kcid": kcid,
        "bcid": bcid,
        "candidate_index": candidate_index,
        "min_buildings": min_buildings,
    }
    grid_summary = powerflow_headline_summary_db(**kwargs)
    tail_values = powerflow_tail_values_db(**kwargs)
    percentile_profile = powerflow_percentile_profile_db(**kwargs)
    return grid_summary, tail_values, percentile_profile


def powerflow_tail_values_db(
    input_id: str | None = None,
    run_name: str = "baseline_static_pre_powerflow",
    stage: str = "pre",
    scenario_id: int | None = None,
    ags: str | int | None = None,
    plz: int | None = None,
    kcid: int | None = None,
    bcid: int | None = None,
    candidate_index: int = 0,
    min_buildings: int = 5,
) -> pd.DataFrame:
    """Read DB-backed p99/p01 tail-hour values at transformer/cable/bus level."""
    grid_summary = powerflow_headline_summary_db(
        input_id=input_id,
        run_name=run_name,
        stage=stage,
        scenario_id=scenario_id,
        ags=ags,
        plz=plz,
        kcid=kcid,
        bcid=bcid,
        candidate_index=candidate_index,
        min_buildings=min_buildings,
    )
    run_ids = grid_summary["powerflow_run_id"].astype(int).tolist()
    if not run_ids:
        return pd.DataFrame()

    db = SurroGridDatabase()
    query = text(
        """
        SELECT powerflow_run_id, stage, metric, asset_type, asset_id, tail,
               threshold_value, t_index, value
        FROM surrogrid.powerflow_tail_value
        WHERE powerflow_run_id = ANY(:run_ids)
          AND stage = :stage
        """
    )
    with db.engine.connect() as conn:
        tail_rows = pd.read_sql_query(query, conn, params={"run_ids": run_ids, "stage": stage})

    meta_cols = [
        "grid",
        "powerflow_run_id",
        "run_name",
        "scenario_id",
        "scenario_key",
        "stage",
        "ags",
        "plz",
        "kcid",
        "bcid",
        "pylovo_grid_result_id",
    ]
    if tail_rows.empty:
        return pd.DataFrame(
            columns=meta_cols
            + [
                "metric",
                "asset_type",
                "asset_id",
                "asset_label",
                "tail",
                "threshold_value",
                "t_index",
                "value",
            ]
        )

    tail_rows = tail_rows.merge(grid_summary[meta_cols], on=["powerflow_run_id", "stage"], how="left")
    tail_rows["asset_label"] = tail_rows["asset_type"].astype(str) + " " + tail_rows["asset_id"].astype(str)
    tail_rows.loc[tail_rows["asset_type"] == "transformer", "asset_label"] = (
        tail_rows.loc[tail_rows["asset_type"] == "transformer", "grid"] + " transformer"
    )
    tail_rows["value"] = tail_rows["value"].astype(float)
    tail_rows["threshold_value"] = tail_rows["threshold_value"].astype(float)
    tail_rows["t_index"] = tail_rows["t_index"].astype(int)
    return tail_rows[
        meta_cols
        + [
            "metric",
            "asset_type",
            "asset_id",
            "asset_label",
            "tail",
            "threshold_value",
            "t_index",
            "value",
        ]
    ].dropna(subset=["value"]).reset_index(drop=True)


powerflow_headline_asset_summary_db = powerflow_tail_values_db


def tail_threshold_counts(
    tail_values: pd.DataFrame,
    loading_threshold_percent: float = 100.0,
    voltage_threshold_pu: float = 0.90,
) -> pd.DataFrame:
    """Count stored tail hours beyond thresholds, with completeness flags.

    Counts are exact when the requested threshold is at least as extreme as the
    stored asset-specific tail threshold: loading threshold >= p99 threshold, or
    voltage threshold <= p01 threshold. Otherwise the count is a lower bound
    because non-tail hours were not stored.
    """
    if tail_values.empty:
        return pd.DataFrame()

    df = tail_values.copy()
    is_loading = df["metric"].isin(["Transformer", "Cables"])
    is_voltage = df["metric"] == "Voltage"
    df["beyond_threshold"] = False
    df.loc[is_loading, "beyond_threshold"] = df.loc[is_loading, "value"] >= loading_threshold_percent
    df.loc[is_voltage, "beyond_threshold"] = df.loc[is_voltage, "value"] <= voltage_threshold_pu

    df["threshold_used"] = np.nan
    df.loc[is_loading, "threshold_used"] = float(loading_threshold_percent)
    df.loc[is_voltage, "threshold_used"] = float(voltage_threshold_pu)

    df["is_complete_for_threshold"] = False
    df.loc[is_loading, "is_complete_for_threshold"] = (
        loading_threshold_percent >= df.loc[is_loading, "threshold_value"]
    )
    df.loc[is_voltage, "is_complete_for_threshold"] = (
        voltage_threshold_pu <= df.loc[is_voltage, "threshold_value"]
    )

    group_cols = [
        "grid",
        "powerflow_run_id",
        "run_name",
        "scenario_id",
        "scenario_key",
        "stage",
        "ags",
        "plz",
        "kcid",
        "bcid",
        "pylovo_grid_result_id",
        "metric",
        "asset_type",
        "asset_id",
        "asset_label",
        "tail",
    ]
    return (
        df.groupby(group_cols, as_index=False)
        .agg(
            n_tail_hours=("value", "size"),
            n_tail_hours_beyond_threshold=("beyond_threshold", "sum"),
            threshold_used=("threshold_used", "first"),
            stored_tail_threshold=("threshold_value", "first"),
            is_complete_for_threshold=("is_complete_for_threshold", "all"),
        )
        .reset_index(drop=True)
    )


def powerflow_percentile_profile_db(
    input_id: str | None = None,
    run_name: str = "baseline_static_pre_powerflow",
    stage: str = "pre",
    scenario_id: int | None = None,
    ags: str | int | None = None,
    plz: int | None = None,
    kcid: int | None = None,
    bcid: int | None = None,
    candidate_index: int = 0,
    min_buildings: int = 5,
) -> pd.DataFrame:
    """Read per-asset time-percentiles in long form for duration-profile plots."""
    grid_summary = powerflow_headline_summary_db(
        input_id=input_id,
        run_name=run_name,
        stage=stage,
        scenario_id=scenario_id,
        ags=ags,
        plz=plz,
        kcid=kcid,
        bcid=bcid,
        candidate_index=candidate_index,
        min_buildings=min_buildings,
    )
    run_ids = grid_summary["powerflow_run_id"].astype(int).tolist()
    if not run_ids:
        return pd.DataFrame()

    db = SurroGridDatabase()
    cable_query = text(
        """
        SELECT powerflow_run_id, stage, cable AS asset_id,
               cable_loading_p50_time_percent, cable_loading_p90_time_percent,
               cable_loading_p95_time_percent, cable_loading_p99_time_percent,
               cable_loading_max_time_percent
        FROM surrogrid.powerflow_cable_summary
        WHERE powerflow_run_id = ANY(:run_ids)
          AND stage = :stage
        """
    )
    voltage_query = text(
        """
        SELECT powerflow_run_id, stage, bus AS asset_id,
               voltage_p50_time_pu, voltage_p10_time_pu, voltage_p05_time_pu,
               voltage_p01_time_pu, voltage_min_time_pu
        FROM surrogrid.powerflow_bus_voltage_summary
        WHERE powerflow_run_id = ANY(:run_ids)
          AND stage = :stage
        """
    )
    with db.engine.connect() as conn:
        cable_rows = pd.read_sql_query(cable_query, conn, params={"run_ids": run_ids, "stage": stage})
        voltage_rows = pd.read_sql_query(voltage_query, conn, params={"run_ids": run_ids, "stage": stage})

    meta_cols = [
        "grid",
        "powerflow_run_id",
        "run_name",
        "scenario_id",
        "scenario_key",
        "stage",
        "ags",
        "plz",
        "kcid",
        "bcid",
        "pylovo_grid_result_id",
    ]
    meta = grid_summary[meta_cols].copy()
    frames = []

    trafo_map = {
        "p50": "trafo_loading_p50_time_percent",
        "p90": "trafo_loading_p90_time_percent",
        "p95": "trafo_loading_p95_time_percent",
        "p99": "trafo_loading_p99_time_percent",
        "max": "trafo_loading_max_time_percent",
    }
    for order, (percentile, column) in enumerate(trafo_map.items()):
        rows = grid_summary[meta_cols + [column]].rename(columns={column: "value"})
        rows["metric"] = "Transformer"
        rows["asset_type"] = "transformer"
        rows["asset_id"] = 0
        rows["asset_label"] = rows["grid"] + " transformer"
        rows["percentile"] = percentile
        rows["percentile_order"] = order
        frames.append(rows)

    cable_map = {
        "p50": "cable_loading_p50_time_percent",
        "p90": "cable_loading_p90_time_percent",
        "p95": "cable_loading_p95_time_percent",
        "p99": "cable_loading_p99_time_percent",
        "max": "cable_loading_max_time_percent",
    }
    if not cable_rows.empty:
        cable_rows = cable_rows.merge(meta, on=["powerflow_run_id", "stage"], how="left")
        for order, (percentile, column) in enumerate(cable_map.items()):
            rows = cable_rows[meta_cols + ["asset_id", column]].rename(columns={column: "value"})
            rows["metric"] = "Cables"
            rows["asset_type"] = "cable"
            rows["asset_label"] = "cable " + rows["asset_id"].astype(str)
            rows["percentile"] = percentile
            rows["percentile_order"] = order
            frames.append(rows)

    voltage_map = {
        "p50": "voltage_p50_time_pu",
        "p10": "voltage_p10_time_pu",
        "p05": "voltage_p05_time_pu",
        "p01": "voltage_p01_time_pu",
        "min": "voltage_min_time_pu",
    }
    if not voltage_rows.empty:
        voltage_rows = voltage_rows.merge(meta, on=["powerflow_run_id", "stage"], how="left")
        for order, (percentile, column) in enumerate(voltage_map.items()):
            rows = voltage_rows[meta_cols + ["asset_id", column]].rename(columns={column: "value"})
            rows["metric"] = "Voltage"
            rows["asset_type"] = "bus"
            rows["asset_label"] = "bus " + rows["asset_id"].astype(str)
            rows["percentile"] = percentile
            rows["percentile_order"] = order
            frames.append(rows)

    out = pd.concat(frames, ignore_index=True)
    out["value"] = out["value"].astype(float)
    return out[
        meta_cols
        + [
            "metric",
            "asset_type",
            "asset_id",
            "asset_label",
            "percentile",
            "percentile_order",
            "value",
        ]
    ].dropna(subset=["value"]).reset_index(drop=True)



def _real_grid_label_from_row(row: pd.Series) -> str:
    return f"SWF LV_{int(row['lv_id']):03d}"


def real_powerflow_headline_summary_db(
    run_name: str = "baseline_static_pre_powerflow_real_swf_hh_only",
    stage: str = "pre",
    scenario_id: int | None = None,
    plz: int | None = None,
    lv_id: str | int | None = None,
) -> pd.DataFrame:
    """Read compact real SWF DB-backed headline power-flow metrics."""
    db = SurroGridDatabase()
    lv_id_int = None if lv_id is None else int(str(lv_id).removeprefix("LV_"))
    query = text(
        """
        SELECT rpr.real_powerflow_run_id AS powerflow_run_id,
               rpr.run_name,
               rpr.scenario_id,
               sc.scenario_key,
               rgc.source,
               rgc.plz,
               rgc.lv_id,
               rgc.variant,
               rgc.category,
               rgc.load_status,
               rgc.source_file,
               rps.stage,
               rps.n_timesteps,
               rps.n_voltage_buses,
               rps.n_cables,
               rps.transformer_s_rated_mva,
               rps.trafo_loading_p50_time_percent,
               rps.trafo_loading_p90_time_percent,
               rps.trafo_loading_p95_time_percent,
               rps.trafo_loading_p99_time_percent,
               rps.trafo_loading_max_time_percent,
               rps.trafo_loading_hours_above_100,
               rps.cable_loading_p95_asset_percent,
               rps.cable_hours_above_100_p95_asset,
               rps.voltage_p05_load_bus_hour_pu,
               rps.voltage_hours_below_0_90_p95_asset
        FROM surrogrid.real_powerflow_summary rps
        JOIN surrogrid.real_powerflow_run rpr USING (real_powerflow_run_id)
        JOIN surrogrid.scenario sc USING (scenario_id)
        JOIN surrogrid.real_grid_case rgc USING (real_grid_case_id)
        WHERE rpr.run_name = :run_name
          AND rps.stage = :stage
          AND (:scenario_id IS NULL OR rpr.scenario_id = :scenario_id)
          AND (:filter_plz IS NULL OR rgc.plz = :filter_plz)
          AND (:lv_id IS NULL OR rgc.lv_id = CAST(:lv_id AS TEXT))
        ORDER BY rgc.lv_id::INTEGER, rpr.real_powerflow_run_id, rps.stage
        """
    )
    with db.engine.connect() as conn:
        summary = pd.read_sql_query(
            query,
            conn,
            params={
                "run_name": run_name,
                "stage": stage,
                "scenario_id": scenario_id,
                "filter_plz": plz,
                "lv_id": None if lv_id_int is None else str(lv_id_int),
            },
        )

    if summary.empty:
        raise ValueError(f"No compact real-grid DB power-flow summary found for run name {run_name!r}.")

    summary["grid"] = summary.apply(_real_grid_label_from_row, axis=1)
    summary["powerflow_source"] = "real_swf"
    summary["comparison_group"] = "Real SWF"
    summary["ags"] = pd.NA
    summary["kcid"] = pd.NA
    summary["bcid"] = pd.NA
    summary["pylovo_grid_result_id"] = pd.NA
    return summary[
        [
            "grid",
            "powerflow_source",
            "comparison_group",
            "powerflow_run_id",
            "run_name",
            "scenario_id",
            "scenario_key",
            "stage",
            "ags",
            "plz",
            "kcid",
            "bcid",
            "pylovo_grid_result_id",
            "lv_id",
            "source_file",
            "n_timesteps",
            "n_voltage_buses",
            "n_cables",
            "transformer_s_rated_mva",
            "trafo_loading_p50_time_percent",
            "trafo_loading_p90_time_percent",
            "trafo_loading_p95_time_percent",
            "trafo_loading_p99_time_percent",
            "trafo_loading_max_time_percent",
            "trafo_loading_hours_above_100",
            "cable_loading_p95_asset_percent",
            "cable_hours_above_100_p95_asset",
            "voltage_p05_load_bus_hour_pu",
            "voltage_hours_below_0_90_p95_asset",
        ]
    ].reset_index(drop=True)


def real_powerflow_tail_values_db(
    run_name: str = "baseline_static_pre_powerflow_real_swf_hh_only",
    stage: str = "pre",
    scenario_id: int | None = None,
    plz: int | None = None,
    lv_id: str | int | None = None,
) -> pd.DataFrame:
    """Read DB-backed p99/p01 tail-hour values for real SWF grids."""
    grid_summary = real_powerflow_headline_summary_db(
        run_name=run_name,
        stage=stage,
        scenario_id=scenario_id,
        plz=plz,
        lv_id=lv_id,
    )
    run_ids = grid_summary["powerflow_run_id"].astype(int).tolist()
    if not run_ids:
        return pd.DataFrame()

    db = SurroGridDatabase()
    query = text(
        """
        SELECT real_powerflow_run_id AS powerflow_run_id, stage, metric, asset_type,
               asset_id, tail, threshold_value, t_index, value
        FROM surrogrid.real_powerflow_tail_value
        WHERE real_powerflow_run_id = ANY(:run_ids)
          AND stage = :stage
        """
    )
    with db.engine.connect() as conn:
        tail_rows = pd.read_sql_query(query, conn, params={"run_ids": run_ids, "stage": stage})

    meta_cols = [
        "grid",
        "powerflow_source",
        "comparison_group",
        "powerflow_run_id",
        "run_name",
        "scenario_id",
        "scenario_key",
        "stage",
        "ags",
        "plz",
        "kcid",
        "bcid",
        "pylovo_grid_result_id",
        "lv_id",
        "source_file",
    ]
    if tail_rows.empty:
        return pd.DataFrame(
            columns=meta_cols
            + [
                "metric",
                "asset_type",
                "asset_id",
                "asset_label",
                "tail",
                "threshold_value",
                "t_index",
                "value",
            ]
        )

    tail_rows = tail_rows.merge(grid_summary[meta_cols], on=["powerflow_run_id", "stage"], how="left")
    tail_rows["asset_label"] = tail_rows["asset_type"].astype(str) + " " + tail_rows["asset_id"].astype(str)
    tail_rows.loc[tail_rows["asset_type"] == "transformer", "asset_label"] = (
        tail_rows.loc[tail_rows["asset_type"] == "transformer", "grid"] + " transformer"
    )
    tail_rows["value"] = tail_rows["value"].astype(float)
    tail_rows["threshold_value"] = tail_rows["threshold_value"].astype(float)
    tail_rows["t_index"] = tail_rows["t_index"].astype(int)
    return tail_rows[
        meta_cols
        + [
            "metric",
            "asset_type",
            "asset_id",
            "asset_label",
            "tail",
            "threshold_value",
            "t_index",
            "value",
        ]
    ].dropna(subset=["value"]).reset_index(drop=True)


def real_powerflow_percentile_profile_db(
    run_name: str = "baseline_static_pre_powerflow_real_swf_hh_only",
    stage: str = "pre",
    scenario_id: int | None = None,
    plz: int | None = None,
    lv_id: str | int | None = None,
) -> pd.DataFrame:
    """Read real SWF per-asset time-percentiles in long form."""
    grid_summary = real_powerflow_headline_summary_db(
        run_name=run_name,
        stage=stage,
        scenario_id=scenario_id,
        plz=plz,
        lv_id=lv_id,
    )
    run_ids = grid_summary["powerflow_run_id"].astype(int).tolist()
    if not run_ids:
        return pd.DataFrame()

    db = SurroGridDatabase()
    cable_query = text(
        """
        SELECT real_powerflow_run_id AS powerflow_run_id, stage, cable AS asset_id,
               cable_loading_p50_time_percent, cable_loading_p90_time_percent,
               cable_loading_p95_time_percent, cable_loading_p99_time_percent,
               cable_loading_max_time_percent
        FROM surrogrid.real_powerflow_cable_summary
        WHERE real_powerflow_run_id = ANY(:run_ids)
          AND stage = :stage
        """
    )
    voltage_query = text(
        """
        SELECT real_powerflow_run_id AS powerflow_run_id, stage, bus AS asset_id,
               voltage_p50_time_pu, voltage_p10_time_pu, voltage_p05_time_pu,
               voltage_p01_time_pu, voltage_min_time_pu
        FROM surrogrid.real_powerflow_bus_voltage_summary
        WHERE real_powerflow_run_id = ANY(:run_ids)
          AND stage = :stage
        """
    )
    with db.engine.connect() as conn:
        cable_rows = pd.read_sql_query(cable_query, conn, params={"run_ids": run_ids, "stage": stage})
        voltage_rows = pd.read_sql_query(voltage_query, conn, params={"run_ids": run_ids, "stage": stage})

    meta_cols = [
        "grid",
        "powerflow_source",
        "comparison_group",
        "powerflow_run_id",
        "run_name",
        "scenario_id",
        "scenario_key",
        "stage",
        "ags",
        "plz",
        "kcid",
        "bcid",
        "pylovo_grid_result_id",
        "lv_id",
        "source_file",
    ]
    meta = grid_summary[meta_cols].copy()
    frames = []

    trafo_map = {
        "p50": "trafo_loading_p50_time_percent",
        "p90": "trafo_loading_p90_time_percent",
        "p95": "trafo_loading_p95_time_percent",
        "p99": "trafo_loading_p99_time_percent",
        "max": "trafo_loading_max_time_percent",
    }
    for order, (percentile, column) in enumerate(trafo_map.items()):
        rows = grid_summary[meta_cols + [column]].rename(columns={column: "value"})
        rows["metric"] = "Transformer"
        rows["asset_type"] = "transformer"
        rows["asset_id"] = 0
        rows["asset_label"] = rows["grid"] + " transformer"
        rows["percentile"] = percentile
        rows["percentile_order"] = order
        frames.append(rows)

    cable_map = {
        "p50": "cable_loading_p50_time_percent",
        "p90": "cable_loading_p90_time_percent",
        "p95": "cable_loading_p95_time_percent",
        "p99": "cable_loading_p99_time_percent",
        "max": "cable_loading_max_time_percent",
    }
    if not cable_rows.empty:
        cable_rows = cable_rows.merge(meta, on=["powerflow_run_id", "stage"], how="left")
        for order, (percentile, column) in enumerate(cable_map.items()):
            rows = cable_rows[meta_cols + ["asset_id", column]].rename(columns={column: "value"})
            rows["metric"] = "Cables"
            rows["asset_type"] = "cable"
            rows["asset_label"] = "cable " + rows["asset_id"].astype(str)
            rows["percentile"] = percentile
            rows["percentile_order"] = order
            frames.append(rows)

    voltage_map = {
        "p50": "voltage_p50_time_pu",
        "p10": "voltage_p10_time_pu",
        "p05": "voltage_p05_time_pu",
        "p01": "voltage_p01_time_pu",
        "min": "voltage_min_time_pu",
    }
    if not voltage_rows.empty:
        voltage_rows = voltage_rows.merge(meta, on=["powerflow_run_id", "stage"], how="left")
        for order, (percentile, column) in enumerate(voltage_map.items()):
            rows = voltage_rows[meta_cols + ["asset_id", column]].rename(columns={column: "value"})
            rows["metric"] = "Voltage"
            rows["asset_type"] = "bus"
            rows["asset_label"] = "bus " + rows["asset_id"].astype(str)
            rows["percentile"] = percentile
            rows["percentile_order"] = order
            frames.append(rows)

    out = pd.concat(frames, ignore_index=True)
    out["value"] = out["value"].astype(float)
    return out[
        meta_cols
        + [
            "metric",
            "asset_type",
            "asset_id",
            "asset_label",
            "percentile",
            "percentile_order",
            "value",
        ]
    ].dropna(subset=["value"]).reset_index(drop=True)


def comparison_powerflow_data_db(
    plz: int = 91301,
    synthetic_run_name: str = "baseline_static_pre_powerflow",
    real_run_name: str = "baseline_static_pre_powerflow_real_swf_hh_only",
    stage: str = "pre",
    scenario_id: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load synthetic and real SWF compact data with a shared comparison schema."""
    synthetic_summary = powerflow_headline_summary_db(
        run_name=synthetic_run_name,
        stage=stage,
        scenario_id=scenario_id,
        plz=plz,
    ).assign(powerflow_source="synthetic", comparison_group="Synthetic")
    synthetic_tail = powerflow_tail_values_db(
        run_name=synthetic_run_name,
        stage=stage,
        scenario_id=scenario_id,
        plz=plz,
    ).assign(powerflow_source="synthetic", comparison_group="Synthetic")
    synthetic_percentiles = powerflow_percentile_profile_db(
        run_name=synthetic_run_name,
        stage=stage,
        scenario_id=scenario_id,
        plz=plz,
    ).assign(powerflow_source="synthetic", comparison_group="Synthetic")

    real_summary = real_powerflow_headline_summary_db(
        run_name=real_run_name,
        stage=stage,
        scenario_id=scenario_id,
        plz=plz,
    )
    real_tail = real_powerflow_tail_values_db(
        run_name=real_run_name,
        stage=stage,
        scenario_id=scenario_id,
        plz=plz,
    )
    real_percentiles = real_powerflow_percentile_profile_db(
        run_name=real_run_name,
        stage=stage,
        scenario_id=scenario_id,
        plz=plz,
    )

    return (
        pd.concat([synthetic_summary, real_summary], ignore_index=True, sort=False),
        pd.concat([synthetic_tail, real_tail], ignore_index=True, sort=False),
        pd.concat([synthetic_percentiles, real_percentiles], ignore_index=True, sort=False),
    )

def plot_powerflow_headline_asset_violins(
    asset_summary: pd.DataFrame,
    group_col: str | None = None,
    show: bool = True,
):
    """Plot continuous p99/p01 tail-hour values using transformer, cable, and bus rows."""
    df = asset_summary.copy()
    if group_col is None or group_col not in df.columns:
        group_col = "comparison_group"
        df[group_col] = "All tail hours"

    metrics = [
        ("Transformer", "P99 tail loading [%]"),
        ("Cables", "P99 tail loading [%]"),
        ("Voltage", "P01 tail voltage [p.u.]"),
    ]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[title for title, _ in metrics])
    for col_idx, (metric, y_title) in enumerate(metrics, start=1):
        cols = [group_col, "value", "threshold_value", "asset_label", "grid", "t_index"]
        plot_df = df.loc[df["metric"] == metric, cols].dropna()
        fig.add_trace(
            go.Violin(
                x=plot_df[group_col].astype(str),
                y=plot_df["value"].astype(float),
                text=(
                    plot_df["grid"].astype(str)
                    + "<br>"
                    + plot_df["asset_label"].astype(str)
                    + "<br>t="
                    + plot_df["t_index"].astype(str)
                    + "<br>threshold="
                    + plot_df["threshold_value"].round(4).astype(str)
                ),
                hovertemplate="%{text}<br>%{y:.4g}<extra></extra>",
                box_visible=True,
                meanline_visible=True,
                points=False,
                scalemode="width",
                name=y_title,
                showlegend=False,
            ),
            row=1,
            col=col_idx,
        )
        fig.update_yaxes(title_text=y_title, row=1, col=col_idx)

    fig.update_layout(
        title="Critical Tail-Hour Values by Asset",
        violingap=0.12,
        height=430,
        margin={"l": 55, "r": 25, "t": 75, "b": 65},
    )
    if show:
        fig.show()
    return fig


def _hex_to_rgba(color: str, alpha: float) -> str:
    color = color.lstrip("#")
    if len(color) != 6:
        return f"rgba(51, 92, 129, {alpha})"
    r, g, b = (int(color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def plot_powerflow_percentile_profiles(
    profile: pd.DataFrame,
    group_col: str | None = None,
    show: bool = True,
    color_map: dict[str, str] | None = None,
    center_stat: str = "median",
    band_quantiles: tuple[float, float] | None = None,
    title: str = "Annual Percentile Profiles by Asset",
    points: str | bool | None = None,
):
    """Plot percentile profiles with a center line and full asset range band.

    For each metric and time-percentile, values are computed per asset. The line
    is the median or mean across assets. By default the shaded band spans the
    full min-to-max asset range, so critical extremes remain visible without
    plotting every asset as a separate point. Pass ``band_quantiles=(0.10, 0.90)``
    to use a quantile band instead.
    """
    df = profile.copy()
    if group_col is None or group_col not in df.columns:
        group_col = "comparison_group"
        df[group_col] = "All assets"

    center_stat = center_stat.lower().strip()
    if center_stat not in {"mean", "median"}:
        raise ValueError("center_stat must be either 'mean' or 'median'.")
    if band_quantiles is not None:
        lower_q, upper_q = band_quantiles
        if not 0 <= lower_q < upper_q <= 1:
            raise ValueError("band_quantiles must satisfy 0 <= lower < upper <= 1.")

    metrics = [
        ("Transformer", "Loading [%]"),
        ("Cables", "Loading [%]"),
        ("Voltage", "Voltage [p.u.]"),
    ]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[title for title, _ in metrics])
    default_colors = {
        "Synthetic": "#335C81",
        "Real SWF": "#D95D39",
        "synthetic": "#335C81",
        "real_swf": "#D95D39",
    }
    if color_map:
        default_colors.update({str(key): value for key, value in color_map.items()})
    fallback_palette = ["#335C81", "#D95D39", "#2A9D8F", "#6D597A", "#7A8450"]

    for col_idx, (metric, y_title) in enumerate(metrics, start=1):
        metric_df = df[df["metric"] == metric].copy()
        if metric_df.empty:
            continue
        for color_idx, (group, group_df) in enumerate(metric_df.groupby(group_col, sort=False)):
            grouped = group_df.groupby(["percentile_order", "percentile"], as_index=False)["value"]
            if band_quantiles is None:
                stats = grouped.agg(
                    center=center_stat,
                    band_lower="min",
                    band_upper="max",
                )
                band_label = "min-max asset range"
            else:
                lower_q, upper_q = band_quantiles
                stats = grouped.agg(
                    center=center_stat,
                    band_lower=lambda s: s.quantile(lower_q),
                    band_upper=lambda s: s.quantile(upper_q),
                )
                band_label = f"p{int(lower_q * 100):02d}-p{int(upper_q * 100):02d} asset band"
            stats = stats.sort_values("percentile_order")

            group_label = str(group)
            color = default_colors.get(group_label, fallback_palette[color_idx % len(fallback_palette)])
            fig.add_trace(
                go.Scatter(
                    x=stats["percentile"],
                    y=stats["band_upper"],
                    mode="lines",
                    line={"width": 0},
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=col_idx,
            )
            fig.add_trace(
                go.Scatter(
                    x=stats["percentile"],
                    y=stats["band_lower"],
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor=_hex_to_rgba(color, 0.16),
                    name=f"{group_label} {band_label}",
                    legendgroup=f"{group_label} band",
                    showlegend=col_idx == 1,
                    hovertemplate=f"%{{x}}<br>{band_label}<extra></extra>",
                ),
                row=1,
                col=col_idx,
            )
            fig.add_trace(
                go.Scatter(
                    x=stats["percentile"],
                    y=stats["center"],
                    mode="lines+markers",
                    line={"color": color, "width": 2.4},
                    marker={"size": 6, "color": color},
                    name=f"{group_label} {center_stat} across assets",
                    legendgroup=f"{group_label} center",
                    showlegend=col_idx == 1,
                    customdata=stats[["band_lower", "band_upper"]].to_numpy(),
                    hovertemplate=(
                        "%{x}<br>"
                        f"{center_stat}: %{{y:.4g}}<br>"
                        "range: %{customdata[0]:.4g} - %{customdata[1]:.4g}"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=col_idx,
            )
        fig.update_yaxes(title_text=y_title, row=1, col=col_idx)
        fig.update_xaxes(title_text="Time percentile", row=1, col=col_idx)

    fig.update_layout(
        title=title,
        legend={"title": {"text": "Profile summary"}},
        height=430,
        margin={"l": 55, "r": 25, "t": 75, "b": 65},
    )
    if show:
        fig.show()
    return fig

def plot_powerflow_headline_violins(
    summary: pd.DataFrame,
    group_col: str | None = None,
    show: bool = True,
):
    """Plot the three compact headline metrics as comparable violin plots."""
    df = summary.copy()
    if group_col is None or group_col not in df.columns:
        group_col = "comparison_group"
        df[group_col] = "All grids"

    metrics = [
        ("trafo_loading_p95_time_percent", "Transformer", "P95 loading [%]"),
        ("cable_loading_p95_asset_percent", "Cables", "P95 annual max loading [%]"),
        ("voltage_p05_load_bus_hour_pu", "Voltage", "P05 load-bus voltage [p.u.]"),
    ]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[title for _, title, _ in metrics])
    for col_idx, (metric, _, y_title) in enumerate(metrics, start=1):
        plot_df = df[[group_col, metric]].dropna()
        fig.add_trace(
            go.Violin(
                x=plot_df[group_col].astype(str),
                y=plot_df[metric].astype(float),
                box_visible=True,
                meanline_visible=True,
                points="all",
                jitter=0.18,
                scalemode="width",
                name=y_title,
                showlegend=False,
            ),
            row=1,
            col=col_idx,
        )
        fig.update_yaxes(title_text=y_title, row=1, col=col_idx)

    fig.update_layout(
        title="Headline Power-Flow Quality Metrics",
        violingap=0.12,
        height=430,
        margin={"l": 55, "r": 25, "t": 75, "b": 65},
    )
    if show:
        fig.show()
    return fig


def voltage_deviation_summary_db(
    input_id: str | None = None,
    run_name: str = "baseline_static_full_powerflow",
    stages: tuple[str, ...] = ("post",),
    scenario_id: int | None = None,
    ags: str | int | None = None,
    plz: int | None = None,
    kcid: int | None = None,
    bcid: int | None = None,
    candidate_index: int = 0,
    min_buildings: int = 5,
) -> pd.DataFrame:
    """Summarize DB voltage extrema for one grid or a population scope.

    Pass ``input_id`` for one concrete grid. Leave ``input_id`` as ``None`` to
    include all matching results, optionally narrowed by ``scenario_id``, ``ags``, ``plz``,
    ``kcid``, or ``bcid``.
    """
    db = SurroGridDatabase()
    run_id = None
    if input_id is not None:
        grid_ref = _resolve_db_grid(db, input_id, plz, kcid, bcid, candidate_index, min_buildings)
        run = _resolve_powerflow_run(db, grid_ref, run_name, scenario_id)
        run_id = int(run["powerflow_run_id"])

    query = text(
        """
        SELECT pr.powerflow_run_id,
               pr.run_name,
               pr.scenario_id,
               sc.scenario_key,
               gc.ags,
               gc.plz,
               gc.kcid,
               gc.bcid,
               gc.pylovo_grid_result_id,
               pbv.stage,
               MIN(pbv.vm_pu) AS min_vm_pu,
               MAX(pbv.vm_pu) AS max_vm_pu,
               COUNT(DISTINCT pbv.t_index) AS n_timesteps,
               COUNT(DISTINCT pbv.bus) AS n_buses
        FROM surrogrid.powerflow_bus_voltage pbv
        JOIN surrogrid.powerflow_run pr USING (powerflow_run_id)
        JOIN surrogrid.scenario sc USING (scenario_id)
        JOIN surrogrid.grid_case gc USING (grid_case_id)
        WHERE pr.run_name = :run_name
          AND pbv.stage = ANY(:stages)
          AND (:run_id IS NULL OR pbv.powerflow_run_id = :run_id)
          AND (:scenario_id IS NULL OR pr.scenario_id = :scenario_id)
          AND (:ags IS NULL OR gc.ags = :ags)
          AND (:filter_plz IS NULL OR gc.plz = :filter_plz)
          AND (:filter_kcid IS NULL OR gc.kcid = :filter_kcid)
          AND (:filter_bcid IS NULL OR gc.bcid = :filter_bcid)
        GROUP BY pr.powerflow_run_id, pr.run_name, pr.scenario_id, sc.scenario_key, gc.ags, gc.plz, gc.kcid, gc.bcid,
                 gc.pylovo_grid_result_id, pbv.stage
        ORDER BY pr.powerflow_run_id, pbv.stage
        """
    )
    with db.engine.connect() as conn:
        summary = pd.read_sql_query(
            query,
            conn,
            params={
                "run_name": run_name,
                "stages": list(stages),
                "run_id": run_id,
                "scenario_id": scenario_id,
                "ags": _normalize_optional_ags(ags),
                "filter_plz": plz if input_id is None else None,
                "filter_kcid": kcid if input_id is None else None,
                "filter_bcid": bcid if input_id is None else None,
            },
        )

    if summary.empty:
        raise ValueError(f"No DB voltage results found for run name {run_name!r}.")

    summary["grid"] = summary.apply(_grid_label_from_row, axis=1)
    return summary[
        [
            "grid",
            "powerflow_run_id",
            "run_name",
            "scenario_id",
            "scenario_key",
            "stage",
            "ags",
            "plz",
            "kcid",
            "bcid",
            "pylovo_grid_result_id",
            "n_timesteps",
            "n_buses",
            "min_vm_pu",
            "max_vm_pu",
        ]
    ].reset_index(drop=True)


def plot_voltage_deviation_histogram(
    summary: pd.DataFrame,
    lower_limit: float = 0.9,
    upper_limit: float = 1.1,
    bin_size: float = 0.01,
    show: bool = True,
):
    lower_values = summary["min_vm_pu"].astype(float)
    upper_values = summary["max_vm_pu"].astype(float)
    lower_share = (lower_values < lower_limit).mean() * 100.0
    upper_share = (upper_values > upper_limit).mean() * 100.0

    x_min = min(float(lower_values.min()), lower_limit) - 0.03
    x_max = max(float(upper_values.max()), upper_limit) + 0.03

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=upper_values,
            name="Max. Upper Voltage Deviation",
            marker={"color": "#2f92c5", "line": {"color": "white", "width": 0.5}},
            xbins={"start": x_min, "end": x_max, "size": bin_size},
            opacity=0.95,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=lower_values,
            name="Max. Lower Voltage Deviation",
            marker={"color": "#66c2a4", "line": {"color": "white", "width": 0.5}},
            xbins={"start": x_min, "end": x_max, "size": bin_size},
            opacity=0.95,
        )
    )
    fig.add_vline(x=lower_limit, line_color="#3a3a3a", line_dash="dash", line_width=2)
    fig.add_vline(x=upper_limit, line_color="#3a3a3a", line_dash="dash", line_width=2)
    fig.add_annotation(
        x=lower_limit - 0.012,
        y=0.72,
        xref="x",
        yref="paper",
        text=f"< {lower_limit:.1f} p.u.: {lower_share:.1f}%",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#d0d0d0",
        borderwidth=1,
    )
    fig.add_annotation(
        x=upper_limit + 0.012,
        y=0.72,
        xref="x",
        yref="paper",
        text=f"> {upper_limit:.1f} p.u.: {upper_share:.1f}%",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#d0d0d0",
        borderwidth=1,
    )
    fig.update_layout(
        barmode="overlay",
        title="Voltage Magnitude Extremes Across LV Grids",
        xaxis_title="Voltage Magnitude (p.u.)",
        yaxis_title="LV Grid Count",
        yaxis={"type": "log", "rangemode": "tozero"},
        legend={"orientation": "h", "x": 0.02, "y": 1.12},
        margin={"l": 70, "r": 30, "t": 80, "b": 65},
        width=820,
        height=420,
    )
    fig.update_xaxes(range=[x_min, x_max], showgrid=True, gridcolor="#d8d8d8")
    fig.update_yaxes(showgrid=True, gridcolor="#d8d8d8")
    if show:
        fig.show()
    return fig


def transformer_import_distribution_db(
    input_id: str | None = None,
    run_name: str = "baseline_static_full_powerflow",
    stage: str = "post",
    reactive_magnitude: bool = True,
    scenario_id: int | None = None,
    ags: str | int | None = None,
    plz: int | None = None,
    kcid: int | None = None,
    bcid: int | None = None,
    candidate_index: int = 0,
    min_buildings: int = 5,
) -> pd.DataFrame:
    """Read transformer import time series for one grid or a population scope.

    Pass ``input_id`` for one concrete grid. Leave ``input_id`` as ``None`` to
    aggregate all matching DB runs, optionally narrowed by ``scenario_id``, ``ags``, ``plz``,
    ``kcid``, or ``bcid``.
    """
    db = SurroGridDatabase()
    run_id = None
    if input_id is not None:
        grid_ref = _resolve_db_grid(db, input_id, plz, kcid, bcid, candidate_index, min_buildings)
        run = _resolve_powerflow_run(db, grid_ref, run_name, scenario_id)
        run_id = int(run["powerflow_run_id"])

    query = text(
        """
        SELECT pr.powerflow_run_id,
               pr.run_name,
               pr.scenario_id,
               sc.scenario_key,
               gc.ags,
               gc.plz,
               gc.kcid,
               gc.bcid,
               gc.pylovo_grid_result_id,
               pi.stage,
               pi.ts,
               pi.t_index,
               pi.p_mw,
               pi.q_mvar
        FROM surrogrid.powerflow_import pi
        JOIN surrogrid.powerflow_run pr USING (powerflow_run_id)
        JOIN surrogrid.scenario sc USING (scenario_id)
        JOIN surrogrid.grid_case gc USING (grid_case_id)
        WHERE pr.run_name = :run_name
          AND pi.stage = :stage
          AND (:run_id IS NULL OR pi.powerflow_run_id = :run_id)
          AND (:scenario_id IS NULL OR pr.scenario_id = :scenario_id)
          AND (:ags IS NULL OR gc.ags = :ags)
          AND (:filter_plz IS NULL OR gc.plz = :filter_plz)
          AND (:filter_kcid IS NULL OR gc.kcid = :filter_kcid)
          AND (:filter_bcid IS NULL OR gc.bcid = :filter_bcid)
        ORDER BY pr.powerflow_run_id, pi.t_index
        """
    )
    with db.engine.connect() as conn:
        df = pd.read_sql_query(
            query,
            conn,
            params={
                "run_name": run_name,
                "stage": stage,
                "run_id": run_id,
                "scenario_id": scenario_id,
                "ags": _normalize_optional_ags(ags),
                "filter_plz": plz if input_id is None else None,
                "filter_kcid": kcid if input_id is None else None,
                "filter_bcid": bcid if input_id is None else None,
            },
        )

    if df.empty:
        raise ValueError(f"No DB transformer import results found for run name {run_name!r}.")

    df["grid"] = df.apply(_grid_label_from_row, axis=1)
    df["q_import_mvar"] = df["q_mvar"].abs() if reactive_magnitude else df["q_mvar"]
    df["s_import_mva"] = np.hypot(df["p_mw"].astype(float), df["q_mvar"].astype(float))

    mean_s = df.groupby("powerflow_run_id")["s_import_mva"].transform("mean").replace(0.0, np.nan)
    max_s_by_grid = df.groupby("powerflow_run_id")["s_import_mva"].max().replace(0.0, np.nan)
    ldc_scale = float(max_s_by_grid.mean())
    if not np.isfinite(ldc_scale) or ldc_scale == 0.0:
        ldc_scale = np.nan
    df["p_ts_norm"] = df["p_mw"] / mean_s
    df["q_ts_norm"] = df["q_import_mvar"] / mean_s
    df["s_ts_norm"] = df["s_import_mva"] / mean_s
    df["p_ldc_norm"] = df["p_mw"] / ldc_scale
    df["q_ldc_norm"] = df["q_import_mvar"] / ldc_scale
    df["s_ldc_norm"] = df["s_import_mva"] / ldc_scale
    df.attrs["ldc_scale_mva"] = ldc_scale
    return df.reset_index(drop=True)


def _band_by_x(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    return (
        df.groupby(x_col)[y_col]
        .agg(
            expected="mean",
            q02=lambda series: series.quantile(0.02275),
            q16=lambda series: series.quantile(0.15865),
            q84=lambda series: series.quantile(0.84135),
            q98=lambda series: series.quantile(0.97725),
        )
        .reset_index()
        .sort_values(x_col)
    )


def _daily_transformer_bands(df: pd.DataFrame, y_col: str) -> pd.DataFrame:
    daily = df[["powerflow_run_id", "t_index", y_col]].copy()
    daily["day_index"] = daily["t_index"] // 24
    daily = daily.groupby(["powerflow_run_id", "day_index"], as_index=False)[y_col].mean()
    return _band_by_x(daily, "day_index", y_col)


def _ldc_transformer_bands(df: pd.DataFrame, y_col: str, n_points: int = 101) -> pd.DataFrame:
    percent_grid = np.linspace(0.0, 100.0, n_points)
    rows = []
    for run_id, group in df.groupby("powerflow_run_id"):
        values = group[y_col].dropna().sort_values(ascending=False).to_numpy()
        if len(values) == 0:
            continue
        percent = np.linspace(0.0, 100.0, len(values))
        rows.append(
            pd.DataFrame(
                {
                    "powerflow_run_id": run_id,
                    "duration_percent": percent_grid,
                    y_col: np.interp(percent_grid, percent, values),
                }
            )
        )
    if not rows:
        raise ValueError(f"No values available to build LDC bands for {y_col}.")
    return _band_by_x(pd.concat(rows, ignore_index=True), "duration_percent", y_col)


def _add_distribution_panel(
    fig,
    row: int,
    col: int,
    band: pd.DataFrame,
    x_col: str,
    color: str,
    fill68: str,
    fill96: str,
    expected_name: str,
    showlegend: bool,
) -> None:
    fig.add_trace(
        go.Scatter(
            x=band[x_col],
            y=band["q02"],
            mode="lines",
            line={"width": 0, "color": fill96},
            hoverinfo="skip",
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=band[x_col],
            y=band["q98"],
            mode="lines",
            line={"width": 0, "color": fill96},
            fill="tonexty",
            fillcolor=fill96,
            name="96% Percentile Band",
            hoverinfo="skip",
            showlegend=showlegend,
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=band[x_col],
            y=band["q16"],
            mode="lines",
            line={"width": 0, "color": fill68},
            hoverinfo="skip",
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=band[x_col],
            y=band["q84"],
            mode="lines",
            line={"width": 0, "color": fill68},
            fill="tonexty",
            fillcolor=fill68,
            name="68% Percentile Band",
            hoverinfo="skip",
            showlegend=showlegend,
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=band[x_col],
            y=band["expected"],
            mode="lines",
            line={"color": color, "width": 2},
            name=expected_name,
            showlegend=showlegend,
        ),
        row=row,
        col=col,
    )


def plot_transformer_import_distributions(
    df: pd.DataFrame,
    show: bool = True,
):
    series = [
        {
            "key": "p",
            "ts_col": "p_ts_norm",
            "ldc_col": "p_ldc_norm",
            "title": "Net Transformer Active Power P Import",
            "ts_y": "Norm. Active Power Import P_i(t) / <|S_i|>",
            "ldc_y": "Norm. Active Power Import P_i / max|S|",
            "color": "#ef3b2c",
            "fill68": "rgba(239, 59, 44, 0.25)",
            "fill96": "rgba(239, 59, 44, 0.14)",
        },
        {
            "key": "q",
            "ts_col": "q_ts_norm",
            "ldc_col": "q_ldc_norm",
            "title": "Net Transformer Reactive Power |Q| Import",
            "ts_y": "Norm. Reactive Power Import |Q_i(t)| / <|S_i|>",
            "ldc_y": "Norm. Reactive Power Import |Q_i| / max|S|",
            "color": "#5b54ff",
            "fill68": "rgba(91, 84, 255, 0.22)",
            "fill96": "rgba(91, 84, 255, 0.12)",
        },
        {
            "key": "s",
            "ts_col": "s_ts_norm",
            "ldc_col": "s_ldc_norm",
            "title": "Net Transformer Apparent Power |S| Load",
            "ts_y": "Norm. Apparent Power Load |S_i(t)| / <|S_i|>",
            "ldc_y": "Norm. Apparent Power Load |S_i| / max|S|",
            "color": "#174a7e",
            "fill68": "rgba(49, 130, 189, 0.35)",
            "fill96": "rgba(49, 130, 189, 0.18)",
        },
    ]

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=[item for serie in series for item in (serie["title"], serie["title"])],
        horizontal_spacing=0.12,
        vertical_spacing=0.12,
    )

    for row, serie in enumerate(series, start=1):
        ts_band = _daily_transformer_bands(df, serie["ts_col"])
        ldc_band = _ldc_transformer_bands(df, serie["ldc_col"])
        _add_distribution_panel(
            fig,
            row=row,
            col=1,
            band=ts_band,
            x_col="day_index",
            color=serie["color"],
            fill68=serie["fill68"],
            fill96=serie["fill96"],
            expected_name="Expected Timeseries (24 h Agg.)",
            showlegend=(row == 1),
        )
        _add_distribution_panel(
            fig,
            row=row,
            col=2,
            band=ldc_band,
            x_col="duration_percent",
            color=serie["color"],
            fill68=serie["fill68"],
            fill96=serie["fill96"],
            expected_name="Expected LDC (Hourly)",
            showlegend=False,
        )
        fig.update_yaxes(title_text=serie["ts_y"], row=row, col=1)
        fig.update_yaxes(title_text=serie["ldc_y"], row=row, col=2)

    month_days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for row in range(1, 4):
        fig.update_xaxes(tickmode="array", tickvals=month_days, ticktext=month_labels, row=row, col=1)
        fig.update_xaxes(ticksuffix="%", range=[0, 100], row=row, col=2)

    fig.update_layout(
        height=980,
        width=1120,
        hovermode="x unified",
        legend={"orientation": "h", "x": 0.02, "y": 1.05},
        margin={"l": 80, "r": 35, "t": 80, "b": 55},
    )
    fig.update_xaxes(showgrid=True, gridcolor="#d8d8d8")
    fig.update_yaxes(showgrid=True, gridcolor="#d8d8d8", zeroline=True, zerolinecolor="#b8b8b8")
    if show:
        fig.show()
    return fig


def _uses_relative_timeslice_axis(df: pd.DataFrame) -> bool:
    if "t_index" not in df.columns:
        return False
    max_horizon = df.groupby("powerflow_run_id")["t_index"].nunique().max()
    return bool(pd.notna(max_horizon) and int(max_horizon) <= 24 * 14)


def _wide_transformer_frame(
    df: pd.DataFrame,
    value_col: str,
    *,
    relative_axis: bool,
) -> pd.DataFrame:
    index_col = "t_index" if relative_axis else "ts"
    if index_col not in df.columns:
        raise ValueError(f"Transformer import plotting requires a {index_col!r} column.")
    wide = df.pivot_table(
        index=index_col,
        columns="powerflow_run_id",
        values=value_col,
        aggfunc="mean",
    ).sort_index()
    if wide.empty:
        raise ValueError(f"No transformer import values available for {value_col}.")
    if relative_axis:
        wide.index = wide.index.astype(int)
        wide.index.name = "t_index"
    else:
        wide.index = pd.to_datetime(wide.index)
        if wide.index.isna().any():
            wide.index = pd.date_range("2009-01-01", periods=len(wide), freq="h")
    return wide


def _matplotlib_quantile_summary(wide: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "expected": wide.mean(axis=1),
            "q02": wide.quantile(0.02275, axis=1),
            "q16": wide.quantile(0.15865, axis=1),
            "q84": wide.quantile(0.84135, axis=1),
            "q98": wide.quantile(0.97725, axis=1),
        },
        index=wide.index,
    )


def _daily_matplotlib_transformer_bands(
    df: pd.DataFrame,
    value_col: str,
    *,
    relative_axis: bool,
) -> pd.DataFrame:
    wide = _wide_transformer_frame(df, value_col, relative_axis=relative_axis)
    if relative_axis:
        daily = wide.groupby(wide.index // 24).mean()
        daily.index.name = "day_index"
    else:
        daily = wide.resample("24h").mean()
    return _matplotlib_quantile_summary(daily)


def _ldc_matplotlib_transformer_bands(
    df: pd.DataFrame,
    value_col: str,
    n_points: int = 101,
    *,
    relative_axis: bool,
) -> pd.DataFrame:
    wide = _wide_transformer_frame(df, value_col, relative_axis=relative_axis)
    duration_percent = np.linspace(0.0, 100.0, n_points)
    curves = []
    for column in wide.columns:
        values = wide[column].dropna().sort_values(ascending=False).to_numpy()
        if len(values) == 0:
            continue
        source_percent = np.linspace(0.0, 100.0, len(values))
        curves.append(pd.Series(np.interp(duration_percent, source_percent, values), name=column))
    if not curves:
        raise ValueError(f"No transformer import values available for {value_col}.")
    ldc = pd.concat(curves, axis=1)
    ldc.index = duration_percent
    return _matplotlib_quantile_summary(ldc)


def _style_transformer_axis(ax) -> None:
    ax.grid(which="major", axis="y", linestyle="-", linewidth=0.55, alpha=0.45)
    ax.grid(which="major", axis="x", linestyle="--", linewidth=0.45, alpha=0.35)
    ax.tick_params(axis="both", which="both", direction="inout", length=5, width=0.8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def _format_month_axis(ax, index: pd.DatetimeIndex) -> None:
    start = index[0].normalize()
    end = index[-1].normalize() + pd.offsets.MonthBegin(1)
    boundaries = pd.date_range(start, end, freq="MS")
    if len(boundaries) < 2:
        return
    centers = boundaries[:-1] + (boundaries[1:] - boundaries[:-1]) / 2
    labels = [value.strftime("%b") for value in boundaries[:-1]]
    ax.set_xticks(centers)
    ax.set_xticklabels(labels)
    ax.set_xticks(boundaries, minor=True)
    ax.tick_params(axis="x", which="major", length=0)
    ax.tick_params(axis="x", which="minor", direction="out", length=5, width=0.8)
    ax.set_xlim(index[0], index[-1])


def _format_relative_day_axis(ax, day_index: pd.Index) -> None:
    if len(day_index) == 0:
        return
    first_day = int(day_index.min())
    last_day = int(day_index.max())
    n_days = last_day - first_day + 1
    if n_days <= 14:
        ticks = np.arange(first_day, last_day + 1)
        labels = [f"Day {int(day - first_day + 1)}" for day in ticks]
    else:
        month_days = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
        month_labels = np.array([
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ])
        mask = (month_days >= first_day) & (month_days <= last_day)
        ticks = month_days[mask]
        labels = month_labels[mask].tolist()
        if len(ticks) == 0:
            ticks = np.linspace(first_day, last_day, min(6, n_days))
            labels = [f"Day {int(round(day - first_day + 1))}" for day in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(first_day, last_day)


def _plot_matplotlib_band(
    ax,
    band: pd.DataFrame,
    x,
    color: str,
    band68_color: str,
    band96_color: str,
    expected_label: str,
    show_legend: bool,
) -> None:
    ax.fill_between(
        x,
        band["q02"].to_numpy(),
        band["q98"].to_numpy(),
        facecolor=band96_color,
        edgecolor="none",
        alpha=0.55,
        label="96% Percentile Band",
    )
    ax.fill_between(
        x,
        band["q16"].to_numpy(),
        band["q84"].to_numpy(),
        facecolor=band68_color,
        edgecolor="none",
        alpha=0.45,
        label="68% Percentile Band",
    )
    ax.plot(
        x,
        band["expected"].to_numpy(),
        color=color,
        linewidth=1.4,
        alpha=0.95,
        label=expected_label,
    )
    _style_transformer_axis(ax)
    if show_legend:
        legend = ax.legend(
            frameon=True,
            framealpha=0.72,
            fontsize=7,
            loc="upper right",
            handlelength=1.2,
            borderpad=0.25,
            labelspacing=0.25,
        )
        legend.get_frame().set_edgecolor("#cccccc")


def plot_transformer_import_distributions_matplotlib(
    df: pd.DataFrame,
    show: bool = True,
):
    specs = [
        {
            "ts_col": "p_ts_norm",
            "ldc_col": "p_ldc_norm",
            "title": "Net Transformer Active Power P Import",
            "ts_y": r"Norm. Active Power Import $P_i(t) / \langle |S_i| \rangle$",
            "ldc_y": r"Norm. Active Power Import $P_i / \langle \max |S| \rangle$",
            "color": "#ef3b2c",
            "band68": "#fcae91",
            "band96": "#fee0d2",
        },
        {
            "ts_col": "q_ts_norm",
            "ldc_col": "q_ldc_norm",
            "title": "Net Transformer Reactive Power |Q| Import",
            "ts_y": r"Norm. React. Power Import $|Q_i(t)| / \langle |S_i| \rangle$",
            "ldc_y": r"Norm. React. Power Import $|Q_i| / \langle \max |S| \rangle$",
            "color": "#5b54ff",
            "band68": "#bcbddc",
            "band96": "#efedf5",
        },
        {
            "ts_col": "s_ts_norm",
            "ldc_col": "s_ldc_norm",
            "title": "Net Transformer Apparent Power |S| Load",
            "ts_y": r"Norm. Apparent Power Load $|S_i(t)| / \langle |S_i| \rangle$",
            "ldc_y": r"Norm. Apparent Power Load $|S_i| / \langle \max |S| \rangle$",
            "color": "#08306b",
            "band68": "#3182bd",
            "band96": "#9ecae1",
        },
    ]

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "text.usetex": False,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    ):
        fig, axes = plt.subplots(3, 2, figsize=(10, 9.2), dpi=150)
        relative_axis = _uses_relative_timeslice_axis(df)
        for row, spec in enumerate(specs):
            ts_band = _daily_matplotlib_transformer_bands(
                df,
                spec["ts_col"],
                relative_axis=relative_axis,
            )
            ldc_band = _ldc_matplotlib_transformer_bands(
                df,
                spec["ldc_col"],
                relative_axis=relative_axis,
            )

            ax_ts = axes[row, 0]
            ax_ldc = axes[row, 1]
            if relative_axis:
                x_ts = ts_band.index.to_numpy(dtype=float)
            else:
                x_ts = mdates.date2num(ts_band.index.to_pydatetime())
            x_ldc = ldc_band.index.to_numpy(dtype=float)

            _plot_matplotlib_band(
                ax_ts,
                ts_band,
                x_ts,
                spec["color"],
                spec["band68"],
                spec["band96"],
                "Expected Timeseries (24 h Agg.)",
                show_legend=True,
            )
            _plot_matplotlib_band(
                ax_ldc,
                ldc_band,
                x_ldc,
                spec["color"],
                spec["band68"],
                spec["band96"],
                "Expected LDC (Hourly)",
                show_legend=True,
            )

            if relative_axis:
                _format_relative_day_axis(ax_ts, ts_band.index)
            else:
                ax_ts.xaxis_date()
                _format_month_axis(ax_ts, ts_band.index)
            ax_ldc.set_xlim(0, 100)
            ax_ldc.set_xticks(np.arange(0, 101, 20))
            ax_ldc.set_xticklabels([f"{value}%" for value in range(0, 101, 20)])
            ax_ts.set_title(spec["title"])
            ax_ldc.set_title(spec["title"])
            ax_ts.set_ylabel(spec["ts_y"])
            ax_ldc.set_ylabel(spec["ldc_y"])

        fig.tight_layout(h_pad=1.8, w_pad=2.0)
        if show:
            plt.show()
        return fig


def plot_transformer_apparent_power_stage_comparison_matplotlib(
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    stage_titles: tuple[str, str] = ("Pre: electricity demand", "Post: electrification"),
    show: bool = True,
):
    datasets = [
        (pre_df, stage_titles[0]),
        (post_df, stage_titles[1]),
    ]
    color = "#08306b"
    band68 = "#3182bd"
    band96 = "#9ecae1"

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "text.usetex": False,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), dpi=150, sharey=True)
        for ax, (df, title) in zip(axes, datasets):
            relative_axis = _uses_relative_timeslice_axis(df)
            band = _daily_matplotlib_transformer_bands(
                df,
                "s_ts_norm",
                relative_axis=relative_axis,
            )
            if relative_axis:
                x_values = band.index.to_numpy(dtype=float)
            else:
                x_values = mdates.date2num(band.index.to_pydatetime())
            _plot_matplotlib_band(
                ax,
                band,
                x_values,
                color,
                band68,
                band96,
                "Expected Timeseries (24 h Agg.)",
                show_legend=True,
            )
            if relative_axis:
                _format_relative_day_axis(ax, band.index)
            else:
                ax.xaxis_date()
                _format_month_axis(ax, band.index)
            ax.set_title(title)
            ax.set_ylabel(r"Norm. Apparent Power Load $|S_i(t)| / \langle |S_i| \rangle$")

        fig.tight_layout(w_pad=2.0)
        if show:
            plt.show()
        return fig


def _normalize_optional_ags(ags: str | int | None) -> int | None:
    if ags is None:
        return None
    return int(str(ags).lstrip("0") or "0")


def available_powerflow_results_db(
    run_name: str | None = None,
    stages: tuple[str, ...] = ("pre", "post"),
    scenario_id: int | None = None,
    ags: str | int | None = None,
    plz: int | None = None,
    kcid: int | None = None,
    bcid: int | None = None,
) -> pd.DataFrame:
    """List DB-backed power-flow result rows available for plotting.

    The returned ``grid`` column is the bridge-style grid identifier used by
    single-grid heatmap helpers. Population plots can use this catalog to choose
    all results or filter by scenario, AGS, PLZ, KCID, and BCID.
    """
    db = SurroGridDatabase()
    query = text(
        """
        SELECT pr.powerflow_run_id,
               pr.run_name,
               pr.pre_only,
               pr.scenario_id,
               sc.scenario_key,
               sc.scenario_label,
               gc.grid_case_id,
               gc.ags,
               gc.plz,
               gc.kcid,
               gc.bcid,
               gc.cell_id,
               gc.pylovo_grid_result_id,
               MIN(pbv.t_index) AS min_timestep,
               MAX(pbv.t_index) AS max_timestep,
               COUNT(DISTINCT pbv.t_index) AS n_timesteps,
               ARRAY_AGG(DISTINCT pbv.stage ORDER BY pbv.stage) AS stages,
               pr.updated_at
        FROM surrogrid.powerflow_run pr
        JOIN surrogrid.grid_case gc USING (grid_case_id)
        JOIN surrogrid.scenario sc USING (scenario_id)
        JOIN surrogrid.powerflow_bus_voltage pbv USING (powerflow_run_id)
        WHERE (:run_name IS NULL OR pr.run_name = :run_name)
          AND (:scenario_id IS NULL OR pr.scenario_id = :scenario_id)
          AND (:ags IS NULL OR gc.ags = :ags)
          AND (:plz IS NULL OR gc.plz = :plz)
          AND (:kcid IS NULL OR gc.kcid = :kcid)
          AND (:bcid IS NULL OR gc.bcid = :bcid)
          AND pbv.stage = ANY(:stages)
        GROUP BY pr.powerflow_run_id,
                 pr.run_name,
                 pr.pre_only,
                 pr.scenario_id,
                 sc.scenario_key,
                 sc.scenario_label,
                 gc.grid_case_id,
                 gc.ags,
                 gc.plz,
                 gc.kcid,
                 gc.bcid,
                 gc.cell_id,
                 gc.pylovo_grid_result_id,
                 pr.updated_at
        ORDER BY gc.ags, gc.plz, gc.kcid, gc.bcid, pr.run_name, pr.powerflow_run_id
        """
    )
    with db.engine.connect() as conn:
        df = pd.read_sql_query(
            query,
            conn,
            params={
                "run_name": run_name,
                "stages": list(stages),
                "scenario_id": scenario_id,
                "ags": _normalize_optional_ags(ags),
                "plz": plz,
                "kcid": kcid,
                "bcid": bcid,
            },
        )

    if df.empty:
        raise ValueError("No DB power-flow results found for the selected filters.")

    df["grid"] = (
        df["cell_id"].astype(str)
        + "_"
        + df["plz"].astype(int).astype(str)
        + "_"
        + df["kcid"].astype(int).astype(str)
        + "_"
        + df["bcid"].astype(int).astype(str)
        + ".h5"
    )
    df["ags_label"] = df["ags"].astype(int).map(lambda value: str(value).zfill(8))
    df["label"] = (
        df["grid"]
        + " | "
        + df["scenario_key"]
        + " | "
        + df["run_name"]
        + " | "
        + df["n_timesteps"].astype(int).astype(str)
        + " timesteps"
    )
    return df[
        [
            "label",
            "grid",
            "powerflow_run_id",
            "run_name",
            "pre_only",
            "scenario_id",
            "scenario_key",
            "scenario_label",
            "ags",
            "ags_label",
            "plz",
            "kcid",
            "bcid",
            "grid_case_id",
            "pylovo_grid_result_id",
            "min_timestep",
            "max_timestep",
            "n_timesteps",
            "stages",
            "updated_at",
        ]
    ].reset_index(drop=True)


def line_loading_distribution_db(
    input_id: str | None = None,
    run_name: str = "baseline_static_full_powerflow",
    stages: tuple[str, ...] = ("pre", "post"),
    scenario_id: int | None = None,
    ags: str | int | None = None,
    plz: int | None = None,
    kcid: int | None = None,
    bcid: int | None = None,
    candidate_index: int = 0,
    min_buildings: int = 5,
) -> pd.DataFrame:
    """Compute per-line maximum loading for one grid or a population scope.

    Pass ``input_id`` for one concrete grid. Leave ``input_id`` as ``None`` to
    include all matching DB runs, optionally narrowed by ``scenario_id``, ``ags``, ``plz``,
    ``kcid``, or ``bcid``.
    """
    db = SurroGridDatabase()
    run_id = None
    if input_id is not None:
        grid_ref = _resolve_db_grid(db, input_id, plz, kcid, bcid, candidate_index, min_buildings)
        run = _resolve_powerflow_run(db, grid_ref, run_name, scenario_id)
        run_id = int(run["powerflow_run_id"])

    query = text(
        """
        SELECT pr.powerflow_run_id,
               pr.run_name,
               pr.scenario_id,
               sc.scenario_key,
               gc.grid_case_id,
               gc.ags,
               gc.plz,
               gc.kcid,
               gc.bcid,
               gc.cell_id,
               gc.pylovo_grid_result_id,
               gc.pylovo_version_id,
               lr.stage,
               lr.line,
               MAX(ABS(lr.i_from_ka)) AS max_i_from_ka,
               COUNT(*) AS n_timesteps
        FROM surrogrid.powerflow_line_result lr
        JOIN surrogrid.powerflow_run pr USING (powerflow_run_id)
        JOIN surrogrid.scenario sc USING (scenario_id)
        JOIN surrogrid.grid_case gc USING (grid_case_id)
        WHERE pr.run_name = :run_name
          AND lr.stage = ANY(:stages)
          AND (:run_id IS NULL OR pr.powerflow_run_id = :run_id)
          AND (:scenario_id IS NULL OR pr.scenario_id = :scenario_id)
          AND (:ags IS NULL OR gc.ags = :ags)
          AND (:filter_plz IS NULL OR gc.plz = :filter_plz)
          AND (:filter_kcid IS NULL OR gc.kcid = :filter_kcid)
          AND (:filter_bcid IS NULL OR gc.bcid = :filter_bcid)
        GROUP BY pr.powerflow_run_id,
                 pr.run_name,
                 pr.scenario_id,
                 sc.scenario_key,
                 gc.grid_case_id,
                 gc.ags,
                 gc.plz,
                 gc.kcid,
                 gc.bcid,
                 gc.cell_id,
                 gc.pylovo_grid_result_id,
                 gc.pylovo_version_id,
                 lr.stage,
                 lr.line
        ORDER BY gc.ags, gc.plz, gc.kcid, gc.bcid, pr.powerflow_run_id, lr.stage, lr.line
        """
    )
    with db.engine.connect() as conn:
        df = pd.read_sql_query(
            query,
            conn,
            params={
                "run_name": run_name,
                "stages": list(stages),
                "run_id": run_id,
                "scenario_id": scenario_id,
                "ags": _normalize_optional_ags(ags),
                "filter_plz": plz if input_id is None else None,
                "filter_kcid": kcid if input_id is None else None,
                "filter_bcid": bcid if input_id is None else None,
            },
        )

    if df.empty:
        scope = input_id if input_id is not None else "all matching DB runs"
        raise ValueError(f"No DB line loading results found for {scope!r}.")

    df["grid"] = (
        df["cell_id"].astype(str)
        + "_"
        + df["plz"].astype(int).astype(str)
        + "_"
        + df["kcid"].astype(int).astype(str)
        + "_"
        + df["bcid"].astype(int).astype(str)
        + ".h5"
    )
    df["line"] = df["line"].astype(int)

    rating_frames = []
    for run_id_value, group in df.groupby("powerflow_run_id", sort=False):
        first = group.iloc[0]
        net = db.read_pandapower_grid(
            {
                "grid_result_id": int(first["pylovo_grid_result_id"]),
                "version_id": str(first["pylovo_version_id"]),
            }
        )
        ratings = net.line[["max_i_ka"]].copy()
        ratings["line"] = ratings.index.astype(int)
        ratings["powerflow_run_id"] = int(run_id_value)
        if "name" in net.line.columns:
            names = net.line["name"].copy()
            ratings["line_name"] = ratings["line"].map(names.to_dict())
        else:
            ratings["line_name"] = np.nan
        rating_frames.append(ratings.reset_index(drop=True))

    ratings = pd.concat(rating_frames, ignore_index=True)
    df = df.merge(ratings, on=["powerflow_run_id", "line"], how="left")
    df["max_i_ka"] = df["max_i_ka"].astype(float).replace(0.0, np.nan)
    df["loading_percent"] = (df["max_i_from_ka"].astype(float) / df["max_i_ka"]) * 100.0
    df["comparison"] = df["stage"].map(
        {"pre": "status_quo_pre", "post": "full_pipeline_post"}
    ).fillna(df["stage"])
    df["line_name"] = df["line_name"].fillna(df["line"].map(lambda line: f"Line {line}"))
    stage_order = {stage: order for order, stage in enumerate(stages)}
    df["stage_order"] = df["stage"].map(stage_order).fillna(len(stage_order)).astype(int)
    df.sort_values(["grid", "stage_order", "line"], inplace=True)
    return df[
        [
            "grid",
            "powerflow_run_id",
            "run_name",
            "scenario_id",
            "scenario_key",
            "comparison",
            "stage",
            "line",
            "line_name",
            "max_i_from_ka",
            "max_i_ka",
            "loading_percent",
            "n_timesteps",
            "ags",
            "plz",
            "kcid",
            "bcid",
        ]
    ].reset_index(drop=True)


def grid_loading_stress_summary(
    line_loading: pd.DataFrame,
    high_threshold: float = 80.0,
    overload_threshold: float = 100.0,
) -> pd.DataFrame:
    """Summarize line-loading stress per grid, run, and stage.

    The returned table includes robust ranking metrics such as
    ``median_line_max_loading_percent`` and ``p95_line_max_loading_percent``.
    The plotting notebook uses these columns to sort candidate grids before
    opening a single-grid network heatmap.
    """
    rows = []
    group_cols = [
        "grid",
        "powerflow_run_id",
        "run_name",
        "scenario_id",
        "scenario_key",
        "comparison",
        "stage",
        "ags",
        "plz",
        "kcid",
        "bcid",
    ]
    for keys, group in line_loading.groupby(group_cols, dropna=False, sort=False):
        values = group["loading_percent"].dropna().astype(float)
        if values.empty:
            continue
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "n_lines": int(values.size),
                "median_line_max_loading_percent": float(values.median()),
                "p95_line_max_loading_percent": float(values.quantile(0.95)),
                "max_line_loading_percent": float(values.max()),
                "high_loaded_lines": int((values >= high_threshold).sum()),
                "overloaded_lines": int((values >= overload_threshold).sum()),
                "share_lines_above_80_percent": float(
                    (values >= high_threshold).mean() * 100.0
                ),
                "share_lines_above_100_percent": float(
                    (values >= overload_threshold).mean() * 100.0
                ),
            }
        )
        rows.append(row)
    if not rows:
        raise ValueError("No valid line loading values available for grid stress summary.")
    return pd.DataFrame(rows)


def plot_line_loading_ecdf(
    line_loading: pd.DataFrame,
    show: bool = True,
):
    fig = go.Figure()
    colors = {"pre": "#636efa", "post": "#ef553b"}
    for stage, group in line_loading.groupby("stage", sort=False):
        values = group["loading_percent"].dropna().astype(float).sort_values().to_numpy()
        if len(values) == 0:
            continue
        share = np.arange(1, len(values) + 1, dtype=float) / len(values) * 100.0
        fig.add_trace(
            go.Scatter(
                x=values,
                y=share,
                mode="lines",
                name=str(stage),
                line={"width": 2.4, "color": colors.get(str(stage))},
                hovertemplate=(
                    "Stage: %{fullData.name}<br>"
                    "Line max loading: %{x:.2f}%<br>"
                    "Share of line maxima below: %{y:.1f}%<extra></extra>"
                ),
            )
        )
    for threshold, color in [(80.0, "#f39c12"), (100.0, "#c0392b")]:
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color=color,
            annotation_text=f"{threshold:.0f}%",
            annotation_position="top right",
        )
    fig.update_layout(
        title="Distribution of Line Maximum Loadings",
        xaxis_title="Line maximum loading over all timesteps [%]",
        yaxis_title="Share of line maxima below threshold [%]",
        xaxis={"rangemode": "tozero"},
        yaxis={"range": [0, 100]},
        legend_title="Stage",
        margin={"l": 75, "r": 35, "t": 70, "b": 65},
        height=430,
    )
    if show:
        fig.show()
    return fig




def max_line_loading_summary_db(
    input_id: str,
    run_name: str = "baseline_static_full_powerflow",
    stages: tuple[str, ...] = ("pre", "post"),
    scenario_id: int | None = None,
    plz: int | None = None,
    kcid: int | None = None,
    bcid: int | None = None,
    candidate_index: int = 0,
    min_buildings: int = 5,
) -> pd.DataFrame:
    db = SurroGridDatabase()
    grid_ref = _resolve_db_grid(db, input_id, plz, kcid, bcid, candidate_index, min_buildings)
    run = _resolve_powerflow_run(db, grid_ref, run_name, scenario_id)
    net = db.read_pandapower_grid(grid_ref)

    query = text(
        """
        SELECT stage, ts, t_index, line, i_from_ka
        FROM surrogrid.powerflow_line_result
        WHERE powerflow_run_id = :run_id
          AND stage = ANY(:stages)
        """
    )
    with db.engine.connect() as conn:
        df = pd.read_sql_query(
            query,
            conn,
            params={"run_id": int(run["powerflow_run_id"]), "stages": list(stages)},
        )

    if df.empty:
        raise ValueError(f"No DB line results found for run {run['powerflow_run_id']}.")

    max_i_ka = net.line["max_i_ka"].astype(float).replace(0.0, np.nan)
    df["max_i_ka"] = df["line"].map(max_i_ka.to_dict())
    df["loading_percent"] = (df["i_from_ka"] / df["max_i_ka"]) * 100.0

    idx = df.groupby("stage", sort=False)["loading_percent"].idxmax()
    summary = df.loc[idx].copy()
    stage_order = {stage: order for order, stage in enumerate(stages)}
    summary["stage_order"] = summary["stage"].map(stage_order)
    summary.sort_values("stage_order", inplace=True)

    line_names = net.line.get("name", pd.Series(index=net.line.index, dtype=object))
    summary["line_name"] = summary["line"].map(line_names.to_dict()).fillna(
        summary["line"].map(lambda line: f"Line {line}")
    )
    summary.insert(0, "scenario_key", run.get("scenario_key"))
    summary.insert(0, "scenario_id", int(run["scenario_id"]))
    summary.insert(0, "run_name", run["run_name"])
    summary.insert(0, "powerflow_run_id", int(run["powerflow_run_id"]))
    summary.insert(0, "grid", grid_ref["bridge_filename"])
    summary["comparison"] = summary["stage"].map(
        {"pre": "status_quo_pre", "post": "full_pipeline_post"}
    ).fillna(summary["stage"])

    return summary[
        [
            "grid",
            "powerflow_run_id",
            "run_name",
            "scenario_id",
            "scenario_key",
            "comparison",
            "stage",
            "ts",
            "t_index",
            "line",
            "line_name",
            "i_from_ka",
            "max_i_ka",
            "loading_percent",
        ]
    ].reset_index(drop=True)


def plot_max_line_loading_comparison(
    summary: pd.DataFrame,
    show: bool = True,
):
    fig = go.Figure(
        data=[
            go.Bar(
                x=summary["comparison"],
                y=summary["loading_percent"],
                text=summary["loading_percent"].map(lambda value: f"{value:.1f}%"),
                textposition="outside",
                customdata=summary[["stage", "t_index", "line", "i_from_ka", "max_i_ka"]],
                hovertemplate=(
                    "Comparison: %{x}<br>"
                    "Stage: %{customdata[0]}<br>"
                    "Timestep: %{customdata[1]}<br>"
                    "Line: %{customdata[2]}<br>"
                    "Current: %{customdata[3]:.4f} kA<br>"
                    "Rating: %{customdata[4]:.4f} kA<br>"
                    "Loading: %{y:.2f}%<extra></extra>"
                ),
            )
        ]
    )
    fig.add_hline(
        y=100.0,
        line_dash="dash",
        line_color="firebrick",
        annotation_text="100%",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Maximum Line Loading Comparison",
        xaxis_title="Power-flow result",
        yaxis_title="Max line loading [%]",
        yaxis={"rangemode": "tozero"},
        margin={"l": 70, "r": 30, "t": 70, "b": 70},
    )
    if show:
        fig.show()
    return fig


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot GridExpand power-flow heatmaps and DB-backed loading comparisons."
    )
    parser.add_argument(
        "input",
        help=(
            "HDF5 output path when --storage h5, or DB grid identifier such as "
            "9278140-00_94342_1_-1.h5 / 09278140 when --storage db."
        ),
    )
    parser.add_argument(
        "--storage",
        choices=("h5", "db"),
        default="h5",
        help="Read plotting data from HDF5 or DB-backed Step 4 tables (default: h5).",
    )
    parser.add_argument(
        "--stage",
        choices=("pre", "post"),
        default="pre",
        help="Which result stage to plot for heatmaps (default: pre).",
    )
    parser.add_argument("--timestep", type=int, default=0, help="Timestep index (default: 0).")
    parser.add_argument(
        "--run-name",
        default="baseline_static_full_powerflow",
        help="DB power-flow run name (default: baseline_static_full_powerflow).",
    )
    parser.add_argument("--plz", type=int, help="DB mode: pin one PLZ.")
    parser.add_argument("--kcid", type=int, help="DB mode: pin one KCID.")
    parser.add_argument("--bcid", type=int, help="DB mode: pin one BCID.")
    parser.add_argument(
        "--candidate-index",
        type=int,
        default=0,
        help="DB mode: 0-based candidate grid index for the given AGS.",
    )
    parser.add_argument(
        "--min-buildings",
        type=int,
        default=5,
        help="DB mode: minimum buildings required when selecting AGS candidates.",
    )
    parser.add_argument(
        "--compare-max-loading",
        action="store_true",
        help="DB mode: print and plot max line loading for pre versus post stages.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        help="Write the generated Plotly figure to this HTML file.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Build the figure without calling fig.show(). Useful for scripts and CI.",
    )
    parser.add_argument(
        "--on-map",
        action="store_true",
        help="Try mapbox background (auto-falls back if geodata is not lon/lat).",
    )
    parser.add_argument("--map-style", default="light", help="Plotly map style (default: light).")
    parser.add_argument("--cmap", default="Jet", help="Colormap name for pf_res_plotly (default: Jet).")
    parser.add_argument(
        "--climits-volt",
        nargs=2,
        type=float,
        metavar=("VMIN", "VMAX"),
        default=(0.9, 1.1),
        help="Voltage color limits in p.u. (default: 0.9 1.1).",
    )
    parser.add_argument(
        "--climits-load",
        nargs=2,
        type=float,
        metavar=("LMIN", "LMAX"),
        default=(0.0, 100.0),
        help="Line loading color limits in percent (default: 0 100).",
    )
    parser.add_argument(
        "--show-household-buses",
        action="store_true",
        help="Show household/load buses. Default hides them to reduce visual clutter.",
    )
    return parser


def _write_html_if_requested(fig, output_html: Path | None) -> None:
    if output_html is None:
        return
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_html)


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    show = not args.no_show

    if args.storage == "db" and args.compare_max_loading:
        summary = max_line_loading_summary_db(
            input_id=args.input,
            run_name=args.run_name,
            plz=args.plz,
            kcid=args.kcid,
            bcid=args.bcid,
            candidate_index=args.candidate_index,
            min_buildings=args.min_buildings,
        )
        print(summary.to_string(index=False))
        fig = plot_max_line_loading_comparison(summary, show=show)
        _write_html_if_requested(fig, args.output_html)
        return

    if args.storage == "db":
        fig = plot_powerflow_heatmap_db(
            input_id=args.input,
            stage=args.stage,
            timestep=args.timestep,
            on_map=args.on_map,
            map_style=args.map_style,
            cmap=args.cmap,
            climits_volt=tuple(args.climits_volt),
            climits_load=tuple(args.climits_load),
            show_household_buses=args.show_household_buses,
            show=show,
            run_name=args.run_name,
            plz=args.plz,
            kcid=args.kcid,
            bcid=args.bcid,
            candidate_index=args.candidate_index,
            min_buildings=args.min_buildings,
        )
        _write_html_if_requested(fig, args.output_html)
        return

    h5_path = Path(args.input)
    if not h5_path.exists():
        raise FileNotFoundError(f"File not found: {h5_path}")

    fig = plot_powerflow_heatmap(
        h5_path=h5_path,
        stage=args.stage,
        timestep=args.timestep,
        on_map=args.on_map,
        map_style=args.map_style,
        cmap=args.cmap,
        climits_volt=tuple(args.climits_volt),
        climits_load=tuple(args.climits_load),
        show_household_buses=args.show_household_buses,
        show=show,
    )
    _write_html_if_requested(fig, args.output_html)


if __name__ == "__main__":
    main()
