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
from scipy.stats import wasserstein_distance


GRIDEXPAND_DIR = Path(__file__).resolve().parents[2]
if str(GRIDEXPAND_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDEXPAND_DIR))

from database import SurroGridDatabase


def save_plotly_figure(
    fig: go.Figure,
    output_path: str | Path,
    formats: tuple[str, ...] = ("png", "svg"),
    width: int | None = None,
    height: int | None = None,
    scale: float = 2.0,
    active_slider_step: int | str | None = None,
) -> list[Path]:
    """Save a Plotly figure to one or more static image formats.

    ``output_path`` can be either a path without suffix, such as
    ``output/asset-percentiles``, or a concrete file path. Static Plotly export
    requires the ``kaleido`` package, which is included in the Step 5
    environment. For figures with sliders, ``active_slider_step`` selects the
    static state to export by zero-based step index or by exact step label.
    """
    export_fig = go.Figure(fig)
    if active_slider_step is not None:
        _apply_plotly_slider_step(export_fig, active_slider_step)

    output_path = Path(output_path)
    if output_path.suffix:
        base_path = output_path.with_suffix("")
        if not formats:
            formats = (output_path.suffix.lstrip("."),)
    else:
        base_path = output_path
    base_path.parent.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for image_format in formats:
        fmt = image_format.lower().lstrip(".")
        target = base_path.with_suffix(f".{fmt}")
        try:
            export_fig.write_image(str(target), format=fmt, width=width, height=height, scale=scale)
        except ValueError as exc:
            raise RuntimeError(
                "Static Plotly export failed. Make sure the Step 5 environment "
                "contains kaleido by running `uv sync` in GridExpand/5.postprocessing."
            ) from exc
        saved_paths.append(target)
    return saved_paths


def _apply_plotly_slider_step(fig: go.Figure, active_slider_step: int | str) -> None:
    sliders = fig.layout.sliders
    if not sliders:
        raise ValueError("active_slider_step was provided, but the Plotly figure has no sliders.")
    slider = sliders[0]
    steps = list(slider.steps)
    if isinstance(active_slider_step, str):
        labels = [str(step.label) for step in steps]
        try:
            step_index = labels.index(active_slider_step)
        except ValueError as exc:
            available = ", ".join(labels)
            raise ValueError(
                f"Unknown slider step label {active_slider_step!r}. Available labels: {available}."
            ) from exc
    else:
        step_index = int(active_slider_step)
    if step_index < 0 or step_index >= len(steps):
        raise IndexError(f"Slider step index {step_index} is outside [0, {len(steps) - 1}].")

    step = steps[step_index]
    args = list(step.args) if step.args is not None else []
    if args:
        trace_update = args[0] or {}
        visible = trace_update.get("visible") if isinstance(trace_update, dict) else None
        if visible is not None:
            for trace, is_visible in zip(fig.data, visible):
                trace.visible = bool(is_visible)
    if len(args) > 1 and isinstance(args[1], dict):
        fig.update_layout(args[1])
    fig.layout.sliders[0].active = step_index


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
    if "parallel" in net.line.columns:
        parallel = net.line["parallel"].reindex(i_from_ka.index).fillna(1).astype(float)
    else:
        parallel = pd.Series(1.0, index=i_from_ka.index)
    loading_percent = (i_from_ka / (max_i_ka * parallel)) * 100.0
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


def _add_headline_asset_percentiles(
    summary: pd.DataFrame,
    db: SurroGridDatabase,
    *,
    cable_table: str,
    voltage_table: str,
    run_id_column: str,
) -> pd.DataFrame:
    if summary.empty:
        return summary
    run_ids = summary["powerflow_run_id"].dropna().astype(int).unique().tolist()
    if not run_ids:
        return summary

    cable_query = text(
        f"""
        SELECT {run_id_column} AS powerflow_run_id,
               stage,
               percentile_cont(0.50) WITHIN GROUP (ORDER BY cable_loading_max_time_percent) AS cable_loading_p50_asset_percent,
               percentile_cont(0.90) WITHIN GROUP (ORDER BY cable_loading_max_time_percent) AS cable_loading_p90_asset_percent,
               percentile_cont(0.95) WITHIN GROUP (ORDER BY cable_loading_max_time_percent) AS cable_loading_p95_asset_percent_derived,
               percentile_cont(0.99) WITHIN GROUP (ORDER BY cable_loading_max_time_percent) AS cable_loading_p99_asset_percent,
               MAX(cable_loading_max_time_percent) AS cable_loading_max_asset_percent
        FROM {cable_table}
        WHERE {run_id_column} = ANY(:run_ids)
          AND cable_loading_max_time_percent IS NOT NULL
        GROUP BY {run_id_column}, stage
        """
    )
    voltage_query = text(
        f"""
        SELECT {run_id_column} AS powerflow_run_id,
               stage,
               percentile_cont(0.50) WITHIN GROUP (ORDER BY voltage_min_time_pu) FILTER (WHERE voltage_min_time_pu IS NOT NULL) AS voltage_p50_asset_time_pu,
               percentile_cont(0.10) WITHIN GROUP (ORDER BY voltage_min_time_pu) FILTER (WHERE voltage_min_time_pu IS NOT NULL) AS voltage_p10_asset_time_pu,
               percentile_cont(0.05) WITHIN GROUP (ORDER BY voltage_min_time_pu) FILTER (WHERE voltage_min_time_pu IS NOT NULL) AS voltage_p05_asset_time_pu,
               percentile_cont(0.01) WITHIN GROUP (ORDER BY voltage_min_time_pu) FILTER (WHERE voltage_min_time_pu IS NOT NULL) AS voltage_p01_asset_time_pu,
               MIN(voltage_min_time_pu) AS voltage_min_asset_time_pu
        FROM {voltage_table}
        WHERE {run_id_column} = ANY(:run_ids)
        GROUP BY {run_id_column}, stage
        """
    )
    with db.engine.connect() as conn:
        cable = pd.read_sql_query(cable_query, conn, params={"run_ids": run_ids})
        voltage = pd.read_sql_query(voltage_query, conn, params={"run_ids": run_ids})

    out = summary.copy()
    if not cable.empty:
        out = out.merge(cable, on=["powerflow_run_id", "stage"], how="left")
        if "cable_loading_p95_asset_percent_derived" in out.columns:
            out["cable_loading_p95_asset_percent"] = out["cable_loading_p95_asset_percent"].fillna(
                out["cable_loading_p95_asset_percent_derived"]
            )
            out.drop(columns=["cable_loading_p95_asset_percent_derived"], inplace=True)
    if not voltage.empty:
        out = out.merge(voltage, on=["powerflow_run_id", "stage"], how="left")

    for column in (
        "cable_loading_p50_asset_percent",
        "cable_loading_p90_asset_percent",
        "cable_loading_p95_asset_percent",
        "cable_loading_p99_asset_percent",
        "cable_loading_max_asset_percent",
        "voltage_p50_asset_time_pu",
        "voltage_p10_asset_time_pu",
        "voltage_p05_asset_time_pu",
        "voltage_p01_asset_time_pu",
        "voltage_min_asset_time_pu",
    ):
        if column not in out.columns:
            out[column] = pd.NA
    return out


def _add_synthetic_household_scope(summary: pd.DataFrame, db: SurroGridDatabase) -> pd.DataFrame:
    if summary.empty:
        return summary
    run_ids = summary["powerflow_run_id"].dropna().astype(int).unique().tolist()
    if not run_ids:
        return summary

    query = text(
        """
        WITH selected_runs AS (
            SELECT pr.powerflow_run_id, pr.grid_case_id
            FROM surrogrid.powerflow_run pr
            WHERE pr.powerflow_run_id = ANY(:run_ids)
        )
        SELECT sr.powerflow_run_id,
               COUNT(*) FILTER (
                   WHERE lower(COALESCE(gbb.building_use, '')) = 'residential'
               ) AS selected_household_load_rows,
               COUNT(DISTINCT gbb.bus) FILTER (
                   WHERE lower(COALESCE(gbb.building_use, '')) = 'residential'
                     AND gbb.bus IS NOT NULL
               ) AS selected_household_load_buses,
               COALESCE(SUM(gbb.households) FILTER (
                   WHERE lower(COALESCE(gbb.building_use, '')) = 'residential'
               ), 0) AS selected_household_equivalents,
               COUNT(*) FILTER (
                   WHERE lower(COALESCE(gbb.building_use, '')) <> 'residential'
                      OR gbb.building_use IS NULL
               ) AS non_household_load_rows,
               COUNT(DISTINCT gbb.bus) FILTER (
                   WHERE (lower(COALESCE(gbb.building_use, '')) <> 'residential'
                          OR gbb.building_use IS NULL)
                     AND gbb.bus IS NOT NULL
               ) AS non_household_load_buses
        FROM selected_runs sr
        LEFT JOIN surrogrid.grid_building_bus gbb USING (grid_case_id)
        GROUP BY sr.powerflow_run_id
        """
    )
    with db.engine.connect() as conn:
        household_scope = pd.read_sql_query(query, conn, params={"run_ids": run_ids})

    out = summary.merge(household_scope, on="powerflow_run_id", how="left")
    for column in (
        "selected_household_load_rows",
        "selected_household_load_buses",
        "selected_household_equivalents",
        "non_household_load_rows",
        "non_household_load_buses",
    ):
        if column not in out.columns:
            out[column] = pd.NA
    return out


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
               pfs.n_converged_timesteps,
               pfs.n_failed_timesteps,
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

    summary = _add_headline_asset_percentiles(
        summary,
        db,
        cable_table="surrogrid.powerflow_cable_summary",
        voltage_table="surrogrid.powerflow_bus_voltage_summary",
        run_id_column="powerflow_run_id",
    )
    summary = _add_synthetic_household_scope(summary, db)
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
            "selected_household_load_rows",
            "selected_household_load_buses",
            "selected_household_equivalents",
            "non_household_load_rows",
            "non_household_load_buses",
            "n_timesteps",
            "n_converged_timesteps",
            "n_failed_timesteps",
            "n_voltage_buses",
            "n_cables",
            "transformer_s_rated_mva",
            "trafo_loading_p50_time_percent",
            "trafo_loading_p90_time_percent",
            "trafo_loading_p95_time_percent",
            "trafo_loading_p99_time_percent",
            "trafo_loading_max_time_percent",
            "trafo_loading_hours_above_100",
            "cable_loading_p50_asset_percent",
            "cable_loading_p90_asset_percent",
            "cable_loading_p95_asset_percent",
            "cable_loading_p99_asset_percent",
            "cable_loading_max_asset_percent",
            "cable_hours_above_100_p95_asset",
            "voltage_p50_asset_time_pu",
            "voltage_p10_asset_time_pu",
            "voltage_p05_asset_time_pu",
            "voltage_p01_asset_time_pu",
            "voltage_min_asset_time_pu",
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
        "selected_household_load_rows",
        "selected_household_load_buses",
        "n_timesteps",
        "n_converged_timesteps",
        "n_failed_timesteps",
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
        "selected_household_load_rows",
        "selected_household_load_buses",
        "n_timesteps",
        "n_converged_timesteps",
        "n_failed_timesteps",
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


def latest_synthetic_powerflow_summary_run_name(
    stage: str = "pre",
    scenario_id: int | None = None,
    ags: str | int | None = None,
    plz: int | None = None,
    db: SurroGridDatabase | None = None,
) -> str:
    """Return the newest synthetic run name with compact power-flow summaries."""
    db = db or SurroGridDatabase()
    query = text(
        """
        SELECT
            pr.run_name,
            COUNT(DISTINCT pr.powerflow_run_id) AS summary_grids,
            MAX(pfs.created_at) AS latest_summary_at
        FROM surrogrid.powerflow_summary pfs
        JOIN surrogrid.powerflow_run pr USING (powerflow_run_id)
        JOIN surrogrid.grid_case gc USING (grid_case_id)
        WHERE pfs.stage = :stage
          AND (:scenario_id IS NULL OR pr.scenario_id = :scenario_id)
          AND (:ags IS NULL OR gc.ags = :ags)
          AND (:filter_plz IS NULL OR gc.plz = :filter_plz)
        GROUP BY pr.run_name
        ORDER BY latest_summary_at DESC, summary_grids DESC, pr.run_name DESC
        LIMIT 1
        """
    )
    with db.engine.connect() as conn:
        row = conn.execute(
            query,
            {
                "stage": stage,
                "scenario_id": scenario_id,
                "ags": _normalize_optional_ags(ags),
                "filter_plz": plz,
            },
        ).mappings().first()
    if row is None:
        raise ValueError(
            "No compact synthetic power-flow summary run found for the selected filters. "
            "Run the pipeline with --powerflow-output summary or --powerflow-output both first."
        )
    return str(row["run_name"])


def load_synthetic_powerflow_cutoff_profile(
    run_name: str | None = None,
    stage: str = "pre",
    scenario_id: int | None = None,
    ags: str | int | None = None,
    plz: int | None = None,
    kcid: int | None = None,
    bcid: int | None = None,
    min_buildings: int = 5,
) -> pd.DataFrame:
    """Load synthetic asset-percentile profiles for retained-asset cutoff plots."""
    if run_name is None:
        run_name = latest_synthetic_powerflow_summary_run_name(
            stage=stage,
            scenario_id=scenario_id,
            ags=ags,
            plz=plz,
        )
    profile = powerflow_percentile_profile_db(
        run_name=run_name,
        stage=stage,
        scenario_id=scenario_id,
        ags=ags,
        plz=plz,
        kcid=kcid,
        bcid=bcid,
        min_buildings=min_buildings,
    )
    profile = profile.copy()
    profile["comparison_group"] = "Synthetic"
    return profile


def _real_grid_label_from_row(row: pd.Series) -> str:
    return f"SWF LV_{int(row['lv_id']):03d}"


def real_powerflow_headline_summary_db(
    run_name: str = "baseline_static_pre_powerflow_real_swf_hh_only_backbone",
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
               NULLIF(rpr.assumptions ->> 'household_load_rows_before_supply_filter', '')::INTEGER AS household_load_rows_before_supply_filter,
               NULLIF(rpr.assumptions ->> 'household_load_buses_before_supply_filter', '')::INTEGER AS household_load_buses_before_supply_filter,
               NULLIF(rpr.assumptions ->> 'dropped_unsupplied_household_load_rows', '')::INTEGER AS dropped_unsupplied_household_load_rows,
               NULLIF(rpr.assumptions ->> 'dropped_unsupplied_household_load_buses', '')::INTEGER AS dropped_unsupplied_household_load_buses,
               NULLIF(rpr.assumptions ->> 'selected_household_load_rows', '')::INTEGER AS selected_household_load_rows,
               NULLIF(rpr.assumptions ->> 'selected_household_load_buses', '')::INTEGER AS selected_household_load_buses,
               NULLIF(rpr.assumptions ->> 'backbone_voltage_buses', '')::INTEGER AS backbone_voltage_buses,
               NULLIF(rpr.assumptions ->> 'backbone_cables', '')::INTEGER AS backbone_cables,
               rps.stage,
               rps.n_timesteps,
               rps.n_converged_timesteps,
               rps.n_failed_timesteps,
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

    summary = _add_headline_asset_percentiles(
        summary,
        db,
        cable_table="surrogrid.real_powerflow_cable_summary",
        voltage_table="surrogrid.real_powerflow_bus_voltage_summary",
        run_id_column="real_powerflow_run_id",
    )
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
            "household_load_rows_before_supply_filter",
            "household_load_buses_before_supply_filter",
            "dropped_unsupplied_household_load_rows",
            "dropped_unsupplied_household_load_buses",
            "selected_household_load_rows",
            "selected_household_load_buses",
            "backbone_voltage_buses",
            "backbone_cables",
            "n_timesteps",
            "n_converged_timesteps",
            "n_failed_timesteps",
            "n_voltage_buses",
            "n_cables",
            "transformer_s_rated_mva",
            "trafo_loading_p50_time_percent",
            "trafo_loading_p90_time_percent",
            "trafo_loading_p95_time_percent",
            "trafo_loading_p99_time_percent",
            "trafo_loading_max_time_percent",
            "trafo_loading_hours_above_100",
            "cable_loading_p50_asset_percent",
            "cable_loading_p90_asset_percent",
            "cable_loading_p95_asset_percent",
            "cable_loading_p99_asset_percent",
            "cable_loading_max_asset_percent",
            "cable_hours_above_100_p95_asset",
            "voltage_p50_asset_time_pu",
            "voltage_p10_asset_time_pu",
            "voltage_p05_asset_time_pu",
            "voltage_p01_asset_time_pu",
            "voltage_min_asset_time_pu",
            "voltage_p05_load_bus_hour_pu",
            "voltage_hours_below_0_90_p95_asset",
        ]
    ].reset_index(drop=True)


def real_powerflow_tail_values_db(
    run_name: str = "baseline_static_pre_powerflow_real_swf_hh_only_backbone",
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
        "household_load_rows_before_supply_filter",
        "household_load_buses_before_supply_filter",
        "dropped_unsupplied_household_load_rows",
        "dropped_unsupplied_household_load_buses",
        "selected_household_load_rows",
        "selected_household_load_buses",
        "backbone_voltage_buses",
        "backbone_cables",
        "n_timesteps",
        "n_converged_timesteps",
        "n_failed_timesteps",
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
    run_name: str = "baseline_static_pre_powerflow_real_swf_hh_only_backbone",
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
        "household_load_rows_before_supply_filter",
        "household_load_buses_before_supply_filter",
        "dropped_unsupplied_household_load_rows",
        "dropped_unsupplied_household_load_buses",
        "selected_household_load_rows",
        "selected_household_load_buses",
        "backbone_voltage_buses",
        "backbone_cables",
        "n_timesteps",
        "n_converged_timesteps",
        "n_failed_timesteps",
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


def _run_key(df: pd.DataFrame) -> pd.Series:
    return df["powerflow_source"].astype(str) + ":" + df["powerflow_run_id"].astype(int).astype(str)


def _filter_by_run_keys(df: pd.DataFrame, run_keys: set[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.assign(_run_key=_run_key)
    return out[out["_run_key"].isin(run_keys)].drop(columns="_run_key").reset_index(drop=True)


def _split_comparison_groups(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    synthetic = summary[summary["comparison_group"].eq("Synthetic")].reset_index(drop=True)
    real = summary[summary["comparison_group"].eq("Real SWF")].reset_index(drop=True)
    return synthetic, real


def _comparison_convergence_overview(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(
            columns=["comparison_group", "grids_total", "grids_non_converged", "grids_unknown_convergence", "failed_timesteps"]
        )
    return (
        summary.assign(n_failed_timesteps=lambda df: df["n_failed_timesteps"].fillna(0))
        .groupby("comparison_group", as_index=False)
        .agg(
            grids_total=("grid", "nunique"),
            grids_non_converged=("n_failed_timesteps", lambda s: int((s > 0).sum())),
            grids_unknown_convergence=("n_failed_timesteps", lambda s: int(s.isna().sum())),
            failed_timesteps=("n_failed_timesteps", "sum"),
        )
    )


def _comparison_coverage(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(columns=["comparison_group", "grids", "assets_voltage", "assets_cables"])
    return (
        summary.groupby("comparison_group", as_index=False)
        .agg(
            grids=("grid", "nunique"),
            assets_voltage=("n_voltage_buses", "sum"),
            assets_cables=("n_cables", "sum"),
        )
    )


def load_powerflow_comparison_data(
    *,
    plz: int = 91301,
    synthetic_run_name: str = "baseline_synthetic_hh_only",
    real_run_name: str = "baseline_real",
    stage: str = "pre",
    min_selected_household_buses: int = 10,
    filter_non_converged_grids: bool = True,
    scenario_id: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Load and filter the synthetic-vs-real power-flow comparison tables.

    Returns a dictionary with the summary/profile tables used by
    ``powerflow_retained_asset_cutoff_comparison.ipynb`` plus lightweight audit tables:
    ``scope_filter_overview``, ``filtered_grids``, ``convergence_overview`` and
    ``coverage``.
    """
    synthetic_summary_all = powerflow_headline_summary_db(
        run_name=synthetic_run_name,
        stage=stage,
        scenario_id=scenario_id,
        plz=plz,
    ).assign(powerflow_source="synthetic", comparison_group="Synthetic")
    real_summary_all = real_powerflow_headline_summary_db(
        run_name=real_run_name,
        stage=stage,
        scenario_id=scenario_id,
        plz=plz,
    )
    summary_all = pd.concat([synthetic_summary_all, real_summary_all], ignore_index=True, sort=False)

    scope_mask = summary_all["selected_household_load_buses"].fillna(0).ge(min_selected_household_buses)
    scope_filter_overview = (
        summary_all.assign(_kept=scope_mask)
        .groupby("comparison_group", as_index=False)
        .agg(
            criterion=(
                "comparison_group",
                lambda _: f"selected_household_load_buses >= {min_selected_household_buses}",
            ),
            grids_before=("grid", "nunique"),
            grids_filtered=("_kept", lambda s: int((~s).sum())),
            grids_kept=("_kept", lambda s: int(s.sum())),
        )
    )
    filtered_grids = summary_all.loc[
        ~scope_mask,
        [
            "comparison_group",
            "grid",
            "selected_household_load_rows",
            "selected_household_load_buses",
            "n_voltage_buses",
            "n_cables",
        ],
    ].sort_values(["comparison_group", "selected_household_load_buses", "grid"])

    summary = summary_all.loc[scope_mask].reset_index(drop=True)
    synthetic_summary, real_summary = _split_comparison_groups(summary)

    synthetic_profile_all = powerflow_percentile_profile_db(
        run_name=synthetic_run_name,
        stage=stage,
        scenario_id=scenario_id,
        plz=plz,
    ).assign(powerflow_source="synthetic", comparison_group="Synthetic")
    real_profile_all = real_powerflow_percentile_profile_db(
        run_name=real_run_name,
        stage=stage,
        scenario_id=scenario_id,
        plz=plz,
    )
    percentile_profile_all = pd.concat([synthetic_profile_all, real_profile_all], ignore_index=True, sort=False)
    percentile_profile = _filter_by_run_keys(percentile_profile_all, set(_run_key(summary)))

    if filter_non_converged_grids:
        convergence_mask = summary["n_failed_timesteps"].isna() | summary["n_failed_timesteps"].eq(0)
        converged_run_keys = set(_run_key(summary.loc[convergence_mask]))
        summary = _filter_by_run_keys(summary, converged_run_keys)
        percentile_profile = _filter_by_run_keys(percentile_profile, converged_run_keys)
        synthetic_summary, real_summary = _split_comparison_groups(summary)

    return {
        "summary_all": summary_all.reset_index(drop=True),
        "summary": summary.reset_index(drop=True),
        "synthetic_summary": synthetic_summary,
        "real_summary": real_summary,
        "percentile_profile_all": percentile_profile_all.reset_index(drop=True),
        "percentile_profile": percentile_profile.reset_index(drop=True),
        "scope_filter_overview": scope_filter_overview.reset_index(drop=True),
        "filtered_grids": filtered_grids.reset_index(drop=True),
        "convergence_overview": _comparison_convergence_overview(summary),
        "coverage": _comparison_coverage(summary),
    }


def powerflow_comparison_grid_count_summary(
    *,
    plz: int,
    synthetic_run_name: str,
    real_run_name: str,
    stage: str,
    scope_filter_overview: pd.DataFrame,
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize launched, completed, and retained real/synthetic power-flow grids.

    ``load_powerflow_comparison_data`` only sees runs with a written summary row.
    This helper additionally audits launched run rows, so hard failures without a
    summary remain visible in comparison notebooks.
    """
    db = SurroGridDatabase()
    run_audit_query = text(
        """
        WITH synthetic_runs AS (
            SELECT
                'Synthetic' AS comparison_group,
                COUNT(DISTINCT pr.powerflow_run_id) AS launched_powerflow_runs,
                COUNT(DISTINCT pr.powerflow_run_id) FILTER (WHERE pfs.powerflow_run_id IS NOT NULL) AS powerflow_summary_grids
            FROM surrogrid.powerflow_run pr
            JOIN surrogrid.grid_case gc USING (grid_case_id)
            LEFT JOIN surrogrid.powerflow_summary pfs
              ON pfs.powerflow_run_id = pr.powerflow_run_id
             AND pfs.stage = :stage
            WHERE pr.run_name = :synthetic_run_name
              AND gc.plz = :plz
        ),
        real_runs AS (
            SELECT
                'Real SWF' AS comparison_group,
                COUNT(DISTINCT rpr.real_powerflow_run_id) AS launched_powerflow_runs,
                COUNT(DISTINCT rpr.real_powerflow_run_id) FILTER (WHERE rps.real_powerflow_run_id IS NOT NULL) AS powerflow_summary_grids
            FROM surrogrid.real_powerflow_run rpr
            JOIN surrogrid.real_grid_case rgc USING (real_grid_case_id)
            LEFT JOIN surrogrid.real_powerflow_summary rps
              ON rps.real_powerflow_run_id = rpr.real_powerflow_run_id
             AND rps.stage = :stage
            WHERE rpr.run_name = :real_run_name
              AND rgc.plz = :plz
        )
        SELECT * FROM synthetic_runs
        UNION ALL
        SELECT * FROM real_runs
        """
    )
    with db.engine.connect() as conn:
        run_audit = pd.read_sql_query(
            run_audit_query,
            conn,
            params={
                "plz": plz,
                "stage": stage,
                "synthetic_run_name": synthetic_run_name,
                "real_run_name": real_run_name,
            },
        )

    grid_count_summary = (
        run_audit.merge(
            scope_filter_overview.rename(
                columns={
                    "grids_before": "powerflow_summary_grids_before_filter",
                    "grids_filtered": "grids_removed_by_filter",
                    "grids_kept": "powerflow_grids_after_filter",
                }
            ),
            on="comparison_group",
            how="left",
        )
        .merge(
            coverage.rename(
                columns={
                    "grids": "coverage_grids_after_filter",
                    "assets_voltage": "voltage_assets_after_filter",
                    "assets_cables": "cable_assets_after_filter",
                }
            ),
            on="comparison_group",
            how="left",
        )
    )
    grid_count_summary["hard_failed_runs_without_summary"] = (
        grid_count_summary["launched_powerflow_runs"]
        - grid_count_summary["powerflow_summary_grids"]
    )
    grid_count_summary.insert(
        1,
        "run_name",
        grid_count_summary["comparison_group"].map(
            {"Synthetic": synthetic_run_name, "Real SWF": real_run_name}
        ),
    )
    return grid_count_summary[
        [
            "comparison_group",
            "run_name",
            "launched_powerflow_runs",
            "powerflow_summary_grids",
            "hard_failed_runs_without_summary",
            "criterion",
            "grids_removed_by_filter",
            "powerflow_grids_after_filter",
            "voltage_assets_after_filter",
            "cable_assets_after_filter",
        ]
    ]


def powerflow_distribution_similarity_summary(
    profile: pd.DataFrame,
    group_col: str = "comparison_group",
    synthetic_group: str = "Synthetic",
    real_group: str = "Real SWF",
) -> pd.DataFrame:
    """Compare critical synthetic and real power-flow result distributions.

    The table uses the same critical result semantics as the asset-cutoff plots:
    transformer/cable annual maximum loading and annual minimum voltage. Signed
    differences are calculated as synthetic minus real.
    """
    required = {group_col, "metric", "percentile", "value"}
    missing_required = required.difference(profile.columns)
    if missing_required:
        missing = ", ".join(sorted(missing_required))
        raise ValueError(f"Missing column(s) for distribution similarity summary: {missing}.")

    critical_percentiles = {
        "Transformer": "max",
        "Cables": "max",
        "Voltage": "min",
    }
    rows = []
    for metric, percentile in critical_percentiles.items():
        metric_rows = profile[
            profile["metric"].eq(metric)
            & profile["percentile"].eq(percentile)
        ]
        synthetic_values = metric_rows.loc[
            metric_rows[group_col].eq(synthetic_group), "value"
        ].astype(float).dropna()
        real_values = metric_rows.loc[
            metric_rows[group_col].eq(real_group), "value"
        ].astype(float).dropna()
        if synthetic_values.empty or real_values.empty:
            rows.append(
                {
                    "metric": metric,
                    "median_diff": np.nan,
                    "std": np.nan,
                    "wasserstein": np.nan,
                }
            )
            continue
        rows.append(
            {
                "metric": metric,
                "median_diff": synthetic_values.median() - real_values.median(),
                "std": synthetic_values.std(ddof=1) - real_values.std(ddof=1),
                "wasserstein": wasserstein_distance(synthetic_values, real_values),
            }
        )
    return pd.DataFrame(rows, columns=["metric", "median_diff", "std", "wasserstein"])


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


def _powerflow_y_axis_ranges(
    y_axis_limits: tuple[float | None, float | None, float | None] | None,
) -> dict[str, list[float]]:
    if y_axis_limits is None:
        return {}
    if len(y_axis_limits) != 3:
        raise ValueError(
            "y_axis_limits must be a tuple of "
            "(transformer_upper_percent, cable_upper_percent, voltage_lower_pu)."
        )
    transformer_upper, cable_upper, voltage_lower = y_axis_limits
    ranges: dict[str, list[float]] = {}
    if transformer_upper is not None:
        ranges["Transformer"] = [0.0, float(transformer_upper)]
    if cable_upper is not None:
        ranges["Cables"] = [0.0, float(cable_upper)]
    if voltage_lower is not None:
        ranges["Voltage"] = [float(voltage_lower), 1.0]
    return ranges


def _powerflow_y_axis_slider_layout(y_axis_ranges: dict[str, list[float]]) -> dict[str, object]:
    axis_by_metric = {
        "Transformer": "yaxis",
        "Cables": "yaxis2",
        "Voltage": "yaxis3",
    }
    layout: dict[str, object] = {}
    for metric, axis_name in axis_by_metric.items():
        if metric in y_axis_ranges:
            layout[f"{axis_name}.range"] = y_axis_ranges[metric]
            layout[f"{axis_name}.autorange"] = False
        else:
            layout[f"{axis_name}.autorange"] = True
    return layout


def plot_powerflow_pooled_asset_percentile_curves(
    profile: pd.DataFrame,
    group_col: str | None = None,
    show: bool = True,
    color_map: dict[str, str] | None = None,
    asset_percentiles: tuple[float, ...] | None = None,
    asset_cutoff_percentiles: tuple[float, ...] | None = None,
    metrics: tuple[str, ...] = ("Transformer", "Cables", "Voltage"),
    title: str = "Pooled Annual Critical Values by Retained-Asset Cutoff",
    y_axis_limits: tuple[float | None, float | None, float | None] | None = None,
    show_band: bool = True,
):
    """Plot cumulative retained-asset cutoff curves over the pooled asset set.

    Transformer and cable values use each asset's annual maximum loading.
    Voltage values use each retained bus' annual minimum voltage.

    The x-axis is the retained-asset cutoff percentile. At P90 for loading
    metrics, the most critical 10% of assets are excluded and the plotted set is
    all assets with annual max loading up to the original P90 threshold. At P90
    for voltage, the most critical 10% lowest-voltage assets are excluded and
    the plotted set is all assets with annual min voltage at or above the
    original P10 threshold.

    For every x-axis cutoff, the solid line is the median of the retained asset
    values and the shaded band is the min-max range of those retained values.
    ``asset_cutoff_percentiles`` controls the slider and the maximum visible
    x-axis cutoff; it does not rescale the x-axis.
    """
    required = {"metric", "percentile", "value"}
    missing_required = required.difference(profile.columns)
    if missing_required:
        missing = ", ".join(sorted(missing_required))
        raise ValueError(
            "plot_powerflow_pooled_asset_percentile_curves expects the asset-level "
            f"percentile profile dataframe; missing column(s): {missing}."
        )

    center_stat = str(center_stat).strip().lower()
    if center_stat not in {"median", "mean"}:
        raise ValueError("center_stat must be either 'median' or 'mean'.")
    center_label = center_stat.capitalize()

    df = profile.copy()
    if group_col is None:
        group_col = "comparison_group"
    if group_col not in df.columns:
        df[group_col] = "All retained assets"

    if asset_cutoff_percentiles is None:
        asset_cutoff_percentiles = (1.0, 0.99, 0.95, 0.90, 0.50)
    asset_cutoff_percentiles = tuple(
        float(q) / 100 if float(q) > 1 else float(q) for q in asset_cutoff_percentiles
    )
    if any(q <= 0 or q > 1 for q in asset_cutoff_percentiles):
        raise ValueError("asset_cutoff_percentiles values must satisfy 0 < value <= 1, or 0 < value <= 100.")

    if asset_percentiles is None:
        asset_percentiles = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
    asset_percentiles = tuple(float(q) / 100 if float(q) > 1 else float(q) for q in asset_percentiles)
    if any(q <= 0 or q > 1 for q in asset_percentiles):
        raise ValueError("asset_percentiles values must satisfy 0 < value <= 1, or 0 < value <= 100.")
    asset_percentiles = tuple(sorted(set(asset_percentiles).union(asset_cutoff_percentiles)))
    asset_cutoff_percentiles = tuple(sorted(set(asset_cutoff_percentiles), reverse=True))

    metric_order = ["Transformer", "Cables", "Voltage"]
    metric_lookup = {metric.lower(): metric for metric in metric_order}
    selected_metrics = []
    for metric in metrics:
        metric_key = metric_lookup.get(str(metric).strip().lower())
        if metric_key is None:
            available = ", ".join(metric_order)
            raise ValueError(f"Unsupported metric {metric!r}. Available: {available}.")
        if metric_key not in selected_metrics:
            selected_metrics.append(metric_key)

    y_axis_ranges = _powerflow_y_axis_ranges(y_axis_limits)
    critical_percentile = {
        "Transformer": "max",
        "Cables": "max",
        "Voltage": "min",
    }
    critical_direction = {
        "Transformer": "high",
        "Cables": "high",
        "Voltage": "low",
    }
    y_titles = {
        "Transformer": "Max loading [%]",
        "Cables": "Max loading [%]",
        "Voltage": "Min voltage [p.u.]",
    }
    subplot_titles = {
        "Transformer": "Transformer",
        "Cables": "Cables",
        "Voltage": "Voltage",
    }
    x_titles = {
        "Transformer": "Asset cutoff",
        "Cables": "Asset cutoff",
        "Voltage": "Asset cutoff",
    }
    default_colors = {
        "Synthetic": "#335C81",
        "Real SWF": "#D95D39",
        "synthetic": "#335C81",
        "real_swf": "#D95D39",
    }
    if color_map:
        default_colors.update({str(key): value for key, value in color_map.items()})
    fallback_palette = ["#335C81", "#D95D39", "#2A9D8F", "#6D597A", "#7A8450"]

    def _cutoff_label(cutoff: float) -> str:
        if np.isclose(cutoff, 1.0):
            return "Show cutoffs through P100"
        return f"Show cutoffs through P{int(round(cutoff * 100)):02d}"

    def _asset_percentile_label(q: float) -> str:
        return f"P{int(round(q * 100)):02d}" if q < 1 else "P100"

    def _visible_asset_percentiles(cutoff: float) -> tuple[float, ...]:
        visible = tuple(q for q in asset_percentiles if q <= cutoff or np.isclose(q, cutoff))
        if not visible:
            visible = (cutoff,)
        return visible

    def _retained_values(values: pd.Series, metric_name: str, retained_fraction: float) -> pd.Series:
        values = values.astype(float).dropna()
        if values.empty:
            return values
        if np.isclose(retained_fraction, 1.0):
            return values
        if critical_direction[metric_name] == "high":
            threshold = values.quantile(retained_fraction)
            return values[values <= threshold]
        threshold = values.quantile(1 - retained_fraction)
        return values[values >= threshold]

    def _retained_curve(values: pd.Series, metric_name: str, x_values: tuple[float, ...]) -> pd.DataFrame:
        rows = []
        for retained_fraction in x_values:
            retained = _retained_values(values, metric_name, retained_fraction)
            if retained.empty:
                continue
            rows.append(
                {
                    "retained_asset_cutoff": retained_fraction,
                    "center": float(retained.median() if center_stat == "median" else retained.mean()),
                    "band_lower": float(retained.min()),
                    "band_upper": float(retained.max()),
                    "retained_assets": int(retained.size),
                    "total_assets": int(values.dropna().size),
                }
            )
        return pd.DataFrame(rows)

    def _x_range(cutoff: float) -> list[float]:
        x_values = _visible_asset_percentiles(cutoff)
        lower = float(min(x_values))
        upper = float(max(x_values))
        if np.isclose(lower, upper):
            pad = 0.01 if upper >= 0.99 else min(0.01, upper / 2)
            return [max(0.0, lower - pad), min(1.0, upper + pad)]
        return [lower, upper]

    df["percentile_norm"] = df["percentile"].map(_normalize_percentile_label)

    fig = make_subplots(
        rows=1,
        cols=len(selected_metrics),
        subplot_titles=[subplot_titles[metric] for metric in selected_metrics],
    )

    def _slider_axis_layout(cutoff: float) -> dict[str, object]:
        layout: dict[str, object] = {
            "autosize": False,
            "width": 1500,
            "height": 520,
            "margin": {"l": 60, "r": 25, "t": 95, "b": 165},
            "legend": {"title": {"text": "Median and range"}},
        }
        tickvals = _visible_asset_percentiles(cutoff)
        for col_idx, metric in enumerate(selected_metrics, start=1):
            xaxis_name = "xaxis" if col_idx == 1 else f"xaxis{col_idx}"
            yaxis_name = "yaxis" if col_idx == 1 else f"yaxis{col_idx}"
            layout[f"{xaxis_name}.range"] = _x_range(cutoff)
            layout[f"{xaxis_name}.tickvals"] = [float(value) for value in tickvals]
            layout[f"{xaxis_name}.ticktext"] = [_asset_percentile_label(float(value)) for value in tickvals]
            if metric in y_axis_ranges:
                layout[f"{yaxis_name}.range"] = y_axis_ranges[metric]
                layout[f"{yaxis_name}.autorange"] = False
            else:
                layout[f"{yaxis_name}.autorange"] = True
        return layout

    traces_by_cutoff: list[list[int]] = []

    for cutoff_index, cutoff in enumerate(asset_cutoff_percentiles):
        is_visible = cutoff_index == 0
        cutoff_trace_indices: list[int] = []
        cutoff_label = _cutoff_label(cutoff)
        x_values = _visible_asset_percentiles(cutoff)

        for col_idx, metric in enumerate(selected_metrics, start=1):
            metric_df = df[
                (df["metric"] == metric)
                & (df["percentile_norm"] == critical_percentile[metric])
            ].dropna(subset=["value"]).copy()
            if metric_df.empty:
                continue

            for color_idx, (group, group_df) in enumerate(metric_df.groupby(group_col, sort=False)):
                values = group_df["value"].astype(float).dropna()
                if values.empty:
                    continue
                curve = _retained_curve(values, metric, x_values)
                if curve.empty:
                    continue

                group_label = str(group)
                color = default_colors.get(group_label, fallback_palette[color_idx % len(fallback_palette)])
                customdata = np.column_stack(
                    [
                        curve["band_lower"].to_numpy(dtype=float),
                        curve["band_upper"].to_numpy(dtype=float),
                        curve["retained_assets"].to_numpy(dtype=int),
                        curve["total_assets"].to_numpy(dtype=int),
                    ]
                )

                if show_band:
                    for trace in (
                        go.Scatter(
                            x=curve["retained_asset_cutoff"],
                            y=curve["band_upper"],
                            mode="lines",
                            line={"width": 0},
                            showlegend=False,
                            hoverinfo="skip",
                            visible=is_visible,
                        ),
                        go.Scatter(
                            x=curve["retained_asset_cutoff"],
                            y=curve["band_lower"],
                            mode="lines",
                            line={"width": 0},
                            fill="tonexty",
                            fillcolor=_hex_to_rgba(color, 0.16),
                            name=f"{group_label}: min-max of retained assets",
                            legendgroup=f"{group_label} retained asset range",
                            showlegend=col_idx == 1,
                            customdata=customdata,
                            hovertemplate=(
                                "retained asset cutoff %{x:.0%}<br>"
                                "min-max of retained assets: %{customdata[0]:.4g} - %{customdata[1]:.4g}<br>"
                                "retained assets: %{customdata[2]} / %{customdata[3]}<br>"
                                f"{cutoff_label}<extra></extra>"
                            ),
                            visible=is_visible,
                        ),
                    ):
                        fig.add_trace(trace, row=1, col=col_idx)
                        cutoff_trace_indices.append(len(fig.data) - 1)

                fig.add_trace(
                    go.Scatter(
                        x=curve["retained_asset_cutoff"],
                        y=curve["center"],
                        mode="lines+markers",
                        line={"color": color, "width": 2.7},
                        marker={"size": 6, "color": color},
                        name=f"{group_label}: median of retained assets",
                        legendgroup=f"{group_label} retained asset median",
                        showlegend=col_idx == 1,
                        customdata=customdata,
                        hovertemplate=(
                            "retained asset cutoff %{x:.0%}<br>"
                            "median of retained assets: %{y:.4g}<br>"
                            "min-max of retained assets: %{customdata[0]:.4g} - %{customdata[1]:.4g}<br>"
                            "retained assets: %{customdata[2]} / %{customdata[3]}<br>"
                            f"{cutoff_label}<extra></extra>"
                        ),
                        visible=is_visible,
                    ),
                    row=1,
                    col=col_idx,
                )
                cutoff_trace_indices.append(len(fig.data) - 1)

            fig.update_yaxes(
                title_text=y_titles[metric],
                tickformat=".2f" if metric == "Voltage" else None,
                row=1,
                col=col_idx,
            )
            if metric in y_axis_ranges:
                fig.update_yaxes(range=y_axis_ranges[metric], row=1, col=col_idx)
            fig.update_xaxes(
                title_text=x_titles[metric],
                tickangle=-45,
                row=1,
                col=col_idx,
            )
        traces_by_cutoff.append(cutoff_trace_indices)

    slider_steps = []
    n_traces = len(fig.data)
    for cutoff, cutoff_trace_indices in zip(asset_cutoff_percentiles, traces_by_cutoff):
        visible = [False] * n_traces
        for trace_index in cutoff_trace_indices:
            visible[trace_index] = True
        slider_steps.append(
            {
                "label": _cutoff_label(cutoff),
                "method": "update",
                "args": [
                    {"visible": visible},
                    _slider_axis_layout(cutoff),
                ],
            }
        )

    if asset_cutoff_percentiles:
        fig.update_layout(_slider_axis_layout(asset_cutoff_percentiles[0]))

    band_note = (
        " Shaded bands show the min-max range of the assets retained at each x-axis cutoff."
        if show_band
        else " Shaded min-max bands are hidden."
    )
    fig.update_layout(
        title={
            "text": (
                f"{title}<br>"
                "<sup>Solid lines show the median of all retained assets at each cutoff."
                f"{band_note}</sup>"
            )
        },
        autosize=False,
        legend={"title": {"text": "Median and range"}},
        height=520,
        width=1500,
        margin={"l": 60, "r": 25, "t": 95, "b": 165},
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Visible cutoff range: "},
                "x": 0.08,
                "len": 0.84,
                "y": -0.24,
                "pad": {"t": 65},
                "steps": slider_steps,
            }
        ] if len(asset_cutoff_percentiles) > 1 else None,
    )
    if show:
        fig.show()
    return fig


def plot_powerflow_asset_cutoff_overview(
    profile: pd.DataFrame,
    group_col: str | None = None,
    show: bool = True,
    color_map: dict[str, str] | None = None,
    asset_percentiles: tuple[float, ...] | None = None,
    asset_cutoff_percentiles: tuple[float, ...] | None = None,
    metrics: tuple[str, ...] = ("Transformer", "Cables", "Voltage"),
    title: str = "Power-Flow Stress by Retained-Asset Cutoff",
    y_axis_limits: tuple[float | None, float | None, float | None] | None = None,
    center_stat: str = "median",
    show_band: bool = True,
    worst_asset_per_grid: bool = False,
):
    """Plot retained-asset cutoff curves and matching asset distributions.

    Row 1 shows, for every retained-asset cutoff, the selected center statistic
    and min-max range of the retained assets. Row 2 shows the distribution of exactly the retained
    assets at the selected cutoff. ``center_stat`` selects whether the top-row
    center line uses the retained-asset median or mean. Set
    ``worst_asset_per_grid=True`` to draw only each grid's most critical retained
    transformer/cable/bus value in the violin row.
    """
    required = {"metric", "percentile", "value"}
    missing_required = required.difference(profile.columns)
    if missing_required:
        missing = ", ".join(sorted(missing_required))
        raise ValueError(
            "plot_powerflow_asset_cutoff_overview expects the asset-level "
            f"percentile profile dataframe; missing column(s): {missing}."
        )

    center_stat = str(center_stat).strip().lower()
    if center_stat not in {"median", "mean"}:
        raise ValueError("center_stat must be either 'median' or 'mean'.")
    center_label = center_stat.capitalize()

    df = profile.copy()
    if group_col is None:
        group_col = "comparison_group"
    if group_col not in df.columns:
        df[group_col] = "All retained assets"

    if asset_cutoff_percentiles is None:
        asset_cutoff_percentiles = (1.0, 0.99, 0.95, 0.90, 0.50)
    asset_cutoff_percentiles = tuple(float(q) / 100 if float(q) > 1 else float(q) for q in asset_cutoff_percentiles)
    if any(q <= 0 or q > 1 for q in asset_cutoff_percentiles):
        raise ValueError("asset_cutoff_percentiles values must satisfy 0 < value <= 1, or 0 < value <= 100.")

    if asset_percentiles is None:
        asset_percentiles = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
    asset_percentiles = tuple(float(q) / 100 if float(q) > 1 else float(q) for q in asset_percentiles)
    if any(q <= 0 or q > 1 for q in asset_percentiles):
        raise ValueError("asset_percentiles values must satisfy 0 < value <= 1, or 0 < value <= 100.")
    asset_percentiles = tuple(sorted(set(asset_percentiles).union(asset_cutoff_percentiles)))
    asset_cutoff_percentiles = tuple(sorted(set(asset_cutoff_percentiles), reverse=True))

    metric_order = ["Transformer", "Cables", "Voltage"]
    metric_lookup = {metric.lower(): metric for metric in metric_order}
    selected_metrics = []
    for metric in metrics:
        metric_key = metric_lookup.get(str(metric).strip().lower())
        if metric_key is None:
            available = ", ".join(metric_order)
            raise ValueError(f"Unsupported metric {metric!r}. Available: {available}.")
        if metric_key not in selected_metrics:
            selected_metrics.append(metric_key)

    y_axis_ranges = _powerflow_y_axis_ranges(y_axis_limits)
    critical_percentile = {"Transformer": "max", "Cables": "max", "Voltage": "min"}
    critical_direction = {"Transformer": "high", "Cables": "high", "Voltage": "low"}
    y_titles = {
        "Transformer": "Max loading [%]",
        "Cables": "Max loading [%]",
        "Voltage": "Min voltage [p.u.]",
    }
    default_colors = {
        "Synthetic": "#335C81",
        "Real SWF": "#D95D39",
        "synthetic": "#335C81",
        "real_swf": "#D95D39",
    }
    if color_map:
        default_colors.update({str(key): value for key, value in color_map.items()})
    fallback_palette = ["#335C81", "#D95D39", "#2A9D8F", "#6D597A", "#7A8450"]

    def _cutoff_label(cutoff: float) -> str:
        if np.isclose(cutoff, 1.0):
            return "Show cutoffs through P100"
        return f"Show cutoffs through P{int(round(cutoff * 100)):02d}"

    def _asset_percentile_label(q: float) -> str:
        return f"P{int(round(q * 100)):02d}" if q < 1 else "P100"

    def _visible_asset_percentiles(cutoff: float) -> tuple[float, ...]:
        visible = tuple(q for q in asset_percentiles if q <= cutoff or np.isclose(q, cutoff))
        return visible or (cutoff,)

    def _retained_mask(values: pd.Series, metric_name: str, retained_fraction: float) -> pd.Series:
        if np.isclose(retained_fraction, 1.0):
            return pd.Series(True, index=values.index)
        if critical_direction[metric_name] == "high":
            threshold = values.quantile(retained_fraction)
            return values <= threshold
        threshold = values.quantile(1 - retained_fraction)
        return values >= threshold

    def _retained_curve(values: pd.Series, metric_name: str, x_values: tuple[float, ...]) -> pd.DataFrame:
        rows = []
        values = values.astype(float).dropna()
        for retained_fraction in x_values:
            retained = values[_retained_mask(values, metric_name, retained_fraction)]
            if retained.empty:
                continue
            rows.append(
                {
                    "retained_asset_cutoff": retained_fraction,
                    "center": float(retained.median() if center_stat == "median" else retained.mean()),
                    "band_lower": float(retained.min()),
                    "band_upper": float(retained.max()),
                    "retained_assets": int(retained.size),
                    "total_assets": int(values.size),
                }
            )
        return pd.DataFrame(rows)

    def _retained_frame(group_df: pd.DataFrame, metric_name: str, cutoff: float) -> pd.DataFrame:
        values = group_df["value"].astype(float)
        return group_df.loc[_retained_mask(values, metric_name, cutoff)].copy()

    def _select_worst_asset_per_grid(plot_df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
        if not worst_asset_per_grid or plot_df.empty:
            return plot_df
        if "grid" not in plot_df.columns:
            raise ValueError("worst_asset_per_grid=True requires a 'grid' column in the profile dataframe.")
        group_keys = [group_col, "grid"] if group_col in plot_df.columns else ["grid"]
        if critical_direction[metric_name] == "high":
            value_index = plot_df.groupby(group_keys, sort=False)["value"].idxmax()
        else:
            value_index = plot_df.groupby(group_keys, sort=False)["value"].idxmin()
        return plot_df.loc[value_index].reset_index(drop=True)

    def _x_range(cutoff: float) -> list[float]:
        x_values = _visible_asset_percentiles(cutoff)
        lower = float(min(x_values))
        upper = float(max(x_values))
        if np.isclose(lower, upper):
            pad = 0.01 if upper >= 0.99 else min(0.01, upper / 2)
            return [max(0.0, lower - pad), min(1.0, upper + pad)]
        return [lower, upper]

    df["percentile_norm"] = df["percentile"].map(_normalize_percentile_label)
    subplot_titles = selected_metrics + ["" for _ in selected_metrics]
    fig = make_subplots(
        rows=2,
        cols=len(selected_metrics),
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        row_heights=[0.56, 0.44],
    )

    def _axis_layout(cutoff: float) -> dict[str, object]:
        layout: dict[str, object] = {
            "autosize": False,
            "width": 1500,
            "height": 860,
            "margin": {"l": 60, "r": 25, "t": 100, "b": 110},
            "legend": {"title": {"text": f"{center_label}, range, distribution"}},
        }
        tickvals = _visible_asset_percentiles(cutoff)
        n_cols = len(selected_metrics)
        for col_idx, metric in enumerate(selected_metrics, start=1):
            top_xaxis = "xaxis" if col_idx == 1 else f"xaxis{col_idx}"
            top_yaxis = "yaxis" if col_idx == 1 else f"yaxis{col_idx}"
            bottom_yaxis_index = n_cols + col_idx
            bottom_yaxis = "yaxis" if bottom_yaxis_index == 1 else f"yaxis{bottom_yaxis_index}"
            layout[f"{top_xaxis}.range"] = _x_range(cutoff)
            layout[f"{top_xaxis}.tickvals"] = [float(value) for value in tickvals]
            layout[f"{top_xaxis}.ticktext"] = [_asset_percentile_label(float(value)) for value in tickvals]
            for yaxis in (top_yaxis, bottom_yaxis):
                if metric in y_axis_ranges:
                    layout[f"{yaxis}.range"] = y_axis_ranges[metric]
                    layout[f"{yaxis}.autorange"] = False
                else:
                    layout[f"{yaxis}.autorange"] = True
        return layout

    traces_by_cutoff: list[list[int]] = []
    for cutoff_index, cutoff in enumerate(asset_cutoff_percentiles):
        is_visible = cutoff_index == 0
        cutoff_trace_indices: list[int] = []
        cutoff_label = _cutoff_label(cutoff)
        x_values = _visible_asset_percentiles(cutoff)

        for col_idx, metric in enumerate(selected_metrics, start=1):
            metric_df = df[
                (df["metric"] == metric) & (df["percentile_norm"] == critical_percentile[metric])
            ].dropna(subset=["value"]).copy()
            if metric_df.empty:
                continue

            for color_idx, (group, group_df) in enumerate(metric_df.groupby(group_col, sort=False)):
                values = group_df["value"].astype(float).dropna()
                if values.empty:
                    continue
                curve = _retained_curve(values, metric, x_values)
                if curve.empty:
                    continue
                group_label = str(group)
                color = default_colors.get(group_label, fallback_palette[color_idx % len(fallback_palette)])
                customdata = np.column_stack(
                    [
                        curve["band_lower"].to_numpy(dtype=float),
                        curve["band_upper"].to_numpy(dtype=float),
                        curve["retained_assets"].to_numpy(dtype=int),
                        curve["total_assets"].to_numpy(dtype=int),
                    ]
                )

                if show_band:
                    for trace in (
                        go.Scatter(
                            x=curve["retained_asset_cutoff"],
                            y=curve["band_upper"],
                            mode="lines",
                            line={"width": 0},
                            showlegend=False,
                            hoverinfo="skip",
                            visible=is_visible,
                        ),
                        go.Scatter(
                            x=curve["retained_asset_cutoff"],
                            y=curve["band_lower"],
                            mode="lines",
                            line={"width": 0},
                            fill="tonexty",
                            fillcolor=_hex_to_rgba(color, 0.16),
                            name=f"{group_label}: range",
                            legendgroup=f"{group_label} retained asset range",
                            showlegend=col_idx == 1,
                            customdata=customdata,
                            hovertemplate=(
                                "asset cutoff %{x:.0%}<br>"
                                "range: %{customdata[0]:.4g} - %{customdata[1]:.4g}<br>"
                                "retained: %{customdata[2]} / %{customdata[3]}<br>"
                                f"{cutoff_label}<extra></extra>"
                            ),
                            visible=is_visible,
                        ),
                    ):
                        fig.add_trace(trace, row=1, col=col_idx)
                        cutoff_trace_indices.append(len(fig.data) - 1)

                fig.add_trace(
                    go.Scatter(
                        x=curve["retained_asset_cutoff"],
                        y=curve["center"],
                        mode="lines+markers",
                        line={"color": color, "width": 2.7},
                        marker={"size": 6, "color": color},
                        name=f"{group_label}: {center_stat}",
                        legendgroup=f"{group_label} retained asset {center_stat}",
                        showlegend=col_idx == 1,
                        customdata=customdata,
                        hovertemplate=(
                            "asset cutoff %{x:.0%}<br>"
                            f"{center_stat}: %{{y:.4g}}<br>"
                            "range: %{customdata[0]:.4g} - %{customdata[1]:.4g}<br>"
                            "retained: %{customdata[2]} / %{customdata[3]}<br>"
                            f"{cutoff_label}<extra></extra>"
                        ),
                        visible=is_visible,
                    ),
                    row=1,
                    col=col_idx,
                )
                cutoff_trace_indices.append(len(fig.data) - 1)

                violin_df = _retained_frame(group_df, metric, cutoff)
                violin_df = _select_worst_asset_per_grid(violin_df, metric)
                if violin_df.empty:
                    continue
                hover_parts = []
                for col, label in {
                    "grid": "grid",
                    "asset_label": "asset",
                    "asset_id": "asset_id",
                    "n_failed_timesteps": "failed_hours",
                }.items():
                    if col in violin_df.columns:
                        hover_parts.append(label + ": " + violin_df[col].astype(str))
                if hover_parts:
                    violin_df["hover_text"] = hover_parts[0]
                    for part in hover_parts[1:]:
                        violin_df["hover_text"] = violin_df["hover_text"] + "<br>" + part
                    violin_df["hover_text"] = violin_df["hover_text"] + "<br>" + cutoff_label
                else:
                    violin_df["hover_text"] = f"{group_label}<br>{cutoff_label}"
                fig.add_trace(
                    go.Violin(
                        x=violin_df[group_col].astype(str),
                        y=violin_df["value"].astype(float),
                        text=violin_df["hover_text"],
                        hovertemplate="%{text}<br>%{y:.4g}<extra></extra>",
                        box_visible=False,
                        meanline_visible=True,
                        points="all",
                        jitter=0.12,
                        width=0.5,
                        scalemode="width",
                        marker={"color": color, "opacity": 0.45, "size": 3.5},
                        line={"color": color, "width": 1.8},
                        fillcolor=_hex_to_rgba(color, 0.36),
                        opacity=0.82,
                        spanmode="hard",
                        name=f"{group_label}: distribution",
                        legendgroup=f"{group_label} retained asset distribution",
                        showlegend=col_idx == 1,
                        visible=is_visible,
                    ),
                    row=2,
                    col=col_idx,
                )
                cutoff_trace_indices.append(len(fig.data) - 1)

            fig.update_yaxes(
                title_text=y_titles[metric],
                tickformat=".2f" if metric == "Voltage" else None,
                row=1,
                col=col_idx,
            )
            fig.update_yaxes(
                title_text=y_titles[metric],
                tickformat=".2f" if metric == "Voltage" else None,
                row=2,
                col=col_idx,
            )
            if metric in y_axis_ranges:
                fig.update_yaxes(range=y_axis_ranges[metric], row=1, col=col_idx)
                fig.update_yaxes(range=y_axis_ranges[metric], row=2, col=col_idx)
            fig.update_xaxes(title_text="Asset cutoff", tickangle=-45, row=1, col=col_idx)
            fig.update_xaxes(title_text="", row=2, col=col_idx)
        traces_by_cutoff.append(cutoff_trace_indices)

    slider_steps = []
    n_traces = len(fig.data)
    for cutoff, cutoff_trace_indices in zip(asset_cutoff_percentiles, traces_by_cutoff):
        visible = [False] * n_traces
        for trace_index in cutoff_trace_indices:
            visible[trace_index] = True
        slider_steps.append(
            {
                "label": _cutoff_label(cutoff),
                "method": "update",
                "args": [{"visible": visible}, _axis_layout(cutoff)],
            }
        )

    if asset_cutoff_percentiles:
        fig.update_layout(_axis_layout(asset_cutoff_percentiles[0]))

    fig.update_layout(
        title={
            "text": (
                f"{title}<br>"
                f"<sup>Top: retained-asset {center_stat} and min-max range. Bottom: distribution of retained assets at selected cutoff.</sup>"
            )
        },
        autosize=False,
        legend={"title": {"text": f"{center_label}, range, distribution"}},
        height=860,
        width=1500,
        margin={"l": 60, "r": 25, "t": 100, "b": 110},
        violingap=0.12,
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Visible cutoff range: "},
                "x": 0.08,
                "len": 0.84,
                "y": -0.08,
                "pad": {"t": 35},
                "steps": slider_steps,
            }
        ] if len(asset_cutoff_percentiles) > 1 else None,
    )
    if show:
        fig.show()
    return fig


def plot_powerflow_asset_cutoff_overview_static(
    profile: pd.DataFrame,
    group_col: str | None = None,
    color_map: dict[str, str] | None = None,
    asset_cutoff_percentile: float = 1.0,
    asset_percentiles: tuple[float, ...] | None = None,
    metrics: tuple[str, ...] = ("Transformer", "Cables", "Voltage"),
    title: str = "Power-Flow Stress by Retained-Asset Cutoff",
    y_axis_limits: tuple[float | None, float | None, float | None] | None = None,
    center_stat: str = "mean",
    show_band: bool = False,
    worst_asset_per_grid: bool = True,
    save_path: str | Path | None = None,
    save_formats: tuple[str, ...] = ("svg", "pdf"),
):
    """Draw a publication-oriented static retained-asset cutoff overview.

    The figure mirrors :func:`plot_powerflow_asset_cutoff_overview` without a
    slider. ``asset_cutoff_percentile`` selects the retained-asset filter shown
    in the bottom row and the maximum cutoff shown in the top-row curves.
    Static Matplotlib output can be saved as SVG/PDF through ``save_path``.
    """
    required = {"metric", "percentile", "value"}
    missing_required = required.difference(profile.columns)
    if missing_required:
        missing = ", ".join(sorted(missing_required))
        raise ValueError(
            "plot_powerflow_asset_cutoff_overview_static expects the asset-level "
            f"percentile profile dataframe; missing column(s): {missing}."
        )

    center_stat = str(center_stat).strip().lower()
    if center_stat not in {"median", "mean"}:
        raise ValueError("center_stat must be either 'median' or 'mean'.")

    cutoff = float(asset_cutoff_percentile)
    if cutoff > 1:
        cutoff = cutoff / 100
    if cutoff <= 0 or cutoff > 1:
        raise ValueError("asset_cutoff_percentile must satisfy 0 < value <= 1, or 0 < value <= 100.")

    if asset_percentiles is None:
        asset_percentiles = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
    asset_percentiles = tuple(float(q) / 100 if float(q) > 1 else float(q) for q in asset_percentiles)
    if any(q <= 0 or q > 1 for q in asset_percentiles):
        raise ValueError("asset_percentiles values must satisfy 0 < value <= 1, or 0 < value <= 100.")
    x_values = tuple(q for q in sorted(set(asset_percentiles).union({cutoff})) if q <= cutoff or np.isclose(q, cutoff))
    if not x_values:
        x_values = (cutoff,)

    df = profile.copy()
    if group_col is None:
        group_col = "comparison_group"
    if group_col not in df.columns:
        df[group_col] = "All retained assets"
    df["percentile_norm"] = df["percentile"].map(_normalize_percentile_label)

    metric_order = ["Transformer", "Cables", "Voltage"]
    metric_lookup = {metric.lower(): metric for metric in metric_order}
    selected_metrics = []
    for metric in metrics:
        metric_key = metric_lookup.get(str(metric).strip().lower())
        if metric_key is None:
            available = ", ".join(metric_order)
            raise ValueError(f"Unsupported metric {metric!r}. Available: {available}.")
        if metric_key not in selected_metrics:
            selected_metrics.append(metric_key)

    critical_percentile = {"Transformer": "max", "Cables": "max", "Voltage": "min"}
    critical_direction = {"Transformer": "high", "Cables": "high", "Voltage": "low"}
    y_titles = {
        "Transformer": "Max loading [%]",
        "Cables": "Max loading [%]",
        "Voltage": "Min voltage [p.u.]",
    }
    default_colors = {
        "Synthetic": "#335C81",
        "Real SWF": "#D95D39",
        "synthetic": "#335C81",
        "real_swf": "#D95D39",
    }
    if color_map:
        default_colors.update({str(key): value for key, value in color_map.items()})
    fallback_palette = ["#335C81", "#D95D39", "#2A9D8F", "#6D597A", "#7A8450"]
    y_axis_ranges = _powerflow_y_axis_ranges(y_axis_limits)

    def _retained_mask(values: pd.Series, metric_name: str, retained_fraction: float) -> pd.Series:
        values = values.astype(float)
        if np.isclose(retained_fraction, 1.0):
            return pd.Series(True, index=values.index)
        if critical_direction[metric_name] == "high":
            threshold = values.quantile(retained_fraction)
            return values <= threshold
        threshold = values.quantile(1 - retained_fraction)
        return values >= threshold

    def _retained_curve(values: pd.Series, metric_name: str) -> pd.DataFrame:
        values = values.astype(float).dropna()
        rows = []
        for retained_fraction in x_values:
            retained = values[_retained_mask(values, metric_name, retained_fraction)]
            if retained.empty:
                continue
            rows.append(
                {
                    "retained_asset_cutoff": retained_fraction,
                    "center": float(retained.median() if center_stat == "median" else retained.mean()),
                    "band_lower": float(retained.min()),
                    "band_upper": float(retained.max()),
                }
            )
        return pd.DataFrame(rows)

    def _retained_frame(group_df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
        values = group_df["value"].astype(float)
        return group_df.loc[_retained_mask(values, metric_name, cutoff)].copy()

    def _select_worst_asset_per_grid(plot_df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
        if not worst_asset_per_grid or plot_df.empty:
            return plot_df
        if "grid" not in plot_df.columns:
            raise ValueError("worst_asset_per_grid=True requires a 'grid' column in the profile dataframe.")
        group_keys = [group_col, "grid"] if group_col in plot_df.columns else ["grid"]
        if critical_direction[metric_name] == "high":
            value_index = plot_df.groupby(group_keys, sort=False)["value"].idxmax()
        else:
            value_index = plot_df.groupby(group_keys, sort=False)["value"].idxmin()
        return plot_df.loc[value_index].reset_index(drop=True)

    def _cutoff_label(value: float) -> str:
        return f"P{int(round(value * 100)):02d}" if value < 1 else "P100"

    title_fontsize = 20
    panel_title_fontsize = 17
    label_fontsize = 16
    tick_fontsize = 14
    legend_fontsize = 15

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        2,
        len(selected_metrics),
        figsize=(4.6 * len(selected_metrics), 7.2),
        gridspec_kw={"height_ratios": [1.0, 1.15], "hspace": 0.34, "wspace": 0.28},
        squeeze=False,
    )
    groups = list(df[group_col].astype(str).dropna().drop_duplicates())
    group_colors = {
        group: default_colors.get(group, fallback_palette[index % len(fallback_palette)])
        for index, group in enumerate(groups)
    }

    for col_idx, metric in enumerate(selected_metrics):
        metric_df = df[
            (df["metric"] == metric)
            & (df["percentile_norm"] == critical_percentile[metric])
        ].dropna(subset=["value"]).copy()
        if metric_df.empty:
            continue

        ax_curve = axes[0, col_idx]
        ax_dist = axes[1, col_idx]
        for group in groups:
            group_df = metric_df[metric_df[group_col].astype(str) == group]
            values = group_df["value"].astype(float).dropna()
            if values.empty:
                continue
            curve = _retained_curve(values, metric)
            color = group_colors[group]
            ax_curve.plot(
                curve["retained_asset_cutoff"],
                curve["center"],
                marker="o",
                linewidth=2.8,
                markersize=6.5,
                color=color,
                label=group,
            )
            if show_band:
                ax_curve.fill_between(
                    curve["retained_asset_cutoff"].to_numpy(dtype=float),
                    curve["band_lower"].to_numpy(dtype=float),
                    curve["band_upper"].to_numpy(dtype=float),
                    color=color,
                    alpha=0.13,
                    linewidth=0,
                )

        violin_values = []
        violin_labels = []
        violin_colors = []
        for group in groups:
            group_df = metric_df[metric_df[group_col].astype(str) == group]
            retained = _select_worst_asset_per_grid(_retained_frame(group_df, metric), metric)
            values = retained["value"].astype(float).dropna().to_numpy()
            if values.size == 0:
                continue
            violin_values.append(values)
            violin_labels.append(group)
            violin_colors.append(group_colors[group])

        if violin_values:
            positions = np.arange(1, len(violin_values) + 1)
            violins = ax_dist.violinplot(
                violin_values,
                positions=positions,
                widths=0.72,
                showmeans=False,
                showmedians=True,
                showextrema=False,
            )
            for body, color in zip(violins["bodies"], violin_colors):
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.28)
                body.set_linewidth(1.2)
            if "cmedians" in violins:
                violins["cmedians"].set_color("#222222")
                violins["cmedians"].set_linewidth(2.0)
            rng = np.random.default_rng(7)
            for position, values, color in zip(positions, violin_values, violin_colors):
                jitter = rng.normal(0, 0.035, size=values.size)
                ax_dist.scatter(
                    np.full(values.size, position) + jitter,
                    values,
                    s=18,
                    color=color,
                    alpha=0.42,
                    linewidths=0,
                )
            ax_dist.set_xticks(positions)
            ax_dist.set_xticklabels(violin_labels, rotation=0, fontsize=tick_fontsize)

        ax_curve.set_title(metric, fontsize=panel_title_fontsize, fontweight="bold")
        ax_curve.set_xlabel("")
        ax_curve.set_ylabel(f"{center_stat.capitalize()} {y_titles[metric].lower()}", fontsize=label_fontsize)
        ax_curve.set_xticks(list(x_values))
        ax_curve.set_xticklabels([_cutoff_label(q) for q in x_values], rotation=55, ha="right", rotation_mode="anchor", fontsize=tick_fontsize)
        ax_dist.set_xlabel("")
        ax_dist.set_ylabel(y_titles[metric], fontsize=label_fontsize)
        if metric in y_axis_ranges:
            ax_curve.set_ylim(y_axis_ranges[metric])
            ax_dist.set_ylim(y_axis_ranges[metric])
        for ax in (ax_curve, ax_dist):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="both", labelsize=tick_fontsize)
            ax.grid(True, axis="y", color="#d8d8d8", linewidth=0.8)
            ax.grid(False, axis="x")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles and len(labels) > 1:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(labels),
            frameon=False,
            bbox_to_anchor=(0.5, 0.91),
            fontsize=legend_fontsize,
        )
    fig.suptitle(
        f"{title} ({_cutoff_label(cutoff)} retained-asset cutoff)",
        y=0.99,
        fontsize=title_fontsize,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.78, bottom=0.08, left=0.08, right=0.985, hspace=0.38, wspace=0.35)

    if save_path is not None:
        save_path = Path(save_path)
        base_path = save_path.with_suffix("") if save_path.suffix else save_path
        base_path.parent.mkdir(parents=True, exist_ok=True)
        for image_format in save_formats:
            fig.savefig(base_path.with_suffix(f".{image_format.lstrip('.')}"), bbox_inches="tight")
    return fig

def plot_powerflow_percentile_profiles(
    profile: pd.DataFrame,
    group_col: str | None = None,
    show: bool = True,
    color_map: dict[str, str] | None = None,
    center_stat: str = "median",
    band_quantiles: tuple[float, float] | None = None,
    metric_config: dict[str, dict[str, object]] | None = None,
    asset_quantile_lines: dict[str, tuple[float, ...]] | tuple[float, ...] | None = None,
    title: str = "Annual Percentile Profiles by Asset",
    points: str | bool | None = None,
):
    """Plot percentile profiles with a center line and selected asset range band.

    For each metric and time-percentile, values are computed per asset. The line
    is the median or mean across the selected assets. If ``metric_config`` is
    provided, the same ``time_quantile`` and ``asset_quantile`` settings used by
    :func:`plot_powerflow_headline_violins` select the asset population before
    plotting: loading metrics keep assets up to the configured upper asset
    quantile, while voltage keeps assets above the configured lower asset
    quantile. The shaded band spans the selected asset range.

    Default dashed guide lines show P95/P99 for loading metrics and P05/P01 for
    voltage, but only when those guide quantiles are inside the selected asset
    range. ``asset_quantile_lines`` remains available for explicit overrides,
    but should not be combined with ``metric_config``.
    """
    df = profile.copy()
    if group_col is None or group_col not in df.columns:
        group_col = "comparison_group"
        df[group_col] = "All assets"

    center_stat = center_stat.lower().strip()
    if center_stat not in {"mean", "median"}:
        raise ValueError("center_stat must be either 'mean' or 'median'.")
    if metric_config is not None and band_quantiles is not None:
        raise ValueError("band_quantiles is derived from metric_config; pass only metric_config for this plot.")
    if band_quantiles is not None:
        lower_q, upper_q = band_quantiles
        if not 0 <= lower_q < upper_q <= 1:
            raise ValueError("band_quantiles must satisfy 0 <= lower < upper <= 1.")
    if metric_config is not None and asset_quantile_lines is not None:
        raise ValueError("Pass either metric_config or asset_quantile_lines, not both.")

    metric_names = {"transformer": "Transformer", "cables": "Cables", "voltage": "Voltage"}
    critical_direction = {
        "Transformer": "high",
        "Cables": "high",
        "Voltage": "low",
    }
    default_time_quantiles = {
        "Transformer": 1.0,
        "Cables": 1.0,
        "Voltage": 0.0,
    }
    default_asset_quantiles = {
        "Transformer": 1.0,
        "Cables": 1.0,
        "Voltage": 0.0,
    }
    default_guide_quantiles = {
        "Transformer": (0.95, 0.99),
        "Cables": (0.95, 0.99),
        "Voltage": (0.05, 0.01),
    }

    def _normalize_asset_quantile(value) -> float:
        quantile = float(value)
        if quantile > 1:
            quantile = quantile / 100
        if not 0 <= quantile <= 1:
            raise ValueError("asset_quantile values must be between 0 and 1, or 0 and 100.")
        return quantile

    def _metric_config_for(metric_name: str) -> dict[str, object]:
        config = {
            "time_quantile": default_time_quantiles[metric_name],
            "asset_quantile": default_asset_quantiles[metric_name],
        }
        if metric_config:
            raw_config = metric_config.get(metric_name) or metric_config.get(metric_name.lower())
            if raw_config:
                unknown = set(raw_config).difference({"time_quantile", "asset_quantile"})
                if unknown:
                    raise ValueError(f"Unsupported metric_config key(s) for {metric_name}: {', '.join(sorted(unknown))}.")
                config.update(raw_config)
        return config

    def _asset_quantile_range(metric_name: str) -> tuple[float, float]:
        config = _metric_config_for(metric_name)
        q = _normalize_asset_quantile(config["asset_quantile"])
        return (0.0, q) if critical_direction[metric_name] == "high" else (q, 1.0)

    def _selected_assets(group_df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
        if metric_config is None:
            return group_df
        config = _metric_config_for(metric_name)
        time_quantile = _normalize_time_quantile(config["time_quantile"])
        asset_quantile = _normalize_asset_quantile(config["asset_quantile"])
        critical_rows = group_df[group_df["percentile"].map(_normalize_percentile_label) == time_quantile].copy()
        if critical_rows.empty:
            available = sorted(group_df["percentile"].map(_normalize_percentile_label).dropna().unique())
            raise ValueError(f"No {metric_name} rows found for time_quantile {time_quantile!r}. Available: {available}.")
        if critical_direction[metric_name] == "high":
            if asset_quantile == 1.0:
                return group_df
            threshold = critical_rows["value"].quantile(asset_quantile)
            selected = critical_rows[critical_rows["value"] <= threshold]
        else:
            if asset_quantile == 0.0:
                return group_df
            threshold = critical_rows["value"].quantile(asset_quantile)
            selected = critical_rows[critical_rows["value"] >= threshold]
        asset_cols = [col for col in (group_col, "powerflow_run_id", "grid", "asset_type", "asset_id") if col in group_df.columns]
        selected_keys = selected[asset_cols].drop_duplicates()
        return group_df.merge(selected_keys, on=asset_cols, how="inner")

    def _asset_quantiles_for(metric_name: str) -> tuple[float, ...]:
        if metric_config is not None:
            lower_q, upper_q = _asset_quantile_range(metric_name)
            return tuple(q for q in default_guide_quantiles[metric_name] if lower_q <= q <= upper_q)
        if asset_quantile_lines is None:
            return ()
        if isinstance(asset_quantile_lines, dict):
            raw_quantiles = asset_quantile_lines.get(metric_name)
            if raw_quantiles is None:
                raw_quantiles = asset_quantile_lines.get(metric_name.lower())
            if raw_quantiles is None:
                return ()
        else:
            raw_quantiles = asset_quantile_lines
        if isinstance(raw_quantiles, (int, float)):
            raw_quantiles = (raw_quantiles,)
        quantiles = tuple(float(q) / 100 if float(q) > 1 else float(q) for q in raw_quantiles)
        if any(q < 0 or q > 1 for q in quantiles):
            raise ValueError("asset_quantile_lines values must be between 0 and 1, or 0 and 100.")
        if band_quantiles is not None:
            lower_q, upper_q = band_quantiles
            quantiles = tuple(q for q in quantiles if lower_q <= q <= upper_q)
        return quantiles

    def _quantile_label(q: float) -> str:
        return f"P{int(round(q * 100)):02d}"

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
    shown_asset_quantile_legends: set[tuple[str, str]] = set()

    for col_idx, (metric, y_title) in enumerate(metrics, start=1):
        metric_df = df[df["metric"] == metric].copy()
        if metric_df.empty:
            continue
        for color_idx, (group, group_df_all) in enumerate(metric_df.groupby(group_col, sort=False)):
            group_df = _selected_assets(group_df_all, metric)
            if group_df.empty:
                continue
            grouped = group_df.groupby(["percentile_order", "percentile"], as_index=False)["value"]
            if band_quantiles is None:
                stats = grouped.agg(
                    center=center_stat,
                    band_lower="min",
                    band_upper="max",
                )
                if metric_config is None:
                    band_label = "min-max asset range"
                else:
                    lower_q, upper_q = _asset_quantile_range(metric)
                    band_label = f"selected p{int(lower_q * 100):02d}-p{int(upper_q * 100):02d} asset range"
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
                    name=f"{group_label} {center_stat} across selected assets",
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
            for quantile in _asset_quantiles_for(metric):
                quantile_stats = (
                    group_df_all.groupby(["percentile_order", "percentile"], as_index=False)["value"]
                    .quantile(quantile)
                    .sort_values("percentile_order")
                )
                quantile_name = _quantile_label(quantile)
                dash_style = "dash" if round(quantile, 2) in {0.95, 0.05} else "dot"
                legend_key = (group_label, quantile_name)
                show_quantile_legend = legend_key not in shown_asset_quantile_legends
                shown_asset_quantile_legends.add(legend_key)
                fig.add_trace(
                    go.Scatter(
                        x=quantile_stats["percentile"],
                        y=quantile_stats["value"],
                        mode="lines",
                        line={"color": color, "width": 1.6, "dash": dash_style},
                        name=f"{group_label} asset {quantile_name}",
                        legendgroup=f"{group_label} asset {quantile_name}",
                        showlegend=show_quantile_legend,
                        hovertemplate=(
                            "%{x}<br>"
                            f"asset {quantile_name}: %{{y:.4g}}"
                            "<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=col_idx,
                )
        fig.update_yaxes(
            title_text=y_title,
            tickformat=".2f" if metric == "Voltage" else None,
            row=1,
            col=col_idx,
        )
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

def _normalize_percentile_label(value) -> str:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"max", "min"}:
            return text
        if text.startswith("p"):
            suffix = text[1:]
            return f"p{int(suffix):02d}" if suffix.isdigit() else text
        return f"p{int(text):02d}" if text.isdigit() else f"p{text}"
    return f"p{int(value):02d}"


def _normalize_time_quantile(value) -> str:
    if isinstance(value, str):
        return _normalize_percentile_label(value)
    quantile = float(value)
    if quantile == 0:
        return "min"
    if quantile == 1:
        return "max"
    if 0 < quantile < 1:
        return f"p{int(round(quantile * 100)):02d}"
    if 1 < quantile <= 100:
        return f"p{int(round(quantile)):02d}"
    raise ValueError("time_quantile must be 0..1, 0..100, or one of 'min'/'max'.")



def plot_powerflow_headline_violins(
    profile: pd.DataFrame,
    group_col: str | None = None,
    metric_config: dict[str, dict[str, object]] | None = None,
    asset_cutoff_percentiles: tuple[float, ...] | None = None,
    worst_asset_per_grid: bool = False,
    points: str | bool | None = "all",
    show: bool = True,
    y_axis_limits: tuple[float | None, float | None, float | None] | None = None,
):
    """Plot one critical annual value per transformer, cable, or bus.

    By default, transformer and cable values use annual maximum loading and
    voltage values use annual minimum voltage. ``asset_cutoff_percentiles`` adds
    the same value-based outlier slider as
    :func:`plot_powerflow_pooled_asset_percentile_curves`: P99 removes loading assets
    above global P99 and voltage assets below global P01 before drawing.
    ``y_axis_limits`` optionally fixes the panels as
    ``(transformer_upper_percent, cable_upper_percent, voltage_lower_pu)``.
    """
    required = {"metric", "percentile", "value"}
    missing_required = required.difference(profile.columns)
    if missing_required:
        missing = ", ".join(sorted(missing_required))
        raise ValueError(
            "plot_powerflow_headline_violins expects the asset-level percentile "
            f"profile dataframe; missing column(s): {missing}. Pass the dataframe "
            "loaded by powerflow_percentile_profile_db / "
            "real_powerflow_percentile_profile_db instead of the compact summary."
        )

    df = profile.copy()
    if group_col is None or group_col not in df.columns:
        group_col = "comparison_group"
        df[group_col] = "All assets"

    config = {
        "Transformer": {"time_quantile": 1.0, "asset_quantile": 1.0},
        "Cables": {"time_quantile": 1.0, "asset_quantile": 1.0},
        "Voltage": {"time_quantile": 0.0, "asset_quantile": 0.0},
    }
    critical_direction = {
        "Transformer": "high",
        "Cables": "high",
        "Voltage": "low",
    }
    if metric_config:
        metric_names = {metric.lower(): metric for metric in config}
        allowed_keys = {"time_quantile", "asset_quantile"}
        for metric, user_config in metric_config.items():
            metric_key = metric_names.get(str(metric).strip().lower())
            if metric_key is None:
                available = ", ".join(config)
                raise ValueError(f"Unsupported headline metric {metric!r}. Available: {available}.")
            if not isinstance(user_config, dict):
                raise ValueError(
                    "Each metric_config entry must be a dict with keys "
                    "'time_quantile' and/or 'asset_quantile'."
                )
            unknown_keys = set(user_config).difference(allowed_keys)
            if unknown_keys:
                unknown = ", ".join(sorted(unknown_keys))
                raise ValueError(f"Unsupported metric_config key(s) for {metric_key}: {unknown}.")
            config[metric_key].update(user_config)

    if asset_cutoff_percentiles is None:
        asset_cutoff_percentiles = (1.0,)
    asset_cutoff_percentiles = tuple(
        float(q) / 100 if float(q) > 1 else float(q) for q in asset_cutoff_percentiles
    )
    if any(q <= 0 or q > 1 for q in asset_cutoff_percentiles):
        raise ValueError("asset_cutoff_percentiles values must satisfy 0 < value <= 1, or 0 < value <= 100.")
    y_axis_ranges = _powerflow_y_axis_ranges(y_axis_limits)

    def _cutoff_label(cutoff: float) -> str:
        return "All assets" if np.isclose(cutoff, 1.0) else f"P{int(round(cutoff * 100)):02d} cutoff"

    def _normalize_asset_quantile(value) -> float | None:
        if value is None:
            return None
        quantile = float(value)
        if quantile > 1:
            quantile = quantile / 100
        if not 0 <= quantile <= 1:
            raise ValueError("asset_quantile must be between 0 and 1, or 0 and 100.")
        return quantile

    def _filter_asset_quantile(plot_df: pd.DataFrame, metric_name: str, asset_quantile) -> pd.DataFrame:
        quantile = _normalize_asset_quantile(asset_quantile)
        if quantile is None or plot_df.empty:
            return plot_df
        direction = critical_direction[metric_name]
        if direction == "high" and quantile == 1:
            return plot_df
        if direction == "low" and quantile == 0:
            return plot_df
        frames = []
        for _, group_df in plot_df.groupby(group_col, sort=False):
            threshold = group_df["value"].quantile(quantile)
            if direction == "high":
                frames.append(group_df[group_df["value"] <= threshold])
            else:
                frames.append(group_df[group_df["value"] >= threshold])
        return pd.concat(frames, ignore_index=True) if frames else plot_df.iloc[0:0].copy()

    def _filter_by_value_cutoff(plot_df: pd.DataFrame, metric_name: str, cutoff: float) -> pd.DataFrame:
        if np.isclose(cutoff, 1.0) or plot_df.empty:
            return plot_df
        values = plot_df["value"].astype(float)
        if critical_direction[metric_name] == "high":
            threshold = values.quantile(cutoff)
            return plot_df[values <= threshold]
        threshold = values.quantile(1 - cutoff)
        return plot_df[values >= threshold]

    def _worst_asset_per_grid(plot_df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
        if not worst_asset_per_grid or plot_df.empty:
            return plot_df
        if "grid" not in plot_df.columns:
            raise ValueError("worst_asset_per_grid=True requires a 'grid' column in the profile dataframe.")
        direction = critical_direction[metric_name]
        group_keys = [group_col, "grid"] if group_col in plot_df.columns else ["grid"]
        value_index = (
            plot_df.groupby(group_keys, sort=False)["value"].idxmax()
            if direction == "high"
            else plot_df.groupby(group_keys, sort=False)["value"].idxmin()
        )
        return plot_df.loc[value_index].reset_index(drop=True)

    y_titles = {
        "Transformer": "Annual loading [%]",
        "Cables": "Annual loading [%]",
        "Voltage": "Annual voltage [p.u.]",
    }
    df["percentile_norm"] = df["percentile"].map(_normalize_percentile_label)
    metrics = [
        ("Transformer", y_titles["Transformer"]),
        ("Cables", y_titles["Cables"]),
        ("Voltage", y_titles["Voltage"]),
    ]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[title for title, _ in metrics])
    traces_by_cutoff: list[list[int]] = []

    for cutoff_index, cutoff in enumerate(asset_cutoff_percentiles):
        is_visible = cutoff_index == 0
        cutoff_trace_indices: list[int] = []
        cutoff_label = _cutoff_label(cutoff)

        for col_idx, (metric, y_title) in enumerate(metrics, start=1):
            time_quantile = _normalize_time_quantile(config[metric]["time_quantile"])
            asset_quantile = config[metric]["asset_quantile"]
            plot_df = df[(df["metric"] == metric) & (df["percentile_norm"] == time_quantile)].copy()
            if plot_df.empty:
                available = sorted(df.loc[df["metric"] == metric, "percentile_norm"].dropna().unique())
                raise ValueError(f"No {metric} rows found for time_quantile {time_quantile!r}. Available: {available}.")
            plot_df = plot_df.dropna(subset=["value"])
            if asset_cutoff_percentiles == (1.0,):
                plot_df = _filter_asset_quantile(plot_df, metric, asset_quantile)
            plot_df = _filter_by_value_cutoff(plot_df, metric, cutoff)
            plot_df = _worst_asset_per_grid(plot_df, metric)
            hover_parts = []
            hover_labels = {
                "grid": "grid",
                "asset_label": "asset",
                "asset_id": "asset_id",
                "n_failed_timesteps": "failed_hours",
                "n_converged_timesteps": "converged_hours",
            }
            for col, label in hover_labels.items():
                if col in plot_df.columns:
                    hover_parts.append(label + ": " + plot_df[col].astype(str))
            if hover_parts:
                plot_df["hover_text"] = hover_parts[0]
                for part in hover_parts[1:]:
                    plot_df["hover_text"] = plot_df["hover_text"] + "<br>" + part
                plot_df["hover_text"] = plot_df["hover_text"] + "<br>" + cutoff_label
            else:
                plot_df["hover_text"] = metric + "<br>" + cutoff_label
            fig.add_trace(
                go.Violin(
                    x=plot_df[group_col].astype(str),
                    y=plot_df["value"].astype(float),
                    text=plot_df["hover_text"],
                    hovertemplate="%{text}<br>%{y:.4g}<extra></extra>",
                    box_visible=True,
                    meanline_visible=True,
                    points=points,
                    jitter=0.18,
                    scalemode="width",
                    name=f"{metric} {time_quantile}",
                    showlegend=False,
                    visible=is_visible,
                ),
                row=1,
                col=col_idx,
            )
            cutoff_trace_indices.append(len(fig.data) - 1)
            fig.update_yaxes(
                title_text=f"{time_quantile.upper()} {y_title}",
                tickformat=".2f" if metric == "Voltage" else None,
                row=1,
                col=col_idx,
            )
            if metric in y_axis_ranges:
                fig.update_yaxes(range=y_axis_ranges[metric], row=1, col=col_idx)
        traces_by_cutoff.append(cutoff_trace_indices)

    slider_steps = []
    n_traces = len(fig.data)
    for cutoff, cutoff_trace_indices in zip(asset_cutoff_percentiles, traces_by_cutoff):
        visible = [False] * n_traces
        for trace_index in cutoff_trace_indices:
            visible[trace_index] = True
        slider_steps.append(
            {
                "label": _cutoff_label(cutoff),
                "method": "update",
                "args": [
                    {"visible": visible},
                    _powerflow_y_axis_slider_layout(y_axis_ranges),
                ],
            }
        )

    fig.update_layout(
        title="Headline Power-Flow Quality Metrics by Grid" if worst_asset_per_grid else "Headline Power-Flow Quality Metrics by Asset",
        violingap=0.12,
        height=470,
        margin={"l": 55, "r": 25, "t": 75, "b": 150},
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Outlier filter: "},
                "x": 0.08,
                "len": 0.84,
                "y": -0.26,
                "pad": {"t": 65},
                "steps": slider_steps,
            }
        ] if len(asset_cutoff_percentiles) > 1 else None,
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
        compact_query = text(
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
                   pbvs.stage,
                   MIN(pbvs.voltage_min_time_pu) AS min_vm_pu,
                   MAX(pbvs.voltage_max_time_pu) AS max_vm_pu,
                   MAX(pfs.n_timesteps) AS n_timesteps,
                   COUNT(DISTINCT pbvs.bus) AS n_buses
            FROM surrogrid.powerflow_bus_voltage_summary pbvs
            JOIN surrogrid.powerflow_summary pfs
              ON pfs.powerflow_run_id = pbvs.powerflow_run_id
             AND pfs.stage = pbvs.stage
            JOIN surrogrid.powerflow_run pr
              ON pr.powerflow_run_id = pbvs.powerflow_run_id
            JOIN surrogrid.scenario sc
              ON sc.scenario_id = pr.scenario_id
            JOIN surrogrid.grid_case gc
              ON gc.grid_case_id = pr.grid_case_id
            WHERE pr.run_name = :run_name
              AND pbvs.stage = ANY(:stages)
              AND (:run_id IS NULL OR pbvs.powerflow_run_id = :run_id)
              AND (:scenario_id IS NULL OR pr.scenario_id = :scenario_id)
              AND (:ags IS NULL OR gc.ags = :ags)
              AND (:filter_plz IS NULL OR gc.plz = :filter_plz)
              AND (:filter_kcid IS NULL OR gc.kcid = :filter_kcid)
              AND (:filter_bcid IS NULL OR gc.bcid = :filter_bcid)
            GROUP BY pr.powerflow_run_id, pr.run_name, pr.scenario_id, sc.scenario_key, gc.ags, gc.plz, gc.kcid, gc.bcid,
                     gc.pylovo_grid_result_id, pbvs.stage
            ORDER BY pr.powerflow_run_id, pbvs.stage
            """
        )
        with db.engine.connect() as conn:
            summary = pd.read_sql_query(
                compact_query,
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



def _format_pu_limit(value: float) -> str:
    return f"{float(value):.3f}".rstrip("0").rstrip(".")

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
            name="Highest voltage per grid",
            marker={"color": "#2f92c5", "line": {"color": "white", "width": 0.5}},
            xbins={"start": x_min, "end": x_max, "size": bin_size},
            opacity=0.95,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=lower_values,
            name="Lowest voltage per grid",
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
        text=f"< {_format_pu_limit(lower_limit)} p.u.: {lower_share:.1f}%",
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
        text=f"> {_format_pu_limit(upper_limit)} p.u.: {upper_share:.1f}%",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#d0d0d0",
        borderwidth=1,
    )
    fig.update_layout(
        barmode="overlay",
        title="Voltage Magnitude Extremes Across LV Grids",
        xaxis_title="Grid-Level Voltage Extremum [p.u.]",
        yaxis_title="LV Grid Count (log scale)",
        yaxis={
            "type": "log",
            "rangemode": "tozero",
            "tickmode": "array",
            "tickvals": [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
            "ticktext": ["1", "2", "5", "10", "20", "50", "100", "200", "500", "1,000"],
            "minor": {"ticks": "outside"},
        },
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



def plot_voltage_deviation_histogram_comparison(
    summaries: dict[str, pd.DataFrame],
    lower_limit: float = 0.9,
    upper_limit: float = 1.1,
    bin_size: float = 0.01,
    title: str = "Voltage Magnitude Extremes Across LV Grids",
    show: bool = True,
):
    """Plot voltage-extreme histograms for multiple stages in horizontal subplots."""
    if not summaries:
        raise ValueError("At least one summary dataframe is required.")

    cleaned: dict[str, tuple[pd.Series, pd.Series]] = {}
    x_min_values = [lower_limit]
    x_max_values = [upper_limit]
    for label, summary in summaries.items():
        if summary.empty:
            continue
        lower_values = summary["min_vm_pu"].astype(float).dropna()
        upper_values = summary["max_vm_pu"].astype(float).dropna()
        if lower_values.empty or upper_values.empty:
            continue
        cleaned[str(label)] = (lower_values, upper_values)
        x_min_values.append(float(lower_values.min()))
        x_max_values.append(float(upper_values.max()))
    if not cleaned:
        raise ValueError("No finite voltage summary values found.")

    x_min = min(x_min_values) - 0.03
    x_max = max(x_max_values) + 0.03
    fig = make_subplots(
        rows=1,
        cols=len(cleaned),
        subplot_titles=list(cleaned.keys()),
        shared_yaxes=True,
        horizontal_spacing=0.08,
    )
    colors = {
        "highest": "#2f92c5",
        "lowest": "#66c2a4",
    }
    for col_idx, (label, (lower_values, upper_values)) in enumerate(cleaned.items(), start=1):
        lower_share = (lower_values < lower_limit).mean() * 100.0
        upper_share = (upper_values > upper_limit).mean() * 100.0
        fig.add_trace(
            go.Histogram(
                x=upper_values,
                name="Highest voltage per grid",
                marker={"color": colors["highest"], "line": {"color": "white", "width": 0.5}},
                xbins={"start": x_min, "end": x_max, "size": bin_size},
                opacity=0.88,
                legendgroup="highest",
                showlegend=col_idx == 1,
            ),
            row=1,
            col=col_idx,
        )
        fig.add_trace(
            go.Histogram(
                x=lower_values,
                name="Lowest voltage per grid",
                marker={"color": colors["lowest"], "line": {"color": "white", "width": 0.5}},
                xbins={"start": x_min, "end": x_max, "size": bin_size},
                opacity=0.88,
                legendgroup="lowest",
                showlegend=col_idx == 1,
            ),
            row=1,
            col=col_idx,
        )
        fig.add_vline(x=lower_limit, line_color="#3a3a3a", line_dash="dash", line_width=2, row=1, col=col_idx)
        fig.add_vline(x=upper_limit, line_color="#3a3a3a", line_dash="dash", line_width=2, row=1, col=col_idx)
        fig.add_annotation(
            x=lower_limit - 0.012,
            y=0.82,
            xref="x" if col_idx == 1 else f"x{col_idx}",
            yref="paper",
            text=f"< {_format_pu_limit(lower_limit)} p.u.: {lower_share:.1f}%",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#d0d0d0",
            borderwidth=1,
        )
        fig.add_annotation(
            x=upper_limit + 0.012,
            y=0.70,
            xref="x" if col_idx == 1 else f"x{col_idx}",
            yref="paper",
            text=f"> {_format_pu_limit(upper_limit)} p.u.: {upper_share:.1f}%",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#d0d0d0",
            borderwidth=1,
        )

    fig.update_layout(
        barmode="overlay",
        title=title,
        yaxis_title="LV Grid Count (log scale)",
        legend={"orientation": "h", "x": 0.02, "y": 1.16},
        margin={"l": 70, "r": 30, "t": 90, "b": 65},
        width=max(820, 455 * len(cleaned)),
        height=440,
    )
    for col_idx in range(1, len(cleaned) + 1):
        fig.update_xaxes(title_text="Grid-Level Voltage Extremum [p.u.]", range=[x_min, x_max], showgrid=True, gridcolor="#d8d8d8", row=1, col=col_idx)
        fig.update_yaxes(
            type="log",
            rangemode="tozero",
            tickmode="array",
            tickvals=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
            ticktext=["1", "2", "5", "10", "20", "50", "100", "200", "500", "1,000"],
            minor={"ticks": "outside"},
            showgrid=True,
            gridcolor="#d8d8d8",
            row=1,
            col=col_idx,
        )
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
        compact_query = text(
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
                   ptd.stage,
                   ptd.diagnostic,
                   ptd.point_index,
                   ptd.x_value,
                   ptd.t_index,
                   ptd.ts,
                   ptd.p_mw,
                   ptd.q_mvar,
                   ptd.q_abs_mvar,
                   ptd.s_mva,
                   ptd.mean_s_mva,
                   ptd.max_s_mva
            FROM surrogrid.powerflow_transformer_diagnostic ptd
            JOIN surrogrid.powerflow_run pr USING (powerflow_run_id)
            JOIN surrogrid.scenario sc USING (scenario_id)
            JOIN surrogrid.grid_case gc USING (grid_case_id)
            WHERE pr.run_name = :run_name
              AND ptd.stage = :stage
              AND (:run_id IS NULL OR ptd.powerflow_run_id = :run_id)
              AND (:scenario_id IS NULL OR pr.scenario_id = :scenario_id)
              AND (:ags IS NULL OR gc.ags = :ags)
              AND (:filter_plz IS NULL OR gc.plz = :filter_plz)
              AND (:filter_kcid IS NULL OR gc.kcid = :filter_kcid)
              AND (:filter_bcid IS NULL OR gc.bcid = :filter_bcid)
            ORDER BY pr.powerflow_run_id, ptd.diagnostic, ptd.point_index
            """
        )
        with db.engine.connect() as conn:
            df = pd.read_sql_query(
                compact_query,
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
        df["q_import_mvar"] = df["q_abs_mvar"] if reactive_magnitude else df["q_mvar"]
        df["s_import_mva"] = df["s_mva"]
        mean_s = df["mean_s_mva"].replace(0.0, np.nan)
        max_s_by_grid = df.groupby("powerflow_run_id")["max_s_mva"].first().replace(0.0, np.nan)
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
    if "diagnostic" in df.columns:
        daily_df = df[df["diagnostic"] == "daily_mean"].copy()
        if daily_df.empty:
            raise ValueError(f"No compact daily transformer diagnostic values available for {value_col}.")
        index_col = "t_index" if relative_axis else "ts"
        wide = daily_df.pivot_table(
            index=index_col,
            columns="powerflow_run_id",
            values=value_col,
            aggfunc="mean",
        ).sort_index()
        if relative_axis:
            wide.index = (wide.index.astype(int) // 24).astype(int)
            wide.index.name = "day_index"
        else:
            wide.index = pd.to_datetime(wide.index)
        return _matplotlib_quantile_summary(wide)

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
    if "diagnostic" in df.columns:
        ldc_df = df[df["diagnostic"] == "ldc"].copy()
        if ldc_df.empty:
            raise ValueError(f"No compact LDC transformer diagnostic values available for {value_col}.")
        ldc = ldc_df.pivot_table(
            index="x_value",
            columns="powerflow_run_id",
            values=value_col,
            aggfunc="mean",
        ).sort_index()
        return _matplotlib_quantile_summary(ldc)

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
    band68_label: str = "68% Percentile Band",
    band96_label: str = "96% Percentile Band",
) -> None:
    ax.fill_between(
        x,
        band["q02"].to_numpy(),
        band["q98"].to_numpy(),
        facecolor=band96_color,
        edgecolor="none",
        alpha=0.42,
        label=band96_label,
    )
    ax.fill_between(
        x,
        band["q16"].to_numpy(),
        band["q84"].to_numpy(),
        facecolor=band68_color,
        edgecolor="none",
        alpha=0.38,
        label=band68_label,
    )
    ax.plot(
        x,
        band["expected"].to_numpy(),
        color=color,
        linewidth=1.6,
        alpha=0.98,
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



def _select_tsam_week_indices(ts_band: pd.DataFrame, requested: tuple[int, int] | None) -> tuple[int, int]:
    n_weeks = int(len(ts_band) // 7)
    if n_weeks < 2:
        return (0, 0)
    if requested is not None:
        first, second = requested
        first = max(0, min(int(first), n_weeks - 1))
        second = max(0, min(int(second), n_weeks - 1))
        return first, second
    week_ids = pd.Series(np.arange(len(ts_band)) // 7, index=ts_band.index)
    weekly_mean = ts_band["expected"].groupby(week_ids).mean().dropna()
    if weekly_mean.empty:
        return (0, min(1, n_weeks - 1))
    high_week = int(weekly_mean.idxmax())
    low_week = int(weekly_mean.idxmin())
    if high_week == low_week:
        low_week = min(high_week + 1, n_weeks - 1) if high_week == 0 else 0
    return high_week, low_week


def _week_slice(band: pd.DataFrame, week_index: int) -> pd.DataFrame:
    start = int(week_index) * 7
    end = start + 7
    return band.iloc[start:end].copy()


def _x_values_for_tsam_week(band: pd.DataFrame):
    if isinstance(band.index, pd.DatetimeIndex):
        return mdates.date2num(band.index.to_pydatetime()), True
    return np.arange(len(band), dtype=float), False


def _format_tsam_week_axis(ax, band: pd.DataFrame) -> None:
    if isinstance(band.index, pd.DatetimeIndex):
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        for label in ax.get_xticklabels():
            label.set_rotation(35)
            label.set_ha("right")
        ax.set_xlim(band.index[0], band.index[-1])
    else:
        ax.set_xticks(np.arange(len(band), dtype=float))
        ax.set_xticklabels([f"Day {idx + 1}" for idx in range(len(band))])
        ax.set_xlim(0, max(len(band) - 1, 1))


def _plot_transformer_import_tsam_week_panels_matplotlib(
    df: pd.DataFrame,
    spec: dict[str, str],
    grouped_frames: list[tuple[str, pd.DataFrame]],
    default_group_styles: dict[str, dict[str, str]],
    fallback_styles: list[dict[str, str]],
    *,
    tsam_week_indices: tuple[int, int] | None,
    tsam_week_labels: tuple[str, str],
    show: bool,
):
    use_calendar_axis = "ts" in df.columns and df["ts"].notna().any()
    relative_axis = not use_calendar_axis

    reference_band = _daily_matplotlib_transformer_bands(
        grouped_frames[-1][1] if grouped_frames else df,
        spec["ts_col"],
        relative_axis=relative_axis,
    )
    week_indices = _select_tsam_week_indices(reference_band, tsam_week_indices)

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
        fig = plt.figure(figsize=(11.2, 6.2), dpi=150)
        gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], hspace=0.42, wspace=0.30)
        week_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]
        ax_ldc = fig.add_subplot(gs[:, 1])

        for row, (ax_ts, week_index, week_label) in enumerate(zip(week_axes, week_indices, tsam_week_labels)):
            for group_index, (group_label, group_df) in enumerate(grouped_frames):
                if group_df.empty:
                    continue
                style = default_group_styles.get(group_label, fallback_styles[group_index % len(fallback_styles)])
                color = style.get("color", spec["color"])
                band68 = style.get("band68", spec["band68"])
                band96 = style.get("band96", spec["band96"])
                ts_band = _daily_matplotlib_transformer_bands(
                    group_df,
                    spec["ts_col"],
                    relative_axis=relative_axis,
                )
                week_band = _week_slice(ts_band, week_index)
                if week_band.empty:
                    continue
                x_ts, _ = _x_values_for_tsam_week(week_band)
                prefix = f"{group_label} " if group_label else ""
                _plot_matplotlib_band(
                    ax_ts,
                    week_band,
                    x_ts,
                    color,
                    band68,
                    band96,
                    f"{prefix}expected",
                    show_legend=row == 0,
                    band68_label=f"{prefix}68% band",
                    band96_label=f"{prefix}96% band",
                )
            _format_tsam_week_axis(ax_ts, _week_slice(reference_band, week_index))
            ax_ts.set_title(f"{week_label} (representative week {week_index + 1})")
            ax_ts.set_ylabel(spec["ts_y"])

        for group_index, (group_label, group_df) in enumerate(grouped_frames):
            if group_df.empty:
                continue
            style = default_group_styles.get(group_label, fallback_styles[group_index % len(fallback_styles)])
            color = style.get("color", spec["color"])
            band68 = style.get("band68", spec["band68"])
            band96 = style.get("band96", spec["band96"])
            ldc_band = _ldc_matplotlib_transformer_bands(
                group_df,
                spec["ldc_col"],
                relative_axis=relative_axis,
            )
            x_ldc = ldc_band.index.to_numpy(dtype=float)
            prefix = f"{group_label} " if group_label else ""
            _plot_matplotlib_band(
                ax_ldc,
                ldc_band,
                x_ldc,
                color,
                band68,
                band96,
                f"{prefix}expected",
                show_legend=True,
                band68_label=f"{prefix}68% band",
                band96_label=f"{prefix}96% band",
            )
        ax_ldc.set_xlim(0, 100)
        ax_ldc.set_xticks(np.arange(0, 101, 20))
        ax_ldc.set_xticklabels([f"{value}%" for value in range(0, 101, 20)])
        ax_ldc.set_title(f"{spec['title']} duration curve")
        ax_ldc.set_ylabel(spec["ldc_y"])
        fig.suptitle(spec["title"], y=0.99, fontsize=10, fontweight="bold")
        fig.subplots_adjust(top=0.90, bottom=0.10, left=0.08, right=0.98, hspace=0.42, wspace=0.30)
        if show:
            plt.show()
        return fig

def plot_transformer_import_distributions_matplotlib(
    df: pd.DataFrame,
    show: bool = True,
    metrics: tuple[str, ...] = ("p", "q", "s"),
    group_col: str | None = None,
    group_styles: dict[str, dict[str, str]] | None = None,
    tsam_week_panels: bool = False,
    tsam_week_indices: tuple[int, int] | None = None,
    tsam_week_labels: tuple[str, str] = ("Min-temperature week", "Max-solar week"),
):
    specs = [
        {
            "key": "p",
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
            "key": "q",
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
            "key": "s",
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
    metric_aliases = {
        "p": "p",
        "active": "p",
        "active_power": "p",
        "q": "q",
        "reactive": "q",
        "reactive_power": "q",
        "s": "s",
        "apparent": "s",
        "apparent_power": "s",
    }
    selected_keys = []
    for metric in metrics:
        key = metric_aliases.get(str(metric).strip().lower())
        if key is None:
            available = ", ".join(sorted(metric_aliases))
            raise ValueError(f"Unsupported transformer metric {metric!r}. Available aliases: {available}.")
        if key not in selected_keys:
            selected_keys.append(key)
    specs = [spec for spec in specs if spec["key"] in selected_keys]
    if not specs:
        raise ValueError("At least one transformer metric must be selected.")

    default_group_styles = {
        "Pre": {"color": "#335C81", "band68": "#9ecae1", "band96": "#deebf7"},
        "Post-all": {"color": "#D95D39", "band68": "#f4a582", "band96": "#fde0c5"},
        "Post": {"color": "#D95D39", "band68": "#f4a582", "band96": "#fde0c5"},
    }
    if group_styles:
        default_group_styles.update({str(key): value for key, value in group_styles.items()})
    fallback_styles = [
        {"color": "#08306b", "band68": "#3182bd", "band96": "#9ecae1"},
        {"color": "#d73027", "band68": "#fc8d59", "band96": "#fee0d2"},
        {"color": "#1a9850", "band68": "#91cf60", "band96": "#d9ef8b"},
        {"color": "#6a3d9a", "band68": "#b2abd2", "band96": "#e0d6f0"},
    ]

    if group_col is not None and group_col in df.columns:
        group_labels = [str(label) for label in df[group_col].dropna().drop_duplicates()]
        grouped_frames = [(label, df[df[group_col].astype(str) == label].copy()) for label in group_labels]
    else:
        grouped_frames = [("", df.copy())]

    if tsam_week_panels:
        if len(specs) != 1:
            raise ValueError("tsam_week_panels=True currently supports exactly one transformer metric.")
        return _plot_transformer_import_tsam_week_panels_matplotlib(
            df,
            specs[0],
            grouped_frames,
            default_group_styles,
            fallback_styles,
            tsam_week_indices=tsam_week_indices,
            tsam_week_labels=tsam_week_labels,
            show=show,
        )

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
        fig_height = 3.25 * len(specs)
        fig, axes = plt.subplots(len(specs), 2, figsize=(10, fig_height), dpi=150, squeeze=False)
        relative_axis = _uses_relative_timeslice_axis(df)
        for row, spec in enumerate(specs):
            ax_ts = axes[row, 0]
            ax_ldc = axes[row, 1]
            for group_index, (group_label, group_df) in enumerate(grouped_frames):
                if group_df.empty:
                    continue
                style = default_group_styles.get(group_label, fallback_styles[group_index % len(fallback_styles)])
                color = style.get("color", spec["color"])
                band68 = style.get("band68", spec["band68"])
                band96 = style.get("band96", spec["band96"])
                ts_band = _daily_matplotlib_transformer_bands(
                    group_df,
                    spec["ts_col"],
                    relative_axis=relative_axis,
                )
                ldc_band = _ldc_matplotlib_transformer_bands(
                    group_df,
                    spec["ldc_col"],
                    relative_axis=relative_axis,
                )
                if relative_axis:
                    x_ts = ts_band.index.to_numpy(dtype=float)
                else:
                    x_ts = mdates.date2num(ts_band.index.to_pydatetime())
                x_ldc = ldc_band.index.to_numpy(dtype=float)

                prefix = f"{group_label} " if group_label else ""
                _plot_matplotlib_band(
                    ax_ts,
                    ts_band,
                    x_ts,
                    color,
                    band68,
                    band96,
                    f"{prefix}expected",
                    show_legend=True,
                    band68_label=f"{prefix}68% band",
                    band96_label=f"{prefix}96% band",
                )
                _plot_matplotlib_band(
                    ax_ldc,
                    ldc_band,
                    x_ldc,
                    color,
                    band68,
                    band96,
                    f"{prefix}expected",
                    show_legend=True,
                    band68_label=f"{prefix}68% band",
                    band96_label=f"{prefix}96% band",
                )

            if relative_axis:
                sample_band = _daily_matplotlib_transformer_bands(grouped_frames[0][1], spec["ts_col"], relative_axis=relative_axis)
                _format_relative_day_axis(ax_ts, sample_band.index)
            else:
                ax_ts.xaxis_date()
                sample_band = _daily_matplotlib_transformer_bands(grouped_frames[0][1], spec["ts_col"], relative_axis=relative_axis)
                _format_month_axis(ax_ts, sample_band.index)
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
        if "parallel" in net.line.columns:
            ratings["parallel"] = net.line["parallel"].fillna(1).astype(float)
        else:
            ratings["parallel"] = 1.0
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
    df["parallel"] = df["parallel"].fillna(1).astype(float)
    df["loading_percent"] = (
        df["max_i_from_ka"].astype(float) / (df["max_i_ka"] * df["parallel"])
    ) * 100.0
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
    if "parallel" in net.line.columns:
        parallel = net.line["parallel"].fillna(1).astype(float)
    else:
        parallel = pd.Series(1.0, index=net.line.index)
    df["max_i_ka"] = df["line"].map(max_i_ka.to_dict())
    df["parallel"] = df["line"].map(parallel.to_dict()).fillna(1).astype(float)
    df["loading_percent"] = (df["i_from_ka"] / (df["max_i_ka"] * df["parallel"])) * 100.0

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
