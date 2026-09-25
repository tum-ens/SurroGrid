"""Power-flow timestep discovery and heatmap plotting helpers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandapower as pp
import pandas as pd
import plotly.graph_objects as go
from pandapower.plotting import create_generic_coordinates
from pandapower.plotting import plotly as pp_plotly
from sqlalchemy import text

GRIDEXPAND_DIR = Path(__file__).resolve().parents[2]
if str(GRIDEXPAND_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDEXPAND_DIR))

from common.database import SurroGridDatabase  # noqa: E402


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
