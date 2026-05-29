"""Minimal plotting utility for step-4 powerflow output files.

This module reads one timestep from an output .h5 file and uses pandapower's
prebuilt pf_res_plotly() visualization to draw:
- bus voltage magnitudes as a heatmap
- line loadings as a heatmap

Example:
    uv run python plotting/powerflow_plotting.py \
    Output/900_80803_2_-1.h5 --stage pre --timestep 0 --on-map
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pandapower as pp
from pandapower.plotting import plotly as pp_plotly
from pandapower.plotting import create_generic_coordinates


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


def _draw_plotly_heatmap(
    net: pp.pandapowerNet,
    output_html: Path,
    on_map: bool,
    map_style: str,
    cmap: str,
    climits_volt: tuple[float, float],
    climits_load: tuple[float, float],
    show_household_buses: bool,
) -> None:
    cmap = _normalize_cmap_name(cmap)
    on_map = _ensure_geodata(net, on_map)
    use_line_geo = "geo" in net.line.columns and net.line["geo"].notna().all()

    bus_subset = _select_bus_subset(net, show_household_buses)

    bus_traces = pp_plotly.create_bus_trace(
        net,
        buses=bus_subset,
        cmap=cmap,
        cmap_vals=net.res_bus.loc[bus_subset, "vm_pu"].values,
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
        cmin=climits_load[0],
        cmax=climits_load[1],
        cpos=1.14,
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
        filename=str(output_html),
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
        margin={"l": 10, "r": 260, "t": 70, "b": 10},
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
                        "title": {"text": "Line Loading [%]"},
                        "x": 1.14,
                    },
                },
                "hoverinfo": "skip",
                "showlegend": False,
            }
        )

    fig.write_html(str(output_html), auto_open=False)


def plot_powerflow_heatmap(
    h5_path: Path,
    stage: str,
    timestep: int,
    output_html: Path,
    on_map: bool,
    map_style: str,
    cmap: str,
    climits_volt: tuple[float, float],
    climits_load: tuple[float, float],
    show_household_buses: bool = False,
) -> Path:
    net = _read_net(h5_path)
    vm_df, line_df = _read_results(h5_path, stage)

    bus_vm = _extract_bus_vm(vm_df, timestep)
    i_from_ka = _extract_line_current_ka(line_df, timestep)
    line_loading_percent = _line_loading_from_current(net, i_from_ka)
    _set_results_on_net(net, bus_vm, i_from_ka, line_loading_percent)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    try:
        _draw_plotly_heatmap(
            net=net,
            output_html=output_html,
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
            "GridExpand/4.powerflow to install plotly."
        ) from exc
    return output_html


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot line loading and bus voltage heatmaps from a step-4 output .h5 file."
    )
    parser.add_argument(
        "h5_file",
        help="Path to output .h5 file (e.g. Output/900_80803_2_-1.h5)",
    )
    parser.add_argument(
        "--stage",
        choices=("pre", "post"),
        default="pre",
        help="Which result stage to plot (default: pre).",
    )
    parser.add_argument("--timestep", type=int, default=0, help="Timestep index (default: 0).")
    parser.add_argument(
        "--output-html",
        default=None,
        help="Path to HTML output. Default: plotting/<stem>_<stage>_t<timestep>_heatmap.html",
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


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    h5_path = Path(args.h5_file)
    if not h5_path.exists():
        raise FileNotFoundError(f"File not found: {h5_path}")

    if args.output_html is None:
        output_html = Path(__file__).resolve().parent / (
            f"{h5_path.stem}_{args.stage}_t{args.timestep}_heatmap.html"
        )
    else:
        output_html = Path(args.output_html)

    saved_path = plot_powerflow_heatmap(
        h5_path=h5_path,
        stage=args.stage,
        timestep=args.timestep,
        output_html=output_html,
        on_map=args.on_map,
        map_style=args.map_style,
        cmap=args.cmap,
        climits_volt=tuple(args.climits_volt),
        climits_load=tuple(args.climits_load),
        show_household_buses=args.show_household_buses,
    )
    print(f"Saved heatmap plot to: {saved_path}")


if __name__ == "__main__":
    main()
