"""Step 4 entry point: time-series power flow of one scenario file.

The input (``work/powerflow/input/``: a Step 2 file for pre-only runs, a Step 3
result otherwise) is read with ``io.ScenarioResultReader``; pre and post demand
are reconstructed per bus, the pylovo grid is loaded from the database (or from
``raw_data/net``) and every timestep is solved once. From that single pass the
run writes the raw tables (``pwrflw/input/*``, ``pwrflw/output/<stage>/*``) and/or
the compact summary (database only). See ``docs/steps/4_powerflow.md``.
"""

from __future__ import annotations

import argparse
import signal
import sys
from dataclasses import dataclass

import pandas as pd

import gridexpand.powerflow.demands as dmnds
from gridexpand.common.resource_report import resource_report
from gridexpand.common.timeframe import scenario_key_for_timeframe
from gridexpand.optimization.identity import resolve_input_file
from gridexpand.optimization.solver import summarize_audit
from gridexpand.paths import POWERFLOW_INPUT_DIR, POWERFLOW_OUTPUT_DIR
from gridexpand.powerflow import engine, network
from gridexpand.powerflow.io import (
    DbRunSink,
    HdfSink,
    ScenarioResultReader,
    component_audit_path,
    grid_ref_from_case_id,
    require_temporal_method,
    residential_buses,
    temporal_assumptions,
    write_component_audit,
)

OUTPUTS = ("raw", "summary")
INFLEX_ASSUMPTION = (
    "fixed heat and EV profiles; optimized post-flex PV capacity; fixed SWF battery "
    "inventory with causal local PV self-consumption control, no grid charging, and "
    "no battery export; heat split uses the fixed input heatpump_air and "
    "heatpump_booster capacities (inst-cap)"
)
INFLEX_CAPACITY_SOURCE = (
    "heat: input inst-cap; PV: input inst-cap if inst-cap == cap-up, else post-flex cap_pro"
)


def _outputs(value: str) -> tuple[str, ...]:
    names = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    unknown = [name for name in names if name not in OUTPUTS]
    if not names or unknown:
        raise argparse.ArgumentTypeError(f"--outputs takes a comma list of {OUTPUTS}, got {value!r}.")
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gridexpand powerflow", description="Step 4: time-series power flow of one scenario file."
    )
    parser.add_argument(
        "inputfile_id",
        help=(
            f"Input file: a path, a file name in {POWERFLOW_INPUT_DIR} or the unique "
            "id prefix before the first underscore."
        ),
    )
    parser.add_argument("--n_cpu", type=int, default=1,
                        help="Time chunks solved in parallel processes.")
    parser.add_argument(
        "--storage",
        choices=["h5", "db"],
        default="h5",
        help="Write powerflow results to HDF5 or database. DB mode still reads urbs_in/urbs_out from HDF5.",
    )
    parser.add_argument(
        "--pre-only",
        action="store_true",
        help="Run only pre-expansion powerflow from urbs_in/demand; does not require urbs_out/MILP/tau_pro.",
    )
    parser.add_argument(
        "--outputs",
        type=_outputs,
        default=None,
        help=(
            "Comma list of raw,summary (default raw). Both come from one power-flow pass; "
            "the summary is stored in the database only."
        ),
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Same as --outputs summary: compact headline metrics in surrogrid.powerflow_summary.",
    )
    parser.add_argument(
        "--summary-run-name",
        default=None,
        help=(
            "With --outputs raw,summary: run name of the summary (default: the summary is "
            "stored in the raw run)."
        ),
    )
    parser.add_argument(
        "--grid-case-id",
        type=int,
        default=None,
        help=(
            "Explicit synthetic surrogrid.grid_case_id. Required when a paired "
            "scenario HDF filename does not encode the pylovo grid identifier."
        ),
    )
    parser.add_argument(
        "--pylovo-version-id",
        default=None,
        help=(
            "Override the pylovo topology version used to resolve the grid in DB "
            "mode. Defaults to the PYLOVO_VERSION_ID environment variable (.env)."
        ),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional DB powerflow run name (of the raw tables, or of the summary with --summary-only).",
    )
    parser.add_argument(
        "--hh-only",
        action="store_true",
        help=(
            "Synthetic-grid DB mode: restrict demand to the included Residential "
            "component buses from surrogrid.grid_building_component."
        ),
    )
    parser.add_argument(
        "--hh-annual-demand-scale",
        type=float,
        default=1.0,
        help=(
            "Optional multiplier for HH-only pre-expansion electricity and reactive demand. "
            "This is intended for aggregate SWF annual-demand sensitivity checks and requires --hh-only --pre-only."
        ),
    )
    parser.add_argument(
        "--post-demand-mode",
        choices=["flexible", "inflex"],
        default="flexible",
        help=(
            "Post-electrification demand reconstruction. 'flexible' uses optimized URBS net import; "
            "'inflex' derives fixed heat, PV, and capped EV charging; the heat split uses the fixed "
            "input heatpump_air/heatpump_booster capacities."
        ),
    )
    parser.add_argument(
        "--expect-temporal-method",
        choices=("full_year_no_tsam", "shared_weather_tsam"),
        default=None,
        help=(
            "Reject the Step-3 result unless it records this temporal method. "
            "Neither the file name nor the presence of the reduced_data group "
            "proves how a result was produced."
        ),
    )
    parser.add_argument(
        "--inflex-ev-charger-kw",
        type=float,
        default=None,
        help="Optional cross-check of the per-vehicle EV charger rating for --post-demand-mode inflex. Ratings come from the EV session table; a value that disagrees with any vehicle is rejected.",
    )
    parser.add_argument(
        "--max-timesteps",
        type=int,
        default=None,
        help="Optional smoke-test limit; omit for the full horizon.",
    )
    parser.add_argument(
        "--summary-nonconvergence",
        choices=["auto", "raise", "nan"],
        default="auto",
        help=(
            "Power-flow non-convergence handling of summary-only runs. 'raise' aborts the grid, "
            "'nan' records failed timesteps and continues, and 'auto' records and continues. "
            "Runs that write raw tables always raise."
        ),
    )
    parser.add_argument(
        "--summary-grid-scope",
        choices=["full", "backbone"],
        default="full",
        help=(
            "Assets included in summary voltage and cable statistics. "
            "'full' includes terminal buses and service lines (default); "
            "'backbone' excludes terminal service edges and maps terminal voltage "
            "observations one bus upstream."
        ),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.summary_only:
        if args.outputs not in (None, ("summary",)):
            parser.error("--summary-only conflicts with --outputs; use --outputs summary.")
        args.outputs = ("summary",)
    if args.outputs is None:
        args.outputs = ("raw",)
    if "summary" in args.outputs and args.storage != "db":
        parser.error("Summary output requires --storage db.")
    if args.summary_run_name is not None and set(args.outputs) != set(OUTPUTS):
        parser.error("--summary-run-name only applies with --outputs raw,summary.")
    if args.hh_only and args.storage != "db":
        parser.error("--hh-only requires --storage db.")
    if args.hh_annual_demand_scale != 1.0 and not args.hh_only:
        parser.error("--hh-annual-demand-scale requires --hh-only.")
    if args.hh_annual_demand_scale != 1.0 and not args.pre_only:
        parser.error("--hh-annual-demand-scale is only supported for --pre-only HH demand runs.")
    if args.post_demand_mode == "inflex" and args.pre_only:
        parser.error("--post-demand-mode inflex requires a post-electrification run, not --pre-only.")
    if args.inflex_ev_charger_kw is not None and args.post_demand_mode != "inflex":
        parser.error("--inflex-ev-charger-kw requires --post-demand-mode inflex.")
    if args.summary_nonconvergence != "auto" and "summary" not in args.outputs:
        parser.error("--summary-nonconvergence only applies to runs with summary output.")
    if args.n_cpu < 1:
        parser.error("--n_cpu must be at least 1.")
    args.summary_nonconvergence = "nan" if args.summary_nonconvergence == "auto" else args.summary_nonconvergence
    return args


def build_assumptions(args, reader: ScenarioResultReader) -> dict:
    """Run assumptions: timeframe metadata of the input plus the Step 4 settings."""
    assumptions = {
        "post_demand_mode": args.post_demand_mode,
        "summary_grid_scope": args.summary_grid_scope,
        "summary_nonconvergence": args.summary_nonconvergence,
    }
    if args.post_demand_mode == "inflex":
        assumptions.update({
            "inflex_assumption": INFLEX_ASSUMPTION,
            "inflex_ev_charger_kw": (
                None if args.inflex_ev_charger_kw is None else float(args.inflex_ev_charger_kw)
            ),
            "ev_service_model": "dedicated_sessions",
            "inflex_capacity_source": INFLEX_CAPACITY_SOURCE,
        })
    if args.hh_only:
        assumptions.update({
            "demand_scope": "synthetic_hh_only",
            "hh_only_filter": "grid_building_component.included_in_lv AND component_category == Residential",
            "hh_annual_demand_scale": float(args.hh_annual_demand_scale),
        })
    # Identity, not file name, decides whether this result may be consumed.
    if args.expect_temporal_method is not None and not args.pre_only:
        audit = require_temporal_method(reader.path, args.expect_temporal_method)
    else:
        audit = reader.temporal_method()
    assumptions.update(temporal_assumptions(audit))
    if not args.pre_only:
        assumptions.update(summarize_audit(reader.solver_audit()))
    return assumptions


def _filter_demand_to_buses(df, buses: set[int], label: str):
    if df is None:
        return None
    if getattr(df.columns, "nlevels", 1) < 2:
        raise ValueError(f"Cannot apply --hh-only to {label}: expected MultiIndex columns (bus, component).")

    keep_columns = []
    for column in df.columns:
        try:
            keep_columns.append(int(column[0]) in buses)
        except (TypeError, ValueError):
            keep_columns.append(False)
    filtered = df.loc[:, keep_columns].copy()
    if filtered.empty:
        raise ValueError(f"--hh-only removed all {label} demand columns; residential bus mapping and demand table do not match.")
    print(f"HH-only {label}: kept {filtered.shape[1]} of {df.shape[1]} demand columns.", flush=True)
    return filtered


def _scale_hh_annual_demand(df, scale: float, label: str):
    if df is None:
        return None
    scale = float(scale)
    if scale <= 0:
        raise ValueError("--hh-annual-demand-scale must be greater than zero.")
    if scale == 1.0:
        return df
    if getattr(df.columns, "nlevels", 1) < 2:
        raise ValueError(f"Cannot scale {label}: expected MultiIndex columns (bus, component).")
    scaled = df.copy()
    component_level = scaled.columns.get_level_values(1)
    mask = component_level.isin(["electricity", "electricity-reactive"])
    if not mask.any():
        raise ValueError(f"Cannot scale {label}: no electricity or electricity-reactive columns found.")
    before_kwh = float(scaled.loc[:, component_level == "electricity"].sum().sum())
    scaled.loc[:, mask] = scaled.loc[:, mask] * scale
    after_kwh = float(scaled.loc[:, component_level == "electricity"].sum().sum())
    print(
        f"HH annual demand scaling for {label}: factor={scale:.6g}, "
        f"active energy {before_kwh:.1f} -> {after_kwh:.1f} kWh.",
        flush=True,
    )
    return scaled


def demand_buses(*demand_frames) -> set[int]:
    """Buses of the ``(bus, component)`` demand columns."""
    buses = set()
    for frame in demand_frames:
        if frame is None or frame.empty:
            continue
        if getattr(frame.columns, "nlevels", 1) < 2:
            raise ValueError(
                "Scenario demand must use MultiIndex columns (bus, component)."
            )
        for column in frame.columns.to_flat_index():
            try:
                buses.add(int(column[0]))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid scenario demand bus in column {column!r}."
                ) from exc
    if not buses:
        raise ValueError("Scenario demand contains no buses.")
    return buses


def load_demands(args, reader, residential=None):
    """Pre and post demand frames after projection, HH filter, scaling and truncation.

    Returns:
        ``(pre, post, reactive_components, battery_diagnostics)``; post and the
        reactive table are None for pre-only runs.
    """
    if args.pre_only:
        pre, post, reactive, battery = dmnds.obtain_pre_demand(reader), None, None, None
    else:
        pre, post, reactive, battery = dmnds.reconstruct_demands(
            reader, post_demand_mode=args.post_demand_mode, ev_charger_kw=args.inflex_ev_charger_kw
        )
    if reader.metadata.get("optimization_space") == "scenario_unit":
        allocation = reader.get_allocation_plan()
        pre = dmnds.project_scenario_units_to_buses(pre, allocation)
        if post is not None:
            post = dmnds.project_scenario_units_to_buses(post, allocation)
    if residential is not None:
        pre = _filter_demand_to_buses(pre, residential, "pre")
        post = _filter_demand_to_buses(post, residential, "post")
    if args.hh_annual_demand_scale != 1.0:
        pre = _scale_hh_annual_demand(pre, args.hh_annual_demand_scale, "pre")
    if args.max_timesteps is not None:
        pre = pre.iloc[: int(args.max_timesteps)].copy()
        if post is not None:
            post = post.iloc[: int(args.max_timesteps)].copy()
    return pre, post, reactive, battery


def run_plan(outputs, run_name, summary_run_name=None) -> dict:
    """Which ``powerflow_run`` rows a run writes.

    Returns:
        ``{"raw": name}`` and/or ``{"summary": name}``; ``shared`` is True when the
        summary goes into the raw run (``--outputs raw,summary`` without a
        different ``--summary-run-name``). Names may be None (DB default name).
    """
    plan = {}
    if "raw" in outputs:
        plan["raw"] = run_name
    if "summary" in outputs:
        if "raw" not in outputs:
            plan["summary"] = run_name
        elif summary_run_name in (None, run_name):
            plan["summary"], plan["shared"] = run_name, True
        else:
            plan["summary"] = summary_run_name
    return plan


@dataclass
class GridContext:
    """Prepared network and the evaluation scope of one Step 4 run."""

    grid: object
    transformer_s_rated_mva: float
    cable_max_i_ka: pd.Series
    cable_ids: list
    voltage_buses: list


def prepare_grid_context(grid, load_buses, scope) -> GridContext:
    original_load_rows = len(grid.load)
    original_static_p_mw = (
        float(grid.load["p_mw"].fillna(0.0).sum()) if "p_mw" in grid.load.columns else 0.0
    )
    grid = network.set_scenario_load_buses(grid, load_buses)
    print(
        "Scenario grid.load: replaced "
        f"{original_load_rows} static rows ({original_static_p_mw:.6f} MW) "
        f"with {len(grid.load)} zeroed scenario-bus rows.",
        flush=True,
    )
    rating = network.transformer_rating_mva(grid)
    cable_max_i_ka = network.rated_cable_currents(grid)
    grid = network.prepare_synthetic_grid(grid)
    cable_ids, voltage_buses = network.comparison_evaluation_scope(grid, load_buses, scope=scope)
    if not voltage_buses:
        voltage_buses = load_buses or grid.bus.index.tolist()
    print(
        f"Summary grid scope={scope}: {len(voltage_buses)} voltage buses, {len(cable_ids)} cable rows.",
        flush=True,
    )
    return GridContext(grid, rating, cable_max_i_ka, cable_ids, voltage_buses)


def run_stage(context: GridContext, demand, stage, *, outputs, raw_sink, summary_sink, n_workers, on_nonconvergence):
    """Solve one stage once and write its raw tables and/or summary."""
    with resource_report(name=f"{stage.capitalize()}-Expansion Powerflow Run", include_children=True):
        matrices = engine.run_timeseries(
            context.grid,
            demand,
            algorithm="bfsw",
            on_nonconvergence="raise" if "raw" in outputs else on_nonconvergence,
            n_workers=n_workers,
        )
        if "raw" in outputs:
            ext_import, vm, line_loads = engine.raw_tables(matrices)
            raw_sink.save_df(ext_import, f"/pwrflw/output/{stage}/demand_import")
            raw_sink.save_df(vm, f"/pwrflw/output/{stage}/vm")
            raw_sink.save_df(line_loads, f"/pwrflw/output/{stage}/line_loads")
        if "summary" in outputs:
            summary = engine.summarize(
                context.grid,
                matrices,
                transformer_s_rated_mva=context.transformer_s_rated_mva,
                cable_max_i_ka=context.cable_max_i_ka,
                voltage_buses=context.voltage_buses,
                cable_ids=context.cable_ids,
            )
            summary_sink.save_summary(summary, stage)


def main(argv: list[str] | None = None) -> None:
    """Run Step 4 for one input file; see ``gridexpand powerflow --help``."""
    args = parse_args(argv)
    input_path = resolve_input_file(POWERFLOW_INPUT_DIR, args.inputfile_id)
    filename = input_path.name
    output_path = POWERFLOW_OUTPUT_DIR / filename
    reader = ScenarioResultReader(input_path)
    print(
        f"Running input file {filename} (ID {args.inputfile_id}, storage {args.storage}, "
        f"outputs {','.join(args.outputs)}) with {args.n_cpu} CPUs!"
    )
    print(input_path)
    assumptions = {**reader.metadata, **build_assumptions(args, reader)}

    db = grid_ref = None
    if args.storage == "db":
        from gridexpand.db.database import SurroGridDatabase

        db = SurroGridDatabase()
        if args.pylovo_version_id is not None:
            db.pylovo_version_id = str(args.pylovo_version_id)
        grid_ref = (
            grid_ref_from_case_id(db, args.grid_case_id)
            if args.grid_case_id is not None
            else db.resolve_grid_identifier(filename)
        )
    residential = residential_buses(db, grid_ref) if args.hh_only else None

    ##### Demand, grid ##### (a failure here registers no run)
    pre, post, reactive, battery = load_demands(args, reader, residential)
    if db is not None:
        grid = db.read_pandapower_grid(grid_ref)
    else:
        grid = reader.read_net()
        if grid is None:
            from gridexpand.db.database import SurroGridDatabase

            fallback = SurroGridDatabase()
            grid = fallback.read_pandapower_grid(fallback.resolve_grid_identifier(filename))
    load_buses = sorted(demand_buses(pre, post))
    context = prepare_grid_context(grid, load_buses, args.summary_grid_scope)

    ##### Outputs #####
    run_names = []
    raw_sink = summary_sink = None
    if db is not None:
        scenario_key = reader.metadata.get("scenario_key") or _default_scenario_key(reader)

        def register(run_name):
            run_names.append(run_name)
            return DbRunSink(
                db, grid_ref, urbs_input_file=filename, pre_only=args.pre_only,
                scenario_key=scenario_key, run_name=run_name, assumptions=assumptions,
            )

        plan = run_plan(args.outputs, args.run_name, args.summary_run_name)
        if "raw" in plan:
            raw_sink = register(plan["raw"])
        if "summary" in plan:
            summary_sink = raw_sink if plan.get("shared") else register(plan["summary"])
    else:
        raw_sink = HdfSink(input_path, output_path)
        run_names.append(args.run_name)

    if battery is not None and not battery.empty:
        for run_name in dict.fromkeys(run_names):
            location = write_component_audit(
                component_audit_path(output_path, run_name), battery, "inflex_battery_state"
            )
            print(f"INFLEX stationary-battery audit retained at {location}.", flush=True)

    db_sinks = list(dict.fromkeys(s for s in (raw_sink, summary_sink) if isinstance(s, DbRunSink)))
    # Job runners cancel with SIGTERM; turn it into SystemExit so the staging runs are dropped.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(128 + signal.SIGTERM))
    try:
        assets = reader.installed_assets() if db_sinks else None
        if assets is not None:
            for sink in db_sinks:
                sink.save_assets(assets)
        if "raw" in args.outputs:
            if reactive is not None:
                raw_sink.save_df(reactive, "pwrflw/urbs_out/MILP/reactive")
            raw_sink.save_df(pre, "/pwrflw/input/demand_pre")
            if post is not None:
                raw_sink.save_df(post, "/pwrflw/input/demand_post")

        stages = [("pre", pre)] + ([] if args.pre_only else [("post", post)])
        for stage, demand in stages:
            run_stage(
                context, demand, stage,
                outputs=args.outputs, raw_sink=raw_sink, summary_sink=summary_sink,
                n_workers=args.n_cpu, on_nonconvergence=args.summary_nonconvergence,
            )
    except BaseException:
        # Keep the previous results: drop the half-written staging runs.
        for sink in db_sinks:
            try:
                sink.discard()
            except Exception as exc:  # the original error matters more
                print(f"Could not discard staging run {sink.powerflow_run_id}: {exc}", flush=True)
        raise
    for sink in db_sinks:
        sink.promote()
    print("Done!")


def _default_scenario_key(reader) -> str:
    return scenario_key_for_timeframe(reader.metadata.get("timeframe_mode", "full_year"))


if __name__ == "__main__":
    main()
