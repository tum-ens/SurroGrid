"""Status-quo test of real grids at alignment (``gridexpand status-quo-test``).

For each real grid of a provider, the status-quo base electricity (households and GHD with
the current GridExpand model, including the scenario's ``ghd:`` rules) runs through the Step 4
power flow at the grid's peak hours with the Step 4 station-voltage convention (LV busbar at
0.96 p.u., up to two off-load tap steps; :mod:`gridexpand.powerflow.station_voltage`). A grid
is flagged when, in any evaluated hour,

* the transformer exceeds 100 % of its rating (``transformer_overload``),
* a cable exceeds 100 % of its rated current (``cable_overload``),
* a load bus stays below 0.90 p.u. after the taps (``undervoltage``), or
* the power flow does not converge (``nonconvergence``).

The flagged grids form the exclusion list: the alignment removes these real grids and their
buildings from both sides before pylovo generation. Output (``--output-dir``):

``status_quo_<provider>_exclusions.csv``
    ``provider, real_grid_id, reason``: one row per flagged grid; ``real_grid_id`` is the
    bundle's own id (SWF ``station_id`` such as ``LV_059``, ÜZW ``area_id``), ``reason`` the
    flags joined by ``;``.
``status_quo_<provider>_grids.csv``
    every grid with its demand and power-flow metrics.
``status_quo_<provider>_ghd_audit.csv``
    the GHD decisions per building (building input only).
``status_quo_<provider>_metadata.json``
    inputs and settings.

Inputs: the alignment bundle gives the real grid, bus and grid file per building; the building
attributes come read-only from InfDB ``basedata.buildings`` (``--db-env-prefix`` names the
connection variables). The pylovo derivations GridExpand's model needs before pylovo has run
(non-residential use, household fallback, peaks, MV-direct) mirror pylovo 8afedcc
(``infdb_client`` and ``preprocessing_mixin``; constants below). With ``--dataset`` the demand
comes from a prepared paired dataset instead (the persisted component plan of a paired run),
which reproduces that run's ``pre`` stage.

Peak hours: the ``--peak-hours`` largest hours of the grid total plus the 24 largest household
and GHD hours; ``--all-hours`` evaluates the whole year.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gridexpand.common import ghd
from gridexpand.common.building_components import build_building_components
from gridexpand.common.reproducibility import DEFAULT_PROFILE_SEED, stable_seed

PROVIDERS = ("swf", "uzw")
REASONS = ("transformer_overload", "cable_overload", "undervoltage", "nonconvergence")
PLAN_COLUMNS = (
    "component_id", "building_objectid", "component_category", "included_in_lv", "annual_energy_kwh",
    "source_asset_count", "stable_seed", "target_bus", "profile_hash",
)
LOADING_LIMIT_PERCENT = 100.0
EXTRA_PEAK_HOURS = 24

# pylovo 8afedcc values (config_generation.yaml, preprocessing_mixin._household_fallback_parameters).
PYLOVO_PEAK_LOAD_HOUSEHOLD_KW = 16.825
PYLOVO_PEAK_W_PER_M2 = {"Commercial": 79.0, "Public": 29.0, "Unknown": 29.0}
PYLOVO_MV_DIRECT_THRESHOLD_KW = 100.0
PYLOVO_AREA_PER_HOUSEHOLD_M2 = {"MFH": 181.0, "untyped": 181.0, "AB": 146.0}
PYLOVO_MINIMUM_HOUSEHOLDS = {"MFH": 2, "untyped": 1, "AB": 5}

BASEDATA_SQL = """
SELECT objectid, floor_area, floor_number, building_use, building_use_id, building_type,
       residential_floor_area, nonresidential_floor_area, mix_rule, occupants, households,
       CASE
         WHEN COALESCE(nonresidential_floor_area, 0) <= 0 THEN NULL
         WHEN building_use = 'Residential' THEN 'Commercial'
         WHEN building_use = 'Mixed' THEN
           CASE {schema}.classify_building_use(building_use_id)
             WHEN 'Residential' THEN 'Commercial'
             ELSE {schema}.classify_building_use(building_use_id)
           END
         WHEN building_use IN ('Commercial', 'Public', 'Unknown') THEN building_use
       END AS nonresidential_use,
       COALESCE(building_type, building_use) AS type
FROM {schema}.buildings
WHERE objectid = ANY(:ids)
"""


# ---------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------


def bundle_grid_id(provider: str, number: int) -> str:
    """The bundle's own id of a real grid: SWF ``LV_059``, ÜZW ``59``."""
    return f"LV_{int(number):03d}" if provider == "swf" else str(int(number))


def read_only_engine(prefix: str = "DB_"):
    """Read-only engine from ``<prefix>HOST/PORT/NAME/USER/PASSWORD`` (GridExpand ``.env``)."""
    from sqlalchemy import create_engine
    from sqlalchemy.engine import URL

    from gridexpand.db.engine import load_env

    load_env()
    name = (os.getenv(f"{prefix}NAME") or "").strip()
    if not name:
        raise ValueError(f"No database configured: set {prefix}HOST, {prefix}PORT, {prefix}NAME, {prefix}USER and {prefix}PASSWORD.")
    url = URL.create(
        "postgresql+psycopg2",
        username=os.getenv(f"{prefix}USER"),
        password=os.getenv(f"{prefix}PASSWORD"),
        host=os.getenv(f"{prefix}HOST"),
        port=int((os.getenv(f"{prefix}PORT") or "").strip() or "5432"),
        database=name,
    )
    return create_engine(url, pool_pre_ping=True, connect_args={"options": "-c default_transaction_read_only=on"})


def read_basedata(engine, objectids: list[str], schema: str = "basedata") -> pd.DataFrame:
    """InfDB building rows with pylovo 8afedcc's ``nonresidential_use`` and ``type``."""
    from sqlalchemy import text

    if not schema.isidentifier():
        raise ValueError(f"Invalid schema name {schema!r}.")
    with engine.connect() as conn:
        rows = pd.read_sql_query(text(BASEDATA_SQL.format(schema=schema)), conn, params={"ids": list(objectids)})
    missing = sorted(set(objectids) - set(rows["objectid"].astype(str)))
    if missing:
        raise ValueError(f"{len(missing)} bundle buildings are missing from {schema}.buildings: {missing[:10]}")
    return rows


def pylovo_equivalent_buildings(rows: pd.DataFrame) -> pd.DataFrame:
    """Fill what pylovo derives before GridExpand reads a building (8afedcc rules).

    Missing households by type, the non-residential use of plain Residential/typed
    buildings, component peaks, MV-direct above 100 kW and missing occupants
    (households x 2.03, as ``gridexpand.db.grids.read_buildings``).
    """
    from gridexpand.sampling.db_read import impute_missing_occupants

    b = rows.copy()
    b["objectid"] = b["objectid"].astype(str)
    kind = b["type"].astype(str)
    res = pd.to_numeric(b["residential_floor_area"], errors="coerce")
    nonres = pd.to_numeric(b["nonresidential_floor_area"], errors="coerce").fillna(0.0)
    gross = pd.to_numeric(b["floor_area"], errors="coerce") * pd.to_numeric(b["floor_number"], errors="coerce").fillna(1)
    fallback = pd.Series(np.nan, index=b.index)
    fallback[kind.isin(["SFH", "TH"])] = 1
    for name, label in (("MFH", "MFH"), ("AB", "AB")):
        mask = kind.eq(name)
        fallback[mask] = np.maximum(PYLOVO_MINIMUM_HOUSEHOLDS[label], np.rint(res.fillna(gross)[mask] / PYLOVO_AREA_PER_HOUSEHOLD_M2[label]))
    untyped = kind.isin(["Residential", "Mixed"]) & res.fillna(0).gt(0)
    fallback[untyped] = np.maximum(PYLOVO_MINIMUM_HOUSEHOLDS["untyped"], np.rint(res[untyped] / PYLOVO_AREA_PER_HOUSEHOLD_M2["untyped"]))
    fallback[kind.isin(["Commercial", "Public", "Unknown"])] = 1
    b["households"] = pd.to_numeric(b["households"], errors="coerce").where(b["households"].notna(), fallback)
    use = b["nonresidential_use"].where(b["nonresidential_use"].isin(list(PYLOVO_PEAK_W_PER_M2)))
    use = use.where(use.notna(), kind.where(kind.isin(list(PYLOVO_PEAK_W_PER_M2))))
    use = use.where(use.notna() | ~b["building_use"].eq("Residential"), "Commercial")
    b["nonresidential_use"] = use.where(nonres.gt(0))
    b["residential_peak_load_in_kw"] = np.where(res.fillna(0).gt(0), b["households"] * PYLOVO_PEAK_LOAD_HOUSEHOLD_KW, 0.0)
    b["nonresidential_peak_load_in_kw"] = np.where(
        nonres.gt(0), nonres * b["nonresidential_use"].map(PYLOVO_PEAK_W_PER_M2).fillna(0.0) / 1000.0, 0.0
    )
    direct = pd.Series(b["nonresidential_peak_load_in_kw"] > PYLOVO_MV_DIRECT_THRESHOLD_KW, index=b.index)
    b["nonresidential_mv_direct"] = [bool(value) if area > 0 else None for value, area in zip(direct, nonres)]
    b["peak_load_in_kw"] = b["residential_peak_load_in_kw"] + np.where(direct, 0.0, b["nonresidential_peak_load_in_kw"])
    for column in ("mix_score", "mix_confidence"):
        b[column] = np.nan
    return impute_missing_occupants(b)


def bundle_buildings(provider: str, alignment_dir: Path, uzw_grids_dir: Path | None) -> pd.DataFrame:
    """Real grid number, bus and grid file of every bundle building (``aligned_allocation``'s mapping)."""
    from gridexpand.allocation.scenario_calibration.allocation.aligned_allocation import real_building_mapping

    return real_building_mapping(provider, alignment_dir, uzw_grids_dir)


# ---------------------------------------------------------------------------------------
# Demand
# ---------------------------------------------------------------------------------------


def component_plan(
    physical: pd.DataFrame,
    config: ghd.GhdConfig,
    evidence: pd.DataFrame | None,
    *,
    profile_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired-contract component plan of buildings with a ``bus`` (their real bus).

    The same modelling path as the aligned allocation: component manifest, GHD rules,
    household and use-type sampling and annual energies of Step 2's electricity module;
    the paired seeds are those of ``aligned_allocation``.
    """
    import gridexpand.allocation.functions.electricity as electricity

    components = build_building_components(physical)
    components, audit = ghd.apply_ghd_policy(physical, components, config, evidence)
    components["objectid"] = components["objectid"].astype(str)
    if not components["included_in_lv"].astype(bool).any():
        return pd.DataFrame(columns=PLAN_COLUMNS), audit
    sampled = electricity.sample_statistics(components, base_seed=profile_seed)
    profiled, _ = electricity.get_elec_demand(sampled, base_seed=profile_seed)
    residential = profiled["component_category"].eq("Residential")
    annual = pd.to_numeric(profiled["annual_electricity_kwh"], errors="coerce").fillna(0.0)
    plan = pd.DataFrame({
        "component_id": profiled["component_id"].astype(str),
        "building_objectid": profiled["objectid"].astype(str),
        "component_category": profiled["component_category"],
        "included_in_lv": profiled["included_in_lv"].astype(bool) & annual.gt(0),
        "annual_energy_kwh": annual.where(profiled["included_in_lv"].astype(bool), 0.0),
        "source_asset_count": np.where(
            residential, profiled["demand_tot_list"].map(lambda value: len(value) if isinstance(value, list) else 0), 1
        ),
        "stable_seed": [
            stable_seed(int(profile_seed), str(oid), str(category), "electricity", "paired")
            for oid, category in zip(profiled["objectid"], profiled["component_category"])
        ],
        "target_bus": profiled["bus"].astype(int),
        "profile_hash": "status_quo",
    })
    return plan, audit


def bus_demand(plan: pd.DataFrame, *, profile_seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hourly total and GHD base electricity (kW) per target bus of a component plan.

    The paired profile builder (household profiles by annual energy, category-mean GHD
    shapes, daylight-saving shift), as in the ``pre`` stage of paired runs.
    """
    from gridexpand.allocation.scenario_calibration.profiles.paired_profiles import build_paired_base_electric_demand

    active = plan[plan["included_in_lv"].astype(bool) & pd.to_numeric(plan["annual_energy_kwh"], errors="coerce").gt(0)]
    buses = sorted(pd.to_numeric(plan["target_bus"]).astype(int).unique())

    def build(rows: pd.DataFrame) -> pd.DataFrame:
        if rows.empty:
            return pd.DataFrame(0.0, index=range(8760), columns=buses)
        demand, _ = build_paired_base_electric_demand(pd.DataFrame(), seed=profile_seed, component_plan=rows)
        demand.columns = demand.columns.get_level_values(0).astype(int)
        return demand.reindex(columns=buses, fill_value=0.0)

    total = build(active)
    ghd_rows = active[active["component_category"].isin(["Commercial", "Public"])]
    return total, build(ghd_rows)


def peak_hours(total: pd.DataFrame, ghd_part: pd.DataFrame, count: int) -> list[int]:
    """The ``count`` largest hours of the grid total plus the largest household and GHD hours."""
    extra = min(EXTRA_PEAK_HOURS, count)
    hours = set(total.sum(axis=1).nlargest(count).index)
    hours |= set(ghd_part.sum(axis=1).nlargest(extra).index) if ghd_part.to_numpy().any() else set()
    hours |= set((total - ghd_part).sum(axis=1).nlargest(extra).index)
    return sorted(int(hour) for hour in hours)


# ---------------------------------------------------------------------------------------
# Power flow
# ---------------------------------------------------------------------------------------


def evaluate_grid(net, rating_mva: float, demand_kw: pd.DataFrame) -> dict[str, Any]:
    """Status-quo power flow of one real grid for ``demand_kw`` (hours x bus, kW).

    Step 4's grid preparation and ``pf_summary`` (station voltage 0.96 p.u. and taps,
    ``nr`` with ``iwamoto_nr`` fallback, non-converged hours recorded, not raised).
    """
    import gridexpand.powerflow.demands as dmnds
    import gridexpand.powerflow.powerflow as pwrflw
    from gridexpand.powerflow.config import config as pf_config
    from gridexpand.powerflow.run_real_swf_scenario_powerflow import _prepare_real_grid_for_allocation

    buses = [int(bus) for bus in demand_kw.columns]
    grid, trafo_mva, cable_max, voltage_buses, cable_ids, _ = _prepare_real_grid_for_allocation(
        net, buses, "full", rating_mva
    )
    raw = demand_kw.copy()
    raw.columns = pd.MultiIndex.from_tuples([(bus, "electricity") for bus in buses])
    active, reactive = dmnds._process_pre_demands(raw.reset_index(drop=True))
    demand = pd.concat([active, reactive], axis=1).sort_index(axis=1)
    summary = pwrflw.pf_summary(
        grid, demand, transformer_s_rated_mva=trafo_mva, cable_max_i_ka=cable_max, voltage_buses=voltage_buses,
        algorithm=["nr", "iwamoto_nr"], cable_ids=cable_ids, on_nonconvergence="nan",
    )
    grid_summary, cables, voltages = summary["grid_summary"], summary["cable_summary"], summary["bus_voltage_summary"]
    cable_max_percent = pd.to_numeric(cables["cable_loading_max_time_percent"], errors="coerce").dropna()
    voltage_min = pd.to_numeric(voltages["voltage_min_time_pu"], errors="coerce").dropna()
    return {
        "transformer_rated_kva": float(trafo_mva) * 1000.0,
        "evaluated_hours": int(len(demand)),
        "failed_hours": int(grid_summary["n_failed_timesteps"]),
        "peak_demand_kw": float(demand_kw.sum(axis=1).max()),
        "transformer_max_percent": float(grid_summary["trafo_loading_max_time_percent"]),
        "cable_max_percent": float(cable_max_percent.max()) if len(cable_max_percent) else float("nan"),
        "cables_over_100": int(cable_max_percent.gt(LOADING_LIMIT_PERCENT).sum()),
        "min_voltage_pu": float(voltage_min.min()) if len(voltage_min) else float("nan"),
        "buses_below_limit": int(voltage_min.lt(pf_config.MIN_VM_PU - 1e-9).sum()),
        "tap_steps": grid_summary.get("tap_steps"),
        "lv_busbar_vm_pu": grid_summary.get("lv_busbar_vm_pu"),
    }


def flag_reasons(metrics: dict[str, Any]) -> list[str]:
    """The exclusion reasons of one evaluated grid (empty: the grid passes).

    A grid without a transformer rating (loading NaN) is checked for cables, voltage and
    convergence only.
    """
    reasons = []
    transformer = metrics["transformer_max_percent"]
    if math.isfinite(transformer) and transformer > LOADING_LIMIT_PERCENT:
        reasons.append("transformer_overload")
    if metrics["cables_over_100"] > 0:
        reasons.append("cable_overload")
    if metrics["buses_below_limit"] > 0:
        reasons.append("undervoltage")
    if metrics["failed_hours"] > 0:
        reasons.append("nonconvergence")
    return reasons


def _evaluate_job(job: dict[str, Any]) -> dict[str, Any]:
    from gridexpand.powerflow.run_real_swf_scenario_powerflow import load_real_net

    if not job["plan"]["included_in_lv"].astype(bool).any():
        # No LV demand left (e.g. every GHD component gated off): nothing can overload.
        return {**job["row"], "households": 0, "hh_annual_kwh": 0.0, "ghd_annual_kwh": 0.0,
                "evaluated_hours": 0, "failed_hours": 0, "flagged": False, "reason": ""}
    total, ghd_part = bus_demand(job["plan"], profile_seed=job["profile_seed"])
    hours = list(range(len(total))) if job["all_hours"] else peak_hours(total, ghd_part, job["peak_hours"])
    net, rating = load_real_net(Path(job["grid_file"]))
    metrics = evaluate_grid(net, rating, total.loc[hours])
    reasons = flag_reasons(metrics)
    return {
        **job["row"],
        "households": int(job["plan"].loc[job["plan"]["component_category"].eq("Residential"), "source_asset_count"].sum()),
        "hh_annual_kwh": float(total.to_numpy().sum() - ghd_part.to_numpy().sum()),
        "ghd_annual_kwh": float(ghd_part.to_numpy().sum()),
        **metrics,
        "flagged": bool(reasons),
        "reason": ";".join(reasons),
    }


def run_status_quo_test(jobs: list[dict[str, Any]], *, processes: int = 1) -> pd.DataFrame:
    """Evaluate every job (one real grid each); one row per grid."""
    if processes > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=processes) as pool:
            rows = list(pool.map(_evaluate_job, jobs))
    else:
        rows = [_evaluate_job(job) for job in jobs]
    return pd.DataFrame(rows).sort_values(["provider", "real_grid_number"]).reset_index(drop=True)


GHD_CATEGORIES = ("Commercial", "Public")


def households_only(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop the GHD components of every job: the GHD-independent data filter.

    A real grid that already fails under its households' status-quo electricity is a
    data problem whatever the commercial load model assumes.
    """
    for job in jobs:
        plan = job["plan"]
        job["plan"] = plan[~plan["component_category"].isin(GHD_CATEGORIES)].copy()
    return jobs


def exclusion_list(grids: pd.DataFrame) -> pd.DataFrame:
    """``provider, real_grid_id, reason`` of the flagged grids."""
    flagged = grids[grids["flagged"]]
    return flagged[["provider", "real_grid_id", "reason"]].reset_index(drop=True)


# ---------------------------------------------------------------------------------------
# Job preparation
# ---------------------------------------------------------------------------------------


def jobs_from_buildings(
    provider: str,
    mapping: pd.DataFrame,
    buildings: pd.DataFrame,
    config: ghd.GhdConfig,
    evidence: pd.DataFrame | None,
    *,
    profile_seed: int,
    peak_hours_count: int,
    all_hours: bool,
    grids: list[int] | None = None,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """One job per real grid from bundle mapping and pylovo-equivalent building rows."""
    physical = buildings.merge(
        mapping[["building_objectid", "real_grid_id", "real_bus"]].rename(columns={"building_objectid": "objectid"}),
        on="objectid", how="inner", validate="one_to_one",
    )
    physical["bus"] = physical["real_bus"].astype(int)
    jobs, audits = [], []
    for number, group in physical.groupby("real_grid_id"):
        if grids is not None and int(number) not in grids:
            continue
        rows = group.drop(columns=["real_grid_id", "real_bus"]).reset_index(drop=True)
        plan, audit = component_plan(rows, config, evidence, profile_seed=profile_seed)
        audit.insert(0, "real_grid_id", bundle_grid_id(provider, number))
        audits.append(audit)
        grid_file = mapping.loc[mapping["real_grid_id"].eq(number), "real_grid_file"].iloc[0]
        jobs.append({
            "row": {"provider": provider, "real_grid_id": bundle_grid_id(provider, number),
                    "real_grid_number": int(number), "buildings": int(len(rows))},
            "plan": plan, "grid_file": str(grid_file), "profile_seed": int(profile_seed),
            "peak_hours": int(peak_hours_count), "all_hours": bool(all_hours),
        })
    audit = pd.concat(audits, ignore_index=True) if audits else pd.DataFrame(columns=["real_grid_id", *ghd.AUDIT_COLUMNS])
    return jobs, audit


def jobs_from_dataset(
    dataset: Path,
    provider: str,
    *,
    profile_seed: int,
    peak_hours_count: int,
    all_hours: bool,
    grids: list[int] | None = None,
) -> list[dict[str, Any]]:
    """One job per real grid from a prepared paired dataset (its persisted component plan)."""
    components = pd.read_csv(Path(dataset) / "paired_component_scenario_plan.csv")
    allocation = pd.read_csv(Path(dataset) / "paired_real_bus_allocation_plan.csv")
    if "provider" in allocation and not allocation["provider"].eq(provider).all():
        raise ValueError(f"{dataset} is not a {provider} dataset.")
    seeds = pd.to_numeric(components["profile_seed"], errors="coerce").dropna().astype(int).unique()
    if len(seeds) == 1 and int(seeds[0]) != int(profile_seed):
        raise ValueError(f"{dataset} was prepared with profile seed {int(seeds[0])}, not {profile_seed}.")
    jobs = []
    for number, group in components.groupby("real_target_grid_id"):
        if grids is not None and int(number) not in grids:
            continue
        plan = group.assign(target_bus=group["real_target_bus"].astype(int))
        grid_file = allocation.loc[allocation["real_grid_id"].eq(number), "real_grid_file"].iloc[0]
        jobs.append({
            "row": {"provider": provider, "real_grid_id": bundle_grid_id(provider, number),
                    "real_grid_number": int(number), "buildings": int(group["building_objectid"].nunique())},
            "plan": plan, "grid_file": str(grid_file), "profile_seed": int(profile_seed),
            "peak_hours": int(peak_hours_count), "all_hours": bool(all_hours),
        })
    return jobs


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gridexpand status-quo-test", description=__doc__.split("\n\n")[0])
    parser.add_argument("--provider", choices=PROVIDERS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--alignment-dir", type=Path, help="Alignment bundle root (<dir>/<provider>/building-mapping.json).")
    source.add_argument("--dataset", type=Path, help="Prepared paired dataset directory (regression: its persisted demand).")
    parser.add_argument("--uzw-grids-dir", type=Path, default=None, help="Frozen ÜZW delivery (ÜZW bundle input).")
    parser.add_argument("--scenario-config", type=Path, default=None, help="Scenario YAML whose ghd: block applies (bundle input).")
    parser.add_argument("--db-env-prefix", default="DB_", help="Environment prefix of the InfDB connection (default DB_).")
    parser.add_argument("--basedata-schema", default="basedata")
    parser.add_argument("--profile-seed", type=int, default=DEFAULT_PROFILE_SEED)
    parser.add_argument("--peak-hours", type=int, default=60, help="Largest total hours evaluated per grid (default 60).")
    parser.add_argument("--all-hours", action="store_true", help="Evaluate all 8760 hours (slow).")
    parser.add_argument("--grids", default=None, help="Comma-separated real grid numbers (default: all).")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel grid processes.")
    parser.add_argument(
        "--households-only", action="store_true",
        help="Household electricity only (GHD components dropped): the GHD-independent data filter.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    grids = None if args.grids is None else [int(value) for value in args.grids.split(",") if value.strip()]
    if args.peak_hours <= 0:
        raise SystemExit("--peak-hours must be positive.")
    metadata: dict[str, Any] = {
        "provider": args.provider, "profile_seed": int(args.profile_seed), "peak_hours": int(args.peak_hours),
        "all_hours": bool(args.all_hours), "limits": {"loading_percent": LOADING_LIMIT_PERCENT},
    }
    audit = None
    if args.dataset is not None:
        jobs = jobs_from_dataset(
            args.dataset, args.provider, profile_seed=args.profile_seed,
            peak_hours_count=args.peak_hours, all_hours=args.all_hours, grids=grids,
        )
        metadata["input"] = {"dataset": str(args.dataset)}
    else:
        if args.scenario_config is None:
            raise SystemExit("--scenario-config is required with --alignment-dir.")
        from gridexpand.scenario.config_loader import load_scenario_config

        scenario, scenario_hash = load_scenario_config(args.scenario_config)
        engine = read_only_engine(args.db_env_prefix)
        mapping = bundle_buildings(args.provider, args.alignment_dir, args.uzw_grids_dir)
        objectids = sorted(mapping["building_objectid"].astype(str).unique())
        buildings = pylovo_equivalent_buildings(read_basedata(engine, objectids, args.basedata_schema))
        evidence = ghd.load_evidence(scenario.ghd, objectids, engine=engine)
        jobs, audit = jobs_from_buildings(
            args.provider, mapping, buildings, scenario.ghd, evidence, profile_seed=args.profile_seed,
            peak_hours_count=args.peak_hours, all_hours=args.all_hours, grids=grids,
        )
        metadata["input"] = {
            "alignment_dir": str(args.alignment_dir), "basedata_schema": args.basedata_schema,
            "scenario_config": str(args.scenario_config), "scenario_hash": scenario_hash,
            "ghd": {"activity_gating": scenario.ghd.activity_gating,
                    "single_volume_one_storey": scenario.ghd.single_volume_one_storey,
                    "osm_levels": scenario.ghd.osm_levels, **ghd.summarize_ghd_audit(audit)},
        }
    if args.households_only:
        jobs = households_only(jobs)
    metadata["households_only"] = bool(args.households_only)
    result = run_status_quo_test(jobs, processes=max(1, int(args.jobs)))
    exclusions = exclusion_list(result)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / f"status_quo_{args.provider}_grids.csv", index=False)
    exclusions.to_csv(out / f"status_quo_{args.provider}_exclusions.csv", index=False)
    if audit is not None:
        audit.to_csv(out / f"status_quo_{args.provider}_ghd_audit.csv", index=False)
    metadata.update({"grids": int(len(result)), "flagged": int(len(exclusions)),
                     "reasons": {reason: int(result["reason"].str.contains(reason).sum()) for reason in REASONS}})
    (out / f"status_quo_{args.provider}_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(f"{args.provider}: {len(exclusions)} of {len(result)} real grids flagged -> {out}")
    for row in exclusions.itertuples(index=False):
        print(f"  {row.real_grid_id}: {row.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
