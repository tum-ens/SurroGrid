"""Database loaders and summary tables for synthetic/real power-flow comparisons."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sqlalchemy import text

GRIDEXPAND_DIR = Path(__file__).resolve().parents[2]
if str(GRIDEXPAND_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDEXPAND_DIR))

from common.database import SurroGridDatabase

try:
    from plotting.powerflow_heatmaps import (
        _normalize_optional_ags,
        _resolve_db_grid,
        _resolve_powerflow_run,
    )
except ImportError:
    from powerflow_heatmaps import (
        _normalize_optional_ags,
        _resolve_db_grid,
        _resolve_powerflow_run,
    )

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
                   WHERE gbc.included_in_lv
                     AND gbc.component_category = 'Residential'
               ) AS selected_household_load_rows,
               COUNT(DISTINCT gbc.bus) FILTER (
                   WHERE gbc.included_in_lv
                     AND gbc.component_category = 'Residential'
                     AND gbc.bus IS NOT NULL
               ) AS selected_household_load_buses,
               COALESCE(SUM(gbc.households) FILTER (
                   WHERE gbc.included_in_lv
                     AND gbc.component_category = 'Residential'
               ), 0) AS selected_household_equivalents,
               COUNT(*) FILTER (
                   WHERE gbc.included_in_lv
                     AND gbc.component_category IN ('Commercial', 'Public')
               ) AS non_household_load_rows,
               COUNT(DISTINCT gbc.bus) FILTER (
                   WHERE gbc.included_in_lv
                     AND gbc.component_category IN ('Commercial', 'Public')
                     AND gbc.bus IS NOT NULL
               ) AS non_household_load_buses
        FROM selected_runs sr
        LEFT JOIN surrogrid.grid_building_component gbc USING (grid_case_id)
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


def demand_component_exposure_summary_db(
    demand_allocation_run_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Summarize component demand, suppression, and mixed exposure by run.

    The returned table is derived from the compact component audit rather than
    hourly demand rows. It is therefore suitable for stratifying downstream
    power-flow and expansion summaries without assigning nonlinear losses back
    to individual components.
    """
    db = SurroGridDatabase()
    filters = ""
    params: dict[str, object] = {}
    if demand_allocation_run_ids is not None:
        ids = [int(value) for value in demand_allocation_run_ids]
        if not ids:
            return pd.DataFrame()
        filters = "WHERE dca.demand_allocation_run_id = ANY(:run_ids)"
        params["run_ids"] = ids

    query = text(
        f"""
        WITH audited AS (
            SELECT
                dca.demand_allocation_run_id,
                dar.grid_case_id,
                gc.ags,
                gc.plz,
                gc.kcid,
                gc.bcid,
                dca.objectid,
                dca.component_id,
                dca.category,
                dca.annual_energy_kwh,
                dca.included_in_lv,
                dca.mv_direct,
                dca.mix_confidence,
                gbc.effective_floor_area_m2
            FROM surrogrid.demand_component_audit dca
            JOIN surrogrid.demand_allocation_run dar
              USING (demand_allocation_run_id)
            JOIN surrogrid.grid_case gc USING (grid_case_id)
            JOIN surrogrid.grid_building_component gbc
              ON gbc.grid_case_id = dar.grid_case_id
             AND gbc.component_id = dca.component_id
            {filters}
        ), physical AS (
            SELECT
                demand_allocation_run_id,
                COUNT(DISTINCT objectid) AS run_physical_buildings,
                COUNT(DISTINCT objectid) FILTER (
                    WHERE category = 'Residential' AND included_in_lv
                ) AS run_residential_buildings,
                COUNT(DISTINCT objectid) FILTER (
                    WHERE category IN ('Commercial', 'Public') AND included_in_lv
                ) AS run_nonresidential_buildings,
                (
                    COUNT(DISTINCT objectid) FILTER (WHERE category = 'Residential') > 0
                    AND COUNT(DISTINCT objectid) FILTER (
                        WHERE category IN ('Commercial', 'Public')
                    ) > 0
                ) AS mixed_use_buildings_present,
                COUNT(DISTINCT objectid) FILTER (WHERE mv_direct) AS mv_direct_buildings,
                COALESCE(SUM(effective_floor_area_m2), 0.0) AS run_effective_component_area_m2,
                COALESCE(SUM(effective_floor_area_m2) FILTER (WHERE mv_direct), 0.0)
                    AS mv_direct_area_m2,
                COALESCE(SUM(annual_energy_kwh) FILTER (WHERE mv_direct), 0.0)
                    AS mv_direct_energy_kwh,
                COUNT(*) FILTER (
                    WHERE LOWER(COALESCE(mix_confidence, '')) = 'low'
                ) AS low_confidence_component_rows
            FROM audited
            GROUP BY demand_allocation_run_id
        )
        SELECT
            a.demand_allocation_run_id,
            MAX(a.grid_case_id) AS grid_case_id,
            MAX(a.ags) AS ags,
            MAX(a.plz) AS plz,
            MAX(a.kcid) AS kcid,
            MAX(a.bcid) AS bcid,
            a.category,
            COUNT(*) AS component_rows,
            COUNT(*) FILTER (WHERE a.included_in_lv) AS included_component_rows,
            COUNT(*) FILTER (WHERE NOT a.included_in_lv) AS suppressed_component_rows,
            COUNT(DISTINCT a.objectid) AS physical_buildings,
            COALESCE(SUM(a.effective_floor_area_m2) FILTER (WHERE a.included_in_lv), 0.0)
                AS included_effective_floor_area_m2,
            COALESCE(SUM(a.effective_floor_area_m2) FILTER (WHERE NOT a.included_in_lv), 0.0)
                AS suppressed_effective_floor_area_m2,
            COALESCE(SUM(a.annual_energy_kwh) FILTER (WHERE a.included_in_lv), 0.0)
                AS included_annual_energy_kwh,
            COALESCE(SUM(a.annual_energy_kwh) FILTER (WHERE NOT a.included_in_lv), 0.0)
                AS suppressed_annual_energy_kwh,
            MAX(p.run_physical_buildings) AS run_physical_buildings,
            MAX(p.run_residential_buildings) AS run_residential_buildings,
            MAX(p.run_nonresidential_buildings) AS run_nonresidential_buildings,
            MAX(p.mixed_use_buildings_present::int) AS mixed_use_buildings_present,
            MAX(p.mv_direct_buildings) AS mv_direct_buildings,
            MAX(p.run_effective_component_area_m2) AS run_effective_component_area_m2,
            MAX(p.mv_direct_area_m2) AS mv_direct_area_m2,
            MAX(p.mv_direct_energy_kwh) AS mv_direct_energy_kwh,
            MAX(p.low_confidence_component_rows) AS low_confidence_component_rows
        FROM audited a
        JOIN physical p USING (demand_allocation_run_id)
        GROUP BY a.demand_allocation_run_id, a.category
        ORDER BY a.demand_allocation_run_id, a.category
        """
    )
    with db.engine.connect() as conn:
        result = pd.read_sql_query(query, conn, params=params)
    if result.empty:
        return result
    total = result.groupby("demand_allocation_run_id")[
        "included_annual_energy_kwh"
    ].transform("sum")
    result["included_energy_share"] = np.divide(
        result["included_annual_energy_kwh"],
        total,
        out=np.zeros(len(result), dtype=float),
        where=total.to_numpy(dtype=float) > 0.0,
    )
    area_total = result.groupby("demand_allocation_run_id")[
        "run_effective_component_area_m2"
    ].transform("max")
    result["included_area_share"] = np.divide(
        result["included_effective_floor_area_m2"],
        area_total,
        out=np.zeros(len(result), dtype=float),
        where=area_total.to_numpy(dtype=float) > 0.0,
    )
    result["mv_direct_area_share"] = np.divide(
        result["mv_direct_area_m2"],
        area_total,
        out=np.zeros(len(result), dtype=float),
        where=area_total.to_numpy(dtype=float) > 0.0,
    )
    total_energy = (
        result["included_annual_energy_kwh"]
        + result["suppressed_annual_energy_kwh"]
    )
    result["mv_direct_energy_share"] = np.divide(
        result["mv_direct_energy_kwh"],
        total_energy,
        out=np.zeros(len(result), dtype=float),
        where=total_energy.to_numpy(dtype=float) > 0.0,
    )
    return result

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

